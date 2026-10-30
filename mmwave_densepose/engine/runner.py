"""One implementation for direct CLI calls and paper reproduction scripts."""

import argparse
import csv
import hashlib
import json
import random
import time
import os
import pickle
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from mmwave_densepose.data import PoseDataset
from mmwave_densepose.metrics import PoseMetrics
from mmwave_densepose.models import build_model, list_models
from .completion import chamfer_loss, chamfer_per_cloud
from .run_logging import RunLogger, hyperparameters, legacy_scalars, write_joint_metrics
from .artifacts import PROJECT_ROOT, resolve_run_paths
from .execution import current_execution, observe_run
from .terminal_progress import TerminalProgress
from .run_config import default_run_config, save_run_config
from .devices import canonical_gpu_ids, validate_gpu_selection
from mmwave_densepose.data.partitions import normalize_data_config, ensure_partition
from mmwave_densepose.data.installation_state import ensure_complete_dataset


def task_for(config):
    """Completion is a private training task of PCC, never a pose substitute."""
    expected = "completion" if config["model"] == "PCCCompletion" else "pose"
    if config.get("task", expected) != expected:
        raise ValueError(f"{config['model']} requires task={expected!r}")
    if expected == "completion":
        if int(config.get("input_dim", 3)) != 3 or int(config.get("seq_len", 1)) != 1:
            raise ValueError("PCCCompletion requires single-frame XYZ input (input_dim=3, seq_len=1)")
        if config.get("dense_label_mode", "pcc") != "pcc":
            raise ValueError("Only PCC sphere supervision is supported for completion")
    return expected


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The config must contain a mapping")
    if config.get("model") not in list_models():
        raise ValueError(f"model must be a public PascalCase identifier: {list_models()}")
    task_for(config)
    return config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_weights(model, path, expected_name):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    name = checkpoint.get("model_name", checkpoint.get("model"))
    if name and name != expected_name:
        raise ValueError(f"Checkpoint model {name!r} does not match {expected_name!r}")
    weights = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(weights, strict=True)
    return checkpoint


def make_loader(config, split, limit_batches=None):
    completion = task_for(config) == "completion"
    dataset_type, extra = PoseDataset, {}
    if completion:
        from mmwave_densepose.data.completion import CompletionDataset
        dataset_type = CompletionDataset
        extra["pcc_label_radius"] = float(config.get("pcc_label_radius", .05))
    if "split_strategy" in config:
        observer = current_execution()
        extra["prepared"] = ensure_partition(config.get("data_root", "data/mmwave_densepose"), config,
                                             progress=observer.update if observer else None)
    dataset = dataset_type(config.get("data_root", "data/mmwave_densepose"),
                           config.get("split_manifest", "pcc_sequence_s42" if completion else "chunk_s42"),
                           split=split, num_points=int(config.get("num_points", 80 if completion else 300)),
                           input_dim=int(config.get("input_dim", 3 if completion else 5)),
                           seq_len=int(config.get("seq_len", 1 if completion else 5)),
                           cache_size=int(config.get("cache_size", 141)),
                           seed=int(config.get("seed", 42)), **extra)
    if len(dataset) == 0:
        raise ValueError(f"The {split} split is empty")
    batch_size = int(config.get("batch_size", 128))
    if batch_size < 1 or (limit_batches is not None and limit_batches < 1):
        raise ValueError("batch_size and limit_batches must be positive")
    original = dataset
    if limit_batches:
        dataset = Subset(dataset, range(min(len(dataset), limit_batches * batch_size)))
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                        num_workers=int(config.get("num_workers", 0)),
                        worker_init_fn=seed_worker, generator=generator,
                        pin_memory=str(config.get("device", "cpu")).startswith("cuda"),
                        drop_last=False)
    return loader, original


def model_forward(model, points, parallel_model=None, **kwargs):
    """Use the base module for tiny training tails that would make BN shards of one."""
    predictor = parallel_model if parallel_model is not None else model
    device_count = len(getattr(parallel_model, "device_ids", ()))
    if model.training and device_count > 1:
        # Tensor chunks can have a final shard of one even when N >= 2*GPUs
        # (for example seven samples on three devices).
        if any(len(chunk) < 2 for chunk in points.chunk(device_count)):
            predictor = model
    return predictor(points, **kwargs)


def forward_loss(model, points, target, parallel_model=None):
    if model.task_type == "dense":
        pred = model_forward(model, points, parallel_model)
        return pred, chamfer_loss(pred, target)
    loss_fn = torch.nn.MSELoss()
    if hasattr(model, "compute_target_loss"):
        outputs = model_forward(model, points, parallel_model, return_dict=True)
        batch_data = {"pose_target": target}
        loss = model.compute_target_loss(outputs, batch_data, loss_fn)
        auxiliary = model.compute_auxiliary_losses(outputs, batch_data, loss_fn)
        return outputs.get("pred_visual_target", outputs["pred_target"]), loss + sum(auxiliary.values())
    pred = model_forward(model, points, parallel_model)
    if isinstance(pred, dict):
        pred = pred["pred_target"]
    return pred, loss_fn(pred, target)


@torch.no_grad()
def evaluate_model(model, loader, device, prediction_dir=None, visual_records=None, progress_label=None, parallel_model=None):
    model.eval()
    completion = model.task_type == "dense"
    supported = getattr(model, "supported_joints", None)
    metrics = None if completion else PoseMetrics(supported)
    cloud_rows = []
    loss_sum, count, prediction_rows = 0.0, 0, []
    started = time.perf_counter()
    observer = current_execution()
    last_progress = 0.0
    label = progress_label or "Evaluation"
    with TerminalProgress(len(loader), label) as terminal:
        for step, batch in enumerate(loader):
            points, target = batch["points"].to(device), batch["target"].to(device)
            if completion:
                pred = model_forward(model, points, parallel_model)
                values = chamfer_per_cloud(pred, target)
                loss = values.mean()
                cloud_rows.extend({"sample_id": sample_id, "cd_m": float(value), "cd_cm": float(value) * 100}
                                  for sample_id, value in zip(batch["sample_id"], values))
            else:
                pred, loss = forward_loss(model, points, target, parallel_model)
                metrics.update(pred, target)
                if visual_records is not None:
                    # XYZ is already retained in model input space; the final
                    # frame aligns with sample_id's end-1 target frame.
                    visual_records["input"].append(points[:, -1, :, :3].detach().float().cpu().numpy())
                    visual_records["output"].append(pred.detach().float().cpu().numpy())
                    visual_records["gt"].append(target.detach().float().cpu().numpy())
                    visual_records["sample_id"].extend(str(item) for item in batch["sample_id"])
            loss_sum += float(loss) * len(pred)
            count += len(pred)
            now = time.monotonic()
            if observer and (now - last_progress >= 1 or step + 1 == len(loader)):
                update = {"progress": {"kind": "samples", "current": count, "total": len(loader.dataset)}}
                if observer.action != "train":
                    live_metrics = ({"cd_m": loss_sum / count, "cd_cm": loss_sum / count * 100}
                                    if completion else metrics.result()[0])
                    update["metrics"] = {**live_metrics, "loss": loss_sum / count}
                observer.update(**update)
                last_progress = now
            if prediction_dir is not None:
                prediction_dir.mkdir(parents=True, exist_ok=True)
                filename = f"batch_{step:06d}.npz"
                np.savez_compressed(prediction_dir / filename, prediction=pred.cpu().numpy(),
                                    target=target.cpu().numpy(), sample_id=np.asarray(batch["sample_id"]))
                prediction_rows.append({"file": filename, "samples": len(pred)})
            terminal.update(step + 1, loss=loss_sum / count)
            if not terminal.dynamic and step % 50 == 0:
                print(f"Evaluated {count} samples", flush=True)
        if completion:
            overall, per_joint = {"samples": count, "cd_m": loss_sum / count,
                                  "cd_cm": loss_sum / count * 100}, cloud_rows
        else:
            overall, per_joint = metrics.result()
        overall.update(loss=loss_sum / count, elapsed_seconds=time.perf_counter() - started)
        if observer and observer.action != "train":
            observer.update(metrics=overall)
        if prediction_dir is not None:
            write_json(prediction_dir / "manifest.json", {"format": "mmwave-densepose-completion-predictions-v1"
                       if completion else "mmwave-densepose-predictions-v1",
                       "batches": prediction_rows, "samples": count})
        terminal.set_metrics(**({"loss": overall["loss"], "cd": overall["cd_m"]} if completion else
                               {"loss": overall["loss"], "mle": overall["mle_m"], "pck": overall["pck"]}))
        return overall, per_joint


def save_metrics(directory, overall, per_joint):
    write_json(directory / "metrics.json", overall)
    if "cd_m" not in overall:
        write_joint_metrics(directory, overall, per_joint)
        return
    with (directory / "per_cloud_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_joint[0]))
        writer.writeheader()
        writer.writerows(per_joint)


def _public_path(path, root):
    if path is None:
        return None
    path = Path(path).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _validate_training_settings(config):
    if int(config.get("batch_size", 128)) < 2:
        raise ValueError("Training BatchNorm needs at least two samples; adjust batch_size/data")
    if config.get("optimizer", "adam").lower() != "adam":
        raise ValueError("This release currently supports the paper Adam optimizer")
    if int(config.get("epochs", 200)) < 1:
        raise ValueError("epochs must be positive")
    for key, default, positive in (("learning_rate", 5e-4, True), ("weight_decay", 1e-5, False)):
        value = float(config.get(key, default))
        if not np.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"Invalid {key}")
    schedule = config.get("scheduler_name", "cosine")
    if schedule not in ("cosine", "step", "none"):
        raise ValueError("scheduler_name must be cosine, step or none")
    if schedule == "cosine":
        eta = float(config.get("scheduler_eta_min", 1e-6))
        if not np.isfinite(eta) or eta < 0:
            raise ValueError("scheduler_eta_min must be nonnegative and finite")
    if schedule == "step":
        if int(config.get("scheduler_step_size", 30)) < 1:
            raise ValueError("scheduler_step_size must be positive")
        gamma = float(config.get("scheduler_gamma", .1))
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("scheduler_gamma must be positive and finite")


@observe_run
def run(action, config, output=None, checkpoint=None, limit_batches=None, save_predictions=False, allow_partial=False,
        visual=False, artifact_root=None):
    if action not in ("train", "validate", "evaluate", "infer"):
        raise ValueError("action must be train, validate, evaluate or infer")
    if visual and action != "evaluate":
        raise ValueError("Visual test PKL export is available for evaluate only")
    config = dict(config)
    save_predictions = bool(save_predictions or action == "infer")
    # Legacy numeric fixtures can still address their explicit test indexes.
    # Public CLI/GUI always resolve a model + strategy + seed combination.
    if "split_strategy" in config or "split_manifest" not in config:
        config = normalize_data_config(config)
    ensure_complete_dataset(config.get("data_root", "data/mmwave_densepose"))
    root = Path(artifact_root or PROJECT_ROOT).resolve()
    workbench = config.pop("_workbench", {})
    public_data_root = config.pop("_public_data_root", None)
    task = task_for(config)
    completion = task == "completion"
    if visual and completion:
        raise ValueError("PCCCompletion does not produce pose visualization PKLs")
    config = dict(config, task=task)
    if completion:
        for key, value in {"num_points": 80, "num_out_points": 196, "input_dim": 3,
                           "seq_len": 1}.items():
            config.setdefault(key, value)
    completion_checkpoint = config.get("completion_checkpoint")
    if config["model"] != "GlobalLocalNetwork" and completion_checkpoint:
        raise ValueError("completion_checkpoint is only accepted for GlobalLocalNetwork")
    if config["model"] == "GlobalLocalNetwork":
        if config.get("pcc_cache_file"):
            raise ValueError("GlobalLocalNetwork requires its PCC branch; pcc_cache_file is not supported")
        if action == "train" and not completion_checkpoint:
            raise ValueError("GlobalLocalNetwork training requires an explicit completion_checkpoint "
                             "in the config or --completion-checkpoint PATH")
    if completion_checkpoint:
        completion_checkpoint = str(Path(completion_checkpoint).resolve())
        if not Path(completion_checkpoint).is_file():
            raise FileNotFoundError(f"Missing explicit PCC completion checkpoint: {completion_checkpoint}")
        config["completion_checkpoint"] = completion_checkpoint
    seed = int(config.get("seed", 42))
    seed_all(seed)
    device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = canonical_gpu_ids(device_name, config.get("gpu_ids"), action=action)
    if gpu_ids:
        device_name = f"cuda:{gpu_ids[0]}"
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    gpu_ids = validate_gpu_selection(action, device_name, gpu_ids, config.get("batch_size", 128))
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(int(config.get("torch_threads", 4)))
    if action == "train":
        _validate_training_settings(config)
    config = dict(config, device=str(device), gpu_ids=gpu_ids)
    checkpoint = checkpoint if action == "train" else (checkpoint or config.get("checkpoint"))
    config_reference = save_run_config(
        action, dict(config, checkpoint=checkpoint), root=root,
        visual=visual, save_predictions=save_predictions or action == "infer",
        allow_partial=allow_partial, limit_batches=limit_batches)
    observer = current_execution()
    if observer:
        observer.update(config={**config, "data_root": public_data_root or config.get("data_root", "data/mmwave_densepose")},
                        phase="initializing", limit_batches=limit_batches, visual=bool(visual),
                        allow_partial=bool(allow_partial), save_predictions=bool(save_predictions),
                        progress={"kind": "indeterminate", "current": None, "total": None},
                        **config_reference)
    # Validate checkpoint compatibility before dataset/cache I/O or creating output.
    model_config = dict(config)
    model_config.pop("completion_checkpoint", None)
    model = build_model(config["model"], model_config)
    if action != "train" and not checkpoint:
        raise ValueError("Validation/testing requires --checkpoint or checkpoint in the config")
    # Check artifact boundaries before dataset/cache reads, without creating a run.
    resolve_run_paths(action, config, checkpoint=checkpoint, output=output, root=root)
    loaded = load_weights(model, checkpoint, config["model"]) if checkpoint else {}
    if completion_checkpoint:
        existing = {k: v.clone() for k, v in model.model.pcc.state_dict().items()} if action != "train" else None
        model.load_completion_checkpoint(completion_checkpoint)
        if existing is not None and any(not torch.equal(existing[k], v)
                                        for k, v in model.model.pcc.state_dict().items()):
            raise ValueError("Explicit PCC weights conflict with the complete pose checkpoint; "
                             "omit completion_checkpoint for evaluation or use matching weights")
    completion_provenance = None
    if completion_checkpoint:
        completion_provenance = {"path": completion_checkpoint, "sha256": sha256(completion_checkpoint)}
    elif config["model"] == "GlobalLocalNetwork":
        completion_provenance = loaded.get("completion_checkpoint")
    model.to(device)
    parallel_model = (torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
                      if len(gpu_ids) > 1 else None)
    if observer:
        observer.update(phase="preparing-cache")
    split = "train" if action == "train" else ("val" if action == "validate" else "test")
    loader, dataset = make_loader(config, split, limit_batches)
    dataset_manifest = Path(config.get("data_root", "data/mmwave_densepose")) / "manifest.json"
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    partial = bool(manifest.get("partial", False) or dataset.metadata.get("partial", False))
    if partial and not allow_partial:
        raise ValueError("Partial fixture dataset: explicitly pass --allow-partial for smoke workflows")
    val_loader = make_loader(config, "val", limit_batches)[0] if action == "train" else None
    paths = resolve_run_paths(action, config, checkpoint=checkpoint, output=output, root=root, reserve=True)
    output = paths["log_dir"]
    output_checkpoint = paths["checkpoint_path"]
    if observer:
        observer.attach_paths(paths)
        observer.update(partial=partial, sample_count=len(dataset))
    logged_config = dict(config)
    logged_config["data_root"] = public_data_root or config.get("data_root", "data/mmwave_densepose")
    logged_completion = dict(completion_provenance) if completion_provenance else None
    if logged_completion and logged_completion.get("path"):
        logged_completion["path"] = _public_path(logged_completion["path"], root)
    checkpoint_logged = _public_path(output_checkpoint if action == "train" else checkpoint, root)
    run_info = {"action": action, "task": task, "origin": "user", "model": config["model"], "config": logged_config, "seed": seed,
                **config_reference,
                "run_id": paths["run_id"], "log_dir": _public_path(output, root),
                "weight_id": paths["run_id"] if action == "train" else (workbench.get("weight_id") or paths["run_id"]),
                "test_id": paths["test_id"],
                "training_run_id": paths["run_id"], "visual": bool(visual),
                "split": split, "split_manifest": config.get("split_manifest"),
                "split_strategy": config.get("split_strategy", dataset.metadata.get("split_strategy")),
                "split_index_sha256": dataset.metadata.get("index_sha256"),
                "partition_cache_id": dataset.cache_identity,
                "dataset_manifest_sha256": sha256(dataset_manifest), "split_metadata": dataset.metadata,
                "training_cache_identity": dataset.cache_identity,
                "checkpoint": checkpoint_logged, "checkpoint_sha256": sha256(checkpoint) if checkpoint and action != "train" else None,
                "initial_checkpoint": _public_path(checkpoint, root) if action == "train" else None,
                "initial_checkpoint_sha256": sha256(checkpoint) if checkpoint and action == "train" else None,
                "completion_checkpoint": logged_completion,
                "partial": partial, "limit_batches": limit_batches, "torch": str(torch.__version__),
                "cuda": torch.version.cuda, "device": str(device), "gpu_ids": gpu_ids,
                "num_devices": len(gpu_ids) if gpu_ids else (1 if device.type == "cuda" else 0)}
    if observer:
        run_info.update(observer.metadata())
    write_json(output / "run.json", run_info)
    write_json(output / "config.json", logged_config)
    write_json(output / "config-reference.json", config_reference)
    with RunLogger(output, hyperparameters(logged_config, run_info, output)) as run_logger:
        print(f"Run configuration: {config_reference['config_file']}", flush=True)
        if action == "train":
            if config.get("optimizer", "adam").lower() != "adam":
                raise ValueError("This release currently supports the paper Adam optimizer")
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                         lr=float(config.get("learning_rate", 5e-4)),
                                         weight_decay=float(config.get("weight_decay", 1e-5)))
            epochs = int(config.get("epochs", 200))
            if epochs < 1:
                raise ValueError("epochs must be positive")
            print(f"Training {config['model']} on {device}: {epochs} epochs", flush=True)
            print(f"Training logs: {output}", flush=True)
            print(f"Best checkpoint: {output_checkpoint}", flush=True)
            scheduler = None
            schedule = config.get("scheduler_name", "cosine")
            if schedule == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=float(config.get("scheduler_eta_min", 1e-6)))
            elif schedule == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(config.get("scheduler_step_size", 30)),
                                                            gamma=float(config.get("scheduler_gamma", 0.1)))
            elif schedule != "none":
                raise ValueError("scheduler_name must be cosine, step or none")
            best, history = float("inf"), []
            metric = "cd_m" if completion else "mle_m"
            validation_key, best_key = "val_" + metric, "best_val_" + metric
            step = 0
            for epoch in range(epochs):
                last_progress = 0.0
                if observer:
                    observer.update(phase="training", progress={"kind": "batch", "current": 0,
                                    "total": len(loader), "epoch": epoch + 1, "total_epochs": epochs})
                model.train()
                total, count = 0.0, 0
                train_metrics = None if completion else PoseMetrics(getattr(model, "supported_joints", None))
                with TerminalProgress(len(loader), f"Epoch {epoch + 1}/{epochs} Training") as terminal:
                    for batch_index, batch in enumerate(loader):
                        points, target = batch["points"].to(device), batch["target"].to(device)
                        if points.shape[0] == 1:
                            raise ValueError("Training BatchNorm needs at least two samples; adjust batch_size/data")
                        optimizer.zero_grad(set_to_none=True)
                        prediction, loss = forward_loss(model, points, target, parallel_model)
                        if not torch.isfinite(loss):
                            raise ValueError("Non-finite training loss")
                        loss.backward()
                        optimizer.step()
                        total += float(loss.detach()) * len(points)
                        count += len(points)
                        step += 1
                        now = time.monotonic()
                        if observer and (now - last_progress >= 1 or batch_index + 1 == len(loader)):
                            observer.update(metrics={"train_loss": total / count},
                                            progress={"kind": "batch", "current": batch_index + 1,
                                            "total": len(loader), "epoch": epoch + 1, "total_epochs": epochs,
                                            "train_loss": total / count})
                            last_progress = now
                        if train_metrics is not None:
                            train_metrics.update(prediction.detach(), target.detach())
                        terminal.update(batch_index + 1, loss=total / count, lr=optimizer.param_groups[0]["lr"])
                if observer:
                    observer.update(phase="validating")
                validation, _ = evaluate_model(model, val_loader, device,
                                               progress_label=f"Epoch {epoch + 1}/{epochs} Validation", parallel_model=parallel_model)
                row = {"epoch": epoch + 1, "train_loss": total / count, validation_key: validation[metric]}
                training = {"loss": total / count, "cd_m": total / count} if completion else train_metrics.result()[0]
                training["loss"] = total / count
                scalar_row = {**legacy_scalars(training), **legacy_scalars(validation, "val_"),
                              "lr-Adam": optimizer.param_groups[0]["lr"]}
                run_logger.log(scalar_row, epoch=epoch, step=step)
                history.append(row)
                print(json.dumps(row), flush=True)
                if observer:
                    observer.update(metrics={"train_loss": total / count, "val_metric": validation[metric]},
                                    progress={"kind": "epoch", "current": epoch + 1, "total": epochs,
                                    "epoch": epoch + 1, "total_epochs": epochs,
                                    "train_loss": total / count, "val_metric": validation[metric]})
                if validation[metric] < best:
                    best = validation[metric]
                    temporary = output_checkpoint.with_name(f".best.{uuid.uuid4().hex}.tmp")
                    torch.save({"model_name": config["model"], "task": task, "config": logged_config,
                                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                "completion_checkpoint": completion_provenance,
                                "run_id": paths["run_id"], "training_run_id": paths["run_id"],
                                "log_dir": _public_path(output, root),
                                "epoch": epoch + 1, validation_key: best}, temporary)
                    temporary.replace(output_checkpoint)
                if scheduler:
                    scheduler.step()
                write_json(output / "history.json", history)
            if observer:
                observer.update(phase="exporting")
            write_json(output / "training_summary.json", {"task": task, "epochs": epochs, best_key: best,
                       "best_checkpoint": checkpoint_logged, "selection_split": "val", "test_used_for_selection": False})
            run_info["checkpoint_sha256"] = sha256(output_checkpoint)
            write_json(output / "run.json", run_info)
            return {best_key: best, "checkpoint": str(output_checkpoint), "log_dir": str(output), "run_id": paths["run_id"]}
        predictions = output / "predictions" if save_predictions or action == "infer" else None
        if observer:
            observer.update(phase="validating" if action == "validate" else "evaluating")
        visual_records = {"input": [], "output": [], "gt": [], "sample_id": []} if visual else None
        overall, per_joint = evaluate_model(model, loader, device, predictions, visual_records,
            progress_label={"validate": "Validation", "evaluate": "Testing", "infer": "Predicting"}[action])
        expected_samples = len(dataset)
        evaluation_complete = overall["samples"] == expected_samples and not partial
        overall.update(model=config["model"], task=task, split=split, dataset_partial=partial,
                       partial=not evaluation_complete, evaluation_complete=evaluation_complete,
                       expected_samples=expected_samples, evaluated_samples=overall["samples"],
                       limit_batches=limit_batches, log_dir=str(output), run_id=paths["run_id"], test_id=paths["test_id"])
        if observer:
            observer.update(phase="exporting")
        save_metrics(output, overall, per_joint)
        scalar_prefix = "val_" if action == "validate" else "test_" if action == "evaluate" else "predict_"
        run_logger.log(legacy_scalars(overall, scalar_prefix), epoch=0, step=len(loader))
        if visual:
            visual_path = output / "test_results.pkl"
            temporary = output / f".test_results.{uuid.uuid4().hex}.tmp"
            payload = {"input": np.concatenate(visual_records["input"], axis=0),
                       "output": np.concatenate(visual_records["output"], axis=0),
                       "gt": np.concatenate(visual_records["gt"], axis=0),
                       "sample_id": np.asarray(visual_records["sample_id"], dtype=np.str_),
                       "meta": {"format": "mmwave-densepose-test-visual-v1",
                                "model": config["model"], "run_id": run_info["run_id"],
                                "training_run_id": run_info["training_run_id"],
                                "weight_id": run_info["weight_id"], "test_id": run_info["test_id"],
                                "dataset_manifest_sha256": run_info["dataset_manifest_sha256"],
                                "training_cache_identity": run_info["training_cache_identity"],
                                "split_manifest": run_info["split_manifest"],
                                "split_strategy": run_info["split_strategy"],
                                "split_index_sha256": run_info["split_index_sha256"],
                                "partition_cache_id": run_info["partition_cache_id"],
                                "sample_count": overall["samples"], "joints": overall["joints"],
                                "metric_joint_indices": getattr(model, "supported_joints", tuple(range(19))),
                                "input_layout": "target-frame XYZ [samples, points, 3]",
                                "sample_id_layout": "sequence_id:start:end; target frame=end-1",
                                "coordinate_frame": "stored_world_xyz"}}
            try:
                with temporary.open("xb") as stream:
                    pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(visual_path)
            finally:
                if temporary.exists(): temporary.unlink()
            overall["visual_pkl"] = "test_results.pkl"
            overall["visual_pkl_sha256"] = sha256(visual_path)
            run_info.update(visual=True, visual_pkl=overall["visual_pkl"],
                            visual_pkl_sha256=overall["visual_pkl_sha256"])
            write_json(output / "run.json", run_info)
            write_json(output / "metrics.json", overall)
        print(json.dumps(overall, indent=2), flush=True)
        return overall


def main(action, argv=None):
    parser = argparse.ArgumentParser(description=f"{action.title()} a paper pose model or PCC completion")
    parser.add_argument("--config", help="Optional JSON/YAML configuration; explicit CLI values override it")
    parser.add_argument("--model", choices=list_models())
    parser.add_argument("--split-strategy", choices=("sequence_chunk", "subject"),
                        help="Partition protocol; model and seed determine the generated cache")
    parser.add_argument("--partition", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--split-chunk-size", type=int)
    parser.add_argument("--max-timestamp-gap-ms", type=float)
    parser.add_argument("--min-valid-points", type=int)
    parser.add_argument("--num-points", type=int)
    parser.add_argument("--input-dim", type=int, choices=(3, 4, 5))
    parser.add_argument("--num-joints", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--optimizer", choices=("adam",))
    parser.add_argument("--ablation", choices=("full", "spatial_only", "baseline"))
    parser.add_argument("--scheduler-name", choices=("cosine", "step", "none"))
    parser.add_argument("--scheduler-eta-min", type=float)
    parser.add_argument("--scheduler-step-size", type=int)
    parser.add_argument("--scheduler-gamma", type=float)
    parser.add_argument("--num-out-points", type=int)
    parser.add_argument("--pcc-label-radius", type=float)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--completion-checkpoint", help="Explicit PCC weights required for GlobalLocalNetwork training")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--gpu-ids", nargs="+", type=int, help="Visible CUDA indices; training may use multiple GPUs, evaluation/prediction use one")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--cache-size", type=int)
    parser.add_argument("--dense-label-mode", choices=("pcc",))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visual", action=argparse.BooleanOptionalAction, default=None, help="Export test-frame inputs, predictions and ground truth to test_results.pkl")
    parser.add_argument("--allow-partial", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    if args.list_models:
        print("\n".join(list_models()))
        return 0
    if not args.config and not args.model:
        parser.error("Without --config, --model is required")
    try:
        if args.config:
            config = load_config(args.config)
        else:
            config = default_run_config(args.model, split_strategy=args.split_strategy or "sequence_chunk")
        # A shared YAML can be launched directly; explicit flags still win.
        options = {}
        for name, default in (("visual", False), ("save_predictions", False),
                              ("allow_partial", False), ("limit_batches", None)):
            configured = config.pop(name, default)
            options[name] = getattr(args, name) if getattr(args, name) is not None else configured
        for key in ("model", "split_strategy", "partition", "split_chunk_size", "max_timestamp_gap_ms",
                    "min_valid_points", "num_points", "input_dim", "num_joints",
                    "seq_len", "seed", "learning_rate", "weight_decay", "optimizer", "ablation",
                    "scheduler_name", "scheduler_eta_min", "scheduler_step_size", "scheduler_gamma",
                    "num_out_points", "pcc_label_radius", "data_root", "device", "gpu_ids", "batch_size",
                    "num_workers", "torch_threads", "cache_size", "dense_label_mode", "epochs", "completion_checkpoint"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        config = normalize_data_config(config)
        workbench_metadata = os.environ.get("MMWAVE_WORKBENCH_METADATA")
        if workbench_metadata:
            workbench = json.loads(workbench_metadata)
            if not isinstance(workbench, dict):
                raise ValueError("Workbench metadata must be an object")
            config["_workbench"] = workbench
        if os.environ.get("MMWAVE_PUBLIC_DATA_ROOT"):
            config["_public_data_root"] = os.environ["MMWAVE_PUBLIC_DATA_ROOT"]
        checkpoint = args.checkpoint
        if checkpoint is None and config.get("action") == action:
            checkpoint = config.get("checkpoint")
        result = run(action, config, args.output_dir, checkpoint, options["limit_batches"],
                     options["save_predictions"], options["allow_partial"], options["visual"])
        if action == "train":
            print(json.dumps(result, indent=2), flush=True)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0
