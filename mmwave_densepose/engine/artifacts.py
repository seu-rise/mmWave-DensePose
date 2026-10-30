"""Shared, standard-library-only user run naming and artifact path allocation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def artifact_roots(root=PROJECT_ROOT):
    root = Path(root).resolve()
    package = root / "mmwave_densepose"
    return {"checkpoints": package / "checkpoints", "train": package / "logs/train",
            "test": package / "logs/test", "runtime": root / ".local/runtime"}


def _component(value, label="run name"):
    value = str(value)
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,139}", value)
            or value.endswith(".")
            or value.split(".", 1)[0].upper() in {"CON", "PRN", "AUX", "NUL",
               *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}):
        raise ValueError(f"Invalid {label}: use a short alphanumeric name without path separators")
    return value


def _positive(value, name, integer=True):
    number = float(value)
    if not math.isfinite(number) or number <= 0 or (integer and not number.is_integer()):
        raise ValueError(f"{name} must be positive{' and integer' if integer else ''}")
    return int(number) if integer else number


def _learning_rate(value):
    value = _positive(value, "learning_rate", integer=False)
    if value < .001 or value >= 1000:
        mantissa, exponent = f"{value:.7e}".split("e")
        return mantissa.rstrip("0").rstrip(".") + "e" + str(int(exponent))
    return f"{value:.8g}"


def base_run_name(config):
    """Readable major parameters; exact settings still come from config.json."""
    completion = config.get("model") == "PCCCompletion"
    split = Path(str(config.get("split_strategy") or config.get("split_manifest", "pcc_sequence_s42" if completion else "chunk_s42"))).stem
    split = re.sub(r"_s\d+$", "", split.removeprefix("pcc_"))
    split = re.sub(r"[^A-Za-z0-9_-]", "-", split).strip("-_")
    fields = [split or "split", f"t{_positive(config.get('seq_len', 1 if completion else 5), 'seq_len')}",
              f"p{_positive(config.get('num_points', 80 if completion else 300), 'num_points')}",
              f"d{_positive(config.get('input_dim', 3 if completion else 5), 'input_dim')}"]
    if completion:
        fields.append(f"o{_positive(config.get('num_out_points', 196), 'num_out_points')}")
    seed = float(config.get("seed", 42))
    if not math.isfinite(seed) or seed < 0 or not seed.is_integer():
        raise ValueError("seed must be a nonnegative integer")
    fields.extend([f"e{_positive(config.get('epochs', 100 if completion else 200), 'epochs')}",
                   f"b{_positive(config.get('batch_size', 128), 'batch_size')}",
                   "lr" + _learning_rate(config.get("learning_rate", 5e-4)), f"s{int(seed)}"])
    if config.get("model") == "PASTNet":
        fields.append(str(config.get("ablation", "full")))
    return _component("_".join(fields))


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def checkpoint_log_dir(checkpoint, root=PROJECT_ROOT):
    """Locate the separate training log for a canonical user checkpoint."""
    if not checkpoint:
        return None
    root = Path(root).resolve()
    path = Path(checkpoint)
    path = (path if path.is_absolute() else root / path).resolve()
    try:
        relative = path.relative_to(artifact_roots(root)["checkpoints"])
    except ValueError:
        return None
    if len(relative.parts) != 3:
        return None
    model, run_id, _filename = relative.parts
    return artifact_roots(root)["train"] / _component(model, "model") / _component(run_id)


def _official_training_log(checkpoint, config, root):
    """Resolve a prepared log without confusing its folder marker with run identity."""
    from mmwave_densepose.workflows.catalog import official_asset_for_log
    try:
        relative = checkpoint.relative_to(root / "paper/experiments")
    except ValueError:
        return None
    if len(relative.parts) != 3 or relative.parts[1] != "checkpoints":
        return None
    train_root = artifact_roots(root)["train"] / _component(config["model"], "model")
    for metadata in sorted(train_root.glob("*/paper-training.json")):
        if metadata.is_symlink() or metadata.parent.is_symlink() or not _inside(metadata, train_root):
            continue
        try:
            document = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(document, dict) or document.get("asset_id") != relative.parts[0]:
            continue
        asset = official_asset_for_log(metadata.parent, root, phase="train", metadata=document)
        if not asset or asset.get("model") != config["model"]:
            continue
        asset_checkpoint = Path(str(asset.get("path", "")))
        if (asset_checkpoint if asset_checkpoint.is_absolute() else root / asset_checkpoint).resolve() != checkpoint:
            continue
        identity = document.get("training_run_id") or document.get("run_id")
        if identity:
            return metadata.parent, _component(identity)
    return None


def training_identity(checkpoint, config, root=PROJECT_ROOT):
    """Get the selected weight's training identity, never the test batch settings."""
    root = Path(root).resolve()
    path = Path(checkpoint)
    path = (path if path.is_absolute() else root / path).resolve()
    log = checkpoint_log_dir(path, root)
    if log is not None:
        if log.parent.name != config["model"]:
            raise ValueError("Checkpoint directory model does not match the selected model")
        return log.name
    official = _official_training_log(path, config, root)
    if official is not None:
        return official[1]
    try:
        relative = path.relative_to(root / "paper/experiments")
        if len(relative.parts) == 3 and relative.parts[1] == "checkpoints":
            return _component("official_" + relative.parts[0])
    except ValueError:
        pass
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(block)
    return "weights_" + digest.hexdigest()[:12]


def resolve_run_paths(action, config, *, checkpoint=None, output=None, run_id=None,
                      root=PROJECT_ROOT, reserve=False):
    """Preview or atomically reserve one run, without modifying official assets.

    Explicit output overrides select a new directory in the correct model/run
    log tree. Default training allocations version the whole log/checkpoint pair.
    Validation/test allocations reuse the selected checkpoint's training identity.
    """
    if action not in ("train", "validate", "evaluate", "infer"):
        raise ValueError("Unknown artifact action")
    root = Path(root).resolve()
    roots = artifact_roots(root)
    model = _component(config["model"], "model")
    training_log = None
    if action == "train":
        name = _component(run_id) if run_id else base_run_name(config)
        parent = roots["train"] / model
        prefix = None
    else:
        if not checkpoint:
            raise ValueError("Validation/testing requires a checkpoint")
        name = training_identity(checkpoint, config, root)
        checkpoint_file = Path(checkpoint)
        checkpoint_file = (checkpoint_file if checkpoint_file.is_absolute() else root / checkpoint_file).resolve()
        official = _official_training_log(checkpoint_file, config, root)
        training_log = official[0] if official else roots["train"] / model / name
        parent = (training_log / "validation" if action == "validate"
                  else roots["test"] / model / name)
        prefix = {"validate": "val", "evaluate": "test", "infer": "predict"}[action]
    if not _inside(parent, root):
        raise ValueError("Artifact directory must stay inside the selected repository")
    explicit = output is not None and str(output).strip() != ""
    if explicit:
        log = Path(output)
        log = (log if log.is_absolute() else root / log).resolve()
        if log.parent != parent.resolve():
            raise ValueError(f"Output directory must be a direct child of {parent}")
        _component(log.name)
        if action == "train":
            name = log.name
        elif not re.fullmatch(prefix + r"_\d{3,}", log.name):
            raise ValueError(f"Output directory name must use {prefix}_001 numbering")
    number = 1
    while True:
        if not explicit:
            suffix = "" if number == 1 else f"_r{number:02d}"
            log = parent / (name + suffix if action == "train" else f"{prefix}_{number:03d}")
        identity = log.name if action == "train" else name
        checkpoint_dir = roots["checkpoints"] / model / identity if action == "train" else None
        if checkpoint_dir is not None and not _inside(checkpoint_dir, root):
            raise ValueError("Checkpoint directory must stay inside the selected repository")
        checkpoint_path = checkpoint_dir / "best.ckpt" if checkpoint_dir else Path(checkpoint)
        # A prepared official record keeps its numbered test identity even when
        # its physical folder ends in _official. Never reuse that same number.
        occupied = (log.exists() or (action == "evaluate" and log.with_name(log.name + "_official").exists())
                    or (checkpoint_dir is not None and checkpoint_dir.exists()))
        if occupied:
            if explicit:
                raise FileExistsError(f"Output directory already exists: {log}. Choose a new run directory.")
            number += 1
            continue
        if reserve:
            log.parent.mkdir(parents=True, exist_ok=True)
            try:
                log.mkdir()
            except FileExistsError:
                if explicit:
                    raise
                number += 1
                continue
            if checkpoint_dir is not None:
                checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
                try:
                    checkpoint_dir.mkdir()
                except FileExistsError:
                    log.rmdir()  # Only remove the empty directory this call created.
                    if explicit:
                        raise
                    number += 1
                    continue
        return {"log_dir": log, "checkpoint_dir": checkpoint_dir,
                "checkpoint_path": checkpoint_path, "run_id": identity,
                "test_id": log.name if action == "evaluate" else None,
                "training_log_dir": training_log or roots["train"] / model / identity}
