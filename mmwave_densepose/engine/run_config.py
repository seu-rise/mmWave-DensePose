"""Complete reusable run snapshots and the explicit CLI view of their settings."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile

import yaml

from mmwave_densepose.data.partitions import _lock, normalize_data_config
from .devices import canonical_gpu_ids


_ACTIONS = {"train", "validate", "evaluate", "infer"}
_RUN_FIELDS = {
    "config_file", "config_id", "config_sha256", "source_config", "source_config_file",
    "config_path", "configuration_file", "run_id", "training_run_id", "job_id", "test_id",
    "weight_id", "output", "output_dir", "log_dir", "log_directory", "execution_id",
    "report_id", "request_id", "origin", "checkpoint_sha256", "initial_checkpoint_sha256",
}
_PATH_FIELDS = {"data_root", "checkpoint", "completion_checkpoint"}
_INTEGER_FIELDS = {
    "batch_size", "epochs", "num_workers", "torch_threads", "cache_size",
    "scheduler_step_size", "num_out_points",
}
_FLOAT_FIELDS = {
    "learning_rate", "weight_decay", "scheduler_eta_min", "scheduler_gamma",
    "pcc_label_radius",
}
_CLI_FIELDS = (
    "model", "data_root", "checkpoint", "split_strategy", "seed", "partition",
    "split_chunk_size", "max_timestamp_gap_ms", "min_valid_points", "num_points",
    "input_dim", "num_joints", "seq_len", "num_out_points", "pcc_label_radius",
    "dense_label_mode", "completion_checkpoint", "device", "gpu_ids", "batch_size", "num_workers",
    "torch_threads", "cache_size", "epochs", "learning_rate", "weight_decay", "optimizer",
    "ablation", "scheduler_name", "scheduler_eta_min", "scheduler_step_size", "scheduler_gamma",
)
_OPTIONS = {"visual": False, "save_predictions": False, "allow_partial": False, "limit_batches": None}


def default_run_config(model, *, device=None, split_strategy="sequence_chunk"):
    """Resolve the CLI's model-aware defaults without reading experiment files."""
    if device is None:
        device = "auto"
    completion = model == "PCCCompletion"
    config = {
        "model": model, "data_root": "data/mmwave_densepose", "split_strategy": split_strategy,
        "num_points": 80 if completion else 300, "input_dim": 3 if completion else 5,
        "num_joints": 19, "seed": 42, "batch_size": 128,
        "epochs": 100 if completion else 200, "learning_rate": .0005,
        "weight_decay": .00001, "optimizer": "adam", "ablation": "full",
        "scheduler_name": "cosine", "scheduler_eta_min": .000001,
        "scheduler_step_size": 30, "scheduler_gamma": .1, "num_workers": 0,
        "torch_threads": 4, "cache_size": 141, "device": str(device), "gpu_ids": [],
        "checkpoint": None, "completion_checkpoint": None,
    }
    if completion:
        config.update(num_out_points=196, pcc_label_radius=.05, dense_label_mode="pcc")
    return normalize_data_config(config)


def _plain(value):
    """Canonicalize equivalent integer/float inputs and reject opaque values."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Run configuration numbers must be finite")
        return int(value) if value.is_integer() else value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise ValueError(f"Unsupported run configuration value: {type(value).__name__}")


def _public_path(value, root):
    if value is None or value == "":
        return None
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else resolved.as_posix()


def prepare_run_config(action, config, *, root, visual=False, save_predictions=False,
                       allow_partial=False, limit_batches=None):
    """Return all public settings; allocated run paths and identities stay in logs."""
    if action not in _ACTIONS:
        raise ValueError("Unknown run action")
    root = Path(root).resolve()
    provided = {key: value for key, value in dict(config).items()
                if not str(key).startswith("_") and key not in _RUN_FIELDS and key not in _OPTIONS}
    model = provided.get("model")
    result = default_run_config(model, split_strategy=provided.get("split_strategy", "sequence_chunk"))
    result.update(provided)
    result["gpu_ids"] = canonical_gpu_ids(result.get("device", "auto"), result.get("gpu_ids"), action=action)
    if result["gpu_ids"]:
        result["device"] = f"cuda:{result['gpu_ids'][0]}"
    if "split_manifest" in provided:
        # Internal prepared-cache fixtures carry their actual cached sequence
        # shape. Recording them must not turn them into a public partition run.
        if "split_strategy" not in provided:
            result.pop("split_strategy", None)
        if "data_family" not in provided:
            result.pop("data_family", None)
    else:
        result = normalize_data_config(result)
    for key in _INTEGER_FIELDS & result.keys():
        value = result[key]
        if value is None:
            continue
        number = float(value)
        if isinstance(value, bool) or not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{key} must be an integer")
        result[key] = int(number)
    for key in _FLOAT_FIELDS & result.keys():
        if result[key] is not None:
            result[key] = float(result[key])
    for key in _PATH_FIELDS:
        result[key] = _public_path(result.get(key), root)
    if limit_batches in (None, 0, "", "0"):
        limit_batches = None
    else:
        number = float(limit_batches)
        if isinstance(limit_batches, bool) or not number.is_integer() or number < 1:
            raise ValueError("limit_batches must be a positive integer")
        limit_batches = int(number)
    for name, value in (("visual", visual), ("save_predictions", save_predictions), ("allow_partial", allow_partial)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false")
    result.update(action=action, visual=visual, save_predictions=save_predictions or action == "infer",
                  allow_partial=allow_partial, limit_batches=limit_batches)
    return _plain(result)


def _baseline(config, *, device=None):
    # Subject's default chunk is None, but choosing subject is itself an override.
    baseline = default_run_config(config["model"], device=device,
                                  split_strategy=config.get("split_strategy", "sequence_chunk"))
    baseline["split_strategy"] = "sequence_chunk"
    baseline.update(_OPTIONS)
    baseline.pop("model")
    return _plain(baseline)


def _snapshot(action, config, **options):
    full = prepare_run_config(action, config, **options)
    # Snapshot bytes must remain identical when a configuration is moved between
    # GPU and CPU machines. Store the resolved device explicitly; automatic
    # device selection is the portable default for grouping these YAML entries.
    baseline = _baseline(full, device="auto")
    ordered = ["model", "action", *[key for key in _CLI_FIELDS if key != "model"], *sorted(full)]
    keys = list(dict.fromkeys(key for key in ordered if key in full))
    changed = {key: full[key] for key in keys if key not in baseline or full[key] != baseline[key]}
    defaults = {key: full[key] for key in keys if key in baseline and full[key] == baseline[key]}
    rendered = ("# Non-default parameters\n" + yaml.safe_dump(changed, sort_keys=False, allow_unicode=True)
                + "\n# Default parameters\n" + yaml.safe_dump(defaults, sort_keys=False, allow_unicode=True))
    canonical = json.dumps(full, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf8")
    identity = hashlib.sha256(canonical).hexdigest()
    raw = rendered.encode("utf8")
    reference = {"config_file": f"configs/runs/{full['model']}/{action}-{identity}.yaml",
                 "config_id": identity, "config_sha256": hashlib.sha256(raw).hexdigest()}
    return reference, raw


def describe_run_config(action, config, *, root, visual=False, save_predictions=False,
                        allow_partial=False, limit_batches=None):
    """Describe the exact snapshot filename and hashes without writing a file."""
    return _snapshot(action, config, root=root, visual=visual, save_predictions=save_predictions,
                     allow_partial=allow_partial, limit_batches=limit_batches)[0]


def _not_linked(path):
    info = path.lstat()
    return not (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400)


def save_run_config(action, config, *, root, visual=False, save_predictions=False,
                    allow_partial=False, limit_batches=None):
    """Atomically publish an immutable snapshot, reusing an identical one."""
    root = Path(root).resolve()
    reference, raw = _snapshot(action, config, root=root, visual=visual, save_predictions=save_predictions,
                               allow_partial=allow_partial, limit_batches=limit_batches)
    destination = root / reference["config_file"]
    current = root
    for part in destination.relative_to(root).parts[:-1]:
        current = current / part
        current.mkdir(exist_ok=True)
        if not _not_linked(current) or not current.is_dir() or not current.resolve().is_relative_to(root):
            raise ValueError("Run configuration storage must not contain linked paths")
    def verify_existing():
        if not _not_linked(destination) or not destination.is_file() or destination.read_bytes() != raw:
            raise ValueError("Stored run configuration was changed; refusing to overwrite it")
    if destination.exists() or destination.is_symlink():
        verify_existing()
        return reference
    lock = destination.parent / '.write.lock'
    if lock.exists() and not _not_linked(lock):
        raise ValueError("Run configuration storage must not contain linked paths")
    with _lock(lock):
        if destination.exists() or destination.is_symlink():
            verify_existing()
            return reference
        descriptor, temporary = tempfile.mkstemp(prefix=".run-config-", suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            # The existing cache lock utility serializes processes and threads;
            # publish a complete file only after checking for an existing snapshot.
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return reference


def explicit_run_arguments(action, config, *, root, output_dir=None, visual=False,
                           save_predictions=False, allow_partial=False, limit_batches=None):
    """Build flags from the same settings as snapshots, omitting actual defaults."""
    full = prepare_run_config(action, config, root=root, visual=visual, save_predictions=save_predictions,
                              allow_partial=allow_partial, limit_batches=limit_batches)
    baseline = _baseline(full)
    unsupported = set(full) - set(_CLI_FIELDS) - set(_OPTIONS) - {"action", "task", "data_family"}
    if unsupported:
        raise ValueError("Explicit CLI does not support parameters: " + ", ".join(sorted(unsupported)))
    arguments = []
    for key in _CLI_FIELDS:
        if key not in full or key in baseline and full[key] == baseline[key]:
            continue
        value = full[key]
        if value is None:
            continue
        arguments.append("--" + key.replace("_", "-"))
        arguments.extend(str(item) for item in value) if isinstance(value, list) else arguments.append(str(value))
    if output_dir is not None:
        arguments.extend(["--output-dir", str(output_dir)])
    for key in ("visual", "save_predictions", "allow_partial"):
        if full[key]:
            arguments.append("--" + key.replace("_", "-"))
    if full["limit_batches"] is not None:
        arguments.extend(["--limit-batches", str(full["limit_batches"])])
    return arguments


__all__ = ["default_run_config", "prepare_run_config", "describe_run_config", "save_run_config", "explicit_run_arguments"]
