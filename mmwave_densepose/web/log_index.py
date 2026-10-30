"""Path-bounded index and recoverable removal for local experiment logs.

New runner JSON/CSV files take precedence over legacy filenames and hparams.
TensorBoard event files are detected, but scalar payloads are not fabricated
when TensorBoard's optional event reader is unavailable.
"""
from __future__ import annotations

import ast
from bisect import bisect_right
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import uuid
from typing import Any, Iterable, Mapping
from mmwave_densepose.workflows.catalog import official_asset_for_log

RUN_MODELS = (
    "GlobalLocalNetwork", "DualPathCNNBiLSTM", "PointTransformer", "PointNet2Baseline",
    "PoseGraphNet", "PartTokenFormer", "PartSkeletonSpatialNet", "PntPoseAT",
    "MMPoseFK", "PointMLP", "DGCNN", "PASTNet", "PointNet",
)
_TIMESTAMP_SUFFIX = re.compile(r"_(\d{8})_(\d{6})$")
_TEMP_PARTS = {"__pycache__", ".git", ".local", ".tmp", "tmp", "cache", "caches"}
_TRANSIENT = {"last.ckpt", "latest", ".DS_Store"}
_METADATA_FILES = {"run.json", "history.json", "training_summary.json", "metrics.json",
                   "metrics.csv", "per_joint_metrics.csv", "per_joint_mle.csv", "per_cloud_metrics.csv",
                   "hparams.yaml", "hparams.yml", "paper-training.json", "events.out.tfevents"}
_NUMERIC_CSV_LIMIT = 32 * 1024 * 1024
_JSON_LIMIT = 16 * 1024 * 1024
_DEFAULT_RUN_LIMIT = 100
_MAX_RUN_LIMIT = 500
_MAX_SERIES_POINTS = 5000
_TAG_ALIASES = {
    "train_loss": "train/loss", "loss": "train/loss",
    "val_mle": "val/MLE (cm)", "val_mle_m": "val/MLE (cm)", "val_mle_cm": "val/MLE (cm)", "val/MLE (cm)": "val/MLE (cm)",
    "val_pa_mpjpe": "val/PA-MPJPE (cm)", "val_pa_mpjpe_m": "val/PA-MPJPE (cm)", "val_pa_mpjpe_cm": "val/PA-MPJPE (cm)",
    "val_pck": "val/PCK@15cm (%)", "val_pck_percent": "val/PCK@15cm (%)",
    "val_auc": "val/AUC@15cm (%)", "val_auc_percent": "val/AUC@15cm (%)",
    "test_mle": "test/MLE (cm)", "test_mle_m": "test/MLE (cm)", "test_mle_cm": "test/MLE (cm)",
    "test_pa_mpjpe": "test/PA-MPJPE (cm)", "test_pa_mpjpe_m": "test/PA-MPJPE (cm)",
    "test_pa_mpjpe_cm": "test/PA-MPJPE (cm)", "test_pck": "test/PCK@15cm (%)",
    "test_pck_percent": "test/PCK@15cm (%)", "test_auc": "test/AUC@15cm (%)",
    "test_auc_percent": "test/AUC@15cm (%)", "mle_cm": "MLE (cm)",
    "pa_mpjpe_cm": "PA-MPJPE (cm)", "pck_percent": "PCK@15cm (%)",
    "auc_percent": "AUC@15cm (%)", "cd_cm": "Chamfer distance (cm)",
    "mle": "MLE (cm)", "mle_m": "MLE (cm)", "pa_mpjpe": "PA-MPJPE (cm)",
    "pa_mpjpe_m": "PA-MPJPE (cm)", "pck": "PCK@15cm (%)", "auc": "AUC@15cm (%)",
}


def default_roots(workspace_root: str | Path | None = None) -> dict[str, Path]:
    """Return the repository's training-log directory."""
    root = Path(workspace_root).resolve() if workspace_root is not None else Path(__file__).resolve().parents[2]
    roots: dict[str, Path] = {}
    roots["training_logs"] = root / "mmwave_densepose" / "logs" / "train"
    return roots


def _paper_table_map(workspace_root: Path) -> dict[str, list[str]]:
    """Read the literal experiment table map without importing or executing run.py."""
    source = workspace_root / "paper" / "run.py"
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                names = ([n.id for n in node.targets if isinstance(n, ast.Name)]
                         if isinstance(node, ast.Assign) else [node.target.id] if isinstance(node.target, ast.Name) else [])
                if "TABLES" in names:
                    value = node.value
                    data = ast.literal_eval(value)
                    return {str(k): list(map(str, v)) for k, v in data.items() if isinstance(v, (list, tuple))}
    except (OSError, SyntaxError, ValueError, TypeError, MemoryError):
        return {}
    return {}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_transient_dir(name: str) -> bool:
    part = name.casefold()
    return part in _TEMP_PARTS or part.startswith((".tmp-", "tmp-", "smoke-"))


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        if path.stat().st_size > _JSON_LIMIT:
            return None
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, (dict, list)) else None
    except (OSError, ValueError, UnicodeError, RecursionError):
        return None


def _read_scalar_config(path: Path) -> dict[str, Any]:
    """Read ordinary YAML scalar hparams without executing tags or adding a YAML dependency."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return {}
    result: dict[str, Any] = {}
    for line in text.splitlines():
        if not line or line.lstrip().startswith(("#", "-", "!")) or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip().strip("\"'")
        raw = raw.split(" #", 1)[0].strip()
        if not key or not raw or raw.startswith(("{", "[", "!", "&", "*")):
            continue
        raw = raw.strip("\"'")
        low = raw.casefold()
        if low in ("true", "false"):
            value: Any = low == "true"
        elif low in ("none", "null", "~"):
            value = None
        else:
            try:
                value = float(raw) if any(c in raw for c in ".eE") else int(raw)
            except ValueError:
                value = raw
        result.setdefault(key, value)
    return result


def _public_model(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if value in RUN_MODELS:
        return value
    folded = value.casefold().replace("_", "").replace("-", "").replace(" ", "")
    return next((name for name in RUN_MODELS
                 if name.casefold().replace("_", "").replace("-", "") == folded), value)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def _numeric_csv_rows(path: Path, limit: int = _NUMERIC_CSV_LIMIT) -> tuple[list[str], list[dict[str, Any]]]:
    """Stream scalar CSV values; malformed rows are skipped without aborting the run index."""
    try:
        if path.stat().st_size > limit:
            return [], []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = [str(c) for c in (reader.fieldnames or [])]
            for index, row in enumerate(reader):
                if index >= 2_000_000:
                    break
                values: dict[str, Any] = {}
                for name, value in row.items():
                    if name is None or value is None:
                        continue
                    try:
                        number = float(value.strip())
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(number):
                        values[name] = number
                if values:
                    rows.append(values)
        return columns, rows
    except (OSError, UnicodeError, csv.Error, ValueError, MemoryError):
        return [], []


def _new_run_info(config: dict[str, Any] | None, hparams: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(hparams or {})
    if isinstance(config, dict):
        merged.update(config)
    model = _public_model(merged.get("model", merged.get("model_name")))
    action = merged.get("action", merged.get("task"))
    if action not in ("train", "validate", "evaluate", "infer", "predict", "test"):
        action = None
    split = merged.get("split_strategy") or merged.get("split_manifest") or merged.get("split", merged.get("data_split"))
    if not isinstance(split, str):
        split = None
    seed = merged.get("seed")
    try:
        seed = int(seed) if seed is not None else None
    except (ValueError, TypeError, OverflowError):
        seed = None
    return {
        "model": model, "action": action, "split": split, "seed": seed,
        "num_points": _safe_meta_number(merged.get("num_points", merged.get("num_in_points"))),
        "input_dim": _safe_meta_number(merged.get("input_dim", merged.get("feature_dim"))),
        "seq_len": _safe_meta_number(merged.get("seq_len", merged.get("t_max"))),
        "batch_size": _safe_meta_number(merged.get("batch_size")),
        "ablation": merged.get("ablation") if isinstance(merged.get("ablation"), str) else None,
        "paper_tables": sorted(set(map(str, merged.get("paper_tables", [])))) if isinstance(merged.get("paper_tables"), list) else [],
        "config": _safe_config({k: v for k, v in merged.items()
                                if isinstance(v, (str, int, float, bool, type(None)))
                                or k == "partition" and isinstance(v, list)
                                and all(isinstance(n, (int, float)) and math.isfinite(n) for n in v)}),
    }

def _safe_config(config: dict[str, Any]) -> dict[str, Any]:
    """Keep useful values while omitting absolute host-specific paths from browser responses."""
    result: dict[str, Any] = {}
    path_names = {"data_root", "dataset_root", "root", "output_dir", "checkpoint",
                  "completion_checkpoint", "pcc_cache_file", "cache_root", "index_file", "log_dir"}
    for key, value in config.items():
        if not isinstance(value, str):
            result[key] = value
            continue
        normalized = value.strip().replace("\\", "/")
        absolute = normalized.startswith("/") or bool(re.match(r"^[A-Za-z]:/", normalized)) or normalized.startswith("//")
        if not absolute:
            result[key] = value
        elif key.casefold() in {"data_root", "dataset_root", "root"}:
            result[key] = "<configured root>"
        elif key.casefold() in {"output_dir", "log_dir"}:
            result[key] = "<run output>"
        elif key.casefold() in path_names:
            result[key] = PurePosixPath(normalized).name or "<configured path>"
        else:
            result[key] = "<redacted path>"
    return result



def _safe_details(value):
    if isinstance(value, dict):
        return {str(key): _safe_details(item) if isinstance(item, (dict, list)) else
                _safe_config({str(key): item})[str(key)] for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_details(item) for item in value]
    return value


def _display_metadata(config, info, legacy, timestamp, status, relative):
    """Actual parameters first; readable directory tokens only fill absent keys."""
    aliases = {"epochs": ("epochs", "effective_max_epochs", "max_epochs"),
               "batch_size": ("batch_size",), "learning_rate": ("learning_rate", "lr"),
               "num_points": ("num_points", "radar_point_size", "num_in_points"),
               "input_dim": ("input_dim", "sample_dim", "feature_dim"),
               "seq_len": ("seq_len", "t_max", "t-max"),
               "variant": ("ablation", "ablation_mode", "variant")}
    result = {"time": timestamp, "status": status, "seed": info.get("seed"), "split": info.get("split")}
    stem = Path(relative).name
    tokens = dict(re.findall(r"(?:^|_)(t|p|d|e|b|lr|s)([^_]+)", stem))
    directory_keys = {"seq_len": "t", "num_points": "p", "input_dim": "d", "epochs": "e", "batch_size": "b", "learning_rate": "lr", "seed": "s"}
    for target, names in aliases.items():
        present = next((name for name in names if name in config), None)
        if present is not None:
            result[target] = config[present]
        elif directory_keys.get(target) in tokens:
            raw = tokens[directory_keys[target]]
            try:
                number = float(raw)
                result[target] = int(number) if number.is_integer() else number
            except ValueError:
                result[target] = None
        else:
            result[target] = None
    if "device" in config:
        result["device"] = config["device"]
    else:
        result["device"] = config.get("accelerator")
    selected_gpus = config.get("gpu_ids")
    result["num_devices"] = config.get("num_devices", config.get("devices"))
    if result["num_devices"] is None and isinstance(selected_gpus, list) and selected_gpus:
        result["num_devices"] = len(selected_gpus)
    if result["seed"] is None and "seed" not in config and "s" in tokens:
        try: result["seed"] = int(tokens["s"])
        except ValueError: pass
    if result["split"] is None and "split_manifest" not in config:
        split_token = stem.split("_", 1)[0]
        if split_token in ("chunk", "subject", "sequence", "intra"):
            result["split"] = split_token
    return result


def _safe_meta_number(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _status_from_artifacts(directory: Path, run: dict[str, Any], metrics: dict[str, Any] | None,
                           history: dict[str, Any] | None, training: dict[str, Any] | None) -> tuple[str, str]:
    if run.get("partial"):
        return "partial", "partial_dataset"
    limit = run.get("limit_batches")
    if limit not in (None, 0, False, ""):
        return "smoke", "limited_run"
    action = run.get("action")
    if action == "train":
        if training is not None:
            return "complete", "training_summary"
        configured = (run.get("config") or {}).get("epochs")
        epochs = history.get("history", history.get("epochs")) if isinstance(history, dict) else history if isinstance(history, list) else None
        if isinstance(configured, int) and isinstance(epochs, list):
            if len(epochs) >= configured:
                return "complete", "history"
        return "incomplete", "training_summary_missing"
    if metrics is not None:
        complete = metrics.get("evaluation_complete")
        expected = metrics.get("expected_samples")
        count = metrics.get("samples")
        if complete is True or (isinstance(expected, int) and expected > 0 and count == expected):
            return "complete", "metrics"
        if complete is False or (isinstance(count, int) and count >= 0):
            return "incomplete", "evaluation_incomplete"
    return "incomplete", "metrics_missing"


def _legacy_name(path: Path) -> dict[str, Any]:
    """Use the timestamp and known model suffix; never split a composite name on every underscore."""
    name = path.parent.name if path.name.startswith("version_") else path.name
    timestamp = _TIMESTAMP_SUFFIX.search(name)
    parsed_time = None
    prefix = name
    if timestamp:
        parsed_time = timestamp.group(1) + "_" + timestamp.group(2)
        prefix = name[:timestamp.start()]
    if prefix.startswith("test_"):
        prefix = prefix[5:]
    if prefix.startswith("predict_"):
        prefix = prefix[8:]
    model = None
    for candidate in sorted(RUN_MODELS, key=len, reverse=True):
        if prefix.endswith("_" + candidate) or prefix == candidate:
            model = candidate
            break
    actions = {"train": ("train",), "test": ("test", "evaluate"), "predict": ("predict", "infer")}
    task = next((task for directory, values in actions.items()
                 if f"/{directory}/" in "/" + path.as_posix().casefold()
                 for task in values), None)
    return {"model": model, "action": task, "timestamp": parsed_time,
            "legacy_name": name, "classification": "filename" if model or parsed_time or task else "unknown"}


def _parse_names(metric_rows: list[dict[str, Any]], *, x_axis: str, smoothing: float,
                 max_points: int, tag_aliases: dict[str, str] | None = None) -> dict[str, list[float | None]]:
    aliases = tag_aliases or {}
    result: dict[str, list[float | None]] = {}
    for row in metric_rows:
        for key, value in row.items():
            if key.casefold() in {"epoch", "step", "steps", "wall_time", "walltime", "time", "global_step"}:
                continue
            number = _number(value)
            if number is None:
                continue
            name = aliases.get(key, key)
            if name in ("val_mle_m", "mle_m", "mle"):
                number *= 100
                name = "val/MLE (cm)" if name == "val_mle_m" else "MLE (cm)"
            elif name in ("val_pa_mpjpe_m", "pa_mpjpe_m"):
                number *= 100
                name = "val/PA-MPJPE (cm)" if name == "val_pa_mpjpe_m" else "PA-MPJPE (cm)"
            elif name in ("pck", "pck_percent"):
                if name == "pck":
                    number *= 100
                name = "val/PCK@15cm (%)"
            elif name in ("auc", "auc_percent"):
                if name == "auc":
                    number *= 100
                name = "val/AUC@15cm (%)"
            elif name in ("train_loss", "loss"):
                name = "train/loss"
            result.setdefault(name, []).append(number)
    return result


def _lttb(points: list[tuple[float, float]], threshold: int) -> list[tuple[float, float]]:
    """Largest-triangle-three-buckets downsampling, preserving first/last samples."""
    length = len(points)
    if threshold >= length or threshold < 3:
        return points if threshold >= length else (points[:1] + points[-1:])
    sampled = [points[0]]
    every = (length - 2) / (threshold - 2)
    anchor = 0
    for i in range(threshold - 2):
        avg_start = int((i + 1) * every) + 1
        avg_end = min(int((i + 2) * every) + 1, length)
        next_bucket = points[avg_start:avg_end]
        avg_x = sum(p[0] for p in next_bucket) / max(1, len(next_bucket))
        avg_y = sum(p[1] for p in next_bucket) / max(1, len(next_bucket))
        range_start = int(i * every) + 1
        range_end = min(int((i + 1) * every) + 1, length - 1)
        avg_point = points[anchor]
        best, max_area, best_index = range_start, -1.0, range_start
        for j in range(range_start, max(range_start + 1, range_end)):
            point = points[j]
            area = abs((avg_point[0] - avg_x) * (point[1] - avg_point[1])
                       - (avg_point[0] - point[0]) * (avg_y - avg_point[1]))
            if area > max_area:
                best, max_area, best_index = point, area, j
        sampled.append(best)
        anchor = best_index
    sampled.append(points[-1])
    return sampled

import base64
import ast
import csv
import io
import threading
import time


class LogIndex:
    """Read allowed run folders and return stable, workspace-relative log records."""

    def __init__(self, roots: Mapping[str, str | Path] | None = None, *, workspace_root: str | Path | None = None, paper_experiments: Mapping[str, Iterable[str]] | None = None,
                 cache_seconds: float = 2.0, actions: Iterable[str] | None = None):
        if isinstance(actions, str):
            raise ValueError("invalid log action scope")
        self.actions = frozenset(actions) if actions is not None else None
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else None
        if self.actions is not None and not self.actions <= {"train", "validate", "evaluate", "infer"}:
            raise ValueError("invalid log action scope")
        if roots is None:
            roots = default_roots(workspace_root)
        if not isinstance(roots, Mapping):
            raise TypeError("roots must be a mapping of fixed source names to directories")
        self.roots: dict[str, Path] = {}
        for name, raw_path in roots.items():
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", name):
                raise ValueError("invalid configured log source")
            try:
                root = Path(raw_path).expanduser().resolve()
            except (OSError, TypeError, ValueError):
                continue
            if not root.exists() or root.is_dir():
                self.roots[name] = root
        if self.workspace_root is None:
            for root in self.roots.values():
                candidates = (root, *root.parents)
                self.workspace_root = next((candidate for candidate in candidates
                    if (candidate / 'paper/manifest.json').is_file()), None)
                if self.workspace_root is not None:
                    break
        self.paper_experiments = ({str(table): set(map(str, values)) for table, values in paper_experiments.items()}
                                  if paper_experiments is not None else {})
        if paper_experiments is None and workspace_root is not None:
            self.paper_experiments = {k: set(v) for k, v in _paper_table_map(Path(workspace_root)).items()}
        if paper_experiments is None and not self.paper_experiments:
            for root in self.roots.values():
                candidate = root.parent if root.name == "outputs" else root
                if (candidate / "paper" / "run.py").is_file():
                    self.paper_experiments = {k: set(v) for k, v in _paper_table_map(candidate).items()}
                    break
        self.cache_seconds = max(0.0, float(cache_seconds))
        self._cache: list[dict[str, Any]] = []
        self._cache_at = 0.0
        self._record_cache = {}
        self._lock = threading.RLock()
        self._native_executable = importlib.util.find_spec("tensorboard") is not None

    def _stable_id(self, source: str, relative: str) -> str:
        return hashlib.sha256((source + "\0" + relative).encode("utf-8")).hexdigest()[:32]

    def _resolve_id(self, run_id: Any) -> dict[str, Any]:
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("unknown run id")
        for record in self._records():
            if record["id"] == run_id:
                return record
        raise KeyError("run not found")

    def _register_directory(self, source: str, root: Path, directory: Path, *, legacy: bool) -> dict[str, Any] | None:
        try:
            resolved = directory.resolve(strict=True)
        except OSError:
            return None
        if not _inside(resolved, root) or not resolved.is_dir():
            return None
        rel = resolved.relative_to(root).as_posix()
        if not rel or any(_is_transient_dir(part) for part in Path(rel).parts) or (source == "training_logs" and any(part in ("validation", "predictions") for part in Path(rel).parts)):
            return None
        meta_path = resolved / "run.json"
        run = _read_json(meta_path) if meta_path.is_file() and not meta_path.is_symlink() else None
        hparams = None
        if not run:
            for filename in ("hparams.yaml", "hparams.yml"):
                candidate = resolved / filename
                if candidate.is_file() and not candidate.is_symlink():
                    hparams = _read_scalar_config(candidate)
                    break
        if not isinstance(run, dict):
            run = {}
        config = run.get("config", {}) if isinstance(run, dict) else {}
        if not isinstance(config, dict):
            config = {}
        merged = dict(config)
        merged.update({k: v for k, v in run.items() if k in (
            "model", "action", "task", "split", "split_manifest", "split_strategy", "seed", "partial", "limit_batches",
            "partition_cache_id", "split_index_sha256",
            "evaluation_complete", "expected_samples", "checkpoint", "checkpoint_sha256", "dataset_manifest_sha256",
            "training_cache_identity", "started_at", "finished_at", "test_id", "training_run_id", "weight_id",
            "paper_table", "record_type", "status", "returncode")})
        if hparams:
            for key, value in hparams.items():
                if key not in merged or merged[key] in (None, ""):
                    merged[key] = value
        paper_path = resolved / "paper-training.json"
        paper = (_read_json(paper_path) if paper_path.is_file()
                 and not paper_path.is_symlink() else None)
        official_asset = (official_asset_for_log(resolved, self.workspace_root, phase='train', metadata=paper)
                          if isinstance(paper, dict) and paper.get("schema_version") == 1
                          and self.workspace_root is not None else None)
        official = bool(official_asset)
        if official:
            actual_config = paper.get("config")
            if isinstance(actual_config, dict):
                merged.update(actual_config)
            merged.update({key: paper[key] for key in
                           ("model", "run_id", "training_run_id", "weight_id", "checkpoint", "paper_tables")
                           if key in paper})
            merged["action"] = "train"
        else:
            paper = {}
        new_info = _new_run_info(merged, hparams)
        files: dict[str, str] = {}
        sizes: dict[str, int] = {}
        for filename in ("run.json", "history.json", "training_summary.json", "metrics.json",
                         "metrics.csv", "per_joint_metrics.csv", "per_joint_mle.csv", "per_cloud_metrics.csv",
                         "hparams.yaml", "hparams.yml", "paper-training.json", "config-reference.json"):
            candidate = resolved / filename
            try:
                if candidate.is_file() and not candidate.is_symlink() and _inside(candidate.resolve(strict=True), root):
                    files[filename] = f"{rel}/{filename}"
                    sizes[filename] = candidate.stat().st_size
            except OSError:
                continue
        metric_csv = files.get("metrics.csv")
        if metric_csv is None:
            for child in sorted(resolved.glob("version_*")):
                try:
                    if child.is_dir() and not child.is_symlink() and _inside(child.resolve(strict=True), root):
                        candidate = child / "metrics.csv"
                        if candidate.is_file() and not candidate.is_symlink():
                            metric_csv = f"{rel}/{child.name}/metrics.csv"
                            files["metrics.csv"] = metric_csv
                            sizes["metrics.csv"] = candidate.stat().st_size
                            for filename in ("hparams.yaml", "hparams.yml"):
                                hp = child / filename
                                if hp.is_file() and not hp.is_symlink():
                                    files[filename] = f"{rel}/{child.name}/{filename}"
                                    if not hparams:
                                        hparams = _read_scalar_config(hp)
                            break
                except OSError:
                    continue
        event_files = self._events_in(resolved, root, rel)
        has_history = "history.json" in files
        has_metrics = "metrics.json" in files or metric_csv is not None
        has_training_summary = "training_summary.json" in files
        metrics = _read_json(root / files["metrics.json"]) if "metrics.json" in files else None
        history = _read_json(root / files["history.json"]) if has_history else None
        training = _read_json(root / files["training_summary.json"]) if has_training_summary else None
        if run:
            status, integrity = _status_from_artifacts(resolved, run, metrics, history, training)
        else:
            csv_columns, csv_rows = self._read_csv(root / metric_csv, keep_last=1) if metric_csv else ([], [])
            status, integrity = self._legacy_status(rel, csv_columns, csv_rows, bool(event_files))
            new_info = _new_run_info(merged, hparams)
            if paper and event_files and status == "incomplete":
                status, integrity = "recorded", "source_recorded"
        original_name = str(paper.get("training_run_id") or paper.get("run_id") or rel.rsplit("/", 1)[-1])
        legacy_info = _legacy_name(Path(original_name) if paper else resolved) if legacy or not run else {}
        model = new_info.get("model") or legacy_info.get("model")
        action = new_info.get("action") or legacy_info.get("action")
        if not run and action in ("test", "evaluate"):
            action = "evaluate"
        if action == "predict":
            action = "infer"
        if action is None and not run and source in ("official_train", "training_logs"):
            action = "train"
        split = new_info.get("split")
        seed = new_info.get("seed")
        timestamp = ((run.get("started_at") if run else None) or paper.get("started_at")
                     or paper.get("timestamp") or merged.get("created_at") or legacy_info.get("timestamp"))
        config_mapping = new_info.get("config", {})
        event_scalars = self._event_scalars(root, event_files) if event_files and not metric_csv and not has_history else {}
        tags = sorted(self._tags(metrics, history, root / metric_csv if metric_csv else None) |
                      {key for key in event_scalars if key not in {"epoch", "step", "global_step"}})
        paper_tables = set(map(str, official_asset.get("paper_tables", new_info.get("paper_tables", [])))) if official else set(new_info.get("paper_tables", []))
        dependency_tables = sorted(set(map(str, official_asset.get("paper_dependency_tables", paper.get("paper_dependency_tables", []))))) if official else []
        for table, experiments in self.paper_experiments.items():
            experiment = original_name
            if experiment in experiments:
                paper_tables.add(table)
        explicit_table = merged.get("paper_table")
        if isinstance(explicit_table, (str, int)):
            paper_tables.add(str(explicit_table))
        try:
            modified = max((root / value).stat().st_mtime for value in files.values()) if files else resolved.stat().st_mtime
        except OSError:
            modified = resolved.stat().st_mtime
        row = dict(run, **metrics) if run and metrics else run or {}
        record_id = self._stable_id(source, rel)
        return {
            "id": record_id, "run_id": str(merged.get("training_run_id") or merged.get("run_id") or original_name),
            "source": ("events" if event_files else "CSV" if metric_csv else "JSON"),
            "source_root": source, "origin": "official" if official else "user", "legacy": legacy or not bool(run), "original_name": original_name,
            "display_name": self._display_name(model, original_name, legacy_info),
            "action": action, "task": action, "model": model,
            "split": split, "seed": seed, "timestamp": str(timestamp) if timestamp else None,
            "status": status, "integrity": integrity,
            "partial": bool(run.get("partial", False)) if run else status in ("smoke", "partial"),
            "limited": bool(run.get("limit_batches")) if run else status == "smoke",
            "sample_count": self._row_samples(row, metrics, root / metric_csv if metric_csv else None),
            "joints": _safe_meta_number((metrics or {}).get("joints")) if metrics else None,
            "threshold_m": _safe_meta_number((metrics or {}).get("threshold_m")) if metrics else None,
            "metrics": self._public_metrics(metrics),
            "files": files, "sizes": sizes,
            "event_files": event_files, "has_events": bool(event_files),
            "paper_tables": sorted(paper_tables, key=lambda value: int(value) if str(value).isdigit() else 99), "paper_dependency_tables": dependency_tables, "tags": tags,
            "name_classification": "metadata" if run or hparams else legacy_info.get("classification", "unknown"),
            "display_metadata": _display_metadata(config_mapping, new_info, legacy_info, timestamp, status, rel),
            "asset_id": paper.get("asset_id"), "weight_id": merged.get("weight_id"),
            "checkpoint": merged.get("checkpoint"),
            "paper_configuration": _safe_details(paper.get("paper_configuration", {})),
            "protocol": split, "config": config_mapping,
            "checkpoint_sha256": merged.get("checkpoint_sha256"),
            "config_file": run.get("config_file"), "config_id": run.get("config_id"),
            "config_sha256": run.get("config_sha256"),
            "gpu_ids": run.get("gpu_ids", config.get("gpu_ids")), "num_devices": run.get("num_devices"),
            "dataset_identity": merged.get("dataset_manifest_sha256") or merged.get("training_cache_identity"),
            "_modified": modified,
            "_source_root": source,
            "_relative_directory": rel,
            "_metric_source": {"json": files.get("metrics.json"), "csv": metric_csv,
                               "history": files.get("history.json")},
            "_root_path": str(root), "_event_scalars": event_scalars,
        }

    @staticmethod
    def _display_name(model: str | None, original: str, legacy_info: dict[str, Any]) -> str:
        if model and legacy_info.get("timestamp"):
            return f"{model} · {legacy_info['timestamp']}"
        if model:
            return model
        return original

    @staticmethod
    def _row_samples(row: dict[str, Any], metrics: dict[str, Any] | None,
                     csv_path: Path | None) -> int | None:
        for value in (row.get("samples"), row.get("evaluated_samples"), row.get("sample_count")):
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        if csv_path:
            _, rows = LogIndex._read_csv(csv_path, keep_last=1)
            if rows:
                for name in ("samples", "sample_count", "evaluated_samples"):
                    if name in rows[-1] and isinstance(rows[-1][name], (int, float)):
                        return int(rows[-1][name])
        return None

    @staticmethod
    def _public_metrics(metrics: dict[str, Any] | None) -> dict[str, float | int]:
        if not isinstance(metrics, dict):
            return {}
        keys = ("mle_cm", "pa_mpjpe_cm", "pck_percent", "auc_percent", "loss", "cd_cm", "samples")
        result = {}
        for key in keys:
            n = _number(metrics.get(key))
            if n is not None:
                result[key] = n
        if not result:
            aliases = {"mle_m": (100, "mle_cm"), "pa_mpjpe_m": (100, "pa_mpjpe_cm"),
                       "pck": (100, "pck_percent"), "auc": (100, "auc_percent"), "cd_m": (100, "cd_cm")}
            for source, (scale, target) in aliases.items():
                n = _number(metrics.get(source))
                if n is not None:
                    result[target] = n * scale
        return result

    @staticmethod
    def _legacy_status(relative: str, columns: list[str], rows: list[dict[str, Any]], has_events: bool) -> tuple[str, str]:
        if not rows:
            return ("incomplete", "events_only" if has_events else "logs_missing_metrics")
        completed = [row.get("status", row.get("state")) for row in rows]
        state = next((v for v in reversed(completed) if isinstance(v, str)), None)
        if state and state.casefold() in ("complete", "completed", "success", "finished"):
            return "complete", "legacy_metrics_csv"
        if state and state.casefold() in ("failed", "error", "interrupted"):
            return "failed", "legacy_metrics_csv"
        return "available", "legacy_metrics_csv"

    @staticmethod
    def _read_csv(path: Path, *, keep_last: int | None = None) -> tuple[list[str], list[dict[str, Any]]]:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > _NUMERIC_CSV_LIMIT:
            return [], []
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                columns = [str(x) for x in (reader.fieldnames or [])]
                for row in reader:
                    cooked: dict[str, Any] = {}
                    for key, value in row.items():
                        if key is None or value is None:
                            continue
                        try:
                            parsed = float(value.strip())
                        except (ValueError, TypeError):
                            text = value.strip()
                            if key in ("status", "state", "tag", "filename") and text:
                                cooked[key] = text[:256]
                            continue
                        if math.isfinite(parsed):
                            cooked[key] = parsed
                    if cooked:
                        rows.append(cooked)
                    if keep_last is not None and len(rows) > keep_last:
                        rows.pop(0)
        except (OSError, UnicodeError, csv.Error, ValueError, MemoryError):
            return [], []
        return columns, rows

    def _events_in(self, directory: Path, root: Path, relative: str) -> list[str]:
        found: list[str] = []
        try:
            for base, dirs, files in os.walk(directory, followlinks=False):
                base_path = Path(base)
                dirs[:] = sorted(d for d in dirs if not _is_transient_dir(d)
                                 and d not in {'validation', 'evaluations', 'predictions', 'test', 'tests'}
                                 and not (base_path / d).is_symlink()
                                 and _inside((base_path / d).resolve(), root))
                for filename in files:
                    if not filename.startswith("events.out.tfevents."):
                        continue
                    candidate = base_path / filename
                    if candidate.is_symlink():
                        continue
                    resolved = candidate.resolve()
                    if _inside(resolved, root) and candidate.stat().st_size:
                        found.append(resolved.relative_to(root).as_posix())
        except OSError:
            return found
        return sorted(set(found))

    @staticmethod
    def _event_scalars(root: Path, event_files: list[str]) -> dict[str, list[tuple[int, float, float]]]:
        """Read scalar summaries from allowlisted event files, with bounded samples.

        This supports the source training logger without converting its events to
        fabricated CSV or modifying imported files. Parsed records reuse the
        existing file-signature cache.
        """
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            from google.protobuf.message import DecodeError
        except ImportError:
            return {}
        values: dict[str, dict[int, tuple[int, float, float]]] = {}
        for relative in sorted(event_files):
            path = root / relative
            try:
                if path.is_symlink() or not path.is_file() or not _inside(path.resolve(strict=True), root):
                    continue
                accumulator = EventAccumulator(str(path), size_guidance={"scalars": 10000})
                accumulator.Reload()
                for raw_tag in accumulator.Tags().get("scalars", []):
                    tag = _TAG_ALIASES.get(raw_tag.casefold(), raw_tag)
                    samples = values.setdefault(tag, {})
                    for event in accumulator.Scalars(raw_tag):
                        value = _number(event.value)
                        if value is None:
                            continue
                        if "(cm)" in tag:
                            value *= 1 if raw_tag.endswith("_cm") or raw_tag == tag else 100
                        elif "PCK" in tag or "AUC" in tag:
                            value *= 1 if raw_tag.endswith("_percent") or raw_tag == tag else 100
                        point = (int(event.step), float(event.wall_time), value)
                        previous = samples.get(point[0])
                        if previous is None or point[1] >= previous[1]:
                            samples[point[0]] = point
            except (OSError, ValueError, KeyError, RuntimeError, DecodeError):
                continue
        return {tag: sorted(samples.values()) for tag, samples in values.items() if samples}

    @staticmethod
    def _tags(metrics: dict[str, Any] | None, history: dict[str, Any] | None, csv_path: Path | None) -> set[str]:
        tags: set[str] = set()
        scalar_skip = {"action", "model", "split", "seed", "config", "checkpoint", "device", "torch", "cuda",
                       "dataset_manifest_sha256", "training_cache_identity", "epoch", "epochs", "samples", "joints",
                       "evaluation_complete", "expected_samples", "threshold_m", "partial", "limit_batches"}
        for source in (metrics,):
            if isinstance(source, dict):
                for key, value in source.items():
                    if key not in scalar_skip and _number(value) is not None:
                        name = str(key)
                        tags.add(_TAG_ALIASES.get(name.casefold(), name))
        if isinstance(history, (dict, list)):
            records = history.get("history", history.get("epochs", [])) if isinstance(history, dict) else history
            if isinstance(records, list):
                for row in records[:1000]:
                    if isinstance(row, dict):
                        for key, value in row.items():
                            if key not in scalar_skip and _number(value) is not None:
                                name = str(key)
                                tags.add(_TAG_ALIASES.get(name.casefold(), name))
        if csv_path:
            columns, _ = LogIndex._read_csv(csv_path, keep_last=1)
            for key in columns:
                if key.casefold() not in {"step", "epoch", "wall_time", "walltime", "time", "global_step", "status", "state"}:
                    tags.add(_TAG_ALIASES.get(key.casefold(), key))
        return tags

    @staticmethod
    def _series_data(record: dict[str, Any], tag: str, x_axis: str) -> tuple[list[float], list[float], str]:
        x: list[float] = []
        y: list[float] = []
        source = record.get("_metric_source", {})
        csv_rel = source.get("csv")
        if csv_rel:
            columns, rows = LogIndex._read_csv(Path(record["_root_path"]) / csv_rel)
            aliases = {"train/loss": (("train_loss", "loss"), 1.0),
                       "val/MLE (cm)": (("val_mle", "val_mle_m", "val_mle_cm", "val/MLE (cm)"), 100.0),
                       "val/PA-MPJPE (cm)": (("val_pa_mpjpe", "val_pa_mpjpe_m", "val_pa_mpjpe_cm", "val/PA-MPJPE (cm)"), 100.0),
                       "val/PCK@15cm (%)": (("val_pck", "val_pck_percent", "val/PCK@15cm (%)"), 100.0),
                       "val/AUC@15cm (%)": (("val_auc", "val_auc_percent", "val/AUC@15cm (%)"), 100.0),
                       "test/MLE (cm)": (("test_mle", "test_mle_m", "test_mle_cm"), 100.0),
                       "test/PA-MPJPE (cm)": (("test_pa_mpjpe", "test_pa_mpjpe_m", "test_pa_mpjpe_cm"), 100.0),
                       "test/PCK@15cm (%)": (("test_pck", "test_pck_percent"), 100.0),
                       "test/AUC@15cm (%)": (("test_auc", "test_auc_percent"), 100.0),
                       "MLE (cm)": (("mle_cm", "mle_m", "mle"),1.0), "PA-MPJPE (cm)": (("pa_mpjpe_cm", "pa_mpjpe_m", "pa_mpjpe"),1.0),
                       "PCK@15cm (%)": (("pck_percent", "pck"),1.0), "AUC@15cm (%)": (("auc_percent", "auc"),1.0),
                       "Chamfer distance (cm)": (("cd_cm",),1.0)}
            candidates,scale=aliases.get(tag,((tag,),1.0))
            x_key = {"step": ("step", "global_step", "epoch"), "epoch": ("epoch", "step"),
                     "wall_time": ("wall_time", "walltime", "time"),
                     "relative_time": ("wall_time", "walltime", "time", "elapsed_seconds")}.get(x_axis, ("epoch", "step", "global_step"))
            for row in rows:
                metric_key = next((key for key in candidates if _number(row.get(key)) is not None), None)
                value = _number(row.get(metric_key)) if metric_key else None
                xv = next((_number(row.get(key)) for key in x_key if _number(row.get(key)) is not None), None)
                if value is None or xv is None:
                    continue
                if tag in ("val/MLE (cm)", "val/PA-MPJPE (cm)", "test/MLE (cm)", "test/PA-MPJPE (cm)", "MLE (cm)", "PA-MPJPE (cm)"):
                    scale = 100.0 if metric_key.endswith("_m") or metric_key in ("val_mle", "val_pa_mpjpe", "test_mle", "test_pa_mpjpe", "mle", "pa_mpjpe") else 1.0
                elif "PCK" in tag or "AUC" in tag:
                    scale = 100.0 if metric_key in ("val_pck", "val_auc", "test_pck", "test_auc", "pck", "auc") and abs(value) <= 1 else 1.0
                x.append(float(xv)); y.append(value*scale)
            if x_axis == "relative_time" and x:
                base = x[0]
                x = [item-base for item in x]
        history_rel = source.get("history")
        if not y and history_rel:
            history = _read_json(Path(record["_root_path"]) / history_rel)
            rows = history.get("history", history.get("epochs", [])) if isinstance(history, dict) else history if isinstance(history, list) else []
            if isinstance(rows, list):
                raw_key, scale = _history_key(tag)
                for index, row in enumerate(rows):
                    if not isinstance(row, dict):
                        continue
                    value = _number(row.get(raw_key))
                    row_scale = scale
                    if value is None:
                        legacy_history = {"val/MLE (cm)": ("val_mle_m", "val_mle"),
                                          "val/PA-MPJPE (cm)": ("val_pa_mpjpe_m", "val_pa_mpjpe"),
                                          "val/PCK@15cm (%)": ("val_pck",), "val/AUC@15cm (%)": ("val_auc",)}
                        found = next((_number(row.get(key)) for key in legacy_history.get(tag, ())
                                      if _number(row.get(key)) is not None), None)
                        if found is None:
                            continue
                        value, row_scale = found, 100.0
                    xvalue = _number(row.get("epoch", row.get("step", row.get("global_step"))))
                    if x_axis in ("wall_time", "relative_time"):
                        elapsed = _number(row.get("wall_time", row.get("walltime", row.get("elapsed_seconds"))))
                        if elapsed is None:
                            continue
                        xvalue = elapsed
                    if xvalue is None:
                        continue
                    x.append(float(xvalue));y.append(value * row_scale)
                if x_axis == "relative_time" and x:
                    base = x[0]
                    x = [item-base for item in x]
        if not y:
            event_data = record.get("_event_scalars", {})
            points = event_data.get(tag, [])
            epochs = event_data.get("epoch", [])
            epoch_steps = [item[0] for item in epochs]
            for step, wall_time, value in points:
                if x_axis == "epoch" and epochs:
                    epoch_index = max(0, bisect_right(epoch_steps, step) - 1)
                    position = epochs[epoch_index][2]
                else:
                    position = wall_time if x_axis in ("wall_time", "relative_time") else step
                x.append(float(position)); y.append(value)
            if x_axis == "relative_time" and x:
                first = x[0]
                x = [value-first for value in x]
        unit = "cm" if "(cm)" in tag else ""
        if "PCK" in tag or "AUC" in tag:
            unit = "%"
        if tag.endswith("loss"):
            unit = ""
        return x, y, unit

    @staticmethod
    def _make_series(record: dict[str, Any], tag: str, x_axis: str, smoothing: float,
                     max_points: int) -> dict[str, Any]:
        x, raw, unit = LogIndex._series_data(record, tag, x_axis)
        count = len(raw)
        pairs = [(x[i], raw[i]) for i in range(count)]
        sampled = _lttb(pairs, max_points) if count > max_points else pairs
        sampled_x = [p[0] for p in sampled]
        sampled_y = [p[1] for p in sampled]
        smooth_values: list[float] = []
        average = None
        for value in sampled_y:
            average = value if average is None else smoothing * average + (1 - smoothing) * value
            smooth_values.append(float(average))
        return {"run_id": record["id"], "name": record["display_name"], "tag": tag,
                "x": sampled_x, "raw_values": sampled_y, "values": smooth_values,
                "unit": unit, "x_axis": x_axis, "original_points": count,
                "displayed_points": len(sampled_x), "downsampled": len(sampled_x) < count,
                "source": record["source"]}

    def _records(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            if self._cache and now - self._cache_at < self.cache_seconds:
                return self._cache
            result: dict[str, dict[str, Any]] = {}
            current_cache = {}
            for source, root in self.roots.items():
                if not root.is_dir() or root.is_symlink():
                    continue
                for candidate in self._discover(root):
                    manifest = self.workspace_root / 'paper/manifest.json' if self.workspace_root else None
                    try:
                        stamp = (manifest.stat().st_mtime_ns, manifest.stat().st_size) if manifest else None
                    except OSError:
                        stamp = None
                    key=(source,str(candidate),stamp);signature=self._directory_signature(candidate)
                    cached=self._record_cache.get(key)
                    record=cached[1] if cached and signature is not None and cached[0]==signature else self._register_directory(source,root,candidate,legacy=source.startswith('legacy'))
                    current_cache[key]=(signature,record)
                    if record is not None and (self.actions is None or record["action"] in self.actions):
                        result[record["id"]] = record
            self._cache = sorted(result.values(), key=lambda r: (r.get("_modified", 0), r["id"]), reverse=True)
            self._record_cache=current_cache
            self._cache_at = time.monotonic()
            return self._cache

    @staticmethod
    def _directory_signature(directory):
        """Reuse parsed runs until their small metadata/event files change."""
        try:
            folders=[directory]+[p for p in directory.glob('version_*') if p.is_dir() and not p.is_symlink()]
            stamps=[]
            for folder in folders:
                stamps.append((str(folder),folder.stat().st_mtime_ns,0))
                for path in folder.iterdir():
                    if path.is_symlink() or not path.is_file():continue
                    if path.name in _METADATA_FILES or path.name.startswith('events.out.tfevents.'):
                        stat=path.stat();stamps.append((str(path),stat.st_mtime_ns,stat.st_size))
            return tuple(sorted(stamps))
        except OSError:
            return None

    @staticmethod
    def _discover(root: Path) -> Iterable[Path]:
        """Yield run directories without following links or traversing transient/cache trees."""
        max_depth = 12
        root_depth = len(root.parts)
        try:
            for base, dirs, files in os.walk(root, followlinks=False):
                base_path = Path(base)
                dirs[:] = sorted(d for d in dirs if not _is_transient_dir(d)
                                 and (root.name != 'train' or d not in {'validation', 'evaluations', 'predictions', 'test', 'tests'})
                                 and not (base_path / d).is_symlink()
                                 and _inside((base_path / d).resolve(), root))
                if len(base_path.parts) - root_depth >= max_depth:
                    dirs[:] = []
                if "run.json" in files or "paper-training.json" in files or "hparams.yaml" in files or "hparams.yml" in files:
                    yield base_path
                    dirs[:] = []
                    continue
                names = set(files)
                if ("metrics.csv" in names and any(part.startswith("version_") for part in base_path.parts)
                        ) or any(n.startswith("events.out.tfevents.") for n in names):
                    yield base_path
        except (OSError, PermissionError):
            return

    def _runfile(self, record: dict[str, Any], rel: str) -> Path:
        root = self.roots.get(record["_source_root"])
        if root is None:
            raise ValueError("unknown source root")
        relative = Path(rel)
        if relative.is_absolute() or any(part in ("..", "") for part in relative.parts):
            raise ValueError("invalid run artifact path")
        path = (root / relative).resolve(strict=True)
        if not _inside(path, root) or not path.is_file() or path.is_symlink():
            raise ValueError("artifact path is outside its configured source")
        return path

    def list_runs(self, filters: Mapping[str, Any] | None = None, *, limit: int = _DEFAULT_RUN_LIMIT,
                  cursor: str | None = None) -> dict[str, Any]:
        filters = dict(filters or {})
        limit = max(1, min(int(limit), _MAX_RUN_LIMIT))
        allowed = {"source", "origin", "task", "model", "split", "seed", "status", "integrity", "paper_table", "search", "after", "before"}
        if set(filters) - allowed:
            raise ValueError("unsupported log filter")
        rows = self._filter(filters)
        offset = self._decode_cursor(cursor) if cursor else 0
        page = rows[offset:offset + limit]
        next_offset = offset + len(page)
        return {"runs": [self._public_run(row) for row in page],
                "models": sorted({row["model"] for row in self._records() if row.get("model")}),
                "total": len(rows), "next_cursor": self._encode_cursor(next_offset) if next_offset < len(rows) else None}

    def get_run(self, run_id: str) -> dict[str, Any]:
        record = self._resolve_id(run_id)
        detail = self._public_run(record)
        detail["config"] = record["config"]
        detail["files"] = record["files"]
        detail["event_files"] = record["event_files"]
        detail["capabilities"] = self._capabilities(record)
        for key in ("asset_id", "weight_id", "checkpoint", "paper_configuration"):
            detail[key] = record.get(key)
        return detail

    @staticmethod
    def _no_links(path: Path, base: Path) -> bool:
        """Check the lexical path too: resolving first would hide junctions."""
        try:
            relative = path.relative_to(base)
            current = base
            for part in (None, *relative.parts):
                if part is not None:
                    if part in {".", ".."}:
                        return False
                    current = current / part
                info = current.lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                    return False
            return True
        except (OSError, ValueError):
            return False

    def _protected_training_paths(self) -> set[Path]:
        """Manifest paths stay protected even if per-run metadata is missing."""
        if self.workspace_root is None:
            return set()
        manifest = self.workspace_root / "paper/manifest.json"
        if not manifest.exists():
            return set()
        if not self._no_links(manifest, self.workspace_root):
            raise PermissionError("Paper log protection metadata is unavailable.")
        data = _read_json(manifest)
        if not isinstance(data, dict) or not isinstance(data.get("assets"), list):
            raise PermissionError("Paper log protection metadata is unavailable.")
        expected = self.workspace_root / "mmwave_densepose/logs/train"
        protected = set()
        for asset in data["assets"]:
            value = asset.get("training_log") if isinstance(asset, dict) else None
            if not isinstance(value, str):
                continue
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            path = (self.workspace_root / relative).resolve()
            if path != expected and _inside(path, expected):
                protected.add(path)
        return protected

    def _delete_reason(self, record: dict[str, Any], protected: set[Path]) -> str | None:
        if record.get("origin") != "user":
            return "Official training logs cannot be deleted."
        if record.get("action") != "train" or self.workspace_root is None:
            return "Only local training logs can be deleted."
        expected = self.workspace_root / "mmwave_densepose/logs/train"
        root = self.roots.get(record.get("_source_root"))
        relative = Path(record.get("_relative_directory", ""))
        if root != expected or relative.is_absolute() or len(relative.parts) != 2 or ".." in relative.parts:
            return "Only individual runs in the training-log directory can be deleted."
        directory = root / relative
        if not self._no_links(directory, self.workspace_root) or not directory.is_dir():
            return "The training-log directory is unavailable or contains a linked path."
        if (directory / "paper-training.json").exists() or any(
                _inside(directory, path) or _inside(path, directory) for path in protected):
            return "Official training logs cannot be deleted."
        return None

    def _active_training_paths(self, directories: Iterable[Path]) -> list[Path]:
        """Inspect unbounded execution inventory so older live runs are covered."""
        from mmwave_densepose.engine.process_identity import probe_process
        runtime = self.workspace_root / ".local/runtime/executions"
        directory_set = set(directories)
        snapshots = [directory / "execution.json" for directory in directory_set]
        if runtime.exists():
            if not self._no_links(runtime, self.workspace_root):
                raise PermissionError("Execution monitoring metadata is unavailable.")
            snapshots.extend(runtime.glob("exec_*/status.json"))
        active = []
        for snapshot in snapshots:
            if not snapshot.exists():
                continue
            if not self._no_links(snapshot, self.workspace_root):
                raise PermissionError("Execution monitoring metadata contains a linked path.")
            record = _read_json(snapshot)
            if not isinstance(record, dict):
                raise RuntimeError("Execution monitoring metadata could not be read. Try again after the run has stopped.")
            if record.get("status") not in {"Queued", "Starting", "Running", "Stopping", "Needs review"}:
                continue
            if (record.get("hostname") == socket.gethostname()
                    and isinstance(record.get("process_identity"), dict)):
                try:
                    if probe_process(record["process_identity"]) == "dead":
                        continue
                except (OSError, ValueError, TypeError):
                    pass
            if snapshot.name == "execution.json" and snapshot.parent in directory_set:
                # An active snapshot located in this run is enough evidence,
                # including an interrupted write before its public paths exist.
                active.append(snapshot.parent)
            for key in ("output_dir", "log_dir"):
                value = record.get(key)
                if isinstance(value, str) and value:
                    path = Path(value)
                    active.append((path if path.is_absolute() else self.workspace_root / path).resolve())
        return active

    def delete_runs(self, run_ids: Iterable[str], *, active_paths: Iterable[str | Path] = ()) -> dict[str, Any]:
        """Move a fully validated batch to private recovery storage, never erase it."""
        if (not isinstance(run_ids, (list, tuple)) or not 1 <= len(run_ids) <= _MAX_RUN_LIMIT
                or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value) for value in run_ids)
                or len(set(run_ids)) != len(run_ids)):
            raise ValueError("Select one to 500 distinct indexed training runs.")
        with self._lock:
            # Do not trust cached origin, metadata or a stale inventory for mutations.
            self._cache_at = -math.inf
            self._record_cache = {}
            records = [self._resolve_id(value) for value in run_ids]
            protected = self._protected_training_paths()
            directories = []
            def walk_error(error):
                raise error
            for record in records:
                reason = self._delete_reason(record, protected)
                if reason:
                    raise PermissionError(reason)
                directory = self.roots[record["_source_root"]] / record["_relative_directory"]
                # A run may not contain a junction, a nested official record or an
                # old embedded test/weight tree that would be lost from its UI.
                for base, folders, files in os.walk(directory, followlinks=False, onerror=walk_error):
                    for name in (*folders, *files):
                        path = Path(base) / name
                        if not self._no_links(path, self.workspace_root):
                            raise PermissionError("Training logs containing linked paths cannot be deleted.")
                        if name in {"paper-training.json", "paper-record.json"}:
                            raise PermissionError("Official training logs cannot be deleted.")
                        if name in {"checkpoints", "evaluations", "test", "tests"} and path.is_dir():
                            raise PermissionError("This log directory contains weights or tests. Move those artifacts before deleting its logs.")
                        if path.is_file() and path.suffix.casefold() in {".ckpt", ".pt", ".pth", ".safetensors"}:
                            raise PermissionError("This log directory contains model weights. Move those weights before deleting its logs.")
                directories.append(directory)
            for index, directory in enumerate(directories):
                if any(_inside(directory, other) or _inside(other, directory) for other in directories[:index]):
                    raise ValueError("Selected training-log directories overlap.")
            active = self._active_training_paths(directories)
            for value in active_paths:
                path = Path(value)
                active.append((path if path.is_absolute() else self.workspace_root / path).resolve())
            if any(_inside(directory, path) or _inside(path, directory) for directory in directories for path in active):
                raise RuntimeError("Stop the selected training runs before deleting their logs.")
            trash_root = self.workspace_root / ".local/trash/training-logs"
            # Check existing ancestors before creating anything in recovery storage.
            existing = trash_root
            while not existing.exists():
                existing = existing.parent
            if not self._no_links(existing, self.workspace_root):
                raise PermissionError("Training-log recovery storage contains a linked path.")
            trash_root.mkdir(parents=True, exist_ok=True)
            if not self._no_links(trash_root, self.workspace_root):
                raise PermissionError("Training-log recovery storage contains a linked path.")
            batch = trash_root / uuid.uuid4().hex
            batch.mkdir()
            entries = [{"run_id": record["id"], "source": directory.relative_to(self.workspace_root).as_posix(),
                        "destination": f"runs/{record['id']}"}
                       for record, directory in zip(records, directories)]
            manifest = {"schema_version": 1, "status": "planned", "created_at": time.time(), "entries": entries}
            manifest_path = batch / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            (batch / "runs").mkdir()
            moved = []
            try:
                for directory, entry in zip(directories, entries):
                    if not self._no_links(directory, self.workspace_root):
                        raise PermissionError("A training-log path changed. Refresh the run list and try again.")
                    destination = batch / entry["destination"]
                    directory.rename(destination)
                    moved.append((directory, destination))
                manifest["status"] = "deleted"
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            except OSError:
                failed = []
                for directory, destination in reversed(moved):
                    try:
                        destination.rename(directory)
                    except OSError:
                        failed.append(directory.relative_to(self.workspace_root).as_posix())
                manifest.update(status="recovery_required" if failed else "rolled_back", recovery_required=failed)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                if failed:
                    raise RuntimeError(f"Some logs require recovery using {manifest_path.relative_to(self.workspace_root).as_posix()}.") from None
                raise
            finally:
                self._cache = []
                self._cache_at = -math.inf
                self._record_cache = {}
            return {"deleted_run_ids": list(run_ids), "count": len(run_ids), "trash_id": batch.name,
                    "recovery_manifest": manifest_path.relative_to(self.workspace_root).as_posix()}

    def list_tags(self, run_ids: Iterable[str]) -> dict[str, Any]:
        if not isinstance(run_ids, (list, tuple, set)):
            raise ValueError("run_ids must be a collection")
        runs = []
        for run_id in list(run_ids)[:8]:
            record = self._resolve_id(run_id)
            runs.append({"run_id": record["id"], "tags": record["tags"],
                         "capabilities": self._capabilities(record)})
        return {"runs": runs}

    def series(self, run_ids: Iterable[str], tags: Iterable[str], *, x_axis: str = "epoch",
               smoothing: float = 0.0, max_points: int = _MAX_SERIES_POINTS) -> dict[str, Any]:
        if not isinstance(run_ids, (list, tuple, set)) or not isinstance(tags, (list, tuple, set)):
            raise ValueError("run_ids and tags must be collections")
        if x_axis not in ("step", "epoch", "relative_time", "wall_time"):
            raise ValueError("unsupported x axis")
        smoothing_value = float(smoothing)
        if not math.isfinite(smoothing_value) or not 0 <= smoothing_value <= .95:
            raise ValueError("smoothing must be from 0 through 0.95")
        max_points = max(20, min(int(max_points), _MAX_SERIES_POINTS))
        records = [self._resolve_id(value) for value in list(run_ids)[:8]]
        requested = [str(tag)[:256] for tag in list(tags)[:32]]
        output = []
        missing: list[dict[str, str]] = []
        for record in records:
            for tag in requested:
                if tag not in record["tags"]:
                    missing.append({"run_id": record["id"], "tag": tag})
                    continue
                item = self._make_series(record, tag, x_axis, smoothing_value, max_points)
                if not item["raw_values"]:
                    missing.append({"run_id": record["id"], "tag": tag})
                    continue
                output.append(item)
        return {"series": output, "missing": missing, "smoothing": smoothing_value, "x_axis": x_axis}

    def tensorboard_info(self, run_id: str | None = None) -> dict[str, Any]:
        records = [self._resolve_id(run_id)] if run_id else self._records()
        records = [record for record in records if record.get("action") == "train"]
        roots: list[str] = []
        events: list[dict[str, Any]] = []
        for record in records:
            for relative in record["event_files"]:
                parent = str(Path(relative).parent).replace("\\", "/")
                if parent not in roots:
                    roots.append(parent)
                events.append({"run_id": record["id"], "relative_path": relative})
        executable = self._native_executable
        return {"available": bool(events), "executable_available": bool(executable),
                "status": "ready" if events and executable else ("events_unavailable" if not events else "executable_unavailable"),
                "event_files": events, "log_roots": roots, "url": None,
                "custom_event_parser": self._native_executable,
                "capabilities": {"json_scalars": True, "csv_scalars": True, "event_scalars": self._native_executable,
                                 "event_images": False, "event_histograms": False,
                                 "native_tensorboard": bool(events and executable)}}

    def _filter(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        rows = self._records()
        search = str(filters.get("search", "")).strip().casefold()
        keep = []
        for row in rows:
            if search and search not in " ".join((row.get("original_name", ""), row.get("display_name", ""),
                                                    row.get("model") or "", row.get("_relative_directory", ""))).casefold():
                continue
            checks = {"source": row.get("source"), "origin": row.get("origin"), "task": row.get("action"), "model": row.get("model"),
                      "split": row.get("split"), "seed": row.get("seed"), "status": row.get("status"),
                      "integrity": row.get("integrity")}
            if any(filters.get(k) not in (None, "", "all", "*", value) for k, value in checks.items()):
                continue
            if filters.get("paper_table") not in (None, "", "all", "*") and str(filters["paper_table"]) not in row["paper_tables"]:
                continue
            modified = row.get("_modified", 0)
            after = filters.get("after")
            before = filters.get("before")
            if after is not None and modified < float(after):
                continue
            if before is not None and modified > float(before):
                continue
            keep.append(row)
        return keep

    @staticmethod
    def _decode_cursor(cursor: str) -> int:
        try:
            payload = base64.urlsafe_b64decode(cursor.encode("ascii") + b"===").decode("ascii")
            offset = int(payload)
            if offset < 0:
                raise ValueError
            return offset
        except (ValueError, UnicodeError, OSError):
            raise ValueError("invalid log cursor") from None

    @staticmethod
    def _encode_cursor(offset: int) -> str:
        return base64.urlsafe_b64encode(str(offset).encode("ascii")).decode("ascii").rstrip("=")

    def _public_run(self, record: dict[str, Any]) -> dict[str, Any]:
        keys = ("id", "run_id", "source", "legacy", "display_name", "action", "task", "model",
                "split", "seed", "timestamp", "status", "integrity", "partial", "limited", "sample_count",
                "joints", "threshold_m", "metrics", "sizes", "has_events", "paper_tables", "paper_dependency_tables", "tags", "source_root",
                "name_classification", "protocol", "checkpoint_sha256", "dataset_identity", "origin", "display_metadata",
                "asset_id", "weight_id", "checkpoint", "config_file", "config_id", "config_sha256", "gpu_ids", "num_devices")
        payload={key: record[key] for key in keys if key in record}
        payload.update({key: value for key, value in record.get('display_metadata', {}).items() if key not in ('status', 'time', 'split', 'seed')})
        payload['training_run_id']=record.get('run_id')
        payload['run_id']=record['id']
        try:
            reason = self._delete_reason(record, self._protected_training_paths())
        except PermissionError as exc:
            reason = str(exc)
        payload.update(can_delete=reason is None, delete_reason=reason)
        return payload

    @staticmethod
    def _capabilities(record: dict[str, Any]) -> dict[str, Any]:
        tags = set(record.get("tags", []))
        return {"scalars": bool(tags), "event_scalar_tags": bool(record.get("_event_scalars")),
                "images": any("image" in tag.casefold() for tag in tags),
                "histograms": False, "hparams": "hparams.yaml" in record["files"] or "hparams.yml" in record["files"],
                "tensorboard_events": record["has_events"]}


def _history_key(tag: str) -> tuple[str, float]:
    values = {"train/loss": ("train_loss", 1.0), "train_loss": ("train_loss", 1.0),
              "val/MLE (cm)": ("val_mle_cm", 1.0), "val_mle_m": ("val_mle_m", 1.0),
              "val/PA-MPJPE (cm)": ("val_pa_mpjpe_cm", 1.0), "val_pa_mpjpe_m": ("val_pa_mpjpe_m", 1.0),
              "val/PCK@15cm (%)": ("val_pck_percent", 1.0), "val/AUC@15cm (%)": ("val_auc_percent", 1.0),
              "test/MLE (cm)": ("test_mle", 100.0), "test/PA-MPJPE (cm)": ("test_pa_mpjpe", 100.0),
              "test/PCK@15cm (%)": ("test_pck", 100.0), "test/AUC@15cm (%)": ("test_auc", 100.0)}
    return values.get(tag, (tag, 1.0))


def _metric_key(tag: str) -> tuple[str, float]:
    values = {"MLE (cm)": ("mle_cm", 1.0), "PA-MPJPE (cm)": ("pa_mpjpe_cm", 1.0),
              "PCK@15cm (%)": ("pck_percent", 1.0), "AUC@15cm (%)": ("auc_percent", 1.0),
              "loss": ("loss", 1.0), "CD (cm)": ("cd_cm", 1.0)}
    if tag in values:
        return values[tag]
    canonical = _TAG_ALIASES.get(tag.casefold())
    return values.get(canonical, (tag, 1.0))

__all__ = ["LogIndex", "default_roots"]

# Return only populated scalar tags; event-backed tags require the native TensorBoard reader.
