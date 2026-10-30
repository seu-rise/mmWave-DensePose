"""Read frozen paper-test CSVs without rewriting or executing source artifacts."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import yaml


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def csv_rows(path, root):
    path, root = Path(path), Path(root).resolve()
    try:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            return []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error, UnicodeError):
        return []


def actual_config(directory, metadata, root):
    values = {}
    path = directory / "hparams.yaml"
    try:
        if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root):
            document = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            if isinstance(document, dict):
                values = {key: value for key, value in document.items()
                          if isinstance(key, str) and isinstance(value, (str, int, float, bool, type(None)))}
    except (OSError, ValueError, yaml.YAMLError):
        pass
    for old, new in (("radar_point_size", "num_points"), ("sample_dim", "input_dim"),
                     ("num_keypoints", "num_joints"), ("t_max", "seq_len"),
                     ("max_epochs", "epochs")):
        if old in values and new not in values:
            values[new] = values[old]
    # The sidecar contains audited normalization, never manuscript metric values.
    values.update(metadata.get("config") if isinstance(metadata.get("config"), dict) else {})
    values["model"] = metadata["model"]
    return values


def read_paper_result(directory, metadata, root):
    """Return measured CSV values; unknown population/integrity stays unknown."""
    directory, root = Path(directory), Path(root).resolve()
    config = actual_config(directory, metadata, root)
    completion = metadata.get("task") == "completion" or metadata["model"] == "PCCCompletion"
    aliases = {
        "mle_cm": (("test_mle_cm", 1), ("mle_cm", 1), ("test_mle", 100), ("test_mle_m", 100)),
        "pa_mpjpe_cm": (("test_pa_mpjpe_cm", 1), ("pa_mpjpe_cm", 1), ("test_pa_mpjpe", 100), ("test_pa_mpjpe_m", 100)),
        "pck_percent": (("test_pck_percent", 1), ("pck_percent", 1), ("test_pck", 100)),
        "auc_percent": (("test_auc_percent", 1), ("auc_percent", 1), ("test_auc", 100)),
        "cd_cm": (("test_cd_cm", 1), ("cd_cm", 1), ("test_cd", 100), ("test_cd_m", 100)),
    }
    metrics = {}
    for row in csv_rows(directory / "metrics.csv", root):
        for key, candidates in aliases.items():
            for field, scale in candidates:
                value = number(row.get(field))
                if value is not None:
                    metrics[key] = value * scale
                    break
        for key in ("samples", "joints", "threshold_m"):
            value = number(row.get(key))
            if value is not None:
                metrics[key] = value
    joints, seen = [], set()
    for name in ("per_joint_metrics.csv", "per_joint_mle.csv"):
        rows = csv_rows(directory / name, root)
        for row in rows:
            raw = row.get("joint_index", "")
            if not str(raw).strip().isdigit():
                continue
            index = int(raw)
            if index in seen or not 0 <= index < 19:
                continue
            values = {"index": index, "joint_name": row.get("joint_name")}
            mle = number(row.get("mle_m", row.get("mle")))
            values["mle_cm"] = mle * 100 if mle is not None else number(row.get("mle_cm"))
            pa = number(row.get("pa_mpjpe_m"))
            values["pa_mpjpe_cm"] = pa * 100 if pa is not None else number(row.get("pa_mpjpe_cm"))
            for key, field in (("pck_percent", "pck"), ("auc_percent", "auc")):
                value = number(row.get(field))
                values[key] = value * 100 if value is not None else number(row.get(key))
            count = number(row.get("count"))
            values["count"] = int(count) if count is not None and count > 0 and count.is_integer() else None
            for key in ("pck_threshold_m", "auc_max_threshold_m"):
                threshold = number(row.get(key))
                if threshold is not None and threshold > 0:
                    metrics.setdefault("threshold_m", threshold)
            joints.append(values)
            seen.add(index)
        if joints:
            break
    if joints:
        joints.sort(key=lambda row: row["index"])
        metrics["joints"] = sum(row.get("mle_cm") is not None for row in joints)
        counts = {row["count"] for row in joints}
        if len(counts) == 1 and None not in counts:
            metrics.setdefault("samples", counts.pop())
    threshold = number(config.get("pck_threshold", config.get("pck_threshold_m")))
    if threshold is not None and threshold > 0:
        metrics.setdefault("threshold_m", threshold)
    return {"config": config, "metrics": metrics, "per_joint": joints or None,
            "task": "completion" if completion else "pose"}
