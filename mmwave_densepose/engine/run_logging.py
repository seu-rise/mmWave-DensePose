"""Training hyperparameters, CSV scalars and native TensorBoard events."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import yaml


JOINT_NAMES = (
    "SpineBase", "SpineMid", "Neck", "Head", "ShoulderLeft", "ElbowLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "HandRight", "HipLeft", "KneeLeft", "AnkleLeft",
    "FootLeft", "HipRight", "KneeRight", "AnkleRight", "FootRight", "SpineShoulder",
)
METRICS = {"loss": "loss", "mle_m": "mle", "pa_mpjpe_m": "pa_mpjpe",
           "pck": "pck", "auc": "auc", "cd_m": "cd"}


def legacy_scalars(metrics, prefix=""):
    """Format scalar tags in meters and fractions."""
    return {prefix + tag: float(metrics[key]) for key, tag in METRICS.items()
            if key in metrics and isinstance(metrics[key], (int, float))
            and math.isfinite(metrics[key])}


def hyperparameters(config, run_info, output):
    """Record effective run parameters and their logging fields."""
    completion = run_info["task"] == "completion"
    values = {"seed": 42, "batch_size": 128, "num_workers": 0, "cache_size": 141,
              "num_points": 80 if completion else 300, "input_dim": 3 if completion else 5,
              "seq_len": 1 if completion else 5, **config}
    if not completion:
        values.setdefault("num_joints", 19)
    if run_info["action"] == "train":
        for key, value in {"epochs": 200, "learning_rate": 5e-4, "weight_decay": 1e-5,
                           "optimizer": "adam", "scheduler_name": "cosine",
                           "scheduler_eta_min": 1e-6, "scheduler_step_size": 30,
                           "scheduler_gamma": .1}.items():
            values.setdefault(key, value)
    for key in ("seed", "batch_size", "num_workers", "cache_size", "num_points", "input_dim",
                "seq_len", "num_joints", "epochs", "num_out_points", "torch_threads",
                "scheduler_step_size"):
        if key in values:
            values[key] = int(values[key])
    for key in ("learning_rate", "weight_decay", "scheduler_eta_min", "scheduler_gamma", "pcc_label_radius"):
        if key in values:
            values[key] = float(values[key])
    values.update({"action": run_info["action"], "task": run_info["task"],
                   "run_name": run_info.get("run_id") or Path(output).name,
                   "dataset_name": "mmWave-Densepose",
                   "resolved_model_name": config["model"],
                   "split": run_info["split"], "checkpoint": run_info.get("checkpoint"),
                   "dataset_manifest_sha256": run_info["dataset_manifest_sha256"],
                   "limit_batches": run_info.get("limit_batches"),
                   "partial": run_info.get("partial", False)})
    for source, alias in (("epochs", "max_epochs"), ("num_points", "radar_point_size"),
                          ("input_dim", "sample_dim"), ("num_joints", "num_keypoints"),
                          ("optimizer", "optimizer_name")):
        if source in values:
            values[alias] = values[source]
    if "epochs" in values:
        values["effective_max_epochs"] = values["epochs"]
    metadata = run_info.get("split_metadata", {})
    for key in ("partition", "split_strategy", "split_chunk_size", "source_index",
                "index_sha256", "max_timestamp_gap_ms"):
        if key in metadata:
            values[key] = metadata[key]
    for key in ("run_id", "training_run_id", "weight_id", "test_id", "config_file", "config_id", "config_sha256", "num_devices"):
        if run_info.get(key) is not None:
            values[key] = run_info[key]
    return values


class RunLogger:
    """Write one selected run directory, flushing every scalar row."""

    def __init__(self, directory, hparams):
        from torch.utils.tensorboard import SummaryWriter

        self.directory = Path(directory)
        self.hparams = hparams
        self.writer = SummaryWriter(log_dir=str(self.directory), flush_secs=5)
        self.fields = None
        self.last_metrics = {}
        try:
            temporary = self.directory / ".hparams.yaml.tmp"
            temporary.write_text(yaml.safe_dump(hparams, allow_unicode=True, sort_keys=False), encoding="utf-8")
            temporary.replace(self.directory / "hparams.yaml")
        except BaseException:
            self.writer.close()
            raise

    def log(self, metrics, *, epoch, step):
        scalars = {key: float(value) for key, value in metrics.items()
                   if isinstance(value, (int, float)) and math.isfinite(value)}
        row = {"epoch": epoch, "step": step, **scalars}
        fields = list(row)
        if self.fields is None:
            self.fields = fields
        elif fields != self.fields:
            raise ValueError("Run scalar columns changed during logging")
        with (self.directory / "metrics.csv").open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
        self.writer.add_scalar("epoch", epoch, step)
        for key, value in scalars.items():
            self.writer.add_scalar(key, value, step)
        self.last_metrics = scalars
        self.writer.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.last_metrics:
                # Use the same directory, with distinct final-metric tags so the
                # HParams plugin never inserts false step-zero training values.
                scalar_hparams = {key: value if isinstance(value, (bool, int, float, str)) else str(value)
                                  for key, value in self.hparams.items()}
                self.writer.add_hparams(scalar_hparams,
                                        {"hparam/" + key: value for key, value in self.last_metrics.items()},
                                        run_name=".")
            self.writer.flush()
        finally:
            self.writer.close()
        return False


def write_joint_metrics(directory, overall, per_joint):
    """Export full joint metrics and the original four-column MLE contract."""
    directory = Path(directory)
    rows = []
    for item in per_joint:
        index = item["joint_index"]
        rows.append({**item, "joint_name": JOINT_NAMES[index], "count": overall["samples"],
                     "pck_threshold_m": overall["threshold_m"],
                     "auc_max_threshold_m": overall["threshold_m"]})
    with (directory / "per_joint_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"joint_index": "mean", "joint_name": "overall",
                         **{key: overall[key] for key in ("mle_m", "pa_mpjpe_m", "pck", "auc")},
                         "count": overall["samples"] * len(rows),
                         "pck_threshold_m": overall["threshold_m"],
                         "auc_max_threshold_m": overall["threshold_m"]})
    with (directory / "per_joint_mle.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["joint_index", "joint_name", "mle", "count"])
        writer.writerows((row["joint_index"], row["joint_name"], row["mle_m"], row["count"])
                         for row in rows)
        writer.writerow(["mean", "overall", overall["mle_m"], overall["samples"] * len(rows)])
