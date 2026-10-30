"""Recover execution metadata only after its exact formal log has been removed."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid

from mmwave_densepose.engine.execution import _atomic_json
from mmwave_densepose.engine.process_identity import probe_process

TERMINAL = {"Complete", "Failed", "Stopped", "Interrupted"}
ACTIVE = {"Queued", "Starting", "Running", "Stopping"}
ID = {"jobs": re.compile(r"job_[A-Za-z0-9_-]{1,90}\Z"),
      "executions": re.compile(r"exec_[0-9a-f]{32}\Z")}


def safe_path(root, value):
    """Reject lexical escapes and Windows reparse points before resolving."""
    root = Path(root).resolve()
    path = Path(value)
    path = path if path.is_absolute() else root / path
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError("Execution recovery path is outside the repository") from None
    current = root
    for part in relative.parts:
        if part in (".", ".."):
            raise ValueError("Execution recovery path is not normalized")
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Execution recovery path contains a linked entry")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Execution recovery path escaped the repository")
    return resolved


def output_path(root, value):
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    path = safe_path(root, value)
    parts = path.relative_to(Path(root).resolve()).parts
    if len(parts) not in (5, 6) or parts[:2] != ("mmwave_densepose", "logs"):
        raise ValueError("Only canonical training/evaluation log directories can be forgotten")
    if not (parts[2] == "train" and len(parts) == 5 or parts[2] == "test" and len(parts) == 6):
        raise ValueError("Only exact training/evaluation log leaf directories can be forgotten")
    return path


def record_output(root, record):
    try:
        return output_path(root, record.get("_output_path") or record.get("output_dir") or record.get("log_dir"))
    except (ValueError, OSError, TypeError):
        return None


def read_json(root, path):
    try:
        path = safe_path(root, path)
        if not path.is_file() or not 0 < path.stat().st_size <= 2 * 1024 * 1024:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def receipts(root):
    found = {}
    for kind in ("training-logs", "results"):
        directory = Path(root) / ".local/trash" / kind
        try:
            directory = safe_path(root, directory)
            paths = list(directory.glob("*/manifest.json")) if directory.is_dir() else []
        except (OSError, ValueError):
            continue
        for path in paths:
            record = read_json(root, path)
            if not record or record.get("status") != "deleted":
                continue
            when = record.get("created_at")
            entries = record.get("entries")
            if (not isinstance(when, (int, float)) or isinstance(when, bool) or not math.isfinite(when)
                    or not isinstance(entries, list)):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    log = output_path(root, entry.get("source"))
                except (ValueError, OSError):
                    continue
                if log:
                    found[log] = max(found.get(log, 0), when)
    return found


def observed_record(root, record):
    execution_id = record.get("execution_id")
    if not isinstance(execution_id, str) or not ID["executions"].fullmatch(execution_id):
        return None
    return read_json(root, Path(root) / ".local/runtime/executions" / execution_id / "status.json")


def inactive(record, *, observed=None, process=None):
    if process is not None:
        try:
            if process.poll() is None:
                return False
        except (OSError, ValueError):
            return False
    state = record.get("status")
    if state in ACTIVE or state not in TERMINAL | {"Needs review"}:
        return False
    identity = (observed or record).get("process_identity")
    if isinstance(identity, dict):
        try:
            return probe_process(identity) == "dead"
        except (OSError, ValueError, TypeError):
            return False
    return state in TERMINAL


def was_deleted(root, record, deleted, *, observed=None):
    log = record_output(root, record)
    if log is None or log.exists():
        return False
    created = record.get("created_at")
    receipt = deleted.get(log)
    if receipt is not None and isinstance(created, (int, float)) and math.isfinite(created) and created <= receipt:
        return True
    if record.get("log_published") is True:
        return True
    if record.get("status") == "Complete" and record.get("run_id"):
        return True
    original = observed or record
    formal = record_output(root, original)
    console = original.get("console_path")
    try:
        return bool(formal == log and original.get("run_id") and isinstance(console, str)
                    and safe_path(root, console) == log / "console.ndjson")
    except (OSError, ValueError):
        return False


def archive(root, kind, records, directory_for, *, recovery_root=None):
    """Move only allowlisted runtime record directories; never remove log data."""
    if not records:
        return []
    root = Path(root).resolve()
    batch = safe_path(root, recovery_root or root / ".local/trash/execution-records" / uuid.uuid4().hex)
    if not batch.is_relative_to(root / ".local/trash"):
        raise ValueError("Execution recovery must stay in local trash")
    destination_root = safe_path(root, batch / "execution-records" / kind)
    sources = []
    prefix = root / (".local/runtime/jobs/runs" if kind == "jobs" else ".local/runtime/executions")
    for identifier, record in records:
        if not ID[kind].fullmatch(identifier):
            raise ValueError("Invalid execution recovery identifier")
        source = safe_path(root, directory_for(identifier))
        if source.parent != prefix or source.name != identifier:
            raise ValueError("Execution recovery only accepts canonical runtime records")
        if source.is_dir():
            for base, dirs, files in os.walk(source, followlinks=False):
                for name in (*dirs, *files):
                    safe_path(root, Path(base) / name)
        target = safe_path(root, destination_root / identifier)
        if target.exists():
            raise FileExistsError("Execution recovery destination already exists")
        sources.append((identifier, record, source, target))
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "status": "archiving", "created_at": time.time(), "entries": []}
    manifest_path = destination_root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    moved = []
    try:
        for identifier, record, source, target in sources:
            if source.exists():
                source.rename(target)
                moved.append((source, target))
            else:
                target.mkdir()
                _atomic_json(target / "snapshot.json", record)
            manifest["entries"].append({"id": identifier, "source": source.relative_to(root).as_posix(),
                                         "destination": target.relative_to(batch).as_posix()})
        manifest["status"] = "archived"
        _atomic_json(manifest_path, manifest)
    except OSError:
        for source, target in reversed(moved):
            target.rename(source)
        manifest["status"] = "rolled_back"
        _atomic_json(manifest_path, manifest)
        raise
    return [identifier for identifier, _record in records]
