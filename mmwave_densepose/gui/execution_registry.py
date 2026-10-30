"""Execution discovery and explicit identity-verified stop control."""
from __future__ import annotations

from collections import OrderedDict
import copy
import json
import math
from pathlib import Path
import re
import socket
import threading
import time
import uuid

from mmwave_densepose.engine.process_identity import probe_process, can_control_process, terminate_verified_tree
from mmwave_densepose.engine.execution import _atomic_json
from . import deleted_logs


EXECUTION_ID = re.compile(r"exec_[0-9a-f]{32}\Z")
TERMINAL = {"Complete", "Failed", "Stopped", "Interrupted"}
ACTIVE = {"Starting", "Running", "Stopping"}
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024
MAX_LOG_LINE_BYTES = 64 * 1024
MAX_LOG_PAGE_BYTES = 512 * 1024


def _number(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


class ExecutionRegistry:
    """Discover local execution manifests and handle explicit verified stop requests."""

    def __init__(self, root, *, heartbeat_timeout=15, inventory_interval=1, limit=200, stop_grace=8):
        self.root = Path(root).resolve()
        self.runtime = self.root / ".local/runtime/executions"
        self.logs_root = self.root / "mmwave_densepose/logs"
        self.heartbeat_timeout = heartbeat_timeout
        self.inventory_interval = inventory_interval
        self.limit = limit
        self._lock = threading.RLock()
        self._inventory_at = -math.inf
        self._archive_at = -math.inf
        self._runtime_paths = []
        self._archive_paths = []
        self._json_cache = OrderedDict()
        self._log_positions = OrderedDict()
        self.stop_grace = max(.05, float(stop_grace))
        self._stop_threads = {}

    def _control(self, record):
        path = self._safe(self.runtime / record["execution_id"] / "control.json")
        control = self._json(path) if path is not None else None
        if not control:
            log = self._public_path(record.get("output_dir") or record.get("log_dir"))
            control = self._json(self.root / log / "execution-control.json") if log else None
        if (not control or control.get("action") != "stop"
                or control.get("execution_id") != record["execution_id"]
                or control.get("repository_root") != str(self.root)
                or control.get("process_identity") != record.get("process_identity")):
            return None
        return control

    def _write_control(self, control):
        path = self._safe(self.runtime / control["execution_id"] / "control.json")
        if path is None:
            raise PermissionError("The execution control path is unsafe.")
        _atomic_json(path, control)
        log = self._public_path(control.get("log_dir"))
        if log and (self.root / log).is_dir():
            _atomic_json(self.root / log / "execution-control.json", control)

    def _controllable(self, record):
        identity = record.get("process_identity")
        return (record.get("hostname") == socket.gethostname() and isinstance(identity, dict)
                and record.get("pid") == identity.get("pid") and can_control_process(identity))

    def _safe(self, value):
        path = Path(value)
        path = path if path.is_absolute() else self.root / path
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return None
        current = self.root
        for part in relative.parts:
            if part in (".", ".."):
                return None
            current = current / part
            if current.is_symlink():
                return None
        resolved = path.resolve()
        return resolved if resolved.is_relative_to(self.root) else None

    def _json(self, path):
        path = self._safe(path)
        if path is None:
            return None
        try:
            stat = path.stat()
            if not path.is_file() or not 0 < stat.st_size <= MAX_SNAPSHOT_BYTES:
                return None
            signature = (stat.st_mtime_ns, stat.st_size)
            cached = self._json_cache.get(path)
            if cached and cached[0] == signature:
                self._json_cache.move_to_end(path)
                return copy.deepcopy(cached[1])
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return None
            self._json_cache[path] = (signature, value)
            self._json_cache.move_to_end(path)
            while len(self._json_cache) > 4 * self.limit:
                self._json_cache.popitem(last=False)
            return copy.deepcopy(value)
        except (OSError, ValueError, TypeError, UnicodeError):
            return None

    def _paths(self):
        now = time.monotonic()
        if now - self._inventory_at >= self.inventory_interval:
            paths = []
            if self._safe(self.runtime) is not None and self.runtime.is_dir():
                for directory in self.runtime.iterdir():
                    if EXECUTION_ID.fullmatch(directory.name) and self._safe(directory) is not None:
                        path = directory / "status.json"
                        try:
                            paths.append((path.stat().st_mtime_ns, path))
                        except OSError:
                            continue
            self._runtime_paths = [path for _, path in sorted(paths, reverse=True)[:self.limit]]
            self._inventory_at = now
        if now - self._archive_at >= max(5, self.inventory_interval):
            paths = []
            if self._safe(self.logs_root) is not None:
                for phase in ("train", "test"):
                    for path in (self.logs_root / phase).glob("**/execution.json"):
                        safe = self._safe(path)
                        if safe is None:
                            continue
                        try:
                            paths.append((safe.stat().st_mtime_ns, safe))
                        except OSError:
                            continue
            self._archive_paths = [path for _, path in sorted(paths, reverse=True)[:self.limit]]
            self._archive_at = now
        return self._runtime_paths, self._archive_paths

    def _console(self, record):
        value = record.get("console_path")
        if not isinstance(value, str):
            return None
        path = self._safe(value)
        if path is None or path.name != "console.ndjson":
            return None
        runtime = self.runtime / record["execution_id"]
        if path.parent == runtime:
            return path
        if path.is_relative_to(self.logs_root / "train") or path.is_relative_to(self.logs_root / "test"):
            return path
        return None

    def _public_path(self, value):
        if not isinstance(value, str) or not value:
            return None
        path = self._safe(value)
        if path is None:
            return None
        if not (path.is_relative_to(self.logs_root) or
                path.is_relative_to(self.root / "mmwave_densepose/checkpoints") or
                path.is_relative_to(self.root / "paper/experiments")):
            return None
        return path.relative_to(self.root).as_posix()

    def _final_metrics(self, record):
        if record.get("status") != "Complete" or record.get("action") == "train":
            return {}
        directory = self._public_path(record.get("output_dir") or record.get("log_dir"))
        if not directory:
            return {}
        directory = self.root / directory
        expected = self.logs_root / "test"
        if not directory.is_relative_to(expected):
            return {}
        identity = tuple(record.get(key) for key in ("model", "run_id", "test_id"))
        if any(not isinstance(value, str) or not value for value in identity):
            return {}
        if directory.relative_to(expected).parts != identity:
            return {}
        result = self._json(directory / "metrics.json")
        if not result or any(result.get(key) != record.get(key) for key in ("model", "run_id", "test_id")):
            return {}
        split = "val" if record.get("action") == "validate" else "test"
        if result.get("split") != split:
            return {}
        names = ("loss", "mle_m", "mle_cm", "pa_mpjpe_m", "pa_mpjpe_cm",
                 "pck", "pck_percent", "auc", "auc_percent", "cd_m", "cd_cm")
        return {name: value for name in names if not isinstance(result.get(name), bool)
                and (value := _number(result.get(name))) is not None}

    def _adapt(self, record, path, *, archive=False):
        execution_id = record.get("execution_id")
        if not isinstance(execution_id, str) or not EXECUTION_ID.fullmatch(execution_id):
            return None
        if record.get("launch_source") not in ("cli", "gui"):
            return None
        if record.get("action") not in ("train", "validate", "evaluate", "infer"):
            return None
        if record.get("status") not in ACTIVE | TERMINAL | {"Needs review"}:
            return None
        repository = record.get("repository_root")
        if not isinstance(repository, str) or Path(repository).resolve() != self.root:
            return None
        if archive and record["status"] not in TERMINAL:
            archived_control = self._control(record)
            if not archived_control or archived_control.get("status") != "Stopped":
                return None
        if not archive and path.parent.name != execution_id:
            return None
        config = record.get("config")
        if not isinstance(config, dict) or not isinstance(record.get("model"), str):
            return None
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        metrics = {**metrics, **self._final_metrics(record)}
        status, phase, delayed = record["status"], record.get("phase") or "initializing", False
        error = record.get("error")
        if status in ACTIVE:
            heartbeat = _number(record.get("heartbeat_at"), 0)
            local_host = record.get("hostname") == socket.gethostname()
            if not local_host:
                status, error = "Needs review", "Execution belongs to another host; monitoring is read-only."
            elif time.time() - heartbeat > self.heartbeat_timeout:
                identity = record.get("process_identity")
                try:
                    alive = probe_process(identity) if isinstance(identity, dict) else "unknown"
                except (OSError, ValueError, TypeError):
                    alive = "unknown"
                if alive == "dead":
                    status, phase = "Interrupted", "interrupted"
                    error = "Process exited without a final execution record."
                elif alive == "alive":
                    delayed = True
                else:
                    status, error = "Needs review", "The execution process identity could not be verified."
        control = self._control(record) if record["status"] in ACTIVE else None
        controlled = False
        if control:
            state = probe_process(record.get("process_identity"))
            if state == "dead" and control.get("status") in ("Stopping", "Stopped"):
                status, phase, error = "Stopped", "stopped", None
                controlled = True
            elif state == "alive" and control.get("status") == "Stopping":
                status, phase, error, delayed = "Stopping", "stopping", None, False
            elif control.get("status") == "Failed":
                error = control.get("error") or "The execution could not be stopped."
        public = {key: copy.deepcopy(record.get(key)) for key in (
            "execution_id", "launch_source", "owner_job_id", "action", "model", "run_id", "training_run_id",
            "weight_id", "test_id", "created_at", "updated_at", "started_at", "finished_at", "return_code",
            "visual", "visual_pkl", "sample_count", "current_metrics", "metrics", "command_preview",
            "limit_batches", "partial", "allow_partial", "save_predictions", "config_file", "config_id", "config_sha256")}
        public.update(job_id=execution_id, origin="user", status=status, phase=phase,
                      status_delayed=delayed, error=error,
                      config={key: value for key, value in config.items() if not key.startswith("_")},
                      progress={**copy.deepcopy(metrics),
                                **(copy.deepcopy(record.get("progress")) if isinstance(record.get("progress"), dict)
                                   else {"kind": "indeterminate", "current": None, "total": None})},
                      output_dir=self._public_path(record.get("output_dir") or record.get("log_dir")),
                      checkpoint_path=self._public_path(record.get("checkpoint_path")),
                      log_cursor=max(0, int(_number(record.get("log_cursor"), 0))),
                      log_generation="engine:" + execution_id,
                      capabilities={"can_stop": status in ACTIVE and self._controllable(record),
                                    "can_requeue": False, "can_view_logs": True})
        if controlled:
            public["finished_at"] = control.get("finished_at") or time.time()
            public["return_code"] = 130
        public["metrics"] = copy.deepcopy(metrics)
        if not isinstance(public.get("current_metrics"), dict):
            public["current_metrics"] = {}
        return public

    def _records(self):
        runtime, archive = self._paths()
        result, source = {}, {}
        for archived, paths in ((False, runtime), (True, archive)):
            for path in paths:
                record = self._json(path)
                if record is None:
                    continue
                try:
                    public = self._adapt(record, path, archive=archived)
                except (OSError, ValueError, TypeError, OverflowError):
                    continue
                if public is not None and public["execution_id"] not in result:
                    result[public["execution_id"]] = public
                    source[public["execution_id"]] = record
        return result, source

    def _runtime_records_for_retention(self):
        runtime = self._safe(self.runtime)
        if runtime is None or not runtime.is_dir():
            return []
        records = []
        for directory in runtime.iterdir():
            if not EXECUTION_ID.fullmatch(directory.name):
                continue
            path = directory / "status.json"
            raw = self._json(path)
            if raw is None:
                continue
            try:
                public = self._adapt(raw, path)
            except (OSError, ValueError, TypeError, OverflowError):
                continue
            if public is not None:
                records.append((raw, public))
        return records

    def forget_deleted_outputs(self, paths, *, recovery_root=None):
        outputs = set()
        for value in paths:
            output = deleted_logs.output_path(self.root, value)
            if output is None or output.exists():
                raise ValueError("Only deleted canonical log directories can be forgotten")
            outputs.add(output)
        with self._lock:
            records = [(raw["execution_id"], raw) for raw, public in self._runtime_records_for_retention()
                       if deleted_logs.record_output(self.root, raw) in outputs
                       and deleted_logs.inactive({**raw, "status": public["status"]})]
            removed = deleted_logs.archive(self.root, "executions", records,
                lambda identifier: self.runtime / identifier, recovery_root=recovery_root)
            if removed:
                self._inventory_at = self._archive_at = -math.inf
                self._runtime_paths = []; self._archive_paths = []
                self._json_cache.clear(); self._log_positions.clear()
            return {"deleted_execution_ids": removed}

    def prune_deleted_logs(self, *, force=False):
        with self._lock:
            now = time.monotonic()
            if not force and now - getattr(self, "_retention_at", -math.inf) < 1:
                return {"deleted_execution_ids": []}
            self._retention_at = now
            receipts = deleted_logs.receipts(self.root)
            paths = {deleted_logs.record_output(self.root, raw)
                     for raw, public in self._runtime_records_for_retention()
                     if deleted_logs.inactive({**raw, "status": public["status"]})
                     and deleted_logs.was_deleted(self.root, {**raw, "status": public["status"]}, receipts)}
            try:
                return self.forget_deleted_outputs(paths)
            except (OSError, ValueError):
                return {"deleted_execution_ids": [], "retention_error": "Execution metadata could not be moved to recovery storage."}

    def list_jobs(self):
        self.prune_deleted_logs()
        with self._lock:
            records, _ = self._records()
            return sorted(records.values(), key=lambda item: _number(item.get("created_at"), 0), reverse=True)[:self.limit]

    def status(self, execution_id):
        if not isinstance(execution_id, str) or not EXECUTION_ID.fullmatch(execution_id):
            return None
        self.prune_deleted_logs()
        with self._lock:
            records, _ = self._records()
            return records.get(execution_id)

    def process_state(self, execution_id):
        """Resolve ownership conflicts without exposing private process fields."""
        with self._lock:
            records, raw = self._records()
            record = raw.get(execution_id)
            if execution_id not in records or not record or record.get("hostname") != socket.gethostname():
                return "unknown"
            try:
                return probe_process(record.get("process_identity"))
            except (OSError, ValueError, TypeError):
                return "unknown"

    def request_stop(self, execution_id):
        """Persist one authenticated stop request; never signal an unverified PID."""
        with self._lock:
            records, raw = self._records()
            public, record = records.get(execution_id), raw.get(execution_id)
            if public is None:
                return None
            if public["status"] in TERMINAL:
                return public
            if not record or not self._controllable(record):
                raise PermissionError("The execution's host and process creation identity must be verified before stopping.")
            path = self._safe(self.runtime / execution_id / "control.json")
            if path is None or not path.parent.is_dir():
                raise PermissionError("The execution control directory is unavailable.")
            control = self._control(record)
            if not control or control.get("status") != "Stopping":
                now = time.time()
                control = {"schema_version": 1, "action": "stop", "execution_id": execution_id,
                           "repository_root": str(self.root), "process_identity": record["process_identity"],
                           "request_id": "stop_" + uuid.uuid4().hex, "requested_at": now,
                           "deadline": now + self.stop_grace, "status": "Stopping", "forced": False,
                           "finished_at": None, "error": None,
                           "log_dir": self._public_path(record.get("output_dir") or record.get("log_dir"))}
                self._write_control(control)
            running = self._stop_threads.get(execution_id)
            if running is None or not running.is_alive():
                thread = threading.Thread(target=self._finish_stop,
                    args=(execution_id, copy.deepcopy(control)), name="execution-stop-" + execution_id, daemon=True)
                self._stop_threads[execution_id] = thread
                thread.start()
            public.update(status="Stopping", phase="stopping", error=None)
            return public

    def _finish_stop(self, execution_id, control):
        identity = control["process_identity"]
        path = self.runtime / execution_id / "control.json"
        try:
            while time.time() < control["deadline"]:
                with self._lock:
                    _, raw = self._records()
                    record = raw.get(execution_id)
                    if record and (record.get("output_dir") or record.get("log_dir")):
                        control["log_dir"] = self._public_path(record.get("output_dir") or record.get("log_dir"))
                    if record and record.get("status") in {"Complete", "Failed", "Stopped"}:
                        control.update(status=record["status"], finished_at=record.get("finished_at") or time.time())
                        self._write_control(control)
                        return
                state = probe_process(identity)
                if state == "dead":
                    break
                if state != "alive":
                    raise PermissionError("The process identity can no longer be verified; force stop was refused.")
                time.sleep(.05)
            state = probe_process(identity)
            if state == "alive":
                control["forced"] = bool(terminate_verified_tree(identity))
                deadline = time.monotonic() + 2
                while probe_process(identity) == "alive" and time.monotonic() < deadline:
                    time.sleep(.02)
            if probe_process(identity) != "dead":
                raise PermissionError("The process exit could not be verified after the stop request.")
            control.update(status="Stopped", finished_at=time.time(), error=None)
        except (OSError, ValueError, TypeError) as exc:
            control.update(status="Failed", finished_at=time.time(), error=str(exc))
        try:
            with self._lock:
                current = self._json(path)
                if current and current.get("request_id") == control["request_id"]:
                    self._write_control(control)
        except (OSError, ValueError, TypeError):
            pass

    def logs(self, execution_id, *, after=0, limit=500):
        with self._lock:
            records, raw = self._records()
            record = raw.get(execution_id)
            if record is None:
                return None
            public = records[execution_id]
            path = self._console(record)
            after, limit = max(0, int(after)), max(1, min(int(limit), 1000))
            lines, cursor = [], after
            if path is not None and path.is_file():
                stat = path.stat()
                signature = (stat.st_dev, getattr(stat, "st_ino", 0))
                position = self._log_positions.get((path, after))
                reusable = position and position[0] == signature and position[1] <= stat.st_size
                offset = position[1] if reusable else 0
                discarding = position[2] if reusable else False
                with path.open("rb") as stream:
                    stream.seek(offset)
                    consumed = 0
                    while len(lines) < limit and consumed < MAX_LOG_PAGE_BYTES:
                        start = stream.tell()
                        remaining = MAX_LOG_PAGE_BYTES - consumed
                        if not discarding and remaining < MAX_LOG_LINE_BYTES + 1:
                            break
                        line = stream.readline(min(MAX_LOG_LINE_BYTES + 1, remaining))
                        if not line:
                            break
                        consumed += len(line)
                        if discarding:
                            discarding = not line.endswith(b"\n")
                            continue
                        if not line.endswith(b"\n"):
                            if len(line) <= MAX_LOG_LINE_BYTES:
                                stream.seek(start)
                                break
                            discarding = True
                            continue
                        try:
                            entry = json.loads(line)
                            sequence = entry.get("seq")
                            if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= cursor:
                                continue
                            text = entry.get("text")
                            if not isinstance(text, str):
                                continue
                            lines.append({"seq": sequence, "timestamp": _number(entry.get("timestamp"), 0),
                                          "stream": "stderr" if entry.get("stream") == "stderr" else "stdout",
                                          "text": text[:MAX_LOG_LINE_BYTES]})
                            cursor = sequence
                        except (ValueError, TypeError, AttributeError, UnicodeError):
                            continue
                    self._log_positions[(path, cursor)] = (signature, stream.tell(), discarding)
                    self._log_positions.move_to_end((path, cursor))
                    while len(self._log_positions) > 2048:
                        self._log_positions.popitem(last=False)
            return {"lines": lines, "next": cursor, "caught_up": cursor >= public["log_cursor"],
                    "generation": public["log_generation"]}


class JobMonitor:
    """Merge managed jobs and observed executions without adopting their processes."""

    def __init__(self, root, task_runner=None, *, registry=None):
        self.task_runner = task_runner
        self.registry = registry or ExecutionRegistry(root)

    def _managed_here(self, job_id):
        return bool(self.task_runner and self.task_runner.manages_job(job_id))

    def forget_deleted_outputs(self, paths, *, recovery_root=None, confirmed_dead_job_ids=()):
        paths = list(paths)
        managed = (self.task_runner.forget_deleted_outputs(paths, recovery_root=recovery_root,
                   confirmed_dead_job_ids=confirmed_dead_job_ids) if self.task_runner else {"deleted_job_ids": []})
        observed = self.registry.forget_deleted_outputs(paths, recovery_root=recovery_root)
        return {**managed, **observed}

    def prune_deleted_logs(self, *, force=False):
        prune = getattr(self.task_runner, "prune_deleted_logs", None)
        managed = prune(force=force) if callable(prune) else {"deleted_job_ids": []}
        observed = self.registry.prune_deleted_logs(force=force)
        return {**managed, **observed}

    def list_jobs(self):
        self.prune_deleted_logs()
        managed = self.task_runner.list_jobs() if self.task_runner else []
        result = {}
        execution_owners = {}
        for record in managed:
            item = dict(record, launch_source="gui", log_generation="parent:" + record["job_id"])
            item["capabilities"] = {"can_stop": self._managed_here(record["job_id"]) and
                                    record.get("status") in {"Queued", "Starting", "Running", "Stopping"},
                                    "can_requeue": False, "can_view_logs": True}
            result[record["job_id"]] = item
            if EXECUTION_ID.fullmatch(str(record.get("execution_id", ""))):
                execution_owners[record["execution_id"]] = record["job_id"]
        for observed in self.registry.list_jobs():
            owner = execution_owners.get(observed["execution_id"])
            if owner and observed.get("owner_job_id") == owner and observed.get("launch_source") == "gui":
                parent = result[owner]
                merged = dict(parent)
                merged.update({key: value for key, value in observed.items() if value is not None and key != "job_id"})
                merged["job_id"] = owner
                merged["error"] = observed.get("error")
                merged["capabilities"] = dict(parent["capabilities"],
                    can_stop=merged.get("status") in ACTIVE | {"Queued"} and
                    (self._managed_here(owner) or observed.get("capabilities", {}).get("can_stop", False)))
                parent_terminal = parent.get("status") in {"Failed", "Stopped"}
                artifact_failure = (parent.get("status") == "Failed" and parent.get("phase") == "missing-artifacts"
                                    and observed.get("status") == "Complete")
                observed_unfinished = observed.get("status") not in {"Complete", "Failed", "Stopped"}
                parent_observed_exit = (parent_terminal and parent.get("return_code") is not None and
                                        observed_unfinished and self.registry.process_state(observed["execution_id"]) == "dead")
                if artifact_failure or parent_observed_exit:
                    for key in ("status", "phase", "error", "finished_at", "return_code"):
                        merged[key] = parent.get(key)
                    merged["capabilities"]["can_stop"] = False
                result[owner] = merged
            else:
                result[observed["job_id"]] = observed
        return sorted(result.values(), key=lambda record: _number(record.get("created_at"), 0), reverse=True)[:200]

    def status(self, job_id):
        return next((record for record in self.list_jobs() if record["job_id"] == job_id), None)

    def logs(self, job_id, *, after=0, limit=500):
        record = self.status(job_id)
        if record is None:
            return None
        if record["log_generation"].startswith("engine:"):
            return self.registry.logs(record["execution_id"], after=after, limit=limit)
        if self.task_runner:
            result = self.task_runner.logs(job_id, after=after, limit=limit)
            if result is not None:
                result["generation"] = record["log_generation"]
            return result
        return None

    def stop(self, job_id):
        record = self.status(job_id)
        if record is None:
            return None
        if record.get("launch_source") == "gui" and self._managed_here(job_id):
            return self.task_runner.stop(job_id)
        result = self.registry.request_stop(record.get("execution_id"))
        if result is not None:
            result["job_id"] = job_id
        return result
