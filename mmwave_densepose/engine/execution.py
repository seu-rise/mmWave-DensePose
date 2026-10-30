"""Engine-owned execution snapshots, heartbeat and terminal-preserving journals."""
from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
import io
import inspect
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
import _thread

from .process_identity import current_process_identity


_CURRENT = ContextVar("mmwave_execution", default=None)
_CAPTURE_SUSPENDED = ContextVar("mmwave_capture_suspended", default=False)
_EXECUTION_ID = re.compile(r"exec_[0-9a-f]{32}\Z")
_JOB_ID = re.compile(r"job_[A-Za-z0-9_-]{1,90}\Z")
_MAX_LINE = 16384
_TERMINAL = {"Complete", "Failed", "Stopped"}


def current_execution():
    observer = _CURRENT.get()
    return observer if observer is not None and observer.data["pid"] == os.getpid() else None


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def _replace_file(source, destination):
    """Allow a short Windows read/share window without losing observation."""
    for attempt in range(12):
        try:
            source.replace(destination)
            return
        except OSError as error:
            if (not isinstance(error, PermissionError) and getattr(error, "winerror", None) not in (32, 33)) or attempt == 11:
                raise
            time.sleep(.01 * min(attempt + 1, 3))


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
        # Keep the same completed temporary file; retries total at most ~0.30 s.
        _replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _TerminalTee(io.TextIOBase):
    def __init__(self, terminal, observer, stream_name):
        self.terminal, self.observer, self.stream_name = terminal, observer, stream_name

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.terminal, "errors", "replace")

    def writable(self):
        return True

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()

    def write(self, text):
        result = self.terminal.write(text)
        if not _CAPTURE_SUSPENDED.get():
            self.observer.capture(self.stream_name, text)
        return result

    def flush(self):
        self.terminal.flush()
        if not _CAPTURE_SUSPENDED.get():
            self.observer.flush_stream(self.stream_name)


class ExecutionObserver:
    """Observe a run and honor explicit cancellation without changing numerical RNG."""

    def __init__(self, root, action, config, heartbeat_interval=2):
        self.root = Path(root).resolve()
        self.action, self.config = action, dict(config)
        self.heartbeat_interval = max(.02, float(heartbeat_interval))
        inherited = os.environ.get("MMWAVE_EXECUTION_ID")
        owner = os.environ.get("MMWAVE_OWNER_JOB_ID")
        self.execution_id = inherited or "exec_" + uuid.uuid4().hex
        self.owner_job_id = owner
        self.enabled = False
        self._warned = False
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._token = None
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._tees = None
        self._buffers = {"stdout": "", "stderr": ""}
        self._signal_handler = None
        self._cancel_requested = False
        self.directory = self.root / ".local/runtime/executions" / self.execution_id
        self.console = self.directory / "console.ndjson"
        self.formal_directory = None
        now = time.time()
        self.data = {"schema_version": 1, "execution_id": self.execution_id,
                     "launch_source": "gui" if owner else "cli", "owner_job_id": owner,
                     "repository_root": str(self.root), "hostname": socket.gethostname(),
                     "pid": os.getpid(), "process_identity": current_process_identity(),
                     "action": action, "model": config.get("model"), "config": self._public_config(config),
                     "status": "Starting", "phase": "initializing", "error": None,
                     "progress": {"kind": "indeterminate", "current": None, "total": None},
                     "metrics": {}, "log_cursor": 0, "created_at": now, "started_at": now,
                     "updated_at": now, "heartbeat_at": now, "finished_at": None,
                     "run_id": None, "training_run_id": None, "test_id": None,
                     "output_dir": None, "log_dir": None, "checkpoint_path": None,
                     "console_path": None, "return_code": None,
                     "stop_protocol_version": 1 if threading.current_thread() is threading.main_thread() else 0}

    def _public_config(self, config):
        result = _clean(config)
        if config.get("_public_data_root"):
            result["data_root"] = config["_public_data_root"]
        return result

    def _warn(self, exc):
        self.enabled = False
        self._stop.set()
        if not self._warned:
            self._warned = True
            try:
                self._stderr.write(f"Execution monitoring unavailable: {type(exc).__name__}: {exc}\n")
                self._stderr.flush()
            except (OSError, ValueError):
                pass

    def _relative(self, path):
        path = Path(path)
        path = (path if path.is_absolute() else self.root / path).resolve()
        return path.relative_to(self.root).as_posix()

    def _publish(self):
        if not self.enabled:
            return
        try:
            _atomic_json(self.directory / "status.json", self.data)
            if self.formal_directory is not None:
                _atomic_json(self.formal_directory / "execution.json", self.data)
        except (OSError, ValueError, TypeError) as exc:
            self._warn(exc)

    def __enter__(self):
        try:
            if not _EXECUTION_ID.fullmatch(self.execution_id):
                raise ValueError("Invalid inherited execution ID")
            if self.owner_job_id and not _JOB_ID.fullmatch(self.owner_job_id):
                raise ValueError("Invalid inherited owner job ID")
            parent = self.directory.parent
            if not parent.resolve().is_relative_to(self.root):
                raise ValueError("Execution directory must stay inside the repository")
            parent.mkdir(parents=True, exist_ok=True)
            self.directory.mkdir()  # Never overwrite a prior execution or share one ID between processes.
            self.data["console_path"] = self._relative(self.console)
            self.enabled = True
            self._publish()
            if self.enabled:
                self._token = _CURRENT.set(self)
                self._tees = (_TerminalTee(self._stdout, self, "stdout"), _TerminalTee(self._stderr, self, "stderr"))
                sys.stdout, sys.stderr = self._tees
                self._thread = threading.Thread(target=self._heartbeat, name="execution-heartbeat", daemon=True)
                self._thread.start()
                if threading.current_thread() is threading.main_thread() and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL:
                    self._signal_handler = signal.getsignal(signal.SIGTERM)
                    signal.signal(signal.SIGTERM, self._interrupt)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            self._warn(exc)
        return self

    @staticmethod
    def _interrupt(_signal, _frame):
        raise KeyboardInterrupt("Termination requested")

    def _heartbeat(self):
        while not self._stop.wait(self.heartbeat_interval):
            interrupt = False
            with self._lock:
                if self.data["status"] in _TERMINAL:
                    return
                interrupt = self._check_stop_request()
                self.data["heartbeat_at"] = time.time()
                self._publish()
            if interrupt:
                _thread.interrupt_main()

    def _check_stop_request(self):
        if self._cancel_requested or self.data.get("stop_protocol_version") != 1:
            return False
        path = self.directory / "control.json"
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 16384:
                return False
            request = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(request, dict) or request.get("action") != "stop"
                    or request.get("execution_id") != self.execution_id
                    or request.get("repository_root") != str(self.root)
                    or request.get("process_identity") != self.data["process_identity"]
                    or request.get("status") != "Stopping"):
                return False
            self._cancel_requested = True
            self.data.update(status="Stopping", phase="stopping", stop_request_id=request.get("request_id"),
                             updated_at=time.time())
            return True
        except (OSError, ValueError, TypeError):
            return False

    @contextmanager
    def suspend_console_capture(self):
        """Preserve terminal bar redraws without recording each carriage return."""
        token = _CAPTURE_SUSPENDED.set(True)
        try:
            yield
        finally:
            _CAPTURE_SUSPENDED.reset(token)

    def metadata(self):
        if not self.enabled:
            return {}
        return {key: self.data[key] for key in ("execution_id", "launch_source", "owner_job_id", "config_file", "config_id", "config_sha256") if key in self.data}

    def update(self, **fields):
        if not self.enabled or self.data["pid"] != os.getpid():
            return
        with self._lock:
            try:
                if self.data["status"] in _TERMINAL:
                    return
                if "config" in fields:
                    fields["config"] = self._public_config(fields["config"])
                if "metrics" in fields:
                    fields["metrics"] = {**self.data.get("metrics", {}), **_clean(fields["metrics"])}
                for key in ("phase", "progress", "config", "metrics", "sample_count", "limit_batches", "partial", "visual", "allow_partial", "save_predictions", "config_file", "config_id", "config_sha256"):
                    if key in fields:
                        self.data[key] = _clean(fields[key])
                self.data.update(status="Stopping" if self._cancel_requested else "Running",
                                 updated_at=time.time(), heartbeat_at=time.time())
                if self._cancel_requested:
                    self.data["phase"] = "stopping"
                self._publish()
            except (OSError, ValueError, TypeError) as exc:
                self._warn(exc)

    def attach_paths(self, paths):
        if not self.enabled or self.data["pid"] != os.getpid():
            return
        with self._lock:
            try:
                directory = Path(paths["log_dir"]).resolve()
                relative = self._relative(directory)
                if not directory.is_dir() or directory.is_symlink():
                    raise ValueError("Execution log directory must already exist")
                if self.formal_directory == directory:
                    return
                existing = directory / "execution.json"
                if existing.exists() and json.loads(existing.read_text(encoding="utf-8")).get("execution_id") != self.execution_id:
                    raise ValueError("Another execution owns this log directory")
                destination = directory / "console.ndjson"
                if destination.exists() and destination.stat().st_size:
                    raise ValueError("Refusing to replace an existing execution console")
                for stream in self._buffers:
                    self._flush_locked(stream)
                if self.console.exists():
                    _replace_file(self.console, destination)
                self.console = destination
                self.formal_directory = directory
                checkpoint = paths.get("checkpoint_path")
                try:
                    checkpoint = self._relative(checkpoint) if checkpoint else None
                except ValueError:
                    checkpoint = "external-weight/" + Path(checkpoint).name
                self.data.update(output_dir=relative, log_dir=relative,
                                 console_path=self._relative(destination), checkpoint_path=checkpoint,
                                 run_id=paths.get("run_id"), training_run_id=paths.get("run_id"),
                                 test_id=paths.get("test_id"), updated_at=time.time())
                self._publish()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self._warn(exc)

    def _emit_locked(self, stream, line):
        if not self.enabled:
            return
        self.data["log_cursor"] += 1
        entry = {"seq": self.data["log_cursor"], "timestamp": time.time(), "stream": stream, "text": line}
        with self.console.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()

    def capture(self, stream, text):
        if not self.enabled or self.data["pid"] != os.getpid() or not text:
            return
        with self._lock:
            try:
                buffer = self._buffers[stream] + text
                while buffer:
                    newline = min((index for index in (buffer.find("\n"), buffer.find("\r")) if index >= 0), default=-1)
                    if newline >= 0 and newline <= _MAX_LINE:
                        self._emit_locked(stream, buffer[:newline])
                        buffer = buffer[newline + 1:]
                    elif len(buffer) >= _MAX_LINE:
                        self._emit_locked(stream, buffer[:_MAX_LINE])
                        buffer = buffer[_MAX_LINE:]
                    else:
                        break
                self._buffers[stream] = buffer
            except (OSError, ValueError, TypeError) as exc:
                self._warn(exc)

    def _flush_locked(self, stream):
        if self._buffers[stream]:
            self._emit_locked(stream, self._buffers[stream])
            self._buffers[stream] = ""

    def flush_stream(self, stream):
        if not self.enabled or self.data["pid"] != os.getpid():
            return
        with self._lock:
            try:
                self._flush_locked(stream)
            except (OSError, ValueError, TypeError) as exc:
                self._warn(exc)

    def __exit__(self, exc_type, exc, tb):
        if self.data["pid"] != os.getpid():
            self._restore_context()
            return False
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            try:
                self._thread.join(timeout=self.heartbeat_interval + 1)
            except RuntimeError as error:
                self._warn(error)
        with self._lock:
            try:
                if exc is not None:
                    self.capture("stderr", "".join(traceback.format_exception(exc_type, exc, tb)))
                for stream in self._buffers:
                    self._flush_locked(stream)
                status, code = "Complete", 0
                if isinstance(exc, KeyboardInterrupt) or isinstance(exc, SystemExit) and exc.code == 130:
                    status, code = "Stopped", 130
                elif exc is not None and not (isinstance(exc, SystemExit) and exc.code in (None, 0)):
                    status, code = "Failed", exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
                now = time.time()
                self.data.update(status=status, phase=status.lower(), return_code=code,
                                 updated_at=now, heartbeat_at=now, finished_at=now,
                                 error=None if status in ("Complete", "Stopped") else f"{exc_type.__name__}: {exc}")
                self._publish()
            except (OSError, ValueError, TypeError) as error:
                self._warn(error)
            finally:
                self._restore_context()
        return False

    def _restore_context(self):
        try:
            if self._signal_handler is not None and threading.current_thread() is threading.main_thread():
                if signal.getsignal(signal.SIGTERM) is self._interrupt:
                    signal.signal(signal.SIGTERM, self._signal_handler)
                self._signal_handler = None
            if self._tees:
                if sys.stdout is self._tees[0]: sys.stdout = self._stdout
                if sys.stderr is self._tees[1]: sys.stderr = self._stderr
            if self._token is not None:
                _CURRENT.reset(self._token)
                self._token = None
        except (OSError, RuntimeError, ValueError) as error:
            self._warn(error)


def observe_run(function):
    """Wrap the common engine entrypoint; nested calls share the outer observer."""
    signature = inspect.signature(function)
    @wraps(function)
    def observed(action, config, *args, **kwargs):
        if current_execution() is not None:
            return function(action, config, *args, **kwargs)
        from .artifacts import PROJECT_ROOT
        bound = signature.bind_partial(action, config, *args, **kwargs)
        root = bound.arguments.get("artifact_root", kwargs.get("artifact_root")) or PROJECT_ROOT
        with ExecutionObserver(root, action, config):
            return function(action, config, *args, **kwargs)
    return observed
