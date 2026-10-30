"""Browser workbench with authenticated local and public read-only demo modes.

One server owns one shared dataset sequence. PKL requires operator consent at
startup; the browser can never grant it. Demo mode disables task execution.
"""

import argparse
import hmac
import http.client
from http.cookies import CookieError, SimpleCookie
import importlib.util
import ipaddress
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlsplit
import webbrowser

import numpy as np

from mmwave_densepose.data.inspection import (
    IMAGE_KEYS, SKELETON_EDGES_19, LoadCancelled, load_sequence,
)
from mmwave_densepose.web.settings import WebGuiSettings, load_settings, DEMO_MESSAGE

SUPPORTED = {".pkl", ".pickle", ".npz", ".npy"}
STATIC_ROOT = Path(__file__).with_name("static")
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".json": "application/json", ".png": "image/png",
                 ".svg": "image/svg+xml"}


class ViewerError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


def _inside(path, directory):
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _retire_deleted_run_records(server, index, result):
    """Retire matching runtime records only after the indexed logs reached trash."""
    from mmwave_densepose.web.log_index import LogIndex
    root = Path(getattr(index, "workspace_root", None) or index.repository_root).resolve()
    relative = PurePosixPath(result["recovery_manifest"])
    if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 5
            or relative.parts[:2] != (".local", "trash")
            or relative.parts[2] not in {"training-logs", "results"}
            or relative.name != "manifest.json"):
        raise ViewerError(409, "Deleted-log recovery metadata is invalid")
    path = root / relative
    if not LogIndex._no_links(path, root) or not 0 < path.stat().st_size <= 2 * 1024 * 1024:
        raise ViewerError(409, "Deleted-log recovery metadata is unavailable")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "deleted" or not isinstance(receipt.get("entries"), list):
        raise ViewerError(409, "Log deletion has not completed")
    outputs = []
    for entry in receipt["entries"]:
        name = entry.get("source") if isinstance(entry, dict) else None
        if not isinstance(name, str) or "\\" in name:
            raise ViewerError(409, "Deleted-log source is invalid")
        source = PurePosixPath(name)
        if (source.is_absolute() or ".." in source.parts or len(source.parts) not in {5, 6}
                or source.parts[:2] != ("mmwave_densepose", "logs")
                or source.parts[2] not in {"train", "test"}):
            raise ViewerError(409, "Deleted-log source is invalid")
        outputs.append(root / source)
    result.update(server.job_monitor.forget_deleted_outputs(outputs, recovery_root=path.parent))
    return result


def _safe_json(value):
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class ViewerState:
    """Bounded, cooperative single-loader state shared by all connected clients."""

    def __init__(self, data_root, trust_pickle=False):
        self.data_root = Path(data_root).expanduser().resolve()
        if self.data_root.exists() and not self.data_root.is_dir():
            raise ValueError("Dataset root must be a directory")
        self.root = self.data_root
        for name in ("raw_face_anonymized", "raw"):
            candidate = self.data_root / name
            if candidate.is_dir():
                if not _inside(candidate.resolve(), self.data_root):
                    raise ValueError("Dataset directory must not escape its configured root")
                self.root = candidate.resolve()
                break
        self.trust_pickle = bool(trust_pickle)
        self.lock = threading.RLock()
        self.loading, self.progress, self.message, self.error = False, 0, "Choose a sequence", None
        self.generation, self.record, self.file_id, self.subject_id = 0, None, None, "not stored"
        self.cancelled = threading.Event()
        self.worker = None

    def path_for(self, file_id):
        if not isinstance(file_id, str) or not file_id or "\\" in file_id or "\x00" in file_id:
            raise ViewerError(400, "Invalid sequence ID")
        relative = PurePosixPath(file_id)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.parts[0] in (".", ".."):
            raise ViewerError(400, "Sequence ID must be a listed file")
        path = (self.root / file_id).resolve()
        if not _inside(path, self.root):
            raise ViewerError(403, "Sequence is outside the authorized directory")
        if path.suffix.lower() not in SUPPORTED or not path.is_file() or _is_index(path):
            raise ViewerError(404, "Sequence not found")
        return path

    def files(self):
        # The GUI can start before the first dataset download. Refresh the
        # source location after installation without recreating viewer state.
        for name in ("raw_face_anonymized", "raw"):
            candidate = self.data_root / name
            if candidate.is_dir():
                if not _inside(candidate.resolve(), self.data_root):
                    raise ViewerError(403, "Dataset directory must not escape its configured root")
                self.root = candidate.resolve()
                break
        else:
            self.root = self.data_root
        files = []
        if not self.root.is_dir():
            return {"files": [], "trust_pickle": self.trust_pickle, "root_label": self.root.name}
        for path in self.root.iterdir():
            try:
                resolved = self.path_for(path.name)
                kind = "pkl" if resolved.suffix.lower() == ".pickle" else resolved.suffix.lower()[1:]
                files.append({"id": path.name, "name": path.name,
                              "bytes": resolved.stat().st_size, "kind": kind})
            except (OSError, ViewerError):
                continue
        files.sort(key=lambda item: [int(part) if part.isdigit() else part.lower()
                                    for part in re.split(r"(\d+)", item["name"])])
        return {"files": files, "trust_pickle": self.trust_pickle, "root_label": self.root.name}

    def manifest_metadata(self, path):
        """Read dataset checksums and subject identifiers from installed manifests."""
        directories = [self.root, self.root.parent, self.data_root]
        if self.root.name == "raw_face_anonymized":
            directories.append(self.root.parent / "raw")
        subject = "not stored"
        checksum = None
        for directory in dict.fromkeys(directories):
            manifest_path = directory / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                sequences = manifest["sequences"]
                if not isinstance(sequences, dict):
                    raise ValueError()
                metadata_only = self.root.name == "raw_face_anonymized" and directory == self.root.parent / "raw"
                if manifest.get("dataset_kind") != "full_multimodal":
                    entry = sequences.get(path.name)
                    if entry is not None:
                        if not isinstance(entry, str):
                            raise ValueError()
                        subject = entry or "not stored"
                    continue
                if metadata_only:
                    continue
                for entry in sequences.values():
                    if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                        raise ValueError()
                    if (directory / entry["file"]).resolve() == path:
                        digest = entry.get("sha256")
                        if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
                            raise ValueError()
                        checksum = digest
                        subject = str(entry.get("subject_id") or "not stored")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ViewerError(400, "Invalid dataset manifest") from exc
        return checksum, subject

    def status(self):
        with self.lock:
            summary = None if self.record is None else {
                **self.record.summary, "file_id": self.file_id, "subject_id": self.subject_id}
            return {"loading": self.loading, "progress": self.progress, "message": self.message,
                    "error": self.error, "generation": self.generation, "sequence": summary}

    def open(self, file_id):
        path = self.path_for(file_id)
        if path.suffix.lower() in (".pkl", ".pickle") and not self.trust_pickle:
            raise ViewerError(403, "PKL loading requires server startup with --trust-pickle")
        checksum, subject = self.manifest_metadata(path)
        with self.lock:
            if self.loading:
                raise ViewerError(409, "Another sequence is loading; cancel it first")
            self.generation += 1
            self.record, self.file_id, self.subject_id = None, file_id, subject
            self.loading, self.progress, self.message, self.error = True, 0, "Loading sequence", None
            self.cancelled = threading.Event()
            self.worker = threading.Thread(target=self._load, args=(path, checksum, self.cancelled), daemon=True)
            self.worker.start()

    def _load(self, path, checksum, cancel):
        def progress(percent, message):
            with self.lock:
                self.progress, self.message = min(percent * 9 // 10, 90), message
        try:
            record = load_sequence(path, trusted=self.trust_pickle, expected_sha256=checksum,
                                   progress=progress, cancelled=cancel.is_set)
            bounds = {}
            for index in range(record.frame_count):
                if cancel.is_set():
                    raise LoadCancelled()
                points = record.frame(index)["radar"]
                for channel in (3, 4):
                    if points.shape[1] > channel:
                        finite = points[:, channel][np.isfinite(points[:, channel])]
                        if len(finite):
                            low, high = float(finite.min()), float(finite.max())
                            old = bounds.get(channel, (low, high))
                            bounds[channel] = (min(low, old[0]), max(high, old[1]))
                if index % 64 == 0:
                    with self.lock:
                        self.progress, self.message = 90 + 9 * index // record.frame_count, "Preparing color ranges"
            record.summary["color_bounds"] = bounds
            with self.lock:
                if cancel.is_set():
                    raise LoadCancelled()
                self.record, self.progress, self.message = record, 100, "Sequence ready"
        except LoadCancelled:
            with self.lock:
                self.message, self.error, self.progress = "Loading cancelled", None, 0
        except Exception as exc:
            with self.lock:
                if isinstance(exc, ValueError) and "SHA256 mismatch" in str(exc):
                    self.error = "Sequence SHA256 mismatch; file was not loaded"
                else:
                    self.error = "Unable to load sequence: invalid, unreadable, or unsupported data"
                self.message, self.progress = "Loading failed", 0
        finally:
            with self.lock:
                self.loading = False

    def frame(self, generation, index):
        with self.lock:
            if self.record is None or generation != self.generation:
                raise ViewerError(409, "Sequence changed or is not ready; refresh status")
            record = self.record
        if not 0 <= index < record.frame_count:
            raise ViewerError(400, "Frame index is outside the sequence")
        frame = record.frame(index)
        with self.lock:
            if generation != self.generation:
                raise ViewerError(409, "Sequence changed; refresh status")
        return frame


def _is_index(path):
    return bool(re.search(r"(^|[_\-.])(index|indices|split|splits)([_\-.]|$)", path.stem, re.I))


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    service_name = "mmwave-densepose-webgui"
    # Windows SO_REUSEADDR lets a second live server steal the same port, so
    # the browser can unexpectedly reach another instance with another token.
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(self, address, state, token, *, repository_root=None, settings=None):
        self.state, self.token = state, token
        self.instance_id = secrets.token_urlsafe(24)
        self.repository_root = Path(repository_root or Path(__file__).resolve().parents[2]).resolve()
        self.settings = settings if settings is not None else load_settings(self.repository_root)
        if not isinstance(self.settings, WebGuiSettings):
            raise ValueError("WebGUI settings must be validated operator settings")
        self.output_root = self.repository_root / "mmwave_densepose" / "logs"
        self.runtime_root = self.repository_root / ".local" / "runtime"
        self.dataset_verification = None
        self.dataset_installation = None
        self.paper_installation = None
        self.task_runner = None
        self.job_monitor = None
        self.log_index = None
        self.cache_manager = None
        self.results_index = None
        self.tensorboard = None
        self.avatar_registry = None
        self._paper_tables_lock = threading.Lock()
        self._paper_tables_cached = None
        self._paper_tables_at = 0.0
        self._paper_tables_builder = None
        # Claim the socket before creating queues or touching shared service
        # state. The CLI requests a local graceful restart before this bind;
        # direct duplicate create_server() calls still fail without side effects.
        super().__init__(address, ViewerHandler)
        (self.output_root / "train").mkdir(parents=True, exist_ok=True)
        (self.output_root / "test").mkdir(parents=True, exist_ok=True)
        try:
            from mmwave_densepose.web.dataset_verification import DatasetVerificationService
            self.dataset_verification = DatasetVerificationService(
                self.repository_root, state.data_root, self.runtime_root / "dataset-verification",
                trust_pickle=state.trust_pickle)
        except Exception as exc:
            self.dataset_verification_error = str(exc)
        if self.dataset_verification is not None:
            try:
                from mmwave_densepose.web.dataset_installation import DatasetInstallationService
                self.dataset_installation = DatasetInstallationService(
                    self.dataset_verification, self.runtime_root / "dataset-installation",
                    cache_dir=self.repository_root / "data" / "downloads")
            except Exception as exc:
                self.dataset_installation_error = str(exc)
        if self.settings.allow_task_execution:
            try:
                from mmwave_densepose.gui.task_runner import TaskRunner
                self.task_runner = TaskRunner(state.data_root, trust_pickle=state.trust_pickle,
                                              project_root=self.repository_root)
            except Exception as exc:
                self.task_runner_error = str(exc)
        from mmwave_densepose.web.evaluation_weights import EvaluationWeights
        self.evaluation_weights = EvaluationWeights(self.repository_root, state.data_root)
        if self.settings.allow_task_execution:
            try:
                from mmwave_densepose.gui.execution_registry import JobMonitor
                self.job_monitor = JobMonitor(self.repository_root, self.task_runner)
            except Exception as exc:
                self.job_monitor_error = str(exc)
        try:
            from mmwave_densepose.web.cache_manager import CacheManager
            def cache_jobs():
                if self.job_monitor is None:
                    raise RuntimeError("Training activity could not be checked")
                return self.job_monitor.list_jobs()
            self.cache_manager = CacheManager(state.data_root, repository_root=self.repository_root,
                                              active_jobs=cache_jobs)
        except Exception as exc:
            self.cache_manager_error = str(exc)
        try:
            from mmwave_densepose.web.log_index import LogIndex, default_roots
            roots = default_roots(workspace_root=self.repository_root)
            self.log_index = LogIndex(roots, workspace_root=self.repository_root, actions={"train"})
        except Exception as exc:
            self.log_index_error = str(exc)
        try:
            from mmwave_densepose.web.avatar_registry import AvatarRegistry
            self.avatar_registry = AvatarRegistry()
        except Exception as exc:
            self.avatar_registry_error = str(exc)
        try:
            from mmwave_densepose.web.results_index import ResultsIndex
            self.results_index = ResultsIndex(self.output_root / "test",
                training_logs_root=self.output_root / "train", repository_root=self.repository_root, unified_logs=True,
                data_root=state.data_root, trusted_pickle=state.trust_pickle)
            from mmwave_densepose.web.tensorboard_runtime import NativeTensorBoardManager
            self.tensorboard = NativeTensorBoardManager(self.log_index,
                self.runtime_root / "tensorboard", self.repository_root)
        except Exception as exc:
            self.results_index_error = str(exc)
        try:
            from mmwave_densepose.web.paper_installation import PaperInstallationService
            self.paper_installation = PaperInstallationService(self.repository_root,
                on_complete=self.results_index.invalidate if self.results_index is not None else None)
        except Exception as exc:
            self.paper_installation_error = str(exc)
    def paper_tables(self):
        """Rebuild best-test cells with the same configuration matching as exports."""
        with self._paper_tables_lock:
            if self._paper_tables_cached is not None and time.monotonic()-self._paper_tables_at < 3:
                return self._paper_tables_cached
            if self._paper_tables_builder is None:
                source = self.repository_root / "paper" / "tables" / "paper-browser" / "build_tables.py"
                spec = importlib.util.spec_from_file_location("_workbench_paper_tables", source)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self._paper_tables_builder = module.build_snapshot
            self._paper_tables_cached = self._paper_tables_builder(self.repository_root)
            self._paper_tables_at = time.monotonic()
            return self._paper_tables_cached

    def server_close(self):
        self.state.cancelled.set()
        for service in (self.paper_installation, self.dataset_installation, self.dataset_verification, self.task_runner, self.tensorboard):
            close = getattr(service, "close", None) or getattr(service, "shutdown", None)
            if close is not None:
                try: close()
                except Exception: pass
        super().server_close()


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "LocalDatasetViewer"

    def log_message(self, *_args):
        pass  # Do not leak tokens, file names, or paths into access logs.

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def _send(self, status, body, content_type="application/json", frame_ancestors="'none'", headers=None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                             "img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors " + frame_ancestors)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionError, TimeoutError):
            # Browser AbortController may disconnect during headers or body.
            # In particular Windows raises ConnectionAbortedError, not reset.
            # Do not attempt a second error response on that closed socket.
            self.close_connection = True

    def _json(self, status, value, headers=None):
        self._send(status, json.dumps(_safe_json(value), allow_nan=False).encode("utf-8"), headers=headers)

    def _authorize(self, *, allow_tensorboard_cookie=False):
        host = self.headers.get("Host", "")
        try:
            parsed = urlsplit("http://" + host)
            hostname, port = parsed.hostname, parsed.port or 80
            if (parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment
                    or not hostname or any(c.isspace() for c in host) or "\\" in host):
                raise ValueError()
            public = urlsplit(self.server.settings.public_origin) if self.server.settings.public_origin else None
            public_port = (public.port or (443 if public.scheme == "https" else 80)) if public else None
            proxied = public is not None and hostname.lower() == public.hostname and (parsed.port or public_port) == public_port
            if proxied:
                allowed, scheme = True, public.scheme
            else:
                if port != self.server.server_port:
                    raise ValueError()
                scheme = "http"
                bound = self.server.server_address[0]
                if bound in ("127.0.0.1", "localhost"):
                    allowed = hostname in ("127.0.0.1", "localhost")
                else:
                    address = ipaddress.ip_address(hostname)
                    allowed = address.version == 4 and not address.is_unspecified and (
                        bound == "0.0.0.0" or str(address) == bound)
            if not allowed:
                raise ValueError()
        except ValueError as exc:
            raise ViewerError(403, "Untrusted request host") from exc
        origin = self.headers.get("Origin")
        if origin:
            try:
                parsed_origin = urlsplit(origin)
                origin_port = parsed_origin.port or (443 if parsed_origin.scheme == "https" else 80)
                expected_port = public_port if proxied else port
                if (parsed_origin.scheme != scheme or parsed_origin.hostname != hostname.lower()
                        or origin_port != expected_port or parsed_origin.username or parsed_origin.password
                        or parsed_origin.path or parsed_origin.query or parsed_origin.fragment):
                    raise ValueError()
            except ValueError as exc:
                raise ViewerError(403, "Cross-origin requests are not allowed") from exc
        if self.server.settings.mode == "demo":
            return
        supplied = self.headers.get("X-Viewer-Token", "")
        if hmac.compare_digest(supplied, self.server.token):
            return
        if allow_tensorboard_cookie and self.server.tensorboard:
            cookie = SimpleCookie()
            try:
                cookie.load(self.headers.get("Cookie", ""))
                supplied_cookie = cookie.get(f"ViewerTensorBoard_{self.server.server_port}")
                if supplied_cookie and hmac.compare_digest(supplied_cookie.value, self.server.tensorboard.proxy_cookie):
                    return
            except (CookieError, ValueError):
                pass
        if supplied:
            raise ViewerError(401, "Viewer session token is invalid or expired; the local service may have restarted")
        raise ViewerError(401, "Viewer token required")

    def _authorize_local_control(self):
        # Reverse-proxied visitors must never inherit the proxy's loopback identity.
        host = urlsplit("http://" + self.headers.get("Host", "")).hostname
        if (host not in {"127.0.0.1", "localhost", self.server.server_address[0]}
                or self.headers.get("X-Forwarded-For") or self.headers.get("X-Real-IP")):
            raise ViewerError(403, "Instance restart control is local-only")
        try:
            peer = ipaddress.ip_address(self.client_address[0])
            bound = ipaddress.ip_address(self.server.server_address[0])
            allowed = peer.is_loopback or (not bound.is_unspecified and peer == bound)
        except ValueError:
            allowed = False
        if not allowed:
            raise ViewerError(403, "Instance restart control is local-only")

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def _dispatch(self, method):
        try:
            url = urlsplit(self.path)
            if url.path == "/_local/instance" and method == "GET":
                self._authorize_local_control()
                return self._json(200, {"service": self.server.service_name,
                                        "instance_id": self.server.instance_id})
            if url.path == "/_local/shutdown" and method == "POST":
                self._authorize_local_control()
                supplied = self.headers.get("X-Viewer-Instance", "")
                if not hmac.compare_digest(supplied, self.server.instance_id):
                    raise ViewerError(403, "Invalid instance restart request")
                self._json(202, {"status": "shutting_down"})
                threading.Thread(target=self.server.shutdown, name="viewer-shutdown", daemon=True).start()
            elif url.path.startswith("/tensorboard/"):
                self._authorize(allow_tensorboard_cookie=True)
                if not self.server.settings.allow_task_execution:
                    raise ViewerError(403, DEMO_MESSAGE)
                self._tensorboard_proxy(method, url)
            elif url.path.startswith("/api/"):
                self._authorize()
                self._api(method, url)
            elif method == "GET" and url.path.startswith("/paper-assets/"):
                self._paper_asset(url.path)
            elif method == "GET":
                self._static(url.path)
            else:
                raise ViewerError(405, "Method not allowed")
        except ViewerError as exc:
            self._json(exc.status, {"error": exc.message})
        except (ValueError, TypeError, OverflowError, UnicodeError):
            self._json(400, {"error": "Invalid request"})
        except Exception:
            self._json(500, {"error": "Unable to serve request"})

    def _tensorboard_proxy(self, method, url):
        if method not in ("GET", "POST"):
            raise ViewerError(405, "Method not allowed")
        manager = self.server.tensorboard
        if manager is None:
            raise ViewerError(503, "Native TensorBoard is unavailable")
        try:
            host, port = manager.proxy_target(url.path)
        except ValueError as exc:
            raise ViewerError(404, str(exc)) from exc
        if self.headers.get("Transfer-Encoding"):
            raise ViewerError(400, "Transfer encoding is not supported")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= 8 * 1024 * 1024:
            raise ViewerError(413, "TensorBoard request is too large")
        body = self.rfile.read(length) if length else None
        request_headers = {key: self.headers[key] for key in ("Accept", "Content-Type") if key in self.headers}
        upstream = http.client.HTTPConnection(host, port, timeout=15)
        try:
            path = url.path + ("?" + url.query if url.query else "")
            upstream.request(method, path, body=body, headers=request_headers)
            response = upstream.getresponse()
            payload = response.read()
            self.send_response(response.status)
            for key in ("Content-Type", "Content-Encoding", "Location"):
                value = response.getheader(key)
                if value is not None:
                    if key == "Location":
                        value = value.removeprefix(f"http://{host}:{port}")
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            # TensorBoard owns its application document and bundled script/style policy.
            # The viewer's stricter document CSP is unchanged; only its own frame may embed this app.
            self.send_header("Content-Security-Policy", "frame-ancestors 'self'; object-src 'none'")
            self.end_headers()
            self.wfile.write(payload)
        except (OSError, http.client.HTTPException) as exc:
            self.close_connection = True
            raise ViewerError(502, "TensorBoard is unavailable") from exc
        finally:
            upstream.close()

    def _body(self):
        if self.headers.get("Transfer-Encoding"):
            raise ViewerError(400, "Transfer encoding is not supported")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ViewerError(400, "Invalid content length") from exc
        if not 0 < length <= 4096:
            raise ViewerError(400, "A small JSON object is required")
        if self.headers.get_content_type() != "application/json":
            raise ViewerError(415, "Content type must be application/json")
        try:
            body = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeError) as exc:
            raise ViewerError(400, "Malformed JSON body") from exc
        if not isinstance(body, dict):
            raise ViewerError(400, "JSON body must be an object")
        return body

    def _api(self, method, url):
        state = self.server.state
        parts = [part for part in url.path.split("/") if part]
        if not self.server.settings.allow_task_execution:
            # Browsing uses POST for loading/cancelling a sequence and for pure
            # metric queries. Every other POST remains a server-owned operation.
            if method == "POST" and url.path not in {
                    "/api/open", "/api/cancel", "/api/results/compare", "/api/logs/series"}:
                raise ViewerError(403, DEMO_MESSAGE)
        if url.path == "/api/capabilities" and method == "GET":
            return self._json(200, self.server.settings.capabilities())
        if url.path == "/api/paper/tables" and method == "GET":
            try:
                return self._json(200, self.server.paper_tables())
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ViewerError(503, "Official result records unavailable") from exc
        if url.path == "/api/avatar/models" and method == "GET":
            return self._json(200, self.server.avatar_registry.list_models() if self.server.avatar_registry else {"available": False, "models": []})
        if len(parts) == 5 and parts[:3] == ["api", "avatar", "models"] and parts[4] == "asset" and method == "GET":
            if self.server.avatar_registry is None: raise ViewerError(404, "Avatar model unavailable")
            try:
                body = self.server.avatar_registry.read_asset(parts[3])
            except (KeyError, OSError, ValueError) as exc: raise ViewerError(404, "Avatar model unavailable") from exc
            return self._send(200, body, "model/gltf-binary")
        if url.path == "/api/system/devices" and method == "GET":
            if not self.server.settings.allow_task_execution:
                return self._json(200, {"cuda_available": False, "devices": []})
            from mmwave_densepose.engine.devices import discover_devices
            return self._json(200, discover_devices())
        if url.path == "/api/dataset/cache" and method == "GET":
            manager = getattr(self.server, "cache_manager", None)
            if manager is None:
                raise ViewerError(503, "Dataset cache management unavailable")
            return self._json(200, manager.inventory())
        if url.path == "/api/dataset/cache/detail" and method == "GET":
            manager = getattr(self.server, "cache_manager", None)
            if manager is None:
                raise ViewerError(503, "Dataset cache management unavailable")
            query = parse_qs(url.query)
            if set(query) != {"cache_id"} or len(query["cache_id"]) != 1:
                raise ViewerError(400, "Select one indexed cache")
            try:
                return self._json(200, manager.detail(query["cache_id"][0]))
            except KeyError as exc:
                raise ViewerError(404, "Cache not found; refresh the list") from exc
            except (ValueError, PermissionError) as exc:
                raise ViewerError(400, str(exc)) from exc
        if url.path == "/api/dataset/cache/delete" and method == "POST":
            manager = getattr(self.server, "cache_manager", None)
            if manager is None:
                raise ViewerError(503, "Dataset cache management unavailable")
            body = self._body()
            if set(body) != {"cache_ids"} or not isinstance(body["cache_ids"], list):
                raise ViewerError(400, "Select indexed caches to delete")
            try:
                return self._json(200, manager.delete(body["cache_ids"]))
            except KeyError as exc:
                raise ViewerError(404, "Cache not found; refresh the list") from exc
            except (PermissionError, RuntimeError, TimeoutError) as exc:
                raise ViewerError(409, str(exc)) from exc
            except ValueError as exc:
                raise ViewerError(400, str(exc)) from exc
        if url.path == "/api/paper-assets" and method == "GET":
            if self.server.paper_installation is None:
                raise ViewerError(503, "Official asset installation unavailable")
            query = parse_qs(url.query)
            if set(query) - {"kind", "asset_id", "test_id"} or any(len(v) != 1 for v in query.values()):
                raise ViewerError(400, "Select an official asset kind, experiment and test")
            try:
                return self._json(200, self.server.paper_installation.inventory(**{k: v[0] for k, v in query.items()}))
            except (ValueError, OSError, KeyError) as exc:
                raise ViewerError(400, str(exc)) from exc
        if url.path == "/api/paper-assets/installations" and method == "POST":
            if self.server.paper_installation is None:
                raise ViewerError(503, "Official asset installation unavailable")
            body = self._body()
            if set(body) - {"package_ids", "repair"} or "package_ids" not in body:
                raise ViewerError(400, "Select official package IDs")
            try:
                return self._json(202, self.server.paper_installation.start(body["package_ids"], repair=body.get("repair", False)))
            except RuntimeError as exc:
                raise ViewerError(409, str(exc)) from exc
            except (ValueError, OSError, KeyError) as exc:
                raise ViewerError(400, str(exc)) from exc
        if len(parts) == 4 and parts[:3] == ["api", "paper-assets", "installations"] and method == "GET":
            job = self.server.paper_installation.status(parts[3]) if self.server.paper_installation else None
            if job is None:
                raise ViewerError(404, "Official asset installation not found")
            return self._json(200, job)
        if (len(parts) == 5 and parts[:3] == ["api", "paper-assets", "installations"]
                and parts[4] == "cancel" and method == "POST"):
            if self._body():
                raise ViewerError(400, "Cancellation accepts an empty object")
            job = self.server.paper_installation.cancel(parts[3]) if self.server.paper_installation else None
            if job is None:
                raise ViewerError(404, "Official asset installation not found")
            return self._json(200, job)
        if url.path == "/api/workflows/weights" and method == "GET":
            query=parse_qs(url.query)
            if set(query)!={"model"} or len(query["model"])!=1:
                raise ViewerError(400,"Select a model for evaluation weights")
            try:return self._json(200,self.server.evaluation_weights.list_weights(query["model"][0]))
            except (ValueError,TypeError) as exc:raise ViewerError(400,str(exc)) from exc
        if url.path == "/api/workflows" and method == "GET":
            from mmwave_densepose.workflows.catalog import get_catalog
            return self._json(200, get_catalog())
        if url.path == "/api/logs/runs" and method == "GET":
            if self.server.log_index is None: raise ViewerError(503, "Log index unavailable")
            query=parse_qs(url.query)
            allowed={"source","origin","task","model","split","seed","status","integrity","paper_table","search","after","before","limit","cursor"}
            if set(query)-allowed or any(len(v)!=1 for v in query.values()):raise ViewerError(400,"Invalid log filters")
            limit=int(query.pop("limit",["100"])[0]);cursor=query.pop("cursor",[None])[0]
            filters={k:v[0] for k,v in query.items()}
            return self._json(200,self.server.log_index.list_runs(filters,limit=limit,cursor=cursor))
        if url.path == "/api/logs/run" and method == "GET":
            if self.server.log_index is None: raise ViewerError(503, "Log index unavailable")
            q=parse_qs(url.query)
            if set(q)!={"run_id"} or len(q["run_id"])!=1:raise ViewerError(400,"Invalid run identifier")
            try:return self._json(200,self.server.log_index.get_run(q["run_id"][0]))
            except (KeyError,ValueError) as exc:raise ViewerError(404,"Log run not found") from exc
        if url.path == "/api/logs/runs/delete" and method == "POST":
            if self.server.log_index is None:
                raise ViewerError(503, "Log index unavailable")
            body = self._body()
            if set(body) != {"run_ids"} or not isinstance(body["run_ids"], list):
                raise ViewerError(400, "Select indexed training runs to delete")
            run_ids = body["run_ids"]
            if (not 1 <= len(run_ids) <= 500 or
                    any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value) for value in run_ids)):
                raise ViewerError(400, "Select one to 500 indexed training runs")
            try:
                for run_id in run_ids:
                    # Validate every ID before any mutation, including on servers
                    # where native TensorBoard is unavailable.
                    self.server.log_index.get_run(run_id)
                    if self.server.tensorboard:
                        info = self.server.tensorboard.info(run_id)
                        if info.get("instance_running") or info.get("instance_starting"):
                            raise ViewerError(409, "Stop Native TensorBoard for the selected run before deleting its logs")
                monitor = self.server.job_monitor
                if monitor is None:
                    raise ViewerError(503, "Training activity could not be checked; log deletion is unavailable")
                if not callable(getattr(monitor, "forget_deleted_outputs", None)):
                    raise ViewerError(503, "Run record cleanup is unavailable")
                active_paths = [job.get("output_dir") or job.get("log_dir") for job in monitor.list_jobs()
                                if job.get("status") in {"Queued", "Starting", "Running", "Stopping", "Needs review"}
                                and (job.get("output_dir") or job.get("log_dir"))]
                result = self.server.log_index.delete_runs(run_ids, active_paths=active_paths)
                return self._json(200, _retire_deleted_run_records(self.server, self.server.log_index, result))
            except KeyError as exc:
                raise ViewerError(404, "Training log not found; refresh the run list") from exc
            except PermissionError as exc:
                raise ViewerError(409, str(exc)) from exc
            except RuntimeError as exc:
                raise ViewerError(409, str(exc)) from exc
            except ValueError as exc:
                raise ViewerError(400, str(exc)) from exc
        if url.path == "/api/logs/tags" and method == "GET":
            if self.server.log_index is None: raise ViewerError(503, "Log index unavailable")
            q=parse_qs(url.query);run_ids=q.get("run_id",[""])[0].split(",")
            if not run_ids or any(not value for value in run_ids) or set(q)!={"run_id"} or len(run_ids)>8:raise ViewerError(400,"Select one to eight indexed runs")
            try:return self._json(200,self.server.log_index.list_tags(run_ids))
            except (KeyError,ValueError) as exc:raise ViewerError(404,"Log run not found") from exc
        if url.path == "/api/logs/series" and method == "POST":
            if self.server.log_index is None: raise ViewerError(503, "Log index unavailable")
            body=self._body()
            if set(body)-{"run_ids","tags","x_axis","smoothing","max_points"} or not isinstance(body.get("run_ids"),list) or not isinstance(body.get("tags"),list):raise ViewerError(400,"Invalid scalar query")
            try:return self._json(200,self.server.log_index.series(body["run_ids"],body["tags"],x_axis=body.get("x_axis","epoch"),smoothing=body.get("smoothing",0),max_points=body.get("max_points",1500)))
            except (KeyError,ValueError,TypeError) as exc:raise ViewerError(400,str(exc)) from exc
        if url.path == "/api/logs/tensorboard" and method == "GET":
            if self.server.log_index is None:raise ViewerError(503,"Log index unavailable")
            query=parse_qs(url.query)
            if set(query)-{"run_id"} or any(len(v)!=1 for v in query.values()):raise ViewerError(400,"Invalid TensorBoard query")
            try:
                info=self.server.tensorboard.info(query.get("run_id",[None])[0]) if self.server.tensorboard else self.server.log_index.tensorboard_info(query.get("run_id",[None])[0])
                return self._json(200,info)
            except (KeyError,ValueError) as exc:raise ViewerError(404,"TensorBoard run is unavailable") from exc
        if url.path == "/api/dataset/summary" and method == "GET":
            if self.server.dataset_verification is None:
                raise ViewerError(503, "Release summary unavailable")
            try:
                summary = self.server.dataset_verification.summary()
            except (OSError, ValueError) as exc:
                from mmwave_densepose.web.dataset_verification import VerificationError
                detail=str(exc) if isinstance(exc,VerificationError) else "Release metadata could not be read"
                raise ViewerError(503,f"Release overview unavailable: {detail}") from exc
            return self._json(200, {"release": summary, "target_root_label": state.data_root.name,
                "latest": self.server.dataset_verification.latest()})
        if url.path == "/api/dataset/installation" and method == "GET":
            if self.server.dataset_installation is None:
                raise ViewerError(503, "Dataset installation unavailable")
            return self._json(200, self.server.dataset_installation.availability())
        if url.path == "/api/dataset/repairs" and method == "GET":
            if self.server.dataset_installation is None:
                raise ViewerError(503, "Dataset installation unavailable")
            query = parse_qs(url.query)
            if set(query) != {"verification_id"} or len(query["verification_id"]) != 1:
                raise ViewerError(400, "Select a verification report")
            try:
                return self._json(200, self.server.dataset_installation.repair_plan(query["verification_id"][0]))
            except KeyError as exc:
                raise ViewerError(404, "Verification not found") from exc
            except RuntimeError as exc:
                raise ViewerError(409, str(exc)) from exc
            except ValueError as exc:
                raise ViewerError(400, str(exc)) from exc
        if url.path == "/api/dataset/installations" and method == "POST":
            if self.server.dataset_installation is None:
                raise ViewerError(503, "Dataset installation unavailable")
            body = self._body()
            if set(body) - {"action", "verification_id"} or "action" not in body:
                raise ViewerError(400, "Select download or repair")
            try:
                return self._json(202, self.server.dataset_installation.start(body["action"], body.get("verification_id")))
            except KeyError as exc:
                raise ViewerError(404, "Verification not found") from exc
            except RuntimeError as exc:
                raise ViewerError(409, str(exc)) from exc
            except ValueError as exc:
                raise ViewerError(400, str(exc)) from exc
        if len(parts) == 4 and parts[:3] == ["api", "dataset", "installations"] and method == "GET":
            job = self.server.dataset_installation.status(parts[3]) if self.server.dataset_installation else None
            if job is None:
                raise ViewerError(404, "Dataset installation not found")
            return self._json(200, job)
        if len(parts) == 5 and parts[:3] == ["api", "dataset", "installations"] and parts[4] == "cancel" and method == "POST":
            if self._body():
                raise ViewerError(400, "Cancel request must be empty")
            job = self.server.dataset_installation.cancel(parts[3]) if self.server.dataset_installation else None
            if job is None:
                raise ViewerError(404, "Dataset installation not found")
            return self._json(200, job)
        if url.path == "/api/workflows" and method == "GET":
            from mmwave_densepose.workflows.catalog import get_catalog
            return self._json(200, get_catalog())
        if url.path == "/api/dataset/verifications" and method == "POST":
            if self._body(): raise ViewerError(400, "Verification request must be empty")
            if self.server.dataset_verification is None: raise ViewerError(503, "Verification unavailable")
            try:
                service = self.server.dataset_installation
                report = service.start_verification() if service is not None else self.server.dataset_verification.start()
                return self._json(202, report)
            except (ValueError, RuntimeError) as exc: raise ViewerError(409, str(exc)) from exc
        if len(parts) == 4 and parts[:3] == ["api", "dataset", "verifications"]:
            job_id = parts[3]
            if method == "GET":
                report = self.server.dataset_verification.status(job_id) if self.server.dataset_verification else None
                if report is None: raise ViewerError(404, "Verification not found")
                return self._json(200, report)
        if len(parts) == 5 and parts[:3] == ["api", "dataset", "verifications"] and parts[4] == "cancel" and method == "POST":
            if self._body(): raise ViewerError(400, "Cancel request must be empty")
            report = self.server.dataset_verification.cancel(parts[3]) if self.server.dataset_verification else None
            if report is None: raise ViewerError(404, "Verification not found")
            return self._json(200, report)
        if parts == ["api", "jobs", "preview-layout"] and method == "POST":
            body = self._body()
            if self.server.task_runner is None:
                raise ViewerError(503, "Task layout preview unavailable")
            if set(body) != {"draft"} or not isinstance(body["draft"], dict):
                raise ViewerError(400, "Invalid layout draft")
            try:
                return self._json(200, self.server.task_runner.preview_layout(body["draft"]))
            except (ValueError, TypeError) as exc:
                raise ViewerError(400, str(exc)) from exc
        if parts[:2] == ["api", "jobs"] and method == "POST" and len(parts) == 3 and parts[2] == "validate":
            body = self._body()
            if set(body) != {"draft"} or self.server.task_runner is None: raise ViewerError(400 if self.server.task_runner else 503, "Task validation unavailable")
            try: return self._json(200, self.server.task_runner.validate(body["draft"]))
            except (ValueError, TypeError) as exc: raise ViewerError(400, str(exc)) from exc
        if parts == ["api", "jobs"]:
            if method == "GET":
                return self._json(200, {"jobs": self.server.job_monitor.list_jobs() if self.server.job_monitor else []})
            if method == "POST":
                body = self._body()
                if set(body) != {"draft", "report_id", "request_id"} or self.server.task_runner is None: raise ViewerError(400 if self.server.task_runner else 503, "Task submission unavailable")
                try: return self._json(202, self.server.task_runner.submit(body["draft"], report_id=body["report_id"], request_id=body["request_id"]))
                except (ValueError, TypeError, FileNotFoundError, FileExistsError) as exc: raise ViewerError(409, str(exc)) from exc
        if len(parts) == 3 and parts[:2] == ["api", "jobs"] and method == "GET":
            job = self.server.job_monitor.status(parts[2]) if self.server.job_monitor else None
            if job is None: raise ViewerError(404, "Job not found")
            return self._json(200, job)
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "logs" and method == "GET":
            query = parse_qs(url.query)
            if set(query) - {"after", "limit"} or any(len(v) != 1 for v in query.values()): raise ViewerError(400, "Invalid log cursor")
            after = int(query.get("after", ["0"])[0]); limit = int(query.get("limit", ["500"])[0])
            if after < 0 or not 1 <= limit <= 1000: raise ViewerError(400, "Invalid log pagination")
            if self.server.job_monitor is None: raise ViewerError(503, "Task monitoring unavailable")
            result=self.server.job_monitor.logs(parts[2],after=after,limit=limit)
            if result is None: raise ViewerError(404,"Job not found")
            return self._json(200,result)
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stop" and method == "POST":
            if self._body(): raise ViewerError(400, "Stop request must be empty")
            try:
                job = self.server.job_monitor.stop(parts[2]) if self.server.job_monitor else None
            except PermissionError as exc:
                raise ViewerError(403, str(exc)) from exc
            if job is None: raise ViewerError(404, "Job not found")
            return self._json(200, job)
        if parts == ["api", "logs"] and method == "GET":
            query=parse_qs(url.query)
            if any(len(v)!=1 for v in query.values()): raise ViewerError(400,"Invalid log filters")
            filters={k:v[0] for k,v in query.items() if k in {"source","task","model","split","seed","status","integrity","paper_table","search","after","before"}}
            try: return self._json(200,self.server.log_index.list_runs(filters,limit=int(query.get("limit",["100"])[0]),cursor=query.get("cursor",[None])[0]) if self.server.log_index else {"runs":[],"total":0,"next_cursor":None})
            except ValueError as exc: raise ViewerError(400,str(exc)) from exc
        if url.path == "/api/logs/tags" and method == "GET":
            query=parse_qs(url.query)
            if "run_id" not in query or any(len(v)!=1 for v in query.values()): raise ViewerError(400,"Select log runs")
            try:return self._json(200,self.server.log_index.list_tags(query["run_id"][0].split(",")) if self.server.log_index else {"runs":[]})
            except ValueError as exc:raise ViewerError(400,str(exc)) from exc
        if url.path == "/api/logs/series" and method == "GET":
            query=parse_qs(url.query)
            if not {"run_id","tag"}.issubset(query) or any(len(v)!=1 for v in query.values()): raise ViewerError(400,"Select runs and tags")
            try:return self._json(200,self.server.log_index.series(query["run_id"][0].split(","),query["tag"][0].split(","),x_axis=query.get("x_axis",["epoch"])[0],smoothing=float(query.get("smoothing",["0"])[0]),max_points=int(query.get("max_points",["1500"])[0])) if self.server.log_index else {"series":[],"missing":[]})
            except ValueError as exc:raise ViewerError(400,str(exc)) from exc
        if url.path == "/api/logs/tensorboard" and method == "GET":
            query=parse_qs(url.query)
            if set(query)-{"run_id"} or any(len(v)!=1 for v in query.values()):raise ViewerError(400,"Invalid TensorBoard query")
            if self.server.log_index is None:raise ViewerError(503,"Log index unavailable")
            try:return self._json(200,self.server.tensorboard.info(query.get("run_id",[None])[0]) if self.server.tensorboard else self.server.log_index.tensorboard_info(query.get("run_id",[None])[0]))
            except ValueError as exc:raise ViewerError(404,str(exc)) from exc
        if url.path == "/api/logs/tensorboard" and method == "POST":
            if self.server.tensorboard is None:raise ViewerError(503,"Native TensorBoard is unavailable")
            body=self._body()
            if body.get("action")=="start" and set(body) in ({"action"},{"action","run_id"}):
                try:
                    payload = self.server.tensorboard.start(body.get("run_id"))
                    cookie = f"ViewerTensorBoard_{self.server.server_port}=" + self.server.tensorboard.proxy_cookie + "; Path=/tensorboard/; HttpOnly; SameSite=Strict"
                    return self._json(202, payload, headers={"Set-Cookie": cookie})
                except (ValueError,KeyError,OSError) as exc:raise ViewerError(409,str(exc)) from exc
            if body.get("action")=="stop" and set(body) in ({"action"},{"action","run_id"}):
                return self._json(200,self.server.tensorboard.stop(body.get("run_id")))
            raise ViewerError(400,"Invalid TensorBoard operation")
        if parts[:2] == ["api", "results"] and method == "GET":
            result_query = parse_qs(url.query, keep_blank_values=True)
            if any(len(values) != 1 for values in result_query.values()):
                raise ViewerError(400, "Invalid result query")
            result_source = result_query.pop("source", ["user"])[0]
            if result_source not in ("user", "official"):
                raise ViewerError(400, "Result source must be user or official")
        if parts == ["api", "results", "models"] and method == "GET":
            if self.server.results_index is None: raise ViewerError(503,"Result index unavailable")
            if result_query: raise ViewerError(400,"Invalid result query")
            return self._json(200,self.server.results_index.list_models(source=result_source))
        if len(parts) == 5 and parts[:3] == ["api", "results", "models"] and parts[4] == "weights" and method == "GET":
            if self.server.results_index is None: raise ViewerError(503,"Result index unavailable")
            if result_query: raise ViewerError(400,"Invalid result query")
            return self._json(200,self.server.results_index.list_weights(parts[3],source=result_source))
        if len(parts) == 7 and parts[:3] == ["api", "results", "models"] and parts[4] == "weights" and parts[6] == "tests" and method == "GET":
            if self.server.results_index is None: raise ViewerError(503,"Result index unavailable")
            if result_query: raise ViewerError(400,"Invalid result query")
            return self._json(200,self.server.results_index.list_tests(parts[3],training_run_id=parts[5],source=result_source))
        if url.path == "/api/results/record" and method == "GET":
            query = dict(result_query)
            if set(query) != {"record_id"} or len(query["record_id"]) != 1: raise ViewerError(400, "Invalid record ID")
            record = self.server.results_index.get_record(query["record_id"][0],source=result_source) if self.server.results_index else None
            if record is None: raise ViewerError(404, "Test record not found")
            return self._json(200, record)
        if url.path == "/api/results/samples" and method == "GET":
            query = dict(result_query)
            if "record_id" not in query or any(len(v) != 1 for v in query.values()): raise ViewerError(400, "Invalid sample query")
            record_id=query.pop("record_id")[0]; limit=int(query.pop("limit",["100"])[0]); offset=int(query.pop("offset",["0"])[0]);sequence_id=query.pop("sequence_id",[None])[0]
            if query or not 1<=limit<=500 or offset<0: raise ViewerError(400, "Invalid sample pagination")
            if self.server.results_index is None: raise ViewerError(503,"Result index unavailable")
            if self.server.results_index.get_record(record_id,source=result_source,_refresh=False) is None: raise ViewerError(404,"Test record not found")
            return self._json(200,self.server.results_index.samples(record_id,limit=limit,offset=offset,sequence_id=sequence_id,source=result_source))
        if url.path == "/api/results/sample" and method == "GET":
            query=dict(result_query)
            if set(query)!={"record_id","sample_id"} or any(len(v)!=1 for v in query.values()): raise ViewerError(400, "Invalid sample ID")
            result=self.server.results_index.sample(query["record_id"][0],query["sample_id"][0],source=result_source) if self.server.results_index else None
            if result is None: raise ViewerError(404, "Sample not found")
            if self.server.avatar_registry and self.server.avatar_registry.available:
                record_id=query["record_id"][0];sample_id=query["sample_id"][0]
                session_id=f"{record_id}:{result.get('sequence_id','')}"
                try:
                    result["avatar_pose"]={
                        "gt":self.server.avatar_registry.pose(result.get("target"),session_id=session_id,source="kinect_gt",frame_id=sample_id),
                        "prediction":self.server.avatar_registry.pose(result.get("prediction"),session_id=session_id,source="inference",frame_id=sample_id),
                    }
                except Exception:
                    result["avatar_pose"]={"gt":None,"prediction":None}
            return self._json(200,result)
        if url.path == "/api/results/records/delete" and method == "POST":
            if self.server.results_index is None:
                raise ViewerError(503, "Result index unavailable")
            body = self._body()
            if set(body) != {"source", "record_ids"} or not isinstance(body.get("record_ids"), list):
                raise ViewerError(400, "Select indexed personal evaluation records to delete")
            if body.get("source") != "user":
                raise ViewerError(409, "Official evaluation logs cannot be deleted")
            monitor = getattr(self.server, "job_monitor", None)
            if monitor is None:
                raise ViewerError(503, "Evaluation activity could not be checked; deletion is unavailable")
            try:
                jobs = monitor.list_jobs()
            except (OSError, ValueError, RuntimeError) as exc:
                raise ViewerError(503, "Evaluation activity could not be checked; deletion is unavailable") from exc
            if not isinstance(jobs, (list, tuple)) or any(not isinstance(job, dict) for job in jobs):
                raise ViewerError(503, "Evaluation activity could not be checked; deletion is unavailable")
            if not callable(getattr(monitor, "forget_deleted_outputs", None)):
                raise ViewerError(503, "Run record cleanup is unavailable")
            active_paths = [job.get("output_dir") or job.get("log_dir") for job in jobs
                            if job.get("status") in {"Queued", "Starting", "Running", "Stopping", "Needs review"}
                            and (job.get("output_dir") or job.get("log_dir"))]
            try:
                result = self.server.results_index.delete_records(body["record_ids"], active_paths=active_paths)
                return self._json(200, _retire_deleted_run_records(self.server, self.server.results_index, result))
            except KeyError as exc:
                raise ViewerError(404, "Test record not found; refresh the result list") from exc
            except (PermissionError, RuntimeError) as exc:
                raise ViewerError(409, str(exc)) from exc
            except ValueError as exc:
                raise ViewerError(400, str(exc)) from exc
        if url.path == "/api/results/compare" and method == "POST":
            body=self._body()
            if set(body)-{"record_ids","threshold","source"} or not isinstance(body.get("record_ids"),list): raise ViewerError(400, "Invalid comparison request")
            if self.server.results_index is None: raise ViewerError(503,"Result index unavailable")
            try: result=self.server.results_index.compare(body["record_ids"],threshold=float(body.get("threshold",.15)),source=body.get("source","user"))
            except (TypeError,ValueError) as exc: raise ViewerError(400,str(exc)) from exc
            return self._json(200,result)
        if method == "GET" and url.path == "/api/files":
            return self._json(200, state.files())
        if method == "GET" and url.path == "/api/status":
            return self._json(200, state.status())
        if method == "POST" and url.path == "/api/open":
            body = self._body()
            if set(body) != {"id"}:
                raise ViewerError(400, "Only a sequence ID is accepted")
            state.open(body["id"])
            return self._json(202, state.status())
        if method == "POST" and url.path == "/api/cancel":
            if self._body():
                raise ViewerError(400, "Cancel body must be an empty object")
            state.cancelled.set()
            return self._json(200, state.status())
        if method == "GET" and url.path in ("/api/frame", "/api/image"):
            query = parse_qs(url.query)
            expected = {"generation", "index"} | ({"key"} if url.path.endswith("image") else set())
            if set(query) != expected or any(len(values) != 1 for values in query.values()):
                raise ViewerError(400, "Invalid frame query")
            generation, index = int(query["generation"][0]), int(query["index"][0])
            frame = state.frame(generation, index)
            if url.path.endswith("image"):
                key = query["key"][0]
                if key not in IMAGE_KEYS:
                    raise ViewerError(400, "Unknown image stream")
                image = frame["images"][key]
                if image is None:
                    raise ViewerError(404, "No image in this frame")
                if image.startswith(b"\xff\xd8\xff"):
                    content_type = "image/jpeg"
                elif image.startswith(b"\x89PNG\r\n\x1a\n"):
                    content_type = "image/png"
                else:
                    raise ViewerError(415, "Only encoded JPEG and PNG images can be displayed")
                return self._send(200, image, content_type)
            points = frame["radar"]
            points = points[np.isfinite(points[:, :3]).all(axis=1)]
            images = {key: None if value is None else {"url": f"/api/image?generation={generation}"
                      f"&index={index}&key={quote(key)}"} for key, value in frame["images"].items()}
            count = 0 if frame["joints"] is None else len(frame["joints"])
            return self._json(200, {"generation": generation, "index": index, "radar": points,
                "joints": frame["joints"], "edges": [edge for edge in SKELETON_EDGES_19 if max(edge) < count],
                "radar_timestamp": frame["radar_timestamp"], "kinect_timestamp": frame["kinect_timestamp"],
                "image_timestamps": frame["image_timestamps"], "images": images})
        raise ViewerError(404, "API endpoint not found")

    def _paper_asset(self, request_path):
        relative = request_path.removeprefix("/paper-assets/")
        if not relative or "%" in relative or "\\" in relative or "\x00" in relative:
            raise ViewerError(404, "Official asset not found")
        parts = PurePosixPath(relative)
        if parts.is_absolute() or any(p in (".", "..") for p in parts.parts): raise ViewerError(404, "Official asset not found")
        if parts.parts[0] == "tables":
            root = (self.server.repository_root / "paper" / "tables").resolve()
            parts = PurePosixPath(*parts.parts[1:])
        else:
            root = (self.server.repository_root / "paper" / "figures").resolve()
        if not parts.parts: raise ViewerError(404, "Official asset not found")
        path = (root / Path(*parts.parts)).resolve()
        if not _inside(path, root) or not path.is_file(): raise ViewerError(404, "Official asset not found")
        mime = {".html":"text/html; charset=utf-8", ".js":"text/javascript; charset=utf-8", ".css":"text/css; charset=utf-8", ".json":"application/json; charset=utf-8", ".png":"image/png", ".jpg":"image/jpeg", ".jpeg":"image/jpeg", ".csv":"text/csv; charset=utf-8", ".txt":"text/plain; charset=utf-8"}.get(path.suffix.lower())
        if not mime: raise ViewerError(404, "Official asset type not served")
        frame="'self'" if path.suffix.lower()==".html" else "'none'"
        self._send(200,path.read_bytes(),mime,frame_ancestors=frame)

    def _static(self, request_path):
        # Browser URL-decoding is not used as filesystem input: encoded traversal is rejected.
        # Old prototype links display the same approved frontend, preserving
        # their fragment token without a duplicate HTML/CSS implementation.
        relative = "index.html" if request_path in ("/", "/prototype.html") else request_path.lstrip("/")
        if "%" in relative or "\\" in relative or "\x00" in relative:
            raise ViewerError(404, "Page not found")
        path = (STATIC_ROOT / relative).resolve()
        if not _inside(path, STATIC_ROOT.resolve()) or path.suffix not in CONTENT_TYPES or not path.is_file():
            raise ViewerError(404, "Page not found")
        self._send(200, path.read_bytes(), CONTENT_TYPES[path.suffix])


def _local_probe_host(host):
    return "127.0.0.1" if host in ("localhost", "0.0.0.0") else host


def _port_is_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _stop_existing_instance(host, port):
    """Gracefully replace this WebGUI only; never terminate an unrelated listener."""
    if port == 0:
        return
    probe_host = _local_probe_host(host)
    if not _port_is_open(probe_host, port):
        return

    try:
        connection = http.client.HTTPConnection(probe_host, port, timeout=1.0)
        try:
            connection.request("GET", "/_local/instance")
            response = connection.getresponse()
            server_header = response.getheader("Server", "")
            identity = json.loads(response.read().decode("utf-8")) if response.status == 200 else {}
        finally:
            connection.close()
    except (OSError, http.client.HTTPException, UnicodeError, json.JSONDecodeError) as exc:
        raise OSError(f"Port {port} is occupied but the WebGUI instance could not be confirmed; stop it manually") from exc

    if (not isinstance(identity, dict) or identity.get("service") != ViewerServer.service_name
            or not isinstance(identity.get("instance_id"), str) or not identity["instance_id"]):
        if server_header.startswith("LocalDatasetViewer"):
            raise OSError(f"An older WebGUI instance is using port {port}; stop it once manually, then relaunch")
        raise OSError(f"Port {port} is occupied by another service; refusing to terminate it")

    try:
        connection = http.client.HTTPConnection(probe_host, port, timeout=1.0)
        try:
            connection.request("POST", "/_local/shutdown",
                               headers={"X-Viewer-Instance": identity["instance_id"]})
            response = connection.getresponse()
            response.read()
            if response.status != 202:
                raise OSError(f"The existing WebGUI refused its restart request (HTTP {response.status})")
        finally:
            connection.close()
    except (http.client.HTTPException, OSError) as exc:
        raise OSError("Unable to request a graceful WebGUI restart") from exc

    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if not _port_is_open(probe_host, port):
            return
        time.sleep(0.1)
    raise OSError(f"The existing WebGUI did not release port {port} in time")


def create_server(data_root, host="127.0.0.1", port=8765, trust_pickle=False, token=None, *, repository_root=None, settings=None):
    """Construct without starting; tests can bind an ephemeral port and own lifecycle."""
    if host != "localhost":
        address = ipaddress.ip_address(host)
        if address.version != 4:
            raise ValueError("This viewer currently supports IPv4 hosts only")
    return ViewerServer((host, port), ViewerState(data_root, trust_pickle), token or secrets.token_urlsafe(32),
                        repository_root=repository_root, settings=settings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/mmwave_densepose")
    parser.add_argument("--config", help="Operator YAML settings; defaults to webgui.local.yaml when present")
    parser.add_argument("--host", default="127.0.0.1", help="Loopback by default; 0.0.0.0 explicitly enables LAN access")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--trust-pickle", action="store_true", help="Operator consent to execute trusted dataset PKL files")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(Path(__file__).resolve().parents[2], args.config)
        _stop_existing_instance(args.host, args.port)
        server = create_server(args.data_root, args.host, args.port, args.trust_pickle, settings=settings)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    base = settings.public_origin or f"http://{display_host}:{server.server_port}"
    url = base + ("/" if settings.mode == "demo" else f"/#token={server.token}")
    print(f"WebGUI mode: {settings.mode}. One loaded dataset sequence is shared by all clients.", flush=True)
    if settings.mode == "demo":
        print(DEMO_MESSAGE, flush=True)
    if args.host not in ("localhost", "127.0.0.1") and not settings.public_origin:
        print("WARNING: LAN mode is unencrypted and not suitable for public or production hosting. "
              "Use only a trusted network, keep the token private, and prefer an SSH tunnel.", flush=True)
        if args.host == "0.0.0.0":
            print("For another computer, replace 127.0.0.1 with this machine's LAN IPv4 address.", flush=True)
    print(url, flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping dataset viewer.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
