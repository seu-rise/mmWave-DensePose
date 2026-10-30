"""Owned, loopback-only TensorBoard processes served through the viewer proxy."""
from __future__ import annotations

import http.client
import importlib.util
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import unquote


class NativeTensorBoardManager:
    def __init__(self, log_index, root, repository_root, *, host="127.0.0.1"):
        self.log_index = log_index
        self.root = Path(root).resolve()
        self.repository_root = Path(repository_root).resolve()
        self.host = "127.0.0.1"
        self.proxy_cookie = secrets.token_urlsafe(32)
        self._lock = threading.RLock()
        self._instances = {}
        self.available = importlib.util.find_spec("tensorboard") is not None

    @staticmethod
    def _free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def _refresh(self, instance):
        if instance["process"].poll() is not None:
            instance["status"] = "Stopped" if instance["status"] == "Stopped" else "Failed"
            return
        if instance["status"] != "Starting":
            return
        connection = http.client.HTTPConnection(self.host, instance["port"], timeout=.2)
        try:
            connection.request("GET", instance["prefix"] + "/data/environment")
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                instance["status"] = "Running"
        except (OSError, http.client.HTTPException):
            pass
        finally:
            connection.close()
        if instance["status"] == "Starting" and time.monotonic() - instance["started_at"] > 60:
            instance["process"].terminate()
            instance["status"] = "Failed"

    def _public(self, instance):
        self._refresh(instance)
        running = instance["status"] == "Running"
        return {"instance_running": running, "instance_starting": instance["status"] == "Starting",
                "run_id": instance["run_id"], "port": instance["port"],
                "url": instance["prefix"] + "/" if running else None,
                "status": instance["status"], "embedded": False,
                "scope": "run" if instance["run_id"] else "all", "run_count": instance["run_count"]}

    def info(self, run_id=None):
        if run_id is not None and (not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id)):
            raise ValueError("unknown run id")
        payload = self.log_index.tensorboard_info(run_id)
        payload["executable_available"] = self.available
        payload["capabilities"]["native_tensorboard"] = bool(payload["available"] and self.available)
        if payload["available"]:
            payload["status"] = "ready" if self.available else "executable_unavailable"
        with self._lock:
            instance = self._instances.get(run_id or "__all__")
            public = self._public(instance) if instance else None
            aggregate = self._instances.get("__all__") if run_id else None
            if aggregate and run_id in aggregate["run_ids"]:
                shared = self._public(aggregate)
                if shared["instance_running"] or shared["instance_starting"]:
                    public = shared
            if public:
                payload.update(public)
            else:
                payload.update(instance_running=False, instance_starting=False, run_id=run_id,
                               port=None, url=None, embedded=False,
                               scope="run" if run_id else "all")
        return payload

    def _named_logdirs(self, run_id):
        records = [self.log_index._resolve_id(run_id)] if run_id is not None else self.log_index._records()
        logdirs = {}
        run_ids = set()
        for record in records:
            if record.get("action") != "train":
                if run_id:
                    raise ValueError("Native TensorBoard accepts training logs only")
                continue
            roots = set()
            for relative in record.get("event_files", []):
                resolved = self.log_index._runfile(record, relative)
                if not resolved.is_file() or resolved.is_symlink():
                    raise ValueError("TensorBoard event file changed or is unavailable")
                roots.add(resolved.parent)
            if not roots:
                continue
            run_ids.add(record["id"])
            origin = "Official" if record.get("origin") == "official" else "My training"
            name = "/".join((origin, str(record.get("model") or "Training"),
                             str(record.get("run_id") or record.get("original_name") or record["id"])))
            # TensorBoard uses commas and colons as logdir_spec delimiters.
            name = re.sub(r"[:,\\\x00-\x1f]", "_", name).strip("/")
            name += " [" + record["id"][:8] + "]"
            for index, root in enumerate(sorted(roots)):
                if "," in str(root):
                    raise ValueError("TensorBoard log directory paths cannot contain commas")
                suffix = f"/events-{index + 1}" if len(roots) > 1 else ""
                logdirs.setdefault(str(root), name + suffix)
        if not logdirs:
            raise ValueError("No TensorBoard training event files are available")
        return tuple(sorted(logdirs.items())), frozenset(run_ids)

    def start(self, run_id=None):
        if not self.available:
            raise ValueError("TensorBoard is not installed in the current Python environment")
        logdirs, run_ids = self._named_logdirs(run_id)
        instance_key = run_id or "__all__"
        with self._lock:
            previous = self._instances.get(instance_key)
            if previous and previous["process"].poll() is None:
                if previous["logdirs"] == logdirs:
                    return self.info(run_id)
                self.stop(instance_key)
            port = self._free_port()
            prefix = "/tensorboard/" + secrets.token_hex(16)
            self.root.mkdir(parents=True, exist_ok=True)
            log_path = self.root / f"tensorboard-{port}.log"
            args = [sys.executable, "-m", "tensorboard.main", "--logdir_spec",
                    ",".join(f"{name}:{path}" for path, name in logdirs),
                    "--host", self.host, "--port", str(port), "--load_fast", "false",
                    "--path_prefix", prefix, "--reload_interval", "1"]
            options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                       if os.name == "nt" else {"start_new_session": True})
            try:
                with log_path.open("ab") as output:
                    process = subprocess.Popen(args, cwd=str(self.repository_root), stdin=subprocess.DEVNULL,
                                               stdout=output, stderr=subprocess.STDOUT, shell=False, **options)
            except (OSError, ValueError) as exc:
                raise ValueError(f"Unable to start TensorBoard: {exc.__class__.__name__}") from exc
            self._instances[instance_key] = {"process": process, "run_id": run_id, "port": port,
                                       "prefix": prefix, "log_path": log_path,
                                       "logdirs": logdirs, "run_ids": run_ids, "run_count": len(run_ids),
                                       "started_at": time.monotonic(), "status": "Starting"}
        return self.info(run_id)

    def proxy_target(self, path):
        """Resolve only a registered native process, never a browser-supplied host."""
        if not re.match(r"^/tensorboard/[0-9a-f]{32}/", path):
            raise ValueError("TensorBoard instance not found")
        decoded = unquote(path)
        if chr(92) in decoded or any(part in (".", "..") for part in decoded.split("/")):
            raise ValueError("Invalid TensorBoard path")
        with self._lock:
            instance = next((item for item in self._instances.values()
                             if path.startswith(item["prefix"] + "/")), None)
            if instance is None:
                raise ValueError("TensorBoard instance not found")
            if not self._public(instance)["instance_running"]:
                raise ValueError("TensorBoard is not running")
            return self.host, instance["port"]

    def stop(self, run_id=None):
        if run_id is not None and not isinstance(run_id, str):
            raise ValueError("unknown run id")
        with self._lock:
            instances = [(key, value) for key, value in self._instances.items()
                         if run_id is None or key == run_id]
            stopped = []
            for key, item in instances:
                process = item["process"]
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
                item["status"] = "Stopped"
                stopped.append(key)
            return {"stopped": stopped, "instance_running": any(
                item["process"].poll() is None for item in self._instances.values())}

    def close(self):
        self.stop()
