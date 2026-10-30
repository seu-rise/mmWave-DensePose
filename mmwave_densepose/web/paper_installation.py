"""Authenticated background jobs for official weights and prediction packages."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import threading
import time
import uuid

from mmwave_densepose.data.download import DownloadCancelled
from mmwave_densepose.web.evaluation_weights import safe_path


JOB_ID = re.compile(r"[0-9a-f]{32}\Z")


class PaperInstallationService:
    def __init__(self, repository_root, *, installer=None, on_complete=None):
        self.root = Path(repository_root).resolve()
        if installer is None:
            from mmwave_densepose.data.paper_assets import PaperAssetInstaller
            installer = PaperAssetInstaller(self.root)
        self.installer = installer
        self.report_root = safe_path(self.root, self.root / ".local/runtime/paper-installation")
        self.on_complete = on_complete
        self.lock = threading.RLock()
        self.jobs = {}
        self.closed = False
        self._load_recent()

    @staticmethod
    def _public(job):
        return copy.deepcopy({key: value for key, value in job.items() if not key.startswith("_")})

    def inventory(self, **filters):
        result = self.installer.inventory(**filters)
        with self.lock:
            jobs = list(self.jobs.values())
            active = next((job for job in reversed(jobs) if job["status"] == "running"), None)
            result["active_job"] = self._public(active) if active else None
            identifiers = {item["package_id"] for item in result["items"]}
            matching = [job for job in jobs if identifiers.intersection(job.get("package_ids", []))]
            latest = max(matching, key=lambda job: job["started_at"], default=None)
            result["latest_job"] = self._public(latest) if latest else None
            if latest and latest.get("status") == "failed" and latest.get("error_code") == "installed_conflict":
                for item in result["items"]:
                    if item["package_id"] == latest.get("package_id") and item["availability"] != "installed":
                        item.update(requires_repair=True, can_download=False)
        return result

    def start(self, package_ids, *, repair=False):
        if (not isinstance(package_ids, list) or not 1 <= len(package_ids) <= 50
                or any(not isinstance(value, str) or not value for value in package_ids)
                or len(set(package_ids)) != len(package_ids) or not isinstance(repair, bool)):
            raise ValueError("Select unique official package IDs (maximum 50).")
        with self.lock:
            if self.closed:
                raise RuntimeError("Official asset installation is shutting down.")
            if any(job["status"] == "running" for job in self.jobs.values()):
                raise RuntimeError("An official asset installation is already running.")
            entries = {row["package_id"]: row for row in self.installer.inventory()["items"]}
            selected = []
            for package_id in package_ids:
                item = entries.get(package_id)
                if item is None:
                    raise ValueError("Unknown official package ID.")
                if not item.get("can_download") and not (repair and item.get("can_repair")):
                    raise ValueError(item.get("reason") or "This official package cannot be downloaded.")
                selected.append(item)
            jid = uuid.uuid4().hex
            job = {"id": jid, "package_ids": list(package_ids), "repair": repair,
                   "status": "running", "stage": "starting", "package_id": None,
                   "bytes_downloaded": 0, "bytes_total": sum(item.get("size_bytes") or 0 for item in selected),
                   "completed_packages": 0, "total_packages": len(package_ids),
                   "started_at": time.time(), "finished_at": None, "error": None,
                   "_cancel": threading.Event(), "_sizes": {item["package_id"]: item.get("size_bytes") or 0 for item in selected},
                   "_downloaded": {}, "_completed": set(), "_written_at": 0}
            self._write(job)
            self.jobs[jid] = job
            thread = threading.Thread(target=self._run, args=(job,), name=f"paper-install-{jid[:8]}", daemon=True)
            thread.start()
            return self._public(job)

    def status(self, jid):
        with self.lock:
            job = self.jobs.get(jid) if isinstance(jid, str) and JOB_ID.fullmatch(jid) else None
            return self._public(job) if job else None

    def cancel(self, jid):
        with self.lock:
            job = self.jobs.get(jid) if isinstance(jid, str) and JOB_ID.fullmatch(jid) else None
            if job is None:
                return None
            if job["status"] == "running":
                job["_cancel"].set()
                job["stage"] = "cancelling"
            return self._public(job)

    def _progress(self, job, value):
        with self.lock:
            if job["_cancel"].is_set():
                raise DownloadCancelled("Official asset installation cancelled.")
            job["stage"] = str(value.get("stage") or job["stage"])
            package_id = value.get("package_id") or value.get("archive_id") or job.get("package_id")
            if package_id in job["_sizes"]:
                job["package_id"] = package_id
                amount = value.get("bytes_downloaded", 0)
                if isinstance(amount, (int, float)) and amount >= 0:
                    job["_downloaded"][package_id] = max(job["_downloaded"].get(package_id, 0),
                                                         min(job["_sizes"][package_id], int(amount)))
                if value.get("stage") in {"installed", "skipped"} or value.get("package_complete"):
                    job["_completed"].add(package_id)
                    job["_downloaded"][package_id] = job["_sizes"][package_id]
                job["bytes_downloaded"] = sum(job["_downloaded"].values())
                job["completed_packages"] = len(job["_completed"])
            if time.monotonic() - job["_written_at"] >= .5:
                self._write(job)

    def _run(self, job):
        try:
            self.installer.install(job["package_ids"], repair=job["repair"], cancel=job["_cancel"],
                                   progress=lambda value: self._progress(job, value))
            if job["_cancel"].is_set():
                raise DownloadCancelled("Official asset installation cancelled.")
            if self.on_complete:
                self.on_complete()
            with self.lock:
                job.update(status="completed", stage="completed", bytes_downloaded=job["bytes_total"],
                           completed_packages=job["total_packages"])
        except DownloadCancelled:
            with self.lock:
                job.update(status="cancelled", stage="cancelled")
        except Exception as exc:
            with self.lock:
                job.update(status="failed", stage="failed", error=str(exc) if isinstance(exc, (ValueError, OSError, RuntimeError))
                           else "Official asset installation failed.",
                           error_code="installed_conflict" if isinstance(exc, FileExistsError) else "installation_failed")
        finally:
            with self.lock:
                job["finished_at"] = time.time()
                try:
                    self._write(job)
                except OSError:
                    job["error"] = job.get("error") or "Installation status could not be saved."

    def _write(self, job):
        directory = safe_path(self.root, self.report_root)
        directory.mkdir(parents=True, exist_ok=True)
        path = safe_path(self.root, directory / f"{job['id']}.json")
        temporary = safe_path(self.root, path.with_suffix(".json.tmp"))
        temporary.write_text(json.dumps(self._public(job), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
        job["_written_at"] = time.monotonic()

    def _load_recent(self):
        directory = safe_path(self.root, self.report_root)
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
            try:
                path = safe_path(self.root, path)
                if not JOB_ID.fullmatch(path.stem) or path.stat().st_size > 128 * 1024:
                    continue
                job = json.loads(path.read_text(encoding="utf-8"))
                if (not isinstance(job, dict) or job.get("id") != path.stem
                        or not isinstance(job.get("started_at"), (int, float))
                        or not isinstance(job.get("package_ids"), list)):
                    continue
                if job.get("status") == "running":
                    job.update(status="interrupted", stage="interrupted", finished_at=time.time(),
                               error="The service stopped. Continue download to verify cached packages and finish installation.")
                job["_cancel"] = threading.Event()
                self.jobs[path.stem] = job
            except (OSError, ValueError, TypeError):
                continue

    def close(self):
        with self.lock:
            self.closed = True
            for job in self.jobs.values():
                if job["status"] == "running":
                    job["_cancel"].set()
