"""Explicit, cancellable dataset installation and archive-scoped repair jobs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import threading
import time
import uuid

from mmwave_densepose.data.download import install_dataset, safe_path, read_installation_state, write_installation_state
from mmwave_densepose.web.dataset_verification import VerificationError


JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
ACTIVE = "running"


class InstallationError(ValueError):
    pass


class InstallationBusy(RuntimeError):
    pass


class DatasetInstallationService:
    """Use a pinned release inventory; no file hashing occurs in availability()."""

    def __init__(self, verification, report_root, *, cache_dir=None, installer=None):
        self.verification = verification
        self.repository_root = verification.repository_root
        self.dataset_root = verification.dataset_root
        self.release_path = verification.release_path
        self.report_root = Path(report_root).resolve()
        self.cache_dir = Path(cache_dir or self.repository_root / "data/downloads").resolve()
        self.installer = installer or install_dataset
        # One lock covers reservation of both verification and installation jobs.
        self.lock = verification.lock
        self.jobs = {}
        self.threads = {}
        self.closed = False
        self._load_recent()

    @staticmethod
    def _public(job):
        return copy.deepcopy({key: value for key, value in job.items() if not key.startswith("_")})

    def _release(self):
        release, _manifest, inventory = self.verification._baseline()
        required = set(inventory) | {"manifest.json"}
        archives, owners, filenames = [], {}, set()
        for entry in release.get("archives", []):
            if not isinstance(entry, dict):
                raise InstallationError("Invalid release archive metadata")
            archive_id, filename, members = entry.get("id"), entry.get("filename"), entry.get("files")
            if (not isinstance(archive_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", archive_id)
                    or any(item["id"] == archive_id for item in archives)):
                raise InstallationError("Release archives need unique package identifiers")
            if not isinstance(filename, str) or len(safe_path(filename).parts) != 1 or filename in filenames:
                raise InstallationError("Release archives need unique filenames")
            if not isinstance(entry.get("sha256"), str) or not SHA256.fullmatch(entry["sha256"]):
                raise InstallationError("Release archive checksum is missing")
            if type(entry.get("size_bytes")) is not int or entry["size_bytes"] <= 0:
                raise InstallationError("Release archive size is invalid")
            if not isinstance(members, list) or not members:
                raise InstallationError("This release needs a per-package file inventory before download or repair")
            for member in members:
                if not isinstance(member, str) or safe_path(member).as_posix() != member or member not in required:
                    raise InstallationError("Archive file inventory differs from the frozen release")
                if member in owners:
                    raise InstallationError("A release file belongs to more than one package")
                owners[member] = archive_id
            filenames.add(filename)
            archives.append(entry)
        if not archives:
            return release, inventory, [], {}, "Dataset download links have not been published yet."
        if set(owners) != required:
            raise InstallationError("Package inventory does not cover the complete frozen release")
        if any(not isinstance(entry.get("url"), str) or not entry["url"].strip() for entry in archives):
            return release, inventory, archives, owners, "Some dataset package download links have not been published yet."
        return release, inventory, archives, owners, None

    def _exists(self, relative):
        try:
            path = self.verification._safe_target(relative)
            return path.is_file()
        except (OSError, ValueError):
            return False

    def _matches_size(self, relative, size):
        try:
            path = self.verification._safe_target(relative)
            return path.is_file() and path.stat().st_size == size
        except (OSError, ValueError):
            return False

    def availability(self):
        """Probe only known paths; release summary statistics remain static."""
        with self.lock:
            reason = None
            try:
                release, inventory, archives, _owners, reason = self._release()
            except (OSError, ValueError) as exc:
                release = {}
                inventory, archives = {}, []
                reason = str(exc) if isinstance(exc, (VerificationError, InstallationError)) else "Release metadata unavailable."
            active = next((job for job in reversed(list(self.jobs.values())) if job["status"] == ACTIVE), None)
            latest = max(self.jobs.values(), key=lambda job: job["started_at"], default=None)
            manifest_size = self.verification.baseline_path.stat().st_size if inventory else 0
            required = {"manifest.json": {"size": manifest_size}, **inventory}
            present = sum(self._exists(name) for name in required)
            installed_files = sum(self._matches_size(name, item["size"]) for name, item in required.items())
            sequence_paths = {name for name in inventory if name.startswith("raw/") and Path(name).suffix in (".pkl", ".pickle")}
            installed_sequences = sum(self._matches_size(name, inventory[name]["size"]) for name in sequence_paths)
            cached_archives = 0
            cache_root = self.cache_dir.resolve()
            if self.cache_dir == cache_root and not self.cache_dir.is_symlink():
                for entry in archives:
                    for filename in (entry["filename"], entry["filename"] + ".partial"):
                        candidate = self.cache_dir / filename
                        try:
                            if (candidate.is_file() and not candidate.is_symlink() and candidate.resolve().parent == cache_root
                                    and candidate.stat().st_size == entry["size_bytes"]):
                                cached_archives += 1
                                break
                        except OSError:
                            pass
            try:
                marker = read_installation_state(self.dataset_root, release.get("dataset_manifest_sha256"))
            except (OSError, ValueError):
                marker = {"complete": False}
            # A non-release numeric/synthetic root is validated by its own model
            # loader, not by the official multimodal release's 141-sequence count.
            custom = None
            try:
                candidate = self.verification._safe_target("manifest.json")
                if candidate.is_file() and candidate.stat().st_size < 2 * 1024 * 1024:
                    value = json.loads(candidate.read_text(encoding="utf-8"))
                    if (isinstance(value, dict) and isinstance(value.get("sequences"), dict)
                            and value["sequences"] and value.get("dataset_kind") != "full_multimodal"):
                        custom = value
            except (OSError, ValueError, UnicodeError):
                pass
            requires_full = custom is None
            complete = bool(inventory and installed_files == len(required) and (marker is None or marker.get("complete") is True))
            if custom is not None:
                complete = True
                sequence_paths = set(custom["sequences"])
                installed_sequences = len(sequence_paths)
                installed_files = 0
                required = {}
            can_resume = requires_full and not complete and bool(present or cached_archives or marker or latest)
            availability = "installed" if complete else "partial" if can_resume else "missing"
            return {"availability": availability, "can_download": bool(archives and not reason),
                    "reason": reason, "active_job": self._public(active) if active else None,
                    "latest_job": self._public(latest) if latest else None,
                    "installed_sequences": installed_sequences, "total_sequences": len(sequence_paths),
                    "installed_files": installed_files, "total_files": len(required),
                    "cached_archives": cached_archives, "total_archives": len(archives),
                    "cached_packages_verified": False, "is_complete": complete,
                    "requires_full_dataset": requires_full, "training_ready": complete and not active,
                    "browsing_available": installed_sequences > 0, "can_resume": can_resume,
                    "installation_incomplete": requires_full and not complete}

    def _ensure_idle(self):
        if self.closed:
            raise InstallationBusy("Dataset installation service is shutting down")
        if any(job["status"] == ACTIVE for job in self.jobs.values()):
            raise InstallationBusy("Dataset installation is already running")
        if any(job["status"] == "scanning" for job in self.verification.jobs.values()):
            raise InstallationBusy("Wait for dataset verification to finish")

    def start_verification(self):
        with self.lock:
            self._ensure_idle()
            return self.verification.start()

    def repair_plan(self, verification_id):
        if not isinstance(verification_id, str) or not JOB_ID.fullmatch(verification_id):
            raise InstallationError("Invalid verification identifier")
        with self.lock:
            report = self.verification.status(verification_id)
            if report is None:
                raise KeyError(verification_id)
            release, inventory, archives, owners, reason = self._release()
            if report.get("status") == "scanning":
                raise InstallationBusy("Wait for dataset verification to finish")
            if report.get("baseline_sha256") != release["dataset_manifest_sha256"]:
                raise InstallationError("Verify the current release again before repairing it")
            affected, unrepairable = {}, []
            for finding in report.get("findings", []):
                if finding.get("severity") not in {"error", "unverified"}:
                    continue
                relative = finding.get("path")
                if relative not in set(inventory) | {"manifest.json"}:
                    unrepairable.append({"path": relative, "reason": "This path is outside the frozen release."})
                    continue
                try:
                    target = self.verification._safe_target(relative)
                    if target.exists() and not target.is_file():
                        raise OSError("A directory occupies this release file path")
                except (OSError, ValueError):
                    unrepairable.append({"path": relative, "reason": "Replace the symbolic link or conflicting directory manually, then verify again."})
                    continue
                owner = owners.get(relative)
                if owner is None:
                    unrepairable.append({"path": relative, "reason": "No published package owns this release file."})
                else:
                    affected.setdefault(owner, set()).add(relative)
            selected = [{"id": entry["id"], "filename": entry["filename"], "size_bytes": entry["size_bytes"],
                         "affected_files": sorted(affected[entry["id"]]),
                         "available": isinstance(entry.get("url"), str) and bool(entry["url"].strip())}
                        for entry in archives if entry["id"] in affected]
            available = bool(selected) and all(entry["available"] for entry in selected)
            return {"verification_id": verification_id, "archives": selected, "unrepairable": unrepairable,
                    "can_repair": available, "reason": None if available else reason or "No downloadable repair packages are available."}

    def start(self, action="download", verification_id=None):
        if action not in {"download", "repair"}:
            raise InstallationError("Select download or repair")
        if action == "download" and verification_id is not None:
            raise InstallationError("A download request does not take a verification identifier")
        with self.lock:
            self._ensure_idle()
            release, _inventory, archives, _owners, reason = self._release()
            archive_ids = None
            if action == "repair":
                plan = self.repair_plan(verification_id)
                if not plan["can_repair"]:
                    raise InstallationError(plan["reason"])
                archive_ids = [entry["id"] for entry in plan["archives"]]
                archives = [entry for entry in archives if entry["id"] in archive_ids]
            elif reason:
                raise InstallationError(reason)
            jid = uuid.uuid4().hex
            job = {"id": jid, "action": action, "status": ACTIVE, "stage": "starting", "progress": {},
                   "archive_id": None, "filename": None, "bytes_downloaded": 0,
                   "total_bytes": sum(entry["size_bytes"] for entry in archives), "completed_archives": 0,
                   "total_archives": len(archives), "started_at": time.time(), "finished_at": None,
                   "error": None, "verification_id": None, "baseline_sha256": release["dataset_manifest_sha256"],
                   "_cancel": threading.Event(), "_sizes": {entry["id"]: entry["size_bytes"] for entry in archives},
                   "_downloaded": {}, "_completed": set(), "_written_at": 0}
            self.jobs[jid] = job
            try:
                self._write(job)
            except OSError as exc:
                self.jobs.pop(jid, None)
                raise InstallationError("The installation report directory is not writable.") from exc
            thread = threading.Thread(target=self._run, args=(job, archive_ids), name=f"dataset-install-{jid[:8]}", daemon=True)
            self.threads[jid] = thread
            thread.start()
            return self._public(job)

    def status(self, jid):
        with self.lock:
            return self._public(self.jobs[jid]) if isinstance(jid, str) and JOB_ID.fullmatch(jid) and jid in self.jobs else None

    def cancel(self, jid):
        with self.lock:
            if not isinstance(jid, str) or not JOB_ID.fullmatch(jid) or jid not in self.jobs:
                return None
            job = self.jobs[jid]
            if job["status"] == ACTIVE:
                job["_cancel"].set()
                if job.get("verification_id"):
                    self.verification.cancel(job["verification_id"])
            return self._public(job)

    def _progress(self, job, progress):
        with self.lock:
            if job["_cancel"].is_set():
                from mmwave_densepose.data.download import DownloadCancelled
                raise DownloadCancelled()
            value = {key: item for key, item in progress.items() if not key.startswith("_")}
            job["progress"] = value
            job["stage"] = str(value.get("stage") or job["stage"])
            for key in ("archive_id", "filename"):
                if key in value:
                    job[key] = value[key]
            archive_id = value.get("archive_id")
            if archive_id in job["_sizes"]:
                amount = value.get("bytes_downloaded", 0)
                if isinstance(amount, (int, float)) and amount >= 0:
                    job["_downloaded"][archive_id] = min(job["_sizes"][archive_id], int(amount))
                if value.get("archive_complete") or value.get("stage") == "installed":
                    job["_completed"].add(archive_id)
                    job["_downloaded"][archive_id] = job["_sizes"][archive_id]
                job["bytes_downloaded"] = sum(job["_downloaded"].values())
                job["completed_archives"] = len(job["_completed"])
            if time.monotonic() - job["_written_at"] >= .4:
                self._write(job)

    def _run(self, job, archive_ids):
        from mmwave_densepose.data.download import DownloadCancelled
        try:
            options = {"archive_ids": archive_ids, "repair": job["action"] == "repair",
                       "progress": lambda value: self._progress(job, value), "cancel": job["_cancel"]}
            if self.installer is install_dataset:
                # Keep the durable task guard closed throughout the final scan.
                # Do not briefly publish complete=True and reset it afterwards.
                options["finalize_installation"] = False
            self.installer(self.release_path, self.dataset_root, self.cache_dir, **options)
            if job["_cancel"].is_set():
                raise DownloadCancelled()
            with self.lock:
                job.update(stage="verifying", completed_archives=job["total_archives"], bytes_downloaded=job["total_bytes"])
                verification = self.verification.start()
                job["verification_id"] = verification["id"]
                self._write(job)
            while verification["status"] == "scanning":
                if job["_cancel"].wait(.1):
                    self.verification.cancel(verification["id"])
                verification = self.verification.status(verification["id"])
                with self.lock:
                    job["progress"] = {"stage": "verifying", "files_checked": verification.get("checked", 0),
                                       "files_total": verification.get("total"), "bytes_hashed": verification.get("bytes_hashed", 0)}
            if job["_cancel"].is_set() or verification["status"] == "cancelled":
                raise DownloadCancelled()
            if verification.get("baseline_sha256") != job["baseline_sha256"]:
                raise InstallationError("The release changed during installation. Verify the current release again.")
            if verification["status"] != "consistent":
                raise InstallationError("Installed files did not pass dataset verification. Review the verification report.")
            with self.lock:
                receipt = read_installation_state(self.dataset_root, job["baseline_sha256"]) or {}
                write_installation_state(self.dataset_root, job["baseline_sha256"],
                                         installed_archive_ids=receipt.get("installed_archive_ids", []), complete=True)
                job.update(status="complete", stage="complete")
        except DownloadCancelled:
            with self.lock:
                job.update(status="cancelled", stage="cancelled")
        except Exception as exc:
            with self.lock:
                job.update(status="failed", stage="failed", error=str(exc) if isinstance(exc, (ValueError, OSError)) else "Dataset installation failed.")
        finally:
            with self.lock:
                job["finished_at"] = time.time()
                try:
                    self._write(job)
                except OSError:
                    job["error"] = job.get("error") or "Installation report could not be saved."

    def _write(self, job):
        self.report_root.mkdir(parents=True, exist_ok=True)
        if self.report_root.is_symlink():
            raise OSError("Installation report directory cannot be a symbolic link")
        path = self.report_root / f"{job['id']}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._public(job), ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        job["_written_at"] = time.monotonic()

    def _load_recent(self):
        if not self.report_root.is_dir() or self.report_root.is_symlink():
            return
        for path in sorted(self.report_root.glob("*.json"), key=lambda entry: entry.stat().st_mtime, reverse=True)[:20]:
            if path.is_symlink() or not JOB_ID.fullmatch(path.stem):
                continue
            try:
                if path.stat().st_size > 512 * 1024:
                    continue
                job = json.loads(path.read_text(encoding="utf-8"))
                if job.get("id") != path.stem or job.get("action") not in {"download", "repair"} or not isinstance(job.get("started_at"), (int, float)):
                    continue
                if job.get("status") == ACTIVE:
                    job.update(status="interrupted", stage="interrupted", finished_at=time.time(), error="The service stopped before installation completed. Download or repair again to reuse verified packages.")
                job["_cancel"] = threading.Event()
                self.jobs[path.stem] = job
            except (OSError, ValueError, TypeError, AttributeError):
                continue

    def close(self):
        with self.lock:
            self.closed = True
            for job in self.jobs.values():
                if job["status"] == ACTIVE:
                    job["_cancel"].set()
                    if job.get("verification_id"):
                        self.verification.cancel(job["verification_id"])
        for thread in list(self.threads.values()):
            thread.join(timeout=2)
