"""Download and install registered paper weights and prediction files without loading them."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import stat
import tempfile
from urllib.parse import urlsplit
import zipfile

from .download import check_cancel, direct_download_url, download_archive, extract_archive, file_hash, safe_path
from .partitions import _lock


IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
SHA256 = re.compile(r"[a-fA-F0-9]{64}\Z")
KINDS = {"weights", "predictions"}


class PaperAssetError(ValueError):
    """A package, path or official-asset association is invalid."""


class PaperAssetBusy(RuntimeError):
    """Another process is installing paper assets in this repository."""


def _not_linked(path):
    try:
        information = path.lstat()
    except FileNotFoundError:
        return True
    return not (stat.S_ISLNK(information.st_mode) or getattr(information, "st_file_attributes", 0) & 0x400)


def _directory(value):
    raw = Path(value).absolute()
    for candidate in [*reversed(raw.parents), raw]:
        if not _not_linked(candidate):
            raise PaperAssetError("Paper asset paths must not contain symbolic links or junctions")
    return raw.resolve()


def _relative(value):
    relative = safe_path(value).as_posix()
    if relative != value:
        raise PaperAssetError("Paper asset paths must be canonical repository-relative paths")
    return relative


def _target(root, relative):
    relative = _relative(relative)
    candidate = root
    for component in safe_path(relative).parts:
        candidate /= component
        if not _not_linked(candidate):
            raise PaperAssetError("Paper asset paths must not contain symbolic links or junctions")
    if not candidate.resolve().is_relative_to(root):
        raise PaperAssetError("Paper asset path escaped its installation root")
    return candidate


def _read(path):
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise PaperAssetError(f"Paper asset metadata unavailable: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise PaperAssetError(f"Invalid paper asset metadata: {path.name}") from exc
    if not isinstance(value, dict):
        raise PaperAssetError(f"Invalid paper asset metadata: {path.name}")
    return value


def _checksum(value):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise PaperAssetError("Paper assets require SHA256 checksums")
    return value.lower()


def _size(value):
    if type(value) is not int or value <= 0:
        raise PaperAssetError("Paper assets require positive integer sizes")
    return value


class PaperAssetInstaller:
    """A fixed repository's manifest-driven weight/prediction installer.

    Inventory uses file sizes only. Use verify() or install() to check content
    hashes. The installer never deserializes a checkpoint or prediction PKL.
    """

    def __init__(self, repository_root, manifest_path=None, cache_dir=None):
        self.repository_root = _directory(repository_root)
        manifest_path = Path(manifest_path or self.repository_root / "paper/download-manifest.json")
        self.manifest_path = manifest_path if manifest_path.is_absolute() else self.repository_root / manifest_path
        self.manifest_path = _target(self.repository_root, self.manifest_path.absolute().relative_to(self.repository_root).as_posix())
        self.cache_dir = _directory(cache_dir or self.repository_root / "data/downloads/paper-assets")
        self.runtime_root = _target(self.repository_root, ".local/runtime/paper-assets")

    def _registered(self):
        manifest = _read(_target(self.repository_root, "paper/manifest.json"))
        originals = manifest.get("assets")
        if not isinstance(originals, list):
            raise PaperAssetError("Official asset manifest must contain an assets list")
        assets, records = {}, {}
        for asset in originals:
            if not isinstance(asset, dict):
                raise PaperAssetError("Invalid official asset entry")
            asset_id = asset.get("id")
            if not isinstance(asset_id, str) or not IDENTIFIER.fullmatch(asset_id) or asset_id in assets:
                raise PaperAssetError("Official asset identifiers must be unique")
            expected_path = f"paper/experiments/{asset_id}/checkpoints/best.ckpt"
            if asset.get("path") != expected_path:
                raise PaperAssetError("Official checkpoint path differs from its registered asset")
            _target(self.repository_root, expected_path)
            expected = {"path": expected_path, "size_bytes": _size(asset.get("size_bytes")),
                        "sha256": _checksum(asset.get("sha256"))}
            assets[asset_id] = asset
            records[f"weights-{asset_id}"] = {"kind": "weights", "asset_id": asset_id,
                                              "test_id": None, "files": [expected]}
            evaluations = asset.get("evaluations", [])
            if not isinstance(evaluations, list):
                raise PaperAssetError("Invalid official test records")
            for evaluation in evaluations:
                if not isinstance(evaluation, dict):
                    raise PaperAssetError("Invalid official test entry")
                test_id = evaluation.get("test_id")
                if not isinstance(test_id, str) or not IDENTIFIER.fullmatch(test_id):
                    raise PaperAssetError("Invalid official test identifier")
                directory = _relative(evaluation.get("path"))
                if not directory.startswith("mmwave_densepose/logs/test/") or len(safe_path(directory).parts) < 5:
                    raise PaperAssetError("Official prediction files must belong to a registered test log")
                _target(self.repository_root, directory)
                identifier = f"predictions-{asset_id}-{test_id}"
                if identifier in records:
                    raise PaperAssetError("Official test identifiers must be unique per asset")
                records[identifier] = {"kind": "predictions", "asset_id": asset_id, "test_id": test_id,
                                       "directory": directory, "files": []}
        return assets, records

    def _release(self):
        assets, records = self._registered()
        if not self.manifest_path.exists():
            return assets, records, {}, None
        release = _read(self.manifest_path)
        if release.get("schema_version") != 1 or not isinstance(release.get("packages"), list):
            raise PaperAssetError("Unsupported paper download manifest")
        packages, filenames, members = {}, set(), set()
        for original in release["packages"]:
            if not isinstance(original, dict):
                raise PaperAssetError("Invalid paper package entry")
            package = copy.deepcopy(original)
            identifier = package.get("id")
            if identifier not in records or identifier in packages:
                raise PaperAssetError("Paper packages must reference unique registered assets and tests")
            record = records[identifier]
            if (package.get("kind") != record["kind"] or package.get("asset_id") != record["asset_id"]
                    or package.get("test_id") != record["test_id"]):
                raise PaperAssetError("Package identity differs from the official asset manifest")
            filename = _relative(package.get("filename"))
            if len(safe_path(filename).parts) != 1 or not filename.lower().endswith(".zip") or filename.casefold() in filenames:
                raise PaperAssetError("Packages require unique ZIP basenames")
            package["size_bytes"] = _size(package.get("size_bytes"))
            package["sha256"] = _checksum(package.get("sha256"))
            url = package.get("url")
            if url is not None:
                if not isinstance(url, str) or not url.strip() or not urlsplit(url).hostname:
                    raise PaperAssetError("Package download URL is invalid")
                direct_download_url(url)
            files = package.get("files")
            if not isinstance(files, list) or not files:
                raise PaperAssetError("Paper packages require an explicit file inventory")
            checked = []
            for entry in files:
                if not isinstance(entry, dict):
                    raise PaperAssetError("Invalid package member metadata")
                name = _relative(entry.get("path"))
                if name.casefold() in members:
                    raise PaperAssetError("Paper package members must be unique")
                _target(self.repository_root, name)
                checked.append({"path": name, "size_bytes": _size(entry.get("size_bytes")),
                                "sha256": _checksum(entry.get("sha256"))})
                if record["kind"] == "predictions":
                    if Path(name).parent.as_posix() != record["directory"] or Path(name).suffix != ".pkl":
                        raise PaperAssetError("Prediction packages may contain only PKLs in their registered test directory")
                members.add(name.casefold())
            if record["kind"] == "weights" and checked != record["files"]:
                raise PaperAssetError("Weight package differs from the registered checkpoint checksum or size")
            package["files"] = checked
            packages[identifier] = package
            filenames.add(filename.casefold())
        return assets, records, packages, release.get("release_version")

    def _unpackaged_predictions(self, record):
        directory = _target(self.repository_root, record["directory"])
        if not directory.is_dir():
            return []
        names = {"test_results.pkl", "test_results_sequence_ordered.pkl", "test_dense_result.pkl"}
        metadata_path = _target(self.repository_root, record["directory"] + "/paper-record.json")
        if metadata_path.is_file():
            metadata = _read(metadata_path)
            name = metadata.get("visual_prediction_file")
            if isinstance(name, str) and len(safe_path(name).parts) == 1 and name.endswith(".pkl"):
                names.add(name)
        result = []
        for name in sorted(names):
            relative = record["directory"] + "/" + name
            path = _target(self.repository_root, relative)
            if path.is_file() and path.stat().st_size > 0:
                result.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": None})
        return result

    @staticmethod
    def _package_signature(package):
        pinned = {key: package.get(key) for key in ("id", "sha256", "size_bytes", "files")}
        return hashlib.sha256(json.dumps(pinned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _verified(self, identifier, package):
        """A cheap receipt check; inventory never hashes checkpoints or PKLs."""
        path = _target(self.repository_root, f".local/runtime/paper-assets/verified/{identifier}.json")
        try:
            receipt = _read(path)
            if receipt.get("signature") != self._package_signature(package):
                return False
            actual = []
            for item in package["files"]:
                target = _target(self.repository_root, item["path"])
                if not target.is_file():
                    return False
                stat = target.stat()
                actual.append([item["path"], stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
            return receipt.get("files") == actual
        except (OSError, ValueError):
            return False

    def _record_verified(self, identifier, package):
        path = _target(self.repository_root, f".local/runtime/paper-assets/verified/{identifier}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        files = []
        for item in package["files"]:
            information = _target(self.repository_root, item["path"]).stat()
            files.append([item["path"], information.st_size, information.st_mtime_ns, information.st_ctime_ns])
        temporary = _target(self.repository_root, path.with_suffix(".json.tmp").relative_to(self.repository_root).as_posix())
        temporary.write_text(json.dumps({"signature": self._package_signature(package), "files": files}) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _cache_inventory(self, package):
        if not package:
            return {"cache_available": False, "complete_transfer_available": False, "cached_bytes": 0,
                    "partial_download_bytes": 0, "cache_verification": None}
        archive = _target(self.cache_dir, package["filename"])
        partial = _target(self.cache_dir, package["filename"] + ".partial")
        size = archive.stat().st_size if archive.is_file() else 0
        partial_size = partial.stat().st_size if partial.is_file() else 0
        return {"cache_available": size == package["size_bytes"], "cached_bytes": size,
                "complete_transfer_available": partial_size == package["size_bytes"],
                "partial_download_bytes": partial_size, "cache_verification": "pending" if size else None}

    @staticmethod
    def _filters(records, kind, asset_id, test_id):
        if kind is not None and kind not in KINDS:
            raise PaperAssetError("Select weights or predictions")
        if asset_id is not None and asset_id not in {item["asset_id"] for item in records.values()}:
            raise PaperAssetError("Unknown official asset ID")
        if test_id is not None and test_id not in {item["test_id"] for item in records.values()}:
            raise PaperAssetError("Unknown official test ID")
        return [(identifier, record) for identifier, record in records.items()
                if (kind is None or record["kind"] == kind) and (asset_id is None or record["asset_id"] == asset_id)
                and (test_id is None or record["test_id"] == test_id)]

    def inventory(self, kind=None, asset_id=None, test_id=None):
        assets, records, packages, version = self._release()
        result = []
        for identifier, record in self._filters(records, kind, asset_id, test_id):
            package = packages.get(identifier)
            files = package["files"] if package else record["files"]
            if not files and record["kind"] == "predictions":
                files = self._unpackaged_predictions(record)
            present, matching, conflict = 0, 0, False
            for item in files:
                path = _target(self.repository_root, item["path"])
                if path.exists():
                    present += 1
                    if path.is_file() and path.stat().st_size == item["size_bytes"]:
                        matching += 1
                    else:
                        conflict = True
            published = bool(package and package.get("url"))
            cached = self._cache_inventory(package)
            verified = bool(package and files and matching == len(files) and self._verified(identifier, package))
            local = bool(package and (cached["cache_available"] or cached.get("complete_transfer_available")
                                      or files and matching == len(files)))
            repair_source = bool(package and (published or cached["cache_available"] or cached.get("complete_transfer_available")))
            resumable = bool(present or cached["cached_bytes"] or cached["partial_download_bytes"])
            legacy_present = bool(not package and files and matching == len(files))
            if verified or legacy_present:
                availability, reason = "installed", None
            elif files and matching == len(files):
                availability, reason = "unverified", "Verify the existing files before using this official package."
            elif conflict:
                availability, reason = "partial", "Existing files differ in size; explicitly repair this package to replace them."
            elif present or resumable:
                availability, reason = "partial", "Continue download to verify cached packages and install the remaining files."
            elif published:
                availability, reason = "missing", "Download and install this official package."
            elif package:
                availability, reason = "unpublished", "The download URL for this package has not been published yet."
            elif record["kind"] == "weights":
                availability, reason = "unpublished", "The official weight package has not been published yet."
            else:
                availability, reason = "unavailable", "No prediction package is available for this selected official test."
            result.append({"package_id": identifier, "kind": record["kind"], "asset_id": record["asset_id"],
                           "test_id": record["test_id"], "model": assets[record["asset_id"]].get("model"),
                           "paper_tables": assets[record["asset_id"]].get("paper_tables", []),
                           "availability": availability, "can_download": (published or local) and not conflict and availability != "installed",
                           "can_repair": repair_source and availability != "installed", "requires_repair": conflict,
                           "can_verify": bool(package and files and matching == len(files) and not verified),
                           "published": published, "reason": reason, "verification": "verified" if verified else "size_only" if legacy_present else "pending",
                           "can_resume": resumable, "installed_files": matching if verified or legacy_present else 0, "present_files": present,
                           "total_files": len(files), **cached,
                           "size_bytes": package.get("size_bytes") if package else None,
                           "filename": package.get("filename") if package else None,
                           "files": copy.deepcopy(files)})
        return {"release_version": version, "items": result}

    def _selection(self, package_ids, records):
        if not isinstance(package_ids, (list, tuple)) or not package_ids or len(package_ids) > 256:
            raise PaperAssetError("Select one or more paper package IDs")
        if any(not isinstance(value, str) or value not in records for value in package_ids):
            raise PaperAssetError("Unknown paper package ID")
        if len(set(package_ids)) != len(package_ids):
            raise PaperAssetError("Select each paper package only once")
        return list(package_ids)

    def verify(self, package_ids, *, progress=None, cancel=None):
        _assets, records, packages, _version = self._release()
        selected = self._selection(package_ids, records)
        results = []
        for identifier in selected:
            check_cancel(cancel)
            package = packages.get(identifier)
            files = package["files"] if package else records[identifier]["files"]
            valid = bool(files)
            details = []
            for item in files:
                path = _target(self.repository_root, item["path"])
                okay = path.is_file() and path.stat().st_size == item["size_bytes"] and file_hash(path, cancel=cancel) == item["sha256"]
                valid = valid and okay
                details.append({"path": item["path"], "valid": okay})
                if progress:
                    progress({"package_id": identifier, "stage": "verifying", "file": item["path"]})
            results.append({"package_id": identifier, "valid": valid, "files": details,
                            "reason": None if files else "No pinned prediction package is available."})
        return {"status": "verified" if all(item["valid"] for item in results) else "invalid", "packages": results}

    @staticmethod
    def _zip_inventory(path, package):
        expected = {item["path"]: item for item in package["files"]}
        if not zipfile.is_zipfile(path):
            raise PaperAssetError("Paper packages must be ZIP archives")
        with zipfile.ZipFile(path) as archive:
            seen = set()
            for member in archive.infolist():
                name = safe_path(member.filename).as_posix()
                mode = member.external_attr >> 16
                if member.is_dir():
                    if stat.S_IFMT(mode) not in (0, stat.S_IFDIR) or not any(value.startswith(name + "/") for value in expected):
                        raise PaperAssetError("Archive contains an undeclared directory or special file")
                    continue
                if (name != member.filename or name.casefold() in seen or name not in expected
                        or stat.S_IFMT(mode) not in (0, stat.S_IFREG) or member.file_size != expected[name]["size_bytes"]):
                    raise PaperAssetError("Archive members do not match their pinned file inventory")
                seen.add(name.casefold())
            if seen != {name.casefold() for name in expected}:
                raise PaperAssetError("Archive members do not match their pinned file inventory")

    def install(self, package_ids, archive_dir=None, repair=False, dry_run=False, progress=None, cancel=None):
        _assets, records, packages, version = self._release()
        selected = self._selection(package_ids, records)
        if any(identifier not in packages for identifier in selected):
            raise PaperAssetError("No downloadable package is published for one or more selected assets")
        if type(repair) is not bool or type(dry_run) is not bool:
            raise PaperAssetError("Repair and dry-run must be booleans")
        source_root = _directory(archive_dir) if archive_dir is not None else self.cache_dir
        if dry_run:
            return {"status": "planned", "release_version": version, "package_ids": selected,
                    "bytes_total": sum(packages[value]["size_bytes"] for value in selected)}
        check_cancel(cancel)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        _target(self.repository_root, ".local/runtime/paper-assets/install.lock")
        try:
            with _lock(self.runtime_root / "install.lock", timeout=0):
                return self._install_locked(selected, packages, source_root, archive_dir, repair, progress, cancel)
        except TimeoutError as exc:
            raise PaperAssetBusy("Another paper asset installation is already running") from exc

    def _install_locked(self, selected, packages, source_root, archive_dir, repair, progress, cancel):
        total = sum(packages[value]["size_bytes"] for value in selected)
        completed, installed, skipped = 0, 0, 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Authenticate every complete cached archive before a missing package can
        # block on the network. This also recovers ZIPs left by an interrupted job.
        ready, remaining, cache_stamps = [], [], {}
        for index, identifier in enumerate(selected, 1):
            check_cancel(cancel)
            package = packages[identifier]
            archive = _target(source_root, package["filename"])
            if progress:
                progress({"package_id": identifier, "filename": package["filename"], "stage": "verifying_cache",
                          "bytes_downloaded": 0, "bytes_total": package["size_bytes"],
                          "overall_bytes_downloaded": 0, "overall_bytes_total": total,
                          "cache_checks_done": index, "cache_checks_total": len(selected)})
            partial = _target(source_root, package["filename"] + ".partial")
            if archive_dir is None and partial.is_file() and partial.stat().st_size == package["size_bytes"]:
                # The transfer may have finished just before process shutdown.
                if file_hash(partial, cancel=cancel) == package["sha256"]:
                    partial.replace(archive)
            before = archive.stat() if archive.is_file() else None
            cached = bool(before and before.st_size == package["size_bytes"]
                          and file_hash(archive, cancel=cancel) == package["sha256"])
            after = archive.stat() if cached else None
            if cached and (before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                cache_stamps[identifier] = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                ready.append(identifier)
            else:
                remaining.append(identifier)
        selected = ready + remaining
        for identifier in selected:
            check_cancel(cancel)
            package = packages[identifier]
            context = {"package_id": identifier, "filename": package["filename"], "bytes_downloaded": 0,
                       "bytes_total": package["size_bytes"], "overall_bytes_total": total,
                       "files_total": len(package["files"]), "files_checked": 0}

            def emit(update):
                context.update(update)
                context["overall_bytes_downloaded"] = completed + context["bytes_downloaded"]
                if progress:
                    progress(dict(context))

            already_valid = True
            for item in package["files"]:
                target = _target(self.repository_root, item["path"])
                emit({"stage": "verifying", "file": item["path"]})
                matches = target.is_file() and target.stat().st_size == item["size_bytes"] and file_hash(target, cancel=cancel) == item["sha256"]
                if target.exists() and not target.is_file():
                    raise PaperAssetError("A paper asset destination conflicts with a directory")
                if target.exists() and not matches and not repair:
                    raise FileExistsError(f"Existing paper asset differs; use explicit repair: {item['path']}")
                already_valid = already_valid and matches
            if already_valid:
                self._record_verified(identifier, package)
                skipped += len(package["files"])
                completed += package["size_bytes"]
                emit({"stage": "installed", "bytes_downloaded": 0, "already_installed": True})
                continue
            archive = _target(source_root, package["filename"])
            emit({"stage": "verifying"})
            information = archive.stat() if archive.is_file() else None
            cached = bool(information and cache_stamps.get(identifier) ==
                          (information.st_size, information.st_mtime_ns, information.st_ctime_ns))
            if not cached:
                cached = bool(information and information.st_size == package["size_bytes"]
                              and file_hash(archive, cancel=cancel) == package["sha256"])
            if not cached:
                if archive_dir is not None:
                    raise PaperAssetError(f"Local archive is missing or corrupt: {package['filename']}")
                if not package.get("url"):
                    raise PaperAssetError(f"The download URL has not been published for {identifier}. Use --archive-dir for local packages.")
                partial = _target(self.cache_dir, package["filename"] + ".partial")
                # Keep interrupted transfer bytes for truthful restart status.
                # HTTP byte-range resume is not assumed; retry replaces them.
                emit({"stage": "downloading"})
                download_archive(package["url"], partial, progress=emit, cancel=cancel, expected_size=package["size_bytes"])
                emit({"stage": "verifying"})
                if partial.stat().st_size != package["size_bytes"] or file_hash(partial, cancel=cancel) != package["sha256"]:
                    raise PaperAssetError(f"Package size or SHA256 failed: {identifier}")
                partial.replace(archive)
            emit({"stage": "downloaded", "bytes_downloaded": package["size_bytes"], "cached": cached})
            self._zip_inventory(archive, package)
            with tempfile.TemporaryDirectory(prefix="stage-", dir=self.runtime_root) as temporary:
                stage = Path(temporary)
                emit({"stage": "extracting"})
                extract_archive(archive, stage, cancel=cancel, expected_files={item["path"] for item in package["files"]})
                for count, item in enumerate(package["files"], 1):
                    check_cancel(cancel)
                    path = stage / item["path"]
                    emit({"stage": "verifying", "file": item["path"], "files_checked": count})
                    if not path.is_file() or path.stat().st_size != item["size_bytes"] or file_hash(path, cancel=cancel) != item["sha256"]:
                        raise PaperAssetError(f"Package member size or SHA256 failed: {item['path']}")
                # Every member is authenticated before any destination changes.
                emit({"stage": "installing"})
                for item in package["files"]:
                    check_cancel(cancel)
                    target = _target(self.repository_root, item["path"])
                    if target.exists():
                        matches = target.is_file() and target.stat().st_size == item["size_bytes"] and file_hash(target, cancel=cancel) == item["sha256"]
                        if matches:
                            skipped += 1
                            continue
                        if not repair or not target.is_file():
                            raise FileExistsError(f"Paper asset destination changed during installation: {item['path']}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _target(self.repository_root, item["path"])
                    (stage / item["path"]).replace(target)
                    installed += 1
                self._record_verified(identifier, package)
                emit({"stage": "installed", "package_complete": True})
            completed += package["size_bytes"]
        check_cancel(cancel)
        return {"status": "repaired" if repair else "installed", "package_ids": selected,
                "files_installed": installed, "files_skipped": skipped, "bytes_total": total}
