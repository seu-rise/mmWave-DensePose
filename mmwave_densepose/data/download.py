"""Manifest-driven archive installation shared by local and future cloud releases."""

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def file_hash(path, *, cancel=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            check_cancel(cancel)
            h.update(block)
    return h.hexdigest()


def safe_path(name):
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise ValueError(f"Archive contains a non-portable path: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts or ":" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    reserved = {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}
    if any(part.rstrip(" .") != part or part.split(".")[0].upper() in reserved for part in path.parts):
        raise ValueError(f"Non-portable archive path: {name}")
    return path


def extract_archive(archive, stage, *, cancel=None, expected_files=None):
    """Accept only regular files/directories; reject traversal, links and collisions."""
    seen = set()

    def member_path(name):
        relative = safe_path(name).as_posix()
        if relative in seen:
            raise ValueError(f"Duplicate archive member: {relative}")
        seen.add(relative)
        if expected_files is not None and relative not in expected_files:
            raise ValueError(f"Undeclared archive member: {relative}")
        return stage / relative

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as package:
            members = package.infolist()
            for member in members:
                safe_path(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are not supported")
            for member in members:
                check_cancel(cancel)
                target = (stage / str(safe_path(member.filename))) if member.is_dir() else member_path(member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with package.open(member) as source, target.open("xb") as dest:
                        copy_chunks(source, dest, cancel)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as package:
            for member in package:
                relative = safe_path(member.name)
                check_cancel(cancel)
                target = (stage / str(relative)) if member.isdir() else member_path(member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with package.extractfile(member) as source, target.open("xb") as dest:
                        copy_chunks(source, dest, cancel)
                else:
                    raise ValueError(f"Only regular files are allowed: {member.name}")
    else:
        raise ValueError("Supported archives are ZIP and TAR/TAR.GZ (not TAR.ZST)")


class DownloadCancelled(RuntimeError):
    """An explicit cancellation; installed verified parts may be reused on retry."""


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise DownloadCancelled("Dataset transfer cancelled")


def copy_chunks(source, destination, cancel=None):
    while True:
        check_cancel(cancel)
        block = source.read(1024 * 1024)
        if not block:
            break
        destination.write(block)


def direct_download_url(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
        raise ValueError("Download URLs must use HTTPS or local test HTTP")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Public downloads must use HTTPS")
    if parsed.hostname in ("drive.google.com", "www.drive.google.com"):
        query = urllib.parse.parse_qs(parsed.query)
        file_id = (parsed.path.split("/file/d/", 1)[1].split("/", 1)[0]
                   if "/file/d/" in parsed.path else query.get("id", [None])[0])
        if not file_id or not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
            raise ValueError("Use a Google Drive file sharing link, not a folder link")
        download_query = {"id": file_id, "export": "download", "confirm": "t"}
        resource_keys = query.get("resourcekey", [])
        if len(resource_keys) > 1:
            raise ValueError("A Google Drive file link must have a single resourcekey")
        if resource_keys:
            download_query["resourcekey"] = resource_keys[0]
        return "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(download_query)
    return url


def download_archive(url, destination, *, progress=None, cancel=None, expected_size=None):
    request = urllib.request.Request(direct_download_url(url), headers={"User-Agent": "mmWave-DensePose/0.1"})
    check_cancel(cancel)
    with urllib.request.urlopen(request, timeout=120) as response, Path(destination).open("wb") as stream:
        direct_download_url(response.url)
        if "text/html" in response.headers.get("Content-Type", "").lower():
            raise ValueError("Download returned HTML. Check public Google Drive access, download permissions or quota.")
        total = expected_size or int(response.headers.get("Content-Length", 0)) or None
        downloaded = 0
        while True:
            check_cancel(cancel)
            block = response.read(1024 * 1024)
            if not block:
                break
            if not downloaded and block.lstrip().lower().startswith((b"<!doctype html", b"<html")):
                raise ValueError("Download returned a Google Drive access or quota page instead of an archive")
            stream.write(block)
            downloaded += len(block)
            if expected_size is not None and downloaded > expected_size:
                raise ValueError("Archive download exceeds its declared size")
            if progress:
                progress({"stage": "downloading", "bytes_downloaded": downloaded, "bytes_total": total})
    check_cancel(cancel)


def _install_legacy(manifest_path, output, cache_dir, archive_dir=None, dry_run=False):
    release = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    archives = release.get("archives", [])
    if not archives:
        raise ValueError("No downloadable archives are published in the release manifest yet.")
    for entry in archives:
        name = safe_path(entry["filename"])
        if len(name.parts) != 1 or not entry.get("sha256") or len(entry["sha256"]) != 64:
            raise ValueError("Each archive requires a basename and a SHA256 checksum")
        if int(entry.get("size_bytes", 0)) <= 0:
            raise ValueError("Each archive requires its expected size_bytes")
    if dry_run:
        return {"status": "planned", "archives": len(archives), "output": str(output)}
    output, cache = Path(output).resolve(), Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataset-install-", dir=output.parent) as temporary:
        stage = Path(temporary)
        for entry in archives:
            name = entry["filename"]
            package = (Path(archive_dir) / name) if archive_dir else cache / name
            if not package.is_file():
                if archive_dir:
                    raise FileNotFoundError(package)
                if not entry.get("url"):
                    raise ValueError(f"Cloud URL not published for {name}. Supply --archive-dir for local archives.")
                partial = package.with_suffix(package.suffix + ".partial")
                download_archive(entry["url"], partial)
                if partial.stat().st_size != entry["size_bytes"] or file_hash(partial) != entry["sha256"]:
                    raise ValueError(f"Archive checksum/size failed: {name}")
                partial.replace(package)
            if package.stat().st_size != entry["size_bytes"] or file_hash(package) != entry["sha256"]:
                raise ValueError(f"Archive checksum/size failed: {name}")
            extract_archive(package, stage)
        data_manifest = stage / "manifest.json"
        if not data_manifest.is_file() or not (stage / "raw").is_dir():
            raise ValueError("Archives must contain manifest.json and raw/ at their root")
        expected_manifest = release.get("dataset_manifest_sha256")
        if expected_manifest and file_hash(data_manifest) != expected_manifest:
            raise ValueError("Installed dataset manifest checksum disagrees with the release")
        dataset = json.loads(data_manifest.read_text(encoding="utf-8"))
        sequences = dataset["sequences"]
        for entry in sequences.values():
            relative = safe_path(entry["file"])
            sequence = stage / str(relative)
            if not sequence.is_file() or file_hash(sequence) != entry["sha256"]:
                raise ValueError(f"Missing or corrupt sequence: {relative}")
        if dataset.get("dataset_kind") == "full_multimodal":
            validate_full_manifest(dataset, stage)
        elif dataset.get("synthetic"):
            validate_fixture_splits(dataset, stage)
        else:
            raise ValueError("Numeric-only archives are training caches, not the full multimodal dataset")
        for source in stage.rglob("*"):
            if source.is_file():
                target = output / source.relative_to(stage)
                if not target.resolve().is_relative_to(output):
                    raise ValueError(f"Existing output symlink escapes dataset root: {target}")
                if target.exists() and file_hash(target) != file_hash(source):
                    raise FileExistsError(f"Existing data conflicts with release: {target}")
        for source in sorted(stage.rglob("*"), key=lambda path: path == data_manifest):
            if source.is_file():
                target = output / source.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(source, target)
    return {"status": "installed", "archives": len(archives), "sequences": len(sequences), "output": str(output)}


def _safe_destination(root, name):
    """Reject existing links, including links which stay within the root."""
    relative = safe_path(name)
    current = root
    for part in relative.parts:
        if current.is_symlink():
            raise ValueError(f"Symbolic link in installation path: {current}")
        current = current / part
    if current.is_symlink() or not current.resolve().is_relative_to(root):
        raise ValueError(f"Symbolic link in installation path: {current}")
    return current


def _root_directory(value):
    raw = Path(value).absolute()
    if raw != raw.resolve() or raw.is_symlink():
        raise ValueError(f"Installation directories may not contain symbolic links: {raw}")
    return raw.resolve()


def _manifest_inventory(raw, expected_hash):
    if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("Dataset manifest checksum disagrees with the release")
    dataset = json.loads(raw)
    if not isinstance(dataset, dict) or not isinstance(dataset.get("sequences"), dict):
        raise ValueError("Invalid dataset manifest")
    inventory = {"manifest.json": {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
    for name, entry in dataset.get("files", {}).items():
        relative = safe_path(name).as_posix()
        if relative == "manifest.json" or not isinstance(entry, dict):
            raise ValueError("Invalid dataset file inventory")
        inventory[relative] = entry
    for entry in dataset["sequences"].values():
        relative = safe_path(entry["file"]).as_posix()
        if relative == "manifest.json" or relative in inventory:
            raise ValueError(f"Duplicate dataset file: {relative}")
        inventory[relative] = entry
    for name, entry in inventory.items():
        if (not isinstance(entry.get("size_bytes"), int) or isinstance(entry["size_bytes"], bool)
                or entry["size_bytes"] < 0 or not re.fullmatch(r"[0-9a-fA-F]{64}", str(entry.get("sha256", "")))):
            raise ValueError(f"Invalid dataset file checksum or size: {name}")
    return dataset, inventory


def _mapped_archives(release):
    archives, names, identifiers, members = [], set(), set(), set()
    for original in release.get("archives", []):
        entry = dict(original)
        name = safe_path(entry["filename"])
        identifier = entry.get("id", entry["filename"])
        if (len(name.parts) != 1 or entry["filename"] in names or identifier in identifiers
                or not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", identifier)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", str(entry.get("sha256", "")))
                or not isinstance(entry.get("size_bytes"), int) or isinstance(entry["size_bytes"], bool)
                or entry["size_bytes"] <= 0):
            raise ValueError("Archives require unique IDs/basenames, positive sizes and SHA256 checksums")
        files = entry.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("Selective installation and repair require explicit archive file mappings")
        canonical = [safe_path(value).as_posix() for value in files]
        if len(set(canonical)) != len(canonical) or members.intersection(canonical):
            raise ValueError("Dataset files must belong to exactly one archive")
        if release.get("schema_version", 2) >= 3 and (entry.get("kind") not in ("metadata", "raw") or "id" not in entry or "url" not in entry):
            raise ValueError("Schema 3 archives require id, kind and url")
        entry.update(id=identifier, files=canonical)
        names.add(entry["filename"]); identifiers.add(identifier); members.update(canonical); archives.append(entry)
    if not archives:
        raise ValueError("No downloadable archives are published in the release manifest yet.")
    if "manifest.json" not in members:
        raise ValueError("Archives must declare manifest.json")
    return archives, members


def read_installation_state(output, expected_hash):
    """Read a small receipt; archive presence is never a checksum receipt."""
    path = _safe_destination(Path(output), "cache/installation-state.json")
    if not path.is_file():
        return None
    if path.stat().st_size > 128 * 1024:
        raise ValueError("Dataset installation state is too large; continue installation to rebuild it")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("dataset_manifest_sha256") != expected_hash:
        raise ValueError("Dataset installation state belongs to a different release; continue installation")
    if (not isinstance(value.get("complete"), bool) or not isinstance(value.get("installed_archive_ids"), list)
            or any(not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", item)
                   for item in value.get("installed_archive_ids", []))):
        raise ValueError("Dataset installation state is invalid; continue installation to rebuild it")
    return value


def write_installation_state(output, expected_hash, *, installed_archive_ids=(), complete=False):
    """Atomically checkpoint independently verified package installation."""
    path = _safe_destination(Path(output), "cache/installation-state.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _safe_destination(Path(output), "cache/installation-state.json.tmp")
    value = {"schema_version": 1, "dataset_manifest_sha256": expected_hash,
             "complete": bool(complete), "installed_archive_ids": sorted(set(installed_archive_ids)),
             "updated_at": time.time()}
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return value


def install_dataset(manifest_path, output, cache_dir, archive_dir=None, dry_run=False, *,
                    archive_ids=None, repair=False, progress=None, cancel=None, finalize_installation=True):
    """Install independent parts, or repair selected parts without touching other data."""
    if not isinstance(finalize_installation, bool):
        raise ValueError("finalize_installation must be true or false")
    release_path = Path(manifest_path)
    release = json.loads(release_path.read_text(encoding="utf-8"))
    mapped = any("files" in item for item in release.get("archives", []))
    if not mapped and release.get("schema_version", 2) < 3:
        if archive_ids is not None or repair:
            raise ValueError("Repair requires a release with per-archive file mappings")
        check_cancel(cancel)
        return _install_legacy(manifest_path, output, cache_dir, archive_dir, dry_run)
    archives, declared_files = _mapped_archives(release)
    if archive_ids is not None:
        if not isinstance(archive_ids, (list, tuple, set)) or not archive_ids:
            raise ValueError("Select at least one archive ID")
        if set(archive_ids) - {entry["id"] for entry in archives}:
            raise ValueError("Unknown archive ID")
        selected = [entry for entry in archives if entry["id"] in archive_ids]
    else:
        selected = list(archives)
    selected.sort(key=lambda entry: "manifest.json" not in entry["files"])
    if dry_run:
        return {"status": "planned", "archives": len(selected), "archive_ids": [entry["id"] for entry in selected], "output": str(output)}
    check_cancel(cancel)
    output, cache = _root_directory(output), _root_directory(cache_dir)
    cache.mkdir(parents=True, exist_ok=True); output.parent.mkdir(parents=True, exist_ok=True)
    expected_manifest = release.get("dataset_manifest_sha256")
    if not isinstance(expected_manifest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_manifest):
        raise ValueError("Mapped releases require a pinned dataset manifest SHA256")
    expected_manifest = expected_manifest.lower()
    dataset, inventory, pending_manifest = None, None, None
    baseline = release_path.parent / "release-baseline.json"
    installed_manifest = _safe_destination(output, "manifest.json")
    if installed_manifest.exists():
        if not installed_manifest.is_file():
            raise FileExistsError("Dataset manifest conflicts with an existing directory")
        if not repair and file_hash(installed_manifest, cancel=cancel) != expected_manifest:
            raise FileExistsError("Existing dataset manifest differs from this release; select explicit repair to replace it")
    for candidate in (baseline, installed_manifest):
        if candidate.is_file() and not candidate.is_symlink() and file_hash(candidate, cancel=cancel) == expected_manifest:
            pending_manifest = candidate.read_bytes()
            dataset, inventory = _manifest_inventory(pending_manifest, expected_manifest)
            break
    if inventory is not None and declared_files != set(inventory):
        raise ValueError("Archive mappings do not cover the pinned dataset inventory exactly")
    if repair and archive_ids is None:
        if inventory is None:
            raise ValueError("Automatic repair requires an authenticated baseline or installed dataset manifest")
        selected = []
        for entry in archives:
            for name in entry["files"]:
                target, expected = _safe_destination(output, name), inventory[name]
                if (not target.is_file() or target.stat().st_size != expected["size_bytes"]
                        or file_hash(target, cancel=cancel) != expected["sha256"].lower()):
                    selected.append(entry)
                    break
        selected.sort(key=lambda entry: "manifest.json" not in entry["files"])
    overall_total = sum(entry["size_bytes"] for entry in selected)
    completed_bytes = 0
    try:
        previous_state = read_installation_state(output, expected_manifest)
    except (OSError, ValueError):
        previous_state = None
    installed_ids = set((previous_state or {}).get("installed_archive_ids", []))
    write_installation_state(output, expected_manifest, installed_archive_ids=installed_ids)
    source_root = _root_directory(archive_dir) if archive_dir else cache
    verified_cache = {}
    # Check every finished package first. A later cached raw package must become
    # browsable before the next missing earlier package waits on the network.
    for entry in selected:
        check_cancel(cancel)
        package = _safe_destination(source_root, entry["filename"])
        candidates = [package]
        if not archive_dir:
            candidates.append(_safe_destination(cache, entry["filename"] + ".partial"))
        for candidate in candidates:
            if not candidate.is_file() or candidate.stat().st_size != entry["size_bytes"]:
                continue
            if progress:
                progress({"archive_id": entry["id"], "filename": entry["filename"], "stage": "verifying",
                          "checking_cache": True, "bytes_downloaded": 0, "bytes_total": entry["size_bytes"],
                          "overall_bytes_total": overall_total})
            if file_hash(candidate, cancel=cancel) == entry["sha256"].lower():
                if candidate != package:
                    candidate.replace(package)
                info = package.stat()
                verified_cache[entry["id"]] = (info.st_size, info.st_mtime_ns)
                break
    selected.sort(key=lambda entry: (
        ("manifest.json" not in entry["files"]) if inventory is None else entry["id"] not in verified_cache,
        "manifest.json" not in entry["files"],
    ))

    def publish_manifest():
        if pending_manifest is None:
            return
        output.mkdir(parents=True, exist_ok=True)
        target = _safe_destination(output, "manifest.json")
        if target.is_file() and file_hash(target, cancel=cancel) == expected_manifest:
            return
        if target.exists() and not repair:
            raise FileExistsError("Existing dataset manifest changed during installation; select explicit repair")
        with tempfile.NamedTemporaryFile(prefix=".manifest-", dir=output, delete=False) as stream:
            temporary_manifest = Path(stream.name)
            stream.write(pending_manifest)
        try:
            check_cancel(cancel)
            temporary_manifest.replace(target)
        finally:
            temporary_manifest.unlink(missing_ok=True)

    for entry in selected:
        check_cancel(cancel)
        context = {"archive_id": entry["id"], "filename": entry["filename"], "bytes_downloaded": 0,
                   "bytes_total": entry["size_bytes"], "overall_bytes_total": overall_total,
                   "files_checked": 0, "files_total": len(entry["files"])}

        def emit(update):
            context.update(update)
            context["overall_bytes_downloaded"] = completed_bytes + context["bytes_downloaded"]
            if progress:
                progress(dict(context))

        package = _safe_destination(source_root, entry["filename"])
        info = package.stat() if package.is_file() else None
        valid_cache = bool(info and verified_cache.get(entry["id"]) == (info.st_size, info.st_mtime_ns))
        if not valid_cache:
            if archive_dir:
                raise ValueError(f"Missing/corrupt local archive: {entry['filename']}")
            if not entry.get("url"):
                raise ValueError(f"Cloud URL not published for {entry['filename']}")
            partial = _safe_destination(cache, entry["filename"] + ".partial")
            try:
                emit({"stage": "downloading"})
                download_archive(entry["url"], partial, progress=emit, cancel=cancel, expected_size=entry["size_bytes"])
                emit({"stage": "verifying"})
                if partial.stat().st_size != entry["size_bytes"] or file_hash(partial, cancel=cancel) != entry["sha256"].lower():
                    raise ValueError(f"Archive checksum/size failed: {entry['filename']}")
                partial.replace(package)
            finally:
                if partial.exists() and not partial.is_symlink():
                    partial.unlink()
        emit({"stage": "downloaded", "bytes_downloaded": entry["size_bytes"], "cached": valid_cache})
        with tempfile.TemporaryDirectory(prefix="dataset-part-", dir=output.parent) as temporary:
            stage = Path(temporary)
            emit({"stage": "extracting"})
            extract_archive(package, stage, cancel=cancel, expected_files=set(entry["files"]))
            actual = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
            if actual != set(entry["files"]):
                raise ValueError(f"Archive members do not match their release mapping: {entry['filename']}")
            if "manifest.json" in actual:
                pending_manifest = (stage / "manifest.json").read_bytes()
                dataset, inventory = _manifest_inventory(pending_manifest, expected_manifest)
            if inventory is None:
                raise ValueError("Download the metadata archive before installing data parts")
            if declared_files != set(inventory):
                raise ValueError("Archive mappings do not cover the pinned dataset inventory exactly")
            emit({"stage": "verifying"})
            for count, name in enumerate(sorted(actual), 1):
                check_cancel(cancel)
                path, expected = stage / name, inventory[name]
                if path.stat().st_size != expected["size_bytes"] or file_hash(path, cancel=cancel) != expected["sha256"].lower():
                    raise ValueError(f"Missing/corrupt archive file: {name}")
                target = _safe_destination(output, name)
                if target.exists() and not target.is_file():
                    raise ValueError(f"Release file conflicts with a directory: {name}")
                if target.is_file() and file_hash(target, cancel=cancel) != expected["sha256"].lower() and not repair:
                    raise FileExistsError(f"Existing data conflicts with release: {target}")
                emit({"files_checked": count})
            emit({"stage": "installing"})
            for name in sorted(actual - {"manifest.json"}):
                check_cancel(cancel)
                target = _safe_destination(output, name)
                if target.is_file() and file_hash(target, cancel=cancel) == inventory[name]["sha256"].lower():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                (stage / name).replace(target)
            # Make authenticated metadata and completed sequences available even
            # if the process is stopped before another package is downloaded.
            publish_manifest()
            installed_ids.add(entry["id"])
            write_installation_state(output, expected_manifest, installed_archive_ids=installed_ids)
            emit({"stage": "installing", "archive_complete": True})
        completed_bytes += entry["size_bytes"]
    check_cancel(cancel)
    if archive_ids is None and not repair:
        if dataset.get("dataset_kind") == "full_multimodal":
            validate_full_manifest(dataset, output, strict_archive=False, cancel=cancel)
        elif dataset.get("synthetic"):
            validate_fixture_splits(dataset, output, cancel=cancel)
        else:
            raise ValueError("Numeric-only archives are not the full multimodal dataset")
    publish_manifest()
    check_cancel(cancel)
    complete = archive_ids is None and finalize_installation
    write_installation_state(output, expected_manifest, installed_archive_ids=installed_ids, complete=complete)
    return {"status": "repaired" if selected and repair else "installed" if not repair else "verified", "archives": len(selected),
            "archive_ids": [entry["id"] for entry in selected], "sequences": len(dataset["sequences"]),
            "is_complete": complete, "output": str(output)}


def validate_full_manifest(dataset, stage, strict_archive=True, *, cancel=None):
    """Authenticate full-release files without executing pickle contents."""
    if not dataset.get("face_anonymized") or dataset.get("image_source") != "raw_face_anonymized":
        raise ValueError("Full release must identify its anonymized image source")
    required_modalities = {"color_frames", "depth_frames", "infrared_frames", "body_index_frames",
                           "radar_frames", "keypoint_frames", "radar_timestamp", "kinect_timestamp", "meta"}
    if not required_modalities.issubset(dataset.get("modalities", [])):
        raise ValueError("Full release is missing original modalities")
    files = dataset.get("files", {})
    indexes = dataset.get("frozen_indexes", {})
    from .cache import INDEXES
    from .completion_indexes import COMPLETION_INDEXES
    completion_indexes = dataset.get("completion_indexes", {})
    # Earlier full releases contain pose indexes only. A declared completion
    # extension must include both exact PCC indexes, never a partial mapping.
    if "completion_indexes" in dataset and completion_indexes != COMPLETION_INDEXES:
        raise ValueError("Full release must declare the exact two PCC completion indexes")
    if indexes != INDEXES or set(files) != {"raw/manifest.json", *INDEXES.values(), *completion_indexes.values()}:
        raise ValueError("Full release must include subject mappings and all declared frozen indexes")
    allowed = {"manifest.json"}
    for relative, entry in files.items():
        check_cancel(cancel)
        path = stage / str(safe_path(relative))
        if not path.is_file() or path.stat().st_size != entry["size_bytes"] or file_hash(path, cancel=cancel) != entry["sha256"]:
            raise ValueError(f"Missing/corrupt original metadata or frozen index: {relative}")
        allowed.add(relative)
    subjects = json.loads((stage / "raw/manifest.json").read_text(encoding="utf-8"))["sequences"]
    for entry in dataset["sequences"].values():
        check_cancel(cancel)
        relative = safe_path(entry["file"])
        path = stage / str(relative)
        if relative.suffix != ".pkl" or relative.parent != PurePosixPath("raw"):
            raise ValueError("Full sequence must be an original-format PKL under raw/")
        if path.stat().st_size != entry["size_bytes"] or subjects.get(relative.name) != entry["subject_id"]:
            raise ValueError(f"Full sequence size/subject mismatch: {relative}")
        allowed.add(str(relative))
    if strict_archive:
        actual = {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}
        if actual != allowed:
            raise ValueError(f"Undeclared files in full release: {sorted(actual-allowed)}")


def validate_fixture_splits(dataset, stage, *, cancel=None):
    """Check numeric synthetic fixtures used by offline/HTTP regression examples."""
    import numpy as np
    sequences = dataset["sequences"]
    split_names = dataset.get("splits", [])
    if not split_names:
        raise ValueError("Dataset manifest must declare its frozen splits")
    for name in split_names:
        check_cancel(cancel)
        if len(safe_path(name).parts) != 1:
            raise ValueError("Split identifiers must be basenames")
        metadata_path = stage / "splits" / (name + ".json")
        index_path = metadata_path.with_suffix(".npz")
        if not metadata_path.is_file() or not index_path.is_file():
            raise ValueError(f"Missing frozen split files: {name}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if file_hash(index_path) != metadata["index_sha256"]:
            raise ValueError(f"Frozen split checksum failed: {name}")
        ids = metadata["sequence_ids"]
        if any(sequence_id not in sequences for sequence_id in ids):
            raise ValueError(f"Frozen split references unknown sequences: {name}")
        frame_counts = np.asarray([sequences[s]["frames"] for s in ids])
        with np.load(index_path, allow_pickle=False) as indexes:
            for partition in ("train", "val", "test"):
                windows = indexes[partition]
                if windows.ndim != 2 or windows.shape[1] != 3 or windows.dtype.kind not in "iu":
                    raise ValueError(f"Invalid frozen window schema: {name}/{partition}")
                if len(windows) != metadata["split_num_samples"][partition]:
                    raise ValueError(f"Frozen split count mismatch: {name}/{partition}")
                if len(windows):
                    if np.any(windows[:, 0] < 0) or np.any(windows[:, 0] >= len(ids)):
                        raise ValueError("Invalid sequence index")
                    if (np.any(windows[:, 1] < 0)
                            or np.any(windows[:, 2] > frame_counts[windows[:, 0]])
                            or np.any(windows[:, 2] - windows[:, 1] != metadata["seq_len"])):
                        raise ValueError("Invalid frozen window bounds")
