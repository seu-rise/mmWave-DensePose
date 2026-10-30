"""Verify and target exact process identities without adding a process dependency."""
from __future__ import annotations

import os
from pathlib import Path
import socket
import sys
import signal


def _process_details(pid):
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4))
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ("dead", None, None) if ctypes.get_last_error() in (87, 1168) else ("unknown", None, None)
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return "unknown", None, None
            if code.value != 259:
                return "dead", None, None
            created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(system), ctypes.byref(user)):
                return "unknown", None, None
            if exited.dwHighDateTime or exited.dwLowDateTime:
                return "dead", None, None
            token = str((created.dwHighDateTime << 32) | created.dwLowDateTime)
            return "alive", token, None
        finally:
            kernel.CloseHandle(handle)
    if sys.platform.startswith("linux"):
        try:
            raw = (Path("/proc") / str(pid) / "stat").read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            if fields[0] in ("Z", "X"):
                return "dead", None, None
            try:
                boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            except OSError:
                boot = None
            return "alive", fields[19], boot
        except FileNotFoundError:
            return "dead", None, None
        except (OSError, IndexError):
            return "unknown", None, None
    return "unknown", None, None


def current_process_identity(pid=None):
    """Return JSON-safe host, PID and creation token; unknown tokens stay unknown."""
    pid = os.getpid() if pid is None else int(pid)
    identity = {"pid": pid, "hostname": socket.gethostname(), "platform": sys.platform,
                "start_token": None}
    if pid < 1:
        return identity
    try:
        _state, token, boot = _process_details(pid)
        identity["start_token"] = token
        if boot is not None:
            identity["boot_id"] = boot
    except (OSError, ValueError, OverflowError):
        pass
    return identity


def probe_process(identity):
    """Verify this exact process, not merely another process reusing its PID."""
    if not isinstance(identity, dict):
        return "unknown"
    if identity.get("hostname") != socket.gethostname() or identity.get("platform") != sys.platform:
        return "unknown"
    try:
        pid = int(identity["pid"])
        if pid < 1:
            return "dead"
        state, token, boot = _process_details(pid)
    except (KeyError, TypeError, ValueError, OSError, OverflowError):
        return "unknown"
    if state != "alive":
        return state
    if not token or not identity.get("start_token"):
        return "unknown"
    if token != identity["start_token"]:
        return "dead"
    if identity.get("boot_id") and identity["boot_id"] != boot:
        return "dead" if boot else "unknown"
    return "alive"


def _local_target(identity):
    if (not isinstance(identity, dict) or identity.get("hostname") != socket.gethostname()
            or identity.get("platform") != sys.platform or not identity.get("start_token")):
        raise PermissionError("The execution process identity cannot be verified on this host.")
    pid = int(identity.get("pid", 0))
    if pid < 1 or pid in (os.getpid(), os.getppid()):
        raise PermissionError("Refusing to stop the monitoring service or an invalid process.")
    parents = _parent_processes()
    if os.getpid() not in parents:
        raise PermissionError("The monitoring service ancestry cannot be verified.")
    ancestor, seen = parents.get(os.getpid()), {os.getpid()}
    while ancestor:
        if ancestor == pid:
            raise PermissionError("Refusing to stop a parent of the monitoring service.")
        if ancestor in seen:
            raise PermissionError("The monitoring service ancestry is incomplete or inconsistent.")
        if ancestor not in parents:
            # Windows may retain an already-exited creator PID. A confirmed
            # dead tail has no live process to protect; unknown/access-denied
            # tails still fail closed instead of assuming a complete inventory.
            if _process_details(ancestor)[0] == "dead":
                break
            raise PermissionError("The monitoring service ancestry is incomplete or inconsistent.")
        seen.add(ancestor)
        ancestor = parents.get(ancestor)
    return pid


def terminate_verified_process(identity):
    """Terminate only a pinned, creation-time-verified process; never a terminal group."""
    pid = _local_target(identity)
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4))
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
        kernel.TerminateProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1001, False, pid)  # QUERY_LIMITED_INFORMATION | TERMINATE
        if not handle:
            if probe_process(identity) == "dead":
                return False
            raise PermissionError("The verified execution process cannot be opened for termination.")
        try:
            created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(system), ctypes.byref(user)):
                raise PermissionError("The opened process creation time cannot be verified.")
            token = str((created.dwHighDateTime << 32) | created.dwLowDateTime)
            if token != identity["start_token"]:
                raise PermissionError("The execution PID has been reused; termination was refused.")
            if exited.dwHighDateTime or exited.dwLowDateTime:
                return False
            if not kernel.TerminateProcess(handle, 130):
                raise OSError(ctypes.get_last_error(), "The verified execution process could not be terminated.")
            return True
        finally:
            kernel.CloseHandle(handle)
    if sys.platform.startswith("linux") and hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        try:
            descriptor = os.pidfd_open(pid)
        except ProcessLookupError:
            return False
        try:
            state = probe_process(identity)
            if state == "dead":
                return False
            if state != "alive":
                raise PermissionError("The execution process identity changed before termination.")
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            return True
        finally:
            os.close(descriptor)
    raise OSError("Safe forced termination requires Windows process handles or Linux pidfd support.")


def can_control_process(identity):
    try:
        _local_target(identity)
        return probe_process(identity) == "alive"
    except (OSError, ValueError, TypeError):
        return False


def _parent_processes():
    if sys.platform.startswith("linux"):
        result = {}
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                raw = path.read_text()
                fields = raw[raw.rfind(")") + 2:].split()
                result[int(path.parent.name)] = int(fields[1])
            except (OSError, ValueError, IndexError):
                continue
        return result
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class ProcessEntry(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                        ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel.Process32FirstW.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessEntry))
        kernel.Process32FirstW.restype = wintypes.BOOL
        kernel.Process32NextW.argtypes = kernel.Process32FirstW.argtypes
        kernel.Process32NextW.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.CreateToolhelp32Snapshot(2, 0)
        if handle == ctypes.c_void_p(-1).value:
            return {}
        result, entry = {}, ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        try:
            available = kernel.Process32FirstW(handle, ctypes.byref(entry))
            while available:
                result[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                available = kernel.Process32NextW(handle, ctypes.byref(entry))
            return result
        finally:
            kernel.CloseHandle(handle)
    return {}


def terminate_verified_tree(identity):
    """Stop the selected training process and only verified newer descendants."""
    root_pid = _local_target(identity)
    if probe_process(identity) != "alive":
        return False
    parents = _parent_processes()
    known = {root_pid: identity}
    descendants = []
    while True:
        added = False
        for pid, parent in parents.items():
            if pid in known or parent not in known:
                continue
            child = _verified_child_identity(pid, known[parent])
            if child is None:
                continue
            try:
                newer = int(child.get("start_token") or 0) >= int(known[parent]["start_token"])
            except (ValueError, TypeError):
                newer = False
            if newer and probe_process(child) == "alive":
                known[pid] = child
                descendants.append(child)
                added = True
        if not added:
            break
    stopped = terminate_verified_process(identity)
    if not stopped:
        return False  # The original root exited; its PID ancestry is no longer trustworthy.
    for child in reversed(descendants):
        try:
            terminate_verified_process(child)
        except (OSError, ValueError):
            pass  # Never broaden termination to an unverified PID or process group.
    return stopped


def _verified_child_identity(pid, parent_identity):
    """Bracket fresh parentage with creation tokens; skip reused/uncertain children."""
    parent_before = current_process_identity(parent_identity.get("pid"))
    if parent_before != parent_identity or probe_process(parent_before) != "alive":
        return None
    before = current_process_identity(pid)
    parents = _parent_processes()
    after = current_process_identity(pid)
    parent_after = current_process_identity(parent_identity.get("pid"))
    if (not before.get("start_token") or before != after
            or parents.get(pid) != parent_identity.get("pid")
            or parent_after != parent_before or probe_process(parent_after) != "alive"
            or probe_process(after) != "alive"):
        return None
    return after
