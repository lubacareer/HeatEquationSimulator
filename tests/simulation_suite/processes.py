"""Own each test's native process group, including descendants holding pipes."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import signal
import subprocess


class OwnedProcessGroup:
    def __init__(self, process: subprocess.Popen):
        self.pid = process.pid
        self.handle = None
        if os.name != "nt":
            return
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_int64),
                ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD),
                ("minimum_working_set", ctypes.c_size_t),
                ("maximum_working_set", ctypes.c_size_t),
                ("active_processes", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class IOCounts(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_uint64)
                for name in (
                    "read_ops",
                    "write_ops",
                    "other_ops",
                    "read_bytes",
                    "write_bytes",
                    "other_bytes",
                )
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimits),
                ("io", IOCounts),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t),
                ("peak_job_memory", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            if not kernel.SetInformationJobObject(
                handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            child = kernel.OpenProcess(
                0x0101, False, process.pid
            )  # SET_QUOTA | TERMINATE
            if not child:
                # The process can legitimately finish before assignment.
                if process.poll() is not None:
                    kernel.CloseHandle(handle)
                    return
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if not kernel.AssignProcessToJobObject(handle, child):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                kernel.CloseHandle(child)
        except BaseException:
            kernel.CloseHandle(handle)
            raise
        self.handle, self.kernel = handle, kernel

    def close(self):
        if os.name == "nt":
            if self.handle is not None:
                self.kernel.CloseHandle(self.handle)
                self.handle = None
        else:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def stop_owned_wsl(case_dir: Path) -> str | None:
    """Last resort: target only same-user Linux commands carrying this case path."""
    if os.name != "nt" or not case_dir.drive:
        return None
    root = (
        "/mnt/"
        + case_dir.drive[0].lower()
        + "/"
        + str(case_dir.resolve())[3:].replace("\\", "/")
    )
    code = """import os, pathlib, signal, sys, time
root = sys.argv[1].rstrip('/') + '/'
def owned(pid):
    try:
        path = pathlib.Path('/proc') / str(pid)
        return (pid != os.getpid() and path.stat().st_uid == os.getuid()
                and any(arg.startswith(root) for arg in (path / 'cmdline').read_bytes().decode(errors='replace').split('\\0')))
    except (OSError, ValueError):
        return False
targets = [int(p.name) for p in pathlib.Path('/proc').iterdir() if p.name.isdigit() and owned(int(p.name))]
for sig in (signal.SIGTERM, signal.SIGKILL):
    for pid in targets:
        if owned(pid):
            try: os.kill(pid, sig)
            except ProcessLookupError: pass
    if targets: time.sleep(.1)
print('Owned WSL processes terminated:', len(targets))
"""
    distro = os.environ.get(
        "MAGNETIC_TORUS_WSL_DISTRO", os.environ.get("TORUS_WSL_DISTRO", "Ubuntu")
    )
    try:
        result = subprocess.run(
            ["wsl", "-d", distro, "--", "/usr/bin/python3", "-", root],
            input=code,
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return (
            result.stdout.strip()
            if result.returncode == 0
            else "WSL cleanup could not be confirmed: " + result.stderr.strip()
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"WSL cleanup could not be confirmed: {error}"
