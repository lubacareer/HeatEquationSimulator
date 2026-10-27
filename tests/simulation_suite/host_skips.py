"""Identify host limitations without treating missing required tools as skips."""
from __future__ import annotations

import errno
import re


_SYMLINK_ERRNOS = frozenset((errno.EPERM, errno.EACCES, errno.ENOSYS,
                             errno.ENOTSUP, errno.EOPNOTSUPP))
_LINUX_SYMLINK_ERRNOS = {"EPERM": 1, "EACCES": 13, "ENOSYS": 38,
                         "ENOTSUP": 95, "EOPNOTSUPP": 95}
_MARKER = re.compile(
    r"^\[host-inapplicable:symlink:(none|[0-9]+):(none|[0-9]+)"
    r"(?::linux:([A-Z]+))?\] "
)
_SYMLINK_TEST_IDS = frozenset((
    "test_research_publication.ResearchPublicationTests.test_cleanup_does_not_follow_stage_symlink",
    "test_setup_geant4.Geant4InstallerTests.test_cached_archive_symlink_is_rejected",
))
_WINDOWS_TEST_ID = "test_torus_publication.JobPublicationTests.test_transport_request_paths_survive_publication"


def is_host_limitation(error: OSError, *, origin: str = "native", error_name: str | None = None) -> bool:
    """Recognize unavailable symlink capability or permissions, not generic I/O failures."""
    if origin == "linux":
        return error_name in _LINUX_SYMLINK_ERRNOS and _LINUX_SYMLINK_ERRNOS[error_name] == error.errno
    if origin != "native":
        raise ValueError("Unknown symlink error origin.")
    return error.errno in _SYMLINK_ERRNOS or getattr(error, "winerror", None) == 1314


def host_skip_reason(kind: str, error_number: int | None, winerror: int | None = None, *,
                     origin: str = "native", error_name: str | None = None) -> str:
    """Build an explicitly classified skip only for a known host limitation."""
    if origin == "linux":
        valid = winerror is None and error_name in _LINUX_SYMLINK_ERRNOS and _LINUX_SYMLINK_ERRNOS[error_name] == error_number
    elif origin == "native":
        valid = error_number in _SYMLINK_ERRNOS or winerror == 1314
    else:
        valid = False
    if kind != "symlink" or not valid:
        raise ValueError("Not a recognized host symlink limitation.")
    number = str(error_number) if error_number is not None else "none"
    windows = str(winerror) if winerror is not None else "none"
    suffix = f":linux:{error_name}" if origin == "linux" else ""
    return f"[host-inapplicable:symlink:{number}:{windows}{suffix}] Symlink creation is unsupported or not permitted on this host."


def is_host_inapplicable(test_id: str, reason: str, *, os_name: str) -> bool:
    """Check a known platform-only test or an explicit verified symlink marker."""
    if test_id == _WINDOWS_TEST_ID:
        return os_name != "nt" and reason == "Windows/WSL path representation"
    if test_id not in _SYMLINK_TEST_IDS:
        return False
    match = _MARKER.match(reason)
    if not match:
        return False
    raw_number, raw_windows, error_name = match.groups()
    number, windows = (None if value == "none" else int(value) for value in (raw_number, raw_windows))
    if error_name is not None:
        return windows is None and error_name in _LINUX_SYMLINK_ERRNOS and _LINUX_SYMLINK_ERRNOS[error_name] == number
    return number in _SYMLINK_ERRNOS or windows == 1314
