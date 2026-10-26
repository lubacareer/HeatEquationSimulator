"""Shared Windows/WSL path and distribution configuration."""
from __future__ import annotations

import os
from pathlib import Path


def wsl_path(path: Path) -> str:
    absolute = path.resolve()
    if absolute.drive:
        return "/mnt/" + absolute.drive[0].lower() + "/" + str(absolute)[3:].replace("\\", "/")
    return str(absolute)


def wsl_distribution() -> str:
    return os.environ.get("MAGNETIC_TORUS_WSL_DISTRO", os.environ.get("TORUS_WSL_DISTRO", "Ubuntu"))
