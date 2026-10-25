"""Dependency inspection without loading incompatible OpenMP runtimes together."""
from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys


def inspect_environment() -> dict:
    packages = {}
    for name in ("PyQt6", "pyvista", "pyvistaqt", "vtk", "numpy", "scipy", "matplotlib", "h5py", "imageio-ffmpeg", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    result = {"python": sys.executable, "python_version": sys.version, "packages": packages}
    commands = {"torch": [sys.executable, "-B", "-c", "import torch,json; print(json.dumps({'cuda_available':torch.cuda.is_available(),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}))"]}
    if os.name == "nt":
        commands["fenics"] = ["wsl", "-d", os.environ.get("TORUS_WSL_DISTRO", "Ubuntu"), "--", "/usr/bin/python3", "-c",
                               "import dolfinx,gmsh,petsc4py,json; print(json.dumps({'dolfinx':dolfinx.__version__,'gmsh':gmsh.__version__,'petsc4py':petsc4py.__version__}))"]
        commands["geant4"] = ["wsl", "-d", os.environ.get("TORUS_WSL_DISTRO", "Ubuntu"), "--", "test", "-x",
                               os.environ.get("TORUS_GEANT4_WORKER", "/opt/magnetic-torus/bin/torus-dose-run")]
    for name, command in commands.items():
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=30)
            result[name] = {"available": process.returncode == 0, "output": process.stdout.strip(), "error": process.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = {"available": False, "error": str(error)}
    return result


def main() -> int:
    print(json.dumps(inspect_environment(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
