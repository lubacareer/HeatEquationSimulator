"""Bounded, isolated checks for the runtimes required by the complete suite.

No scientific or graphics runtime is imported in the orchestrating process.
Probes exercise a small kernel/render or inspect installed worker/data files;
they never solve a physical scenario, install packages, or enter the main app.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from typing import Any

from magnetic_torus.backend_environment import wsl_distribution


def _record(
    available: bool, reason: str, *, versions=None, details=None
) -> dict[str, Any]:
    return {
        "available": bool(available),
        "reason": str(reason),
        "versions": dict(versions or {}),
        "details": dict(details or {}),
    }


def _run_json(
    command: list[str], *, timeout: float = 60.0, input_text: str | None = None
) -> tuple[dict[str, Any] | None, str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            input=input_text,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return None, f"Capability probe exceeded {timeout:g} seconds."
    except OSError as error:
        return None, f"Cannot launch capability probe: {error}"
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[-2500:]
        return None, f"Capability probe exited {completed.returncode}: {detail}"
    # Some installed runtimes emit harmless startup messages before our JSON.
    for line in reversed(completed.stdout.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value, ""
    return (
        None,
        "Capability probe returned no JSON result: " + completed.stdout.strip()[-1000:],
    )


_DESKTOP_PROBE = r"""
import json, os
import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor
from vtkmodules.vtkCommonCore import vtkVersion
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
surface = app.primaryScreen()
if surface is None or app.platformName().lower() in {'offscreen', 'minimal'}:
    raise RuntimeError('A real graphical Qt session is required by the desktop suite.')
window = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout(window)
plotter = QtInteractor(window, auto_update=False)
layout.addWidget(plotter)
try:
    window.resize(220, 180)
    window.show()
    plotter.add_mesh(pv.Sphere(theta_resolution=12, phi_resolution=12), color='orange', render=False)
    plotter.set_background('#182d3b')
    plotter.reset_camera()
    app.processEvents()
    plotter.render()
    pixels = plotter.screenshot(return_img=True)
    if pixels is None or pixels.ndim != 3 or min(pixels.shape[:2]) < 16 or np.ptp(pixels) == 0:
        raise RuntimeError('VTK did not produce a nonconstant rendered image.')
    print(json.dumps({'available': True, 'reason': 'Qt screen and VTK render/readback succeeded.',
        'versions': {'qt': QtCore.QT_VERSION_STR, 'pyvista': pv.__version__, 'vtk': vtkVersion.GetVTKVersion()},
        'details': {'qt_platform': app.platformName(), 'screen': [surface.size().width(), surface.size().height()],
                    'render_shape': list(pixels.shape)}}))
finally:
    plotter.close()
    window.close()
    app.processEvents()
"""

_TORCH_PROBE = r"""
import json, platform, torch
cpu = float(torch.ones((2, 2), dtype=torch.float64).sum())
if cpu != 4.: raise RuntimeError('PyTorch CPU kernel returned an unexpected value.')
base = {'torch': torch.__version__, 'python': platform.python_version()}
out = {'torch_cpu': {'available': True, 'reason': 'PyTorch float64 CPU kernel succeeded.', 'versions': base,
                     'details': {'dtype': 'float64'}}}
try:
    if not torch.cuda.is_available():
        out['torch_cuda'] = {'available': False, 'reason': 'PyTorch reports CUDA unavailable.', 'versions': base, 'details': {}}
    else:
        gpu = float(torch.ones((2, 2), dtype=torch.float64, device='cuda').sum().item())
        torch.cuda.synchronize()
        if gpu != 4.: raise RuntimeError('CUDA kernel returned an unexpected value.')
        out['torch_cuda'] = {'available': True, 'reason': 'CUDA device and float64 kernel succeeded.',
            'versions': {**base, 'cuda': torch.version.cuda},
            'details': {'device': torch.cuda.get_device_name(0), 'compute_capability': list(torch.cuda.get_device_capability(0)),
                        'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory}}
except Exception as error:
    out['torch_cuda'] = {'available': False, 'reason': str(error), 'versions': base, 'details': {}}
print(json.dumps(out))
"""

_FENICS_PROBE = r"""
import json, dolfinx, gmsh
from petsc4py import PETSc
print(json.dumps({'available': True, 'reason': 'DOLFINx, Gmsh, and PETSc imported in the FE interpreter.',
    'versions': {'dolfinx': dolfinx.__version__, 'gmsh': gmsh.__version__, 'petsc': '.'.join(str(x) for x in PETSc.Sys.getVersion())},
    'details': {}}))
"""

_TRANSPORT_PROBE = r"""
import hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
override = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('TORUS_GEANT4_WORKER', '')
worker = Path(override or shutil.which('torus-dose-run') or '/opt/magnetic-torus/bin/torus-dose-run')
root = Path('/opt/magnetic-torus/data/ICRPdata')
out = {}
def stat_manifest(paths):
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda item: str(item)):
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        digest.update(json.dumps([str(resolved), stat.st_size, stat.st_mtime_ns], separators=(',', ':')).encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()
fingerprint_note = 'Metadata fingerprint of resolved paths, byte sizes, and nanosecond modification times; file contents are not hashed.'
versions, details, problems = {}, {'worker': str(worker), 'physics_lists': ['Shielding', 'QGSP_BIC_HP']}, []
try:
    if not worker.is_file() or not os.access(worker, os.X_OK):
        raise RuntimeError('Geant4 worker is absent or not executable: ' + str(worker))
    versions['worker_sha256'] = hashlib.sha256(worker.read_bytes()).hexdigest()
    # The protocol requires a request argument. No argument exits before the
    # worker creates PID/output files or initializes a transport simulation.
    started = subprocess.run([str(worker)], capture_output=True, text=True, timeout=20)
    if 'Usage: torus-dose request.json' not in started.stderr + started.stdout:
        raise RuntimeError('Worker startup did not reach its request-protocol usage check: ' + (started.stderr or started.stdout)[-1200:])
    details['worker_startup_checked'] = True
    prefix = Path(os.environ.get('GEANT4_PREFIX', '/opt/geant4'))
    try:
        wrapper = worker.read_text(encoding='utf-8')
        match = re.search(r'^export CONDA_PREFIX="([^"$`]+)"', wrapper, re.M)
        if match: prefix = Path(match.group(1))
        executable = re.search(r'^exec\s+"([^"$`]+)"', wrapper, re.M)
        if executable:
            binary = Path(executable.group(1))
            versions['transport_binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    except (UnicodeError, OSError):
        pass  # A configured native executable need not be a shell wrapper.
    config = prefix / 'bin' / 'geant4-config'
    if not config.is_file():
        located = shutil.which('geant4-config')
        if located: config = Path(located)
    if not config.is_file(): raise RuntimeError('Cannot inspect Geant4 version/datasets: geant4-config is absent.')
    version = subprocess.run([str(config), '--version'], capture_output=True, text=True, timeout=10, check=True)
    versions['geant4'] = version.stdout.strip()
    data = subprocess.run([str(config), '--datasets'], capture_output=True, text=True, timeout=10, check=True)
    datasets = {}
    for line in data.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3: datasets[parts[1]] = parts[2]
    if not datasets: raise RuntimeError('Geant4 reports no installed physics datasets.')
    # Conda activation overrides the paths reported by geant4-config. Resolve
    # only its literal path exports; do not evaluate text as shell commands.
    for activation in sorted((prefix / 'etc' / 'conda' / 'activate.d').glob('*.sh')):
        for line in activation.read_text(encoding='utf-8').splitlines():
            match = re.match(r'^\s*export\s+(G4[A-Z0-9_]+)=(.+?)\s*$', line)
            if match:
                path = match.group(2).strip().strip('"\'')
                path = path.replace('${CONDA_PREFIX}', str(prefix)).replace('$CONDA_PREFIX', str(prefix))
                if '$' not in path: datasets[match.group(1)] = path
    missing, dataset_files = [], []
    for key, value in sorted(datasets.items()):
        directory = Path(value)
        if not directory.is_dir() or not any(directory.iterdir()): missing.append(key + '=' + value)
        else: dataset_files.extend(path for path in directory.rglob('*') if path.is_file())
    details['datasets'] = datasets
    if missing: raise RuntimeError('Missing/empty Geant4 physics datasets: ' + '; '.join(missing))
    versions['dataset_stat_sha256'] = stat_manifest(dataset_files)
    details['dataset_fingerprint'] = fingerprint_note
    details['dataset_file_count'] = len(dataset_files)
    out['geant4'] = {'available': True, 'reason': 'Worker startup, toolkit version, and physics datasets verified.', 'versions': versions, 'details': details}
except Exception as error:
    out['geant4'] = {'available': False, 'reason': str(error), 'versions': versions, 'details': details}
phantoms, missing, phantom_files = {}, [], []
for label, prefix, shape in [('male', 'AM', (254, 127, 222)), ('female', 'AF', (299, 137, 348))]:
    raw = root / 'ICRP110_g4dat' / 'P110_data_V1.2' / prefix
    required = [raw / (prefix + '_organs.dat'), raw / (prefix + '_media.dat')]
    required += [root / 'ICRP110_g4dat' / prefix / (prefix + '_slice' + str(i) + '.g4dat') for i in range(1, shape[2] + 1)]
    phantom_files.extend(required)
    absent = []
    for path in required:
        try:
            if not path.is_file() or path.stat().st_size == 0: absent.append(str(path))
        except OSError:
            absent.append(str(path))
    invalid = []
    if not absent:
        for path in required[2:]:
            try:
                with path.open() as stream:
                    words = []
                    while len(words) < 3:
                        line = stream.readline()
                        if not line: break
                        words.extend(line.split())
                    header = tuple(int(x) for x in words[:3])
                if header != shape: invalid.append(str(path))
            except (OSError, ValueError):
                invalid.append(str(path))
    missing.extend(absent + invalid)
    phantoms[label] = {'expected_slices': shape[2], 'missing_files': absent, 'invalid_headers': invalid}
phantom_versions = {'dataset': 'ICRP110 official Geant4 voxel data'}
if not missing:
    try: phantom_versions['dataset_stat_sha256'] = stat_manifest(phantom_files)
    except OSError as error: missing.append('Cannot fingerprint ICRP110 file metadata: ' + str(error))
out['icrp110'] = {'available': not missing,
    'reason': 'Male and female official tables and all slice headers verified.' if not missing else 'Missing, empty, or inconsistent ICRP110 files: ' + '; '.join(missing[:8]),
    'versions': phantom_versions, 'details': {'root': str(root), 'phantoms': phantoms,
    'dataset_fingerprint': fingerprint_note, 'dataset_file_count': len(phantom_files)}}
print(json.dumps(out))
"""


def _backend_command(*, interpreter: str = "/usr/bin/python3") -> list[str]:
    if sys.platform == "win32":
        return ["wsl", "-d", wsl_distribution(), "--", interpreter, "-"]
    return [interpreter, "-"]


def detect_capabilities() -> dict[str, dict[str, Any]]:
    """Check every required runtime without mutating user runs or environments."""
    versions = {"python": sys.version.split()[0]}
    for package in (
        "numpy",
        "scipy",
        "h5py",
        "torch",
        "PyQt6",
        "pyvista",
        "pyvistaqt",
        "vtk",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    caps = {}
    desktop, error = _run_json([sys.executable, "-B", "-c", _DESKTOP_PROBE])
    caps["desktop"] = desktop or _record(False, error)
    torch, error = _run_json([sys.executable, "-B", "-c", _TORCH_PROBE])
    for name in ("torch_cpu", "torch_cuda"):
        caps[name] = (
            torch.get(name, _record(False, "Torch probe omitted " + name))
            if torch
            else _record(False, error)
        )
    fenics, error = _run_json(
        _backend_command(
            interpreter=os.environ.get("MAGNETIC_TORUS_FE_PYTHON", "/usr/bin/python3"),
        ),
        input_text=_FENICS_PROBE,
    )
    caps["fenics"] = fenics or _record(False, error)
    transport, error = _run_json(
        [*_backend_command(), os.environ.get("TORUS_GEANT4_WORKER", "")],
        timeout=60.0,
        input_text=_TRANSPORT_PROBE,
    )
    for name in ("geant4", "icrp110"):
        caps[name] = (
            transport.get(name, _record(False, "Transport probe omitted " + name))
            if transport
            else _record(False, error)
        )
    for cap in caps.values():
        cap.setdefault("versions", {})
        cap["versions"] = {**versions, **cap["versions"]}
        cap.setdefault("details", {})
        cap["details"]["python_executable"] = sys.executable
    return caps
