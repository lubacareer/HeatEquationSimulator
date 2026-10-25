"""Run test modules in fresh processes, matching the numerical/PINN architecture.

The Windows conda environment contains independent Intel OpenMP runtimes in its
SciPy and PyTorch builds. Keeping scientific jobs isolated avoids loading both
runtimes into one process; no KMP_DUPLICATE_LIB_OK override is used.
"""
from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
failures = []
environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
for path in sorted((root / "tests").glob("test_*.py")):
    print(f"\n{path.name}", flush=True)
    command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", path.name, "-v"]
    result = subprocess.run(command, cwd=root, env=environment)
    if result.returncode:
        failures.append(path.name)
if failures:
    print("Failed modules: " + ", ".join(failures))
    raise SystemExit(1)
print("All isolated test modules passed.")
