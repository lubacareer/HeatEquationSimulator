"""
Lightweight NumPy compatibility shims and diagnostics.

Purpose
-------
Some third-party packages (or older local code) reference symbols that were
removed or renamed in NumPy 2.0 (e.g., `numpy.bool8`). Importing this module
early in your program defines missing aliases to keep such code running
without modifying external dependencies.

Additionally, we provide a tiny health check helper to detect broken NumPy
binary installs (e.g., DLL load failures on Windows due to mixed conda/pip
or incompatible wheels) and present a friendly message before heavy imports.

This file intentionally keeps the surface area tiny to avoid side effects.
Add only the aliases you actually need.
"""
from __future__ import annotations

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - if NumPy is not installed we do nothing
    _np = None  # type: ignore

if _np is not None:
    # NumPy 2.0 removed the `bool8` alias. Many libs still expect it.
    # Map it to the canonical `bool_` type if missing.
    if not hasattr(_np, "bool8"):
        try:  # tolerate read-only module attributes in exotic environments
            setattr(_np, "bool8", _np.bool_)
        except Exception:
            pass

    # If you encounter similar errors (e.g., AttributeError: numpy has no attribute 'XYZ'),
    # you can add small, safe aliases below following the same pattern.


def check_numpy_health() -> tuple[bool, str]:
    """
    Verify that NumPy can import and perform a couple of trivial operations.

    Returns (ok, message). If ok is False, message contains the original error.
    """
    try:
        import importlib

        np = importlib.import_module("numpy")
        # Basic sanity ops that require working C extensions
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = a * 2.0
        _ = (a + b).sum(dtype=np.float64)
        # Touch a core submodule name that often triggers DLL issues if broken
        try:
            importlib.import_module("numpy.core._multiarray_umath")  # type: ignore
        except Exception:
            # On newer NumPy the private path may differ; ignore errors here if basic ops worked
            pass
        return True, f"NumPy {getattr(np, '__version__', '?')} OK"
    except Exception as e:
        return False, str(e)


def friendly_numpy_error(ex_msg: str) -> str:
    """
    Produce a concise, actionable message for NumPy binary import failures.
    """
    lines = [
        "NumPy failed to import its core C-extensions.",
        "This is an environment issue (often due to mixing conda and pip,",
        "or installing a NumPy build incompatible with your Python).",
        "",
        "Recreate a clean Python 3.12 environment for the application you use:",
        "- Magnetic Torus desktop: follow README.md and install",
        "    requirements-desktop.txt plus the appropriate PyTorch build.",
        "",
        "- Legacy TensorFlow CPU: in the clean environment, run",
        "    python -m pip install -r requirements.txt",
        "",
        "- Legacy TensorFlow NVIDIA GPU: use Ubuntu on WSL2 and install",
        "    requirements-gpu.txt as described in docs/LEGACY_README.md.",
        "",
        "Notes:",
        "- Keep the desktop and legacy TensorFlow environments separate.",
        "- Do not repair the environment by downgrading NumPy independently",
        "  of the selected application's dependency requirements.",
        "",
        f"Original import error: {ex_msg}",
    ]
    return "\n".join(lines)
