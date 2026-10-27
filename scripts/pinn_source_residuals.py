"""Evaluate saved FE residuals in a Torch-first isolated process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path.")
    # Keep Torch before NumPy and the repository's optional SciPy imports.
    __import__("torch")
    from magnetic_torus.store import RunStore
    from scripts.pinn_pressure_diagnostics import accepted_residuals

    source = RunStore.load_run(args.source)
    evidence = {str(index): accepted_residuals(source, index) for index in range(len(source.times))}
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
