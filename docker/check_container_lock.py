"""Fail a container build before pip if its shared requirement inputs changed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

INPUTS = ("requirements.txt", "requirements-gpu.txt")


def normalized_digest(path: Path) -> str:
    # Universal newline decoding makes LF and CRLF checkouts equivalent.
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def check(requirements_root: Path, provenance: Path) -> None:
    expected = json.loads(provenance.read_text(encoding="utf-8"))["source_requirements_normalized_sha256"]
    if set(expected) != set(INPUTS):
        raise ValueError("Container lock provenance has unexpected requirement inputs")
    for name in INPUTS:
        if not isinstance(expected[name], str) or not re.fullmatch(r"[0-9a-f]{64}", expected[name]):
            raise ValueError("Container lock provenance has an invalid input digest")
        if normalized_digest(requirements_root / name) != expected[name]:
            raise ValueError(f"Container lock is stale for {name}; regenerate and verify the image lock before installing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements-root", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    args = parser.parse_args()
    try:
        check(args.requirements_root, args.provenance)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(2, f"Container requirement check failed: {error}\n")
    print("Container requirement inputs match the verified lock.")


if __name__ == "__main__":
    main()
