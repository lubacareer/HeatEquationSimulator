"""Fail a container build before pip if its shared requirement inputs changed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

INPUTS = ("requirements.txt", "requirements-gpu.txt")
LOCK_FILES = ("requirements-bootstrap.lock", "requirements-linux-cp312.lock")


def normalized_text_digest(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")).hexdigest()


def normalized_digest(path: Path) -> str:
    # Universal newline decoding makes LF and CRLF checkouts equivalent.
    return normalized_text_digest(path.read_text(encoding="utf-8"))


def verify_files(root: Path, expected: dict, names: tuple[str, ...]) -> None:
    if not isinstance(expected, dict) or set(expected) != set(names):
        raise ValueError("Container lock provenance has unexpected requirement inputs")
    for name in names:
        if not isinstance(expected[name], str) or not re.fullmatch(r"[0-9a-f]{64}", expected[name]):
            raise ValueError("Container lock provenance has an invalid input digest")
        if normalized_digest(root / name) != expected[name]:
            raise ValueError(f"Container lock is stale for {name}; regenerate and verify the image lock before installing")


def check(requirements_root: Path, provenance: Path) -> None:
    record = json.loads(provenance.read_text(encoding="utf-8"))
    verify_files(requirements_root, record["source_requirements_normalized_sha256"], INPUTS)
    verify_files(provenance.parent, record["lock_files_normalized_sha256"], LOCK_FILES)


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
