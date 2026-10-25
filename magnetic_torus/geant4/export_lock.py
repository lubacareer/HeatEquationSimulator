"""Export the installed conda toolkit as a replayable, integrity-pinned lock."""
from pathlib import Path
import json
import sys


def export_lock(prefix: Path, output: Path) -> None:
    packages = [json.loads(path.read_text()) for path in (prefix / "conda-meta").glob("*.json")]
    if not any(package.get("name") == "geant4" for package in packages):
        raise RuntimeError("The requested prefix contains no installed Geant4 package.")
    lines = ["# platform: linux-64", "# Generated from installed package metadata; includes official physics datasets.", "@EXPLICIT"]
    for package in sorted(packages, key=lambda item: item["name"]):
        url, digest = package["url"], package["md5"]
        if not url.startswith("https://") or len(digest) != 32:
            raise RuntimeError("Package metadata lacks a secure URL or integrity digest.")
        lines.append(url + "#" + digest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    export_lock(Path(sys.argv[1]), Path(sys.argv[2]))
