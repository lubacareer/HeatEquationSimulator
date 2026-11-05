"""Pin the proven Linux/CPython 3.12 image without upgrading its packages.

Input is an importlib.metadata distribution inventory plus pip sys_tags captured
from that image. Hashes come from official PyPI release JSON, not re-downloaded
CUDA wheels. The subsequent Docker install must verify the actual wheel bytes.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import stat
from tempfile import TemporaryDirectory
from urllib.parse import quote, urlparse
from urllib.request import urlopen

from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
from pip._vendor.packaging.version import InvalidVersion, Version

from check_container_lock import LOCK_FILES, normalized_digest, normalized_text_digest


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirement(text: str) -> Requirement:
    try:
        requirement = Requirement(text)
    except InvalidRequirement:
        raise ValueError("Requirement syntax is unsupported; no input contents are included in diagnostics") from None
    if requirement.url is not None:
        raise ValueError("Direct or remote requirement URLs are unsupported")
    return requirement


def validate_inventory(root: Path, inventory: dict) -> dict:
    """Check declared inputs and their active dependency/extra closure offline.

    The inventory is operator-supplied evidence from a verified image, not a
    cryptographic attestation. Markers use its captured target, never this host.
    Only the two confined input files and ordinary PEP 508 entries are supported.
    """
    environment = inventory.get("environment")
    expected_target = {"implementation_name": "cpython", "os_name": "posix",
                       "platform_machine": "x86_64", "platform_python_implementation": "CPython",
                       "platform_system": "Linux", "python_version": "3.12", "sys_platform": "linux"}
    environment_names = {"implementation_name", "implementation_version", "os_name", "platform_machine",
                         "platform_python_implementation", "platform_release", "platform_system",
                         "platform_version", "python_full_version", "python_version", "sys_platform"}
    if not isinstance(environment, dict) or set(environment) != environment_names or not all(
            isinstance(value, str) for value in environment.values()):
        raise ValueError("Inventory needs the captured target marker environment")
    if any(environment.get(name) != value for name, value in expected_target.items()):
        raise ValueError("Inventory target must be Linux x86-64 CPython 3.12")
    try:
        full_version = Version(environment["python_full_version"])
        if full_version.release[:2] != (3, 12) or Version(environment["implementation_version"]) != full_version:
            raise ValueError("Inventory target Python versions are inconsistent")
        if full_version != Version(inventory["python"].split()[0]):
            raise ValueError("Inventory Python identity does not match the target environment")
    except (InvalidVersion, KeyError, IndexError, TypeError):
        raise ValueError("Inventory Python identity is invalid") from None

    packages = {}
    for package in inventory["distributions"]:
        name = canonical(package["name"])
        if name in packages:
            raise ValueError("Inventory contains duplicate distribution identities")
        if not all(isinstance(package.get(field), list) and all(isinstance(item, str) for item in package[field])
                   for field in ("requires_dist", "provides_extra")):
            raise ValueError("Inventory needs captured Requires-Dist and Provides-Extra metadata")
        try:
            Version(package["version"])
        except (InvalidVersion, TypeError):
            raise ValueError("Inventory distribution version is invalid") from None
        packages[name] = package

    root = root.resolve()
    allowed = {root / name for name in ("requirements.txt", "requirements-gpu.txt")}
    try:
        for path in allowed:
            info = path.lstat()
            reparse = getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if not stat.S_ISREG(info.st_mode) or path.is_symlink() or reparse or path.resolve() != path:
                raise ValueError("Requirement inputs must be ordinary files at the two canonical root names")
        if (root / "requirements.txt").samefile(root / "requirements-gpu.txt"):
            raise ValueError("The two requirement inputs must not alias the same file")
    except OSError:
        raise ValueError("The two supported requirement inputs must be readable ordinary files") from None
    visited, pending = set(), deque()

    def active(requirement: Requirement, extra: str = "") -> bool:
        try:
            return requirement.marker is None or requirement.marker.evaluate({**environment, "extra": extra})
        except Exception:
            raise ValueError("Requirement marker metadata is invalid") from None

    def read_inputs(path: Path, stack: set[Path]) -> None:
        canonical_path = Path(os.path.abspath(path))
        if canonical_path not in allowed or canonical_path != path.resolve() or canonical_path in stack:
            raise ValueError("Requirement includes must stay within the two supported inputs without cycles")
        path = canonical_path
        if path in visited:
            return
        visited.add(path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            raise ValueError("The two supported requirement inputs must be readable") from None
        for raw in lines:
            line = re.sub(r"\s+#.*$", "", raw).strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-"):
                parts = line.split()
                if len(parts) != 2 or parts[0] not in ("-r", "--requirement"):
                    raise ValueError("Unsupported pip option in requirement inputs")
                read_inputs(path.parent / parts[1], stack | {path})
            else:
                requirement = parse_requirement(line)
                if active(requirement):
                    pending.append(requirement)

    for path in sorted(allowed):
        read_inputs(path, set())
    contexts, checked, requested_extras = set(), set(), {}
    while pending:
        requirement = pending.popleft()
        name = canonical(requirement.name)
        package = packages.get(name)
        if package is None:
            raise ValueError("Inventory is missing a required distribution")
        if not requirement.specifier.contains(package["version"], prereleases=True):
            raise ValueError("Inventory distribution violates a required version constraint")
        extras = {canonical(extra) for extra in requirement.extras}
        if not extras <= {canonical(extra) for extra in package["provides_extra"]}:
            raise ValueError("Inventory does not provide a requested extra")
        checked.add(name)
        requested_extras.setdefault(name, set()).update(extras)
        for extra in ("", *sorted(extras)):
            if (name, extra) in contexts:
                continue
            contexts.add((name, extra))
            for text in package["requires_dist"]:
                dependency = parse_requirement(text)
                if active(dependency, extra):
                    pending.append(dependency)
    return {"environment": environment, "required_distributions": sorted(checked),
            "requested_extras": {name: sorted(extras) for name, extras in sorted(requested_extras.items()) if extras}}


def compatible(filename: str, target_tags: set[str]) -> bool:
    if not filename.endswith(".whl"):
        return False
    parts = filename[:-4].rsplit("-", 3)
    return len(parts) == 4 and any(
        "-".join(tag) in target_tags
        for tag in itertools.product(*(part.split(".") for part in parts[1:]))
    )


def release(package: dict, target_tags: set[str], metadata_dir: Path) -> dict:
    name, version = canonical(package["name"]), package["version"]
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version):
        raise ValueError("Invalid distribution identity in proven-image inventory")
    url = f"https://pypi.org/pypi/{quote(name, safe='')}/{quote(version, safe='')}/json"
    with urlopen(url, timeout=30) as response:
        raw = response.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError(f"Oversized release metadata: {name}")
    data = json.loads(raw)
    if canonical(data["info"]["name"]) != name or data["info"]["version"] != version:
        raise ValueError(f"Release identity mismatch: {name}")
    wheels = []
    for item in data["urls"]:
        if item["packagetype"] != "bdist_wheel" or item.get("yanked") or not compatible(item["filename"], target_tags):
            continue
        digest, download = item["digests"]["sha256"], urlparse(item["url"])
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or download.scheme != "https" or download.hostname != "files.pythonhosted.org":
            raise ValueError(f"Unexpected wheel provenance: {name}")
        wheels.append({"filename": item["filename"], "sha256": digest, "url": item["url"],
                       "requires_python": item.get("requires_python"), "size": item["size"]})
    if not wheels:
        raise ValueError(f"No non-yanked compatible official wheel for {name}=={version}")
    metadata_dir.joinpath(f"{name}-{version}.json").write_bytes(raw)
    return {"name": name, "version": version, "metadata_url": url,
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "wheels": sorted(wheels, key=lambda item: item["filename"])}


def lock_text(packages: list[dict], image: str) -> str:
    lines = ["# Linux x86-64 / CPython 3.12; image-only dependency lock.",
             f"# Proven source image: {image}",
             "# Generated by docker/generate_container_lock.py from official PyPI release metadata.",
             "# No upgrades. See requirements-lock-provenance.json for wheel URLs and metadata digests.", ""]
    for package in packages:
        name = package["name"] + ("[and-cuda]" if package["name"] == "tensorflow" else "")
        hashes = sorted({item["sha256"] for item in package["wheels"]})
        lines.append(f"{name}=={package['version']} \\")
        lines.extend(f"    --hash=sha256:{digest}" + (" \\" if index < len(hashes) - 1 else "")
                     for index, digest in enumerate(hashes))
        lines.append("")
    return "\n".join(lines)


def publish_lock_set(output: Path, bootstrap: str, application: str, provenance: dict) -> None:
    """Stage the set and publish its digest-bearing commit record last.

    Three replacements are not a filesystem-wide transaction. An interrupted
    or competing publication may leave a mixed set, which the build checker
    must reject before pip rather than treating it as verified provenance.
    """
    output = output.resolve()
    bodies = dict(zip(LOCK_FILES, (bootstrap, application)))
    record = {**provenance, "lock_files_normalized_sha256": {
        name: normalized_text_digest(text) for name, text in bodies.items()}}
    with TemporaryDirectory(prefix=".requirements-staging-", dir=output) as directory:
        staged = Path(directory)
        if staged.resolve().parent != output:
            raise ValueError("Lock staging directory escaped its intended output directory")
        for name, text in bodies.items():
            (staged / name).write_text(text, encoding="utf-8", newline="\n")
        record_name = "requirements-lock-provenance.json"
        (staged / record_name).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
        # Only fully staged files are exposed; the provenance is the commit
        # record. Docker copies these files into an immutable build snapshot.
        for name in (*LOCK_FILES, record_name):
            os.replace(staged / name, output / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--metadata-dir", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.source_image):
        parser.error("Use the inspected image's exact sha256 ID")
    raw_inventory = args.inventory.read_bytes()
    inventory = json.loads(raw_inventory.decode("utf-8-sig"))
    tags = set(inventory["tags"])
    packages = inventory["distributions"]
    if not inventory["python"].startswith("3.12.") or "cp312-cp312-linux_x86_64" not in tags:
        parser.error("Inventory must come from the proven Linux x86-64 CPython 3.12 image")
    names = [canonical(item["name"]) for item in packages]
    if len(names) != len(set(names)) or names.count("pip") != 1:
        parser.error("Inventory must contain unique distributions and exactly one pip")
    root = Path(__file__).resolve().parents[1]
    try:
        validation = validate_inventory(root, inventory)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    # No network or output publication occurs until the declared inputs and
    # requested extras are satisfied by the captured target inventory.
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as workers:
        releases = list(workers.map(lambda package: release(package, tags, args.metadata_dir), packages))
    releases.sort(key=lambda package: package["name"])
    output = root / "docker"
    provenance = {"source_image": args.source_image, "python": inventory["python"],
                  "inventory_sha256": hashlib.sha256(raw_inventory).hexdigest(),
                  "fetched_at": datetime.now(timezone.utc).isoformat(),
                  "source_requirements_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                                 for name in ("requirements.txt", "requirements-gpu.txt")},
                  "source_requirements_normalized_sha256": {name: normalized_digest(root / name)
                                                            for name in ("requirements.txt", "requirements-gpu.txt")},
                  "requirements_validation": validation,
                  "validation_inventory_sha256": hashlib.sha256(raw_inventory).hexdigest(),
                  "validation_source_image": args.source_image,
                  "packages": releases}
    # Every official release is validated before staging and publication.
    publish_lock_set(output,
                     lock_text([item for item in releases if item["name"] == "pip"], args.source_image),
                     lock_text([item for item in releases if item["name"] != "pip"], args.source_image),
                     provenance)
    print(f"Pinned {len(releases)} proven distributions; wheel hashes will be verified during cold install.")


if __name__ == "__main__":
    main()
