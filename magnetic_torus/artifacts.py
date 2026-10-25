"""Explicitly captured immutable field artifacts, separate from JSON metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path


FIELD_ARTIFACT_KEYS = ("weak_operator_path", "exterior_field_path", "thermal_operator_path")


@dataclass(frozen=True)
class FieldArtifact:
    """Accepted bytes; a metadata string can never grant filesystem access."""

    data: bytes = field(repr=False)
    origin: Path | None = None
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise TypeError("Field artifacts require immutable bytes.")
        object.__setattr__(self, "digest", sha256(self.data).hexdigest())

    @classmethod
    def capture(cls, path: str | Path, *, root: str | Path) -> "FieldArtifact":
        """Only a trusted producer may explicitly capture its output directory."""
        directory = Path(root).resolve(strict=True)
        source = Path(path).resolve(strict=True)
        if not directory.is_dir() or not source.is_relative_to(directory) or not source.is_file():
            raise ValueError("Field artifact must remain inside its producing directory.")
        return cls(source.read_bytes(), source)
