"""Serializable case descriptions; importing the runner loads no science runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CaseSpec:
    id: str
    category: str
    description: str
    command: tuple[str, ...]
    timeout: float = 300.0
    capabilities: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    kind: str = "functional"
    dependencies: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
