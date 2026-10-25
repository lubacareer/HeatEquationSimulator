"""Qt-free descriptions of the selected scientific fields and scene options."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class DisplaySelection:
    fields: Mapping[str, np.ndarray] | None = None
    difference: bool = False
    source_label: str | None = None


@dataclass(frozen=True)
class SceneOptions:
    cutaway: bool = True
    show_phase: bool = False
    show_flow: bool = False
    show_tracers: bool = False
    show_field: bool = False
    slice_enabled: bool = False
    slice_position: float = 0.0
    visual_detail: str = "Medium"
    opacity: float = 1.0
