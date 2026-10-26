"""Explicit coupled correction branches and scenario-bound checkpoint restoration.

Importing this module never imports PyTorch.  Network classes are constructed
only after a caller supplies the runtime or explicitly requests restoration.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


CORRECTION_ARCHITECTURE = "coupled_correction_branches"
CORRECTION_VERSION = 1
_GROUP_OUTPUTS = {"thermofluid": (0, 1, 2, 3), "elliptic": (10, 11)}
_FREQUENCIES = (1, 2, 4, 8)


def _check_core(torch: Any, core: Any) -> None:
    if not isinstance(core, torch.nn.Sequential) or len(core) != 7:
        raise ValueError(
            "Coupled corrections require the existing 21-64-64-64-12 tanh network."
        )
    for index, widths in ((0, (21, 64)), (2, (64, 64)), (4, (64, 64)), (6, (64, 12))):
        layer = core[index]
        if (
            not isinstance(layer, torch.nn.Linear)
            or (layer.in_features, layer.out_features) != widths
            or layer.bias is None
        ):
            raise ValueError(
                "Coupled corrections require the existing 21-64-64-64-12 tanh network."
            )
    if any(not isinstance(core[index], torch.nn.Tanh) for index in (1, 3, 5)):
        raise ValueError("Coupled correction activations must be tanh.")


def create_correction_model(
    core: Any, active_groups: tuple[str, ...], *, torch: Any = None
) -> Any:
    """Clone a coupled model into independent, initially identical correction branches.

    The old network is frozen without detaching its input graph.  Each selected
    branch begins with the same hidden layers and its original output rows;
    added spatial features have zero initial weights.  No random draws are made.
    """
    if torch is None:
        from .fields import _torch

        torch = _torch()
    _check_core(torch, core)
    if (
        not isinstance(active_groups, tuple)
        or any(
            not isinstance(group, str) or group not in _GROUP_OUTPUTS
            for group in active_groups
        )
        or len(set(active_groups)) != len(active_groups)
    ):
        raise ValueError("Correction groups must be unique thermofluid/elliptic names.")

    def branch(rows: tuple[int, ...]) -> Any:
        result = deepcopy(core)
        first, final = result[0], result[-1]
        first.in_features = 53
        first.weight = torch.nn.Parameter(
            torch.cat(
                (first.weight.detach().clone(), first.weight.new_zeros((64, 32))), dim=1
            )
        )
        final.out_features = len(rows)
        final.weight = torch.nn.Parameter(final.weight.detach()[list(rows)].clone())
        final.bias = torch.nn.Parameter(final.bias.detach()[list(rows)].clone())
        result.requires_grad_(True)
        return result

    class CoupledCorrectionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.active_groups = active_groups
            self.core = deepcopy(core).requires_grad_(False)
            self.thermofluid = (
                branch(_GROUP_OUTPUTS["thermofluid"])
                if "thermofluid" in active_groups
                else None
            )
            self.elliptic = (
                branch(_GROUP_OUTPUTS["elliptic"])
                if "elliptic" in active_groups
                else None
            )

        def branch_parameters(self):
            for selected in (self.thermofluid, self.elliptic):
                if selected is not None:
                    yield from selected.parameters()

        def architecture_metadata(self) -> dict[str, Any]:
            return {
                "kind": CORRECTION_ARCHITECTURE,
                "version": CORRECTION_VERSION,
                "widths": [21, 64, 64, 64, 12],
                "activation": "tanh",
                "dtype": "float64",
                "branch_input_width": 53,
                "active_groups": list(self.active_groups),
                "branch_outputs": {
                    name: list(rows) for name, rows in _GROUP_OUTPUTS.items()
                },
                "spatial_frequencies": list(_FREQUENCIES),
                "spatial_features": "sin/cos(pi*f*[cross_u,cross_v,sin_major,cos_major])",
                "core_frozen": True,
            }

        def forward(self, inputs: Any) -> Any:
            if inputs.ndim != 2 or inputs.shape[1] != 21:
                raise ValueError(
                    "Coupled correction inputs must have shape (samples, 21)."
                )
            output = self.core(inputs)
            if not self.active_groups:
                return output
            coordinates = torch.stack(
                (
                    inputs[:, 3] * inputs[:, 4],
                    inputs[:, 3] * inputs[:, 5],
                    inputs[:, 6],
                    inputs[:, 7],
                ),
                dim=1,
            )
            phase = torch.pi * coordinates[:, :, None] * inputs.new_tensor(_FREQUENCIES)
            enriched = torch.cat(
                (inputs, torch.sin(phase).flatten(1), torch.cos(phase).flatten(1)),
                dim=1,
            )
            if self.thermofluid is not None:
                output = torch.cat((self.thermofluid(enriched), output[:, 4:]), dim=1)
            if self.elliptic is not None:
                output = torch.cat((output[:, :10], self.elliptic(enriched)), dim=1)
            return output

    return CoupledCorrectionModel()


def restore_model_from_checkpoint(
    checkpoint: str | Path | Mapping[str, Any], source: Any, device: str = "cpu"
) -> Any:
    """Restore only known network layouts bound to the explicitly supplied source.

    File loading uses PyTorch's restricted weights-only reader.  Architecture
    metadata selects an in-package layout; it never names executable code.
    """
    from .fields import (
        FORMAT_VERSION,
        ThermalOperator,
        _make_network,
        _operator_fingerprint,
        _torch,
    )
    from .modulation import FILM_ARCHITECTURE, FILM_VERSION, create_film_model
    from .gauge import (TEMPORAL_GAUGE_ARCHITECTURE, TEMPORAL_GAUGE_VERSION,
                        create_temporal_gauge_model)
    import numpy as np

    torch = _torch()
    payload = (
        torch.load(checkpoint, map_location=device, weights_only=True)
        if not isinstance(checkpoint, Mapping)
        else checkpoint
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("Unsupported PINN checkpoint format version.")
    if (
        payload.get("source_run_id") != source.run_id
        or payload.get("config_hash") != source.config.config_hash
    ):
        raise ValueError(
            "Checkpoint belongs to a different source run or scenario configuration."
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata must be a mapping.")
    if metadata.get("format_version", FORMAT_VERSION) != FORMAT_VERSION:
        raise ValueError("Unsupported PINN checkpoint metadata version.")
    for key, expected in (
        ("source_run_id", source.run_id),
        ("source_config_hash", source.config.config_hash),
        ("source_physics_hash", source.config.physics_hash),
    ):
        if key in metadata and metadata[key] != expected:
            raise ValueError(
                "Checkpoint source provenance does not match the supplied run."
            )
    if "source_snapshot_times_s" in metadata and not np.array_equal(
        metadata["source_snapshot_times_s"], source.times
    ):
        raise ValueError(
            "Checkpoint accepted snapshot times do not match the supplied source."
        )
    if "source_operator_sha256" in metadata:
        operator = (
            ThermalOperator.from_result(source)
            if source.config.solver == "thermal_benchmark"
            else None
        )
        if metadata["source_operator_sha256"] != _operator_fingerprint(
            operator, source
        ):
            raise ValueError(
                "Checkpoint physical operator does not match the supplied source."
            )
    thermal = source.config.solver == "thermal_benchmark"
    widths = [4 if thermal else 21, 64, 64, 64, 1 if thermal else 12]
    architecture = metadata.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("Checkpoint has no supported network architecture metadata.")
    if (
        architecture.get("widths") != widths
        or architecture.get("activation") != "tanh"
        or architecture.get("dtype") != "float64"
    ):
        raise ValueError("Unsupported PINN checkpoint network architecture.")
    kind = architecture.get("kind", "sequential_tanh")
    if kind not in {"sequential_tanh", CORRECTION_ARCHITECTURE, FILM_ARCHITECTURE,
                    TEMPORAL_GAUGE_ARCHITECTURE}:
        raise ValueError("Unsupported PINN checkpoint network architecture.")
    # Creating a known architecture must not change the caller's random stream.
    with torch.random.fork_rng(devices=[]):
        if kind == FILM_ARCHITECTURE:
            if thermal or architecture.get("version") != FILM_VERSION:
                raise ValueError("Unsupported PINN FiLM architecture version.")
            model = create_film_model(torch=torch).to(device=device)
        else:
            model = _make_network(torch, widths[-1], widths[0]).to(
                device=device, dtype=torch.float64
            )
    if kind == FILM_ARCHITECTURE:
        if dict(architecture) != model.architecture_metadata():
            raise ValueError("Unsupported PINN FiLM architecture metadata.")
    elif kind == TEMPORAL_GAUGE_ARCHITECTURE:
        if thermal or architecture.get("version") != TEMPORAL_GAUGE_VERSION:
            raise ValueError("Unsupported temporal phi architecture version.")
        times = np.asarray(source.times)
        if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
            raise ValueError("Temporal phi needs strictly increasing accepted source times.")
        expected_times = (times - times[0]) / (times[-1] - times[0])
        if not np.array_equal(architecture.get("normalized_times"), expected_times):
            raise ValueError("Temporal phi knots do not match the source accepted times.")
        model = create_temporal_gauge_model(model, expected_times, torch=torch)
        if dict(architecture) != model.architecture_metadata():
            raise ValueError("Unsupported temporal phi architecture metadata.")
    elif kind == CORRECTION_ARCHITECTURE:
        if thermal or architecture.get("version") != CORRECTION_VERSION:
            raise ValueError("Unsupported PINN correction architecture version.")
        groups = architecture.get("active_groups")
        if not isinstance(groups, list):
            raise ValueError("Checkpoint correction groups must be a list.")
        model = create_correction_model(model, tuple(groups), torch=torch)
        if dict(architecture) != model.architecture_metadata():
            raise ValueError("Unsupported PINN correction architecture metadata.")
    elif (
        set(architecture) - {"kind", "version", "widths", "activation", "dtype"}
        or architecture.get("version", 1) != 1
    ):
        raise ValueError("Unsupported legacy PINN network architecture metadata.")
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str)
        or not torch.is_tensor(value)
        or not bool(torch.isfinite(value).all())
        for name, value in state.items()
    ):
        raise ValueError("Checkpoint network weights are invalid or non-finite.")
    if kind == TEMPORAL_GAUGE_ARCHITECTURE:
        knots = state.get("normalized_times")
        if (not torch.is_tensor(knots) or knots.dtype != torch.float64
                or not torch.equal(knots.to(device=device), model.normalized_times)):
            raise ValueError("Temporal phi state knots do not match the source accepted times.")
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            "Checkpoint weights do not match its declared architecture."
        ) from error
    return model.eval()
