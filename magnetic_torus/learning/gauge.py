"""Research-only temporal electric-potential correction in the source FE gauge.

The scalar correction is spatially constant on the conductor after the outer
field masks. It is learned through the original joint objective; no reference
labels or spatial means initialize it. Importing this module does not load Torch.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


TEMPORAL_GAUGE_ARCHITECTURE = "coupled_temporal_phi"
TEMPORAL_GAUGE_VERSION = 1


def create_temporal_gauge_model(
    core: Any, normalized_times: Any, *, torch: Any = None
) -> Any:
    """Freeze a legacy coupled core and learn additive accepted-time phi biases.

    The bias at the first accepted time is fixed at zero. Between the accepted
    times it is linearly interpolated, with constant endpoint extension. The
    outer ``NeuralField`` retains its existing initial and Dirichlet constraints.
    """
    if torch is None:
        from .fields import _torch

        torch = _torch()
    from .corrections import _check_core

    _check_core(torch, core)
    parameters = tuple(core.parameters())
    if any(parameter.dtype != torch.float64 for parameter in parameters):
        raise ValueError("Temporal phi corrections require a float64 coupled core.")
    if len({parameter.device for parameter in parameters}) != 1:
        raise ValueError("Temporal phi corrections require one core device.")
    if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
        raise ValueError("Temporal phi corrections require finite core weights.")
    try:
        times = (
            torch.as_tensor(
                normalized_times, dtype=torch.float64, device=parameters[0].device
            )
            .detach()
            .clone()
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            "Accepted normalized times must be a finite 1D sequence."
        ) from error
    if (
        times.ndim != 1
        or len(times) < 2
        or not bool(torch.isfinite(times).all())
        or float(times[0]) != 0.0
        or float(times[-1]) != 1.0
        or not bool((times[1:] > times[:-1]).all())
    ):
        raise ValueError(
            "Accepted normalized times must increase strictly from 0 to 1."
        )
    time_values = [float(value) for value in times.detach().cpu().tolist()]

    class TemporalGaugeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = deepcopy(core).requires_grad_(False)
            self.register_buffer("normalized_times", times)
            self.phi_bias = torch.nn.Parameter(times.new_zeros(len(times) - 1))

        def branch_parameters(self):
            yield self.phi_bias

        def architecture_metadata(self) -> dict[str, Any]:
            return {
                "kind": TEMPORAL_GAUGE_ARCHITECTURE,
                "version": TEMPORAL_GAUGE_VERSION,
                "widths": [21, 64, 64, 64, 12],
                "activation": "tanh",
                "dtype": "float64",
                "normalized_times": list(time_values),
                "bias_output": 11,
                "initial_bias": 0.0,
                "interpolation": "linear; constant endpoint extension",
                "core_frozen": True,
                "gauge": "original source FE gauge; no spatial mean projection",
            }

        def forward(self, inputs: Any) -> Any:
            if inputs.ndim != 2 or inputs.shape[1] != 21:
                raise ValueError("Temporal phi inputs must have shape (samples, 21).")
            query = inputs[:, 8].contiguous()
            if not bool(torch.isfinite(query).all()):
                raise ValueError("Temporal phi query times must be finite.")
            values = torch.cat((self.phi_bias.new_zeros(1), self.phi_bias))
            upper = torch.searchsorted(self.normalized_times, query, right=True)
            upper = upper.clamp(1, len(time_values) - 1)
            lower = upper - 1
            weight = (
                (query - self.normalized_times[lower])
                / (self.normalized_times[upper] - self.normalized_times[lower])
            ).clamp(0.0, 1.0)
            bias = values[lower] + weight * (values[upper] - values[lower])
            original = self.core(inputs)
            return torch.cat(
                (original[:, :11], original[:, 11:] + bias[:, None]), dim=1
            )

    return TemporalGaugeModel()
