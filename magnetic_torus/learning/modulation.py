"""Fixed FiLM time-conditioning architecture for explicit research experiments.

The desktop and normal trainer do not select this architecture.  Importing the
module does not import PyTorch, fit a basis, or inspect numerical labels.
"""

from __future__ import annotations

from typing import Any


FILM_ARCHITECTURE = "coupled_film_tanh"
FILM_VERSION = 1


def create_film_model(*, torch: Any = None) -> Any:
    """Build a fixed 21-input, 12-output float64 coupled correction model.

    Inputs retain ``NeuralField``'s existing geometry/time feature convention:
    columns 0:8 are spatial and columns 8:21 are normalized time plus its fixed
    Fourier features.  The zero output starts at the prescribed particular
    solution.  Initial/boundary masks remain the outer field's responsibility.
    No convergence or computational speed improvement is assumed.
    """
    if torch is None:
        from .fields import _torch

        torch = _torch()

    class CoupledFiLMModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spatial = torch.nn.Sequential(
                torch.nn.Linear(8, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
            )
            self.temporal = torch.nn.Sequential(
                torch.nn.Linear(13, 32),
                torch.nn.Tanh(),
                torch.nn.Linear(32, 128),
            )
            self.output = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 12),
            )
            self.double()
            with torch.no_grad():
                self.temporal[-1].weight.zero_()
                self.temporal[-1].bias.zero_()
                self.output[-1].weight.zero_()
                self.output[-1].bias.zero_()

        def architecture_metadata(self) -> dict[str, Any]:
            return {
                "kind": FILM_ARCHITECTURE,
                "version": FILM_VERSION,
                "widths": [21, 64, 64, 64, 12],
                "activation": "tanh",
                "dtype": "float64",
                "spatial_columns": list(range(8)),
                "temporal_columns": list(range(8, 21)),
                "spatial_widths": [8, 64, 64],
                "temporal_widths": [13, 32, 128],
                "output_widths": [64, 64, 12],
                "modulation": "spatial * (1 + gamma(time)) + beta(time)",
                "initial_correction": "zero",
                "experimental": True,
            }

        def forward(self, inputs: Any) -> Any:
            if inputs.ndim != 2 or inputs.shape[1] != 21:
                raise ValueError("Coupled FiLM inputs must have shape (samples, 21).")
            spatial = self.spatial(inputs[:, :8])
            gamma, beta = self.temporal(inputs[:, 8:]).chunk(2, dim=1)
            return self.output(spatial * (1.0 + gamma) + beta)

    return CoupledFiLMModel()
