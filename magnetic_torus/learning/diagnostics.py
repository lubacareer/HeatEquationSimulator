"""Shared validation limits and explanations for current and saved PINN fits."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..models import PINNResult

LOSS_LIMIT = 0.01
FIELD_RMSE_LIMIT = 0.1
RESIDUAL_RMS_LIMIT = 0.1
ENERGY_DEFECT_LIMIT = 0.05


@dataclass(frozen=True)
class ValidationFailure:
    metric: str
    label: str
    value: float
    limit: float


def failed_checks(model: PINNResult) -> tuple[ValidationFailure, ...]:
    """Explain recorded errors without changing a saved model's status."""
    thermal = model.metadata.get(
        "thermal_physics_only", "energy_defect_relative" in model.metrics
    )
    losses = {
        "validation_loss": "held-out loss",
        "physics_loss": "physics loss",
        "boundary_loss": "boundary loss",
    }
    failures = []
    for key, value in model.metrics.items():
        if key in losses:
            label, limit = losses[key], LOSS_LIMIT
        elif key.endswith("_normalized_rmse"):
            label, limit = (
                key.removesuffix("_normalized_rmse").replace("_", " ")
                + " normalized RMSE",
                FIELD_RMSE_LIMIT,
            )
        elif key.endswith("_residual_rms") and (
            not thermal or key == "enthalpy_residual_rms"
        ):
            label, limit = (
                key.removesuffix("_residual_rms").replace("_", " ") + " residual",
                RESIDUAL_RMS_LIMIT,
            )
        elif thermal and key == "energy_defect_relative":
            label, limit = "relative energy defect", ENERGY_DEFECT_LIMIT
        else:
            continue
        if not math.isfinite(value) or value > limit:
            failures.append(ValidationFailure(key, label, value, limit))
    return tuple(failures)


def training_epochs(model: PINNResult) -> int | None:
    value = model.metrics.get("epochs", model.metadata.get("training_epochs"))
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (ValueError, TypeError):
        return None
    return (
        int(numeric)
        if math.isfinite(numeric) and numeric >= 0 and numeric.is_integer()
        else None
    )


def validation_message(model: PINNResult) -> str:
    status = str(model.metadata.get("validation_status", "unvalidated"))
    epochs = training_epochs(model)
    budget = f" after {epochs} epochs" if epochs is not None else ""
    if status != "failed":
        return f"PINN validation: {status}{budget}."
    checks = failed_checks(model)
    details = [
        (
            f"{check.label} {check.value:.3g} > {check.limit:g}"
            if math.isfinite(check.value)
            else f"{check.label} is non-finite (limit {check.limit:g})"
        )
        for check in checks[:4]
    ]
    if len(checks) > 4:
        details.append(f"{len(checks) - 4} more failed checks")
    missing = model.metadata.get("fields_without_heldout_labels", ())
    if missing:
        details.append(
            "missing held-out labels: " + ", ".join(str(name) for name in missing)
        )
    reason = (
        "; ".join(details) or "inspect the saved validation criteria and diagnostics"
    )
    return f"PINN validation failed{budget}: {reason}. Accepted numerical results remain available."
