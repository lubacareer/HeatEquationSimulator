"""Scenario-bound PINNs with lazy PyTorch loading and unchanged public imports.

Field/operator contracts, staged optimization, and independent validation have
separate owners in :mod:`magnetic_torus.learning`.
"""
from .learning.fields import (
    FORMAT_VERSION,
    STEFAN_BOLTZMANN,
    FieldScaling,
    ThermalOperator,
    UnsupportedPINNConfiguration,
    enthalpy_to_state_torch,
    regularized_enthalpy_to_state_torch,
    validate_pinn_source,
)
from .learning.training import train_pinn

__all__ = [
    "FORMAT_VERSION", "STEFAN_BOLTZMANN", "FieldScaling", "ThermalOperator",
    "UnsupportedPINNConfiguration", "enthalpy_to_state_torch",
    "regularized_enthalpy_to_state_torch", "train_pinn", "validate_pinn_source",
]
