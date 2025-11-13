import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, cast

# NumPy compatibility shims (e.g., np.bool8 on NumPy 2)
import np_compat  # noqa: F401
import numpy as np

# Optional SciPy imports with graceful fallback on Windows if binary wheels are missing,
# and with stderr/stdout silenced to avoid noisy DLL ImportError messages on Windows.
SCIPY_AVAILABLE = True
SCIPY_IMPORT_ERROR = None
try:
    import contextlib
    import io

    _scipy_capture = io.StringIO()
    with (
        contextlib.redirect_stderr(_scipy_capture),
        contextlib.redirect_stdout(_scipy_capture),
    ):
        from scipy.sparse import lil_matrix  # type: ignore
        from scipy.sparse.linalg import splu  # type: ignore
        from scipy.special import erf as scipy_erf  # type: ignore
except Exception as e:
    SCIPY_AVAILABLE = False
    # Preserve (but do not print) the import error and any noisy output
    try:
        _scipy_noisy = _scipy_capture.getvalue()
    except Exception:
        _scipy_noisy = ""
    SCIPY_IMPORT_ERROR = f"{e}" if not _scipy_noisy else f"{e}\n{_scipy_noisy.strip()}"
    scipy_erf = None

# Provide a vectorized erf that does not require SciPy
ErfArg = Union[float, np.ndarray]
FloatArray = ErfArg

if SCIPY_AVAILABLE and "scipy_erf" in globals() and scipy_erf is not None:

    def erf(x: ErfArg) -> ErfArg:
        return cast(ErfArg, scipy_erf(x))  # type: ignore[no-any-return]

else:
    from math import erf as _math_erf

    _vectorized_erf = np.vectorize(_math_erf)

    def erf(x: ErfArg) -> ErfArg:
        if isinstance(x, np.ndarray):
            return cast(ErfArg, _vectorized_erf(x))
        return cast(ErfArg, _math_erf(float(x)))


# def compute_mask_arrays(T, cls, tolerance=100, phase_mask=None, boundary_mask=None):
#     if phase_mask is None:
#         phase_mask = np.zeros_like(T, dtype=int)
#     if boundary_mask is None:
#         boundary_mask = np.zeros_like(T, dtype=int)
#
#     T_minus = cls.T_m - tolerance
#     T_plus = cls.T_m + tolerance
#
#     # Initialize phase mask
#     phase_mask[:] = 0
#     phase_mask[T < T_minus] = 0  # Solid phase
#     phase_mask[(T >= T_minus) & (T <= T_plus)] = 1  # Phase transition (melting/freezing)
#     phase_mask[T > T_plus] = 2  # Liquid phase
#
#     # Initialize boundary mask
#     boundary_mask[:] = 0
#     boundary_mask[phase_mask == 1] = 1  # Mark where phase change occurs
#
#     # Debugging information
#     # print(f"T_minus: {T_minus}, T_plus: {T_plus}")
#     # print(f"phase_mask: {np.unique(phase_mask, return_counts=True)}")
#     # print(f"boundary_mask: {np.unique(boundary_mask, return_counts=True)}")
#
#     return phase_mask, boundary_mask
def compute_mask_arrays(T, H, cls, tolerance=10, phase_mask=None, boundary_mask=None):
    # Temperature-based phase and boundary masks around the melting point
    if phase_mask is None:
        phase_mask = np.zeros_like(T, dtype=int)
    if boundary_mask is None:
        boundary_mask = np.zeros_like(T, dtype=int)

    T_minus = cls.T_m - tolerance
    T_plus = cls.T_m + tolerance

    # Phase mask: 0 solid, 1 mushy (near melting), 2 liquid
    phase_mask[:] = 0
    phase_mask[(T >= T_minus) & (T <= T_plus)] = 1
    phase_mask[T > T_plus] = 2

    # Boundary mask marks the mushy zone
    boundary_mask[:] = 0
    boundary_mask[phase_mask == 1] = 1

    return phase_mask, boundary_mask


class PCM(ABC):
    # Common scalar attributes expected across concrete PCM implementations
    dt: float
    dx: float
    T_m: float
    T_a: float
    alpha2: float
    rho: float
    k: float
    c: float
    c_solid: float
    c_liquid: float
    LH: float
    x_max: float
    cycle_duration: float
    heat_source_max: float

    @abstractmethod
    def calcThermalConductivity(self, temp: FloatArray) -> FloatArray: ...

    @abstractmethod
    def calcSpecificHeat(self, temp: FloatArray) -> FloatArray: ...

    @abstractmethod
    def generate_data(self, x_max: float, t_max: float) -> Tuple[np.ndarray, ...]: ...

    @abstractmethod
    def implicitSol(
        self,
        x_arr: np.ndarray,
        t_arr: np.ndarray,
        T_arr: np.ndarray,
        H_arr: np.ndarray,
        cls: "PCM",
        phase_mask_array: Optional[np.ndarray] = None,
        boundary_mask_array: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, ...]: ...

    @abstractmethod
    def explicitNumerical(
        self,
        x_arr: np.ndarray,
        t_arr: np.ndarray,
        T_arr: np.ndarray,
        cls: "PCM",
        phase_mask_array: Optional[np.ndarray] = None,
        boundary_mask_array: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, ...]: ...

    @abstractmethod
    def heat_source_function(
        self,
        x: Union[float, np.ndarray],
        t: float,
        cycle_duration: Optional[float] = None,
        heat_source_max: Optional[float] = None,
    ) -> Union[float, np.ndarray]: ...

    def enforce_stability_dt(self, dt_value: float, safety_limit: float = 0.5) -> float:
        """Ensure the time step respects the diffusion stability criterion alpha*dt/dx^2 <= safety_limit."""
        try:
            alpha = float(getattr(self, "alpha2", 0.0))
            dx = float(getattr(self, "dx", 0.0))
            dt_scalar = float(dt_value)
        except (TypeError, ValueError):
            return float(dt_value)

        if alpha <= 0.0 or dx <= 0.0:
            return dt_scalar
        max_dt = safety_limit * (dx**2) / alpha
        if dt_scalar > max_dt:
            print(
                f"Info: Reducing dt from {dt_scalar:.6g} to {max_dt:.6g} to satisfy stability condition "
                f"(alpha*dt/dx^2 <= {safety_limit})."
            )
            return max_dt
        return dt_scalar

    def alpha(self, k, c, rho):
        # Check for division by zero or small number
        if c * rho == 0 or k == 0:
            raise ValueError("c * rho should not be zero")
        return k / (c * rho)

    def _solve_stefan_lambda(
        self, stefan_number: float, tol: float = 1e-6, max_iter: int = 100
    ) -> float:
        """Solve the transcendental Stefan equation for the similarity parameter λ.

        Uses a simple bisection search to avoid pulling an additional dependency when SciPy is missing.
        """
        if stefan_number <= 0.0:
            return 0.0

        # The Stefan equation for a semi-infinite slab with constant surface temperature is:
        #     Ste = λ * exp(λ^2) * erf(λ)
        # The function is monotonically increasing for λ > 0, so bisection is robust.
        def stefan_function(lmbd: float) -> float:
            return lmbd * math.exp(lmbd * lmbd) * math.erf(lmbd) - stefan_number

        a, b = 1e-8, max(1.0, stefan_number)
        fa, fb = stefan_function(a), stefan_function(b)
        # Expand the upper bracket until a sign change is found or a generous ceiling is hit.
        while fa * fb > 0 and b < 1e6:
            b *= 2.0
            fb = stefan_function(b)

        # If we still failed to bracket the root, fall back to the current upper bound.
        if fa * fb > 0:
            return b

        for _ in range(max_iter):
            mid = 0.5 * (a + b)
            fmid = stefan_function(mid)
            if abs(fmid) < tol:
                return mid
            if fa * fmid <= 0:
                b, fb = mid, fmid
            else:
                a, fa = mid, fmid

        return 0.5 * (a + b)

    def stefan_boundary_location(
        self, t_arr: np.ndarray, surface_temp: Optional[float] = None
    ) -> np.ndarray:
        """Return the analytical Stefan moving-boundary position s(t) for a semi-infinite PCM.

        The closed-form solution assumes a constant surface temperature above the melting point and
        a single moving interface. The boundary location is:
            s(t) = 2 λ sqrt(α t)
        where λ solves the Stefan equation and α is the thermal diffusivity.
        """
        surface_temperature = (
            surface_temp
            if surface_temp is not None
            else (getattr(self, "T_m", 0.0) + 10.0)
        )
        latent_heat = getattr(self, "LH", 0.0)
        if latent_heat <= 0.0:
            return np.zeros_like(t_arr, dtype=float)

        # Use solid specific heat when available; fall back to the generic value.
        cp_solid = getattr(self, "c_solid", getattr(self, "c", 1.0))
        delta_T = max(surface_temperature - getattr(self, "T_m", 0.0), 0.0)
        if delta_T <= 0.0:
            return np.zeros_like(t_arr, dtype=float)

        stefan_number = (cp_solid * delta_T) / latent_heat
        lambda_param = self._solve_stefan_lambda(stefan_number)
        if lambda_param <= 0.0:
            return np.zeros_like(t_arr, dtype=float)

        alpha_eff = getattr(self, "alpha2", None)
        if alpha_eff is None:
            alpha_eff = self.alpha(
                getattr(self, "k", 1.0),
                getattr(self, "c", cp_solid),
                getattr(self, "rho", 1.0),
            )

        sqrt_term = np.sqrt(np.maximum(t_arr, 0.0))
        return 2.0 * lambda_param * np.sqrt(alpha_eff) * sqrt_term

    def solve_stefan_problem_enthalpy(
        self,
        cls,
        x_arr,
        t_arr,
        T_arr,
        H_arr,
        temp_mask_array=None,
        bound_mask_array=None,
    ):
        print("Debug: Entering solve_stefan_problem_enthalpy")

        if temp_mask_array is None or bound_mask_array is None:
            temp_mask_array = np.zeros((len(x_arr), len(t_arr)), dtype=int)
            bound_mask_array = np.zeros((len(x_arr), len(t_arr)), dtype=int)

        boundary_indices = np.full(len(t_arr), -1, dtype=int)

        # Initialize temperature with analytical solution for smooth start
        T_initial = self.analyticalSol(x_arr, t_arr[:1], cls)[:, 0]
        T_arr[:, 0] = T_initial
        H_arr[:, 0] = cls.calcEnthalpy2(T_initial, cls)

        for t_idx in range(1, len(t_arr)):
            H_old = H_arr[:, t_idx - 1]
            T_old = T_arr[:, t_idx - 1]

            # Calculate thermal properties
            k_vals = np.array(
                [cls.calcThermalConductivity(T_old[i]) for i in range(len(x_arr))]
            )
            c_vals = np.array(
                [cls.calcSpecificHeat(T_old[i]) for i in range(len(x_arr))]
            )
            alpha_vals = k_vals / (cls.rho * c_vals)
            lmbda_vals = cls.dt / cls.dx**2 * alpha_vals

            # Apply heat source term
            heat_source = np.array(
                [cls.heat_source_function(x, t_arr[t_idx]) for x in x_arr]
            )

            # Update enthalpy considering heat source
            H_new_internal = (
                H_old[1:-1]
                + (cls.dt / cls.dx**2)
                * (
                    lmbda_vals[1:-1] * (H_old[2:] - H_old[1:-1])
                    - lmbda_vals[:-2] * (H_old[1:-1] - H_old[:-2])
                )
                + cls.dt * heat_source[1:-1]
            )

            # Apply boundary conditions to H_new_internal
            H_new = np.zeros_like(H_old)
            H_new[1:-1] = H_new_internal
            # Set boundary enthalpy based on the boundary temperature (units consistency)
            H_new[0] = cls.calcEnthalpy2(cls.T_m + 10, cls)
            H_new[-1] = H_old[-1]  # Keep the last value as in the previous time step

            # Update temperature from the new enthalpy
            T_new = self.update_temperature(H_new, cls)

            # Store the new values in the arrays
            H_arr[:, t_idx] = H_new
            T_arr[:, t_idx] = T_new

            # Update phase mask based on the new temperature
            temp_mask_array[:, t_idx], bound_mask_array[:, t_idx] = compute_mask_arrays(
                T_new, H_new, cls
            )

            # Calculate the boundary index for the current time step
            phase_change_indices = np.where(np.abs(T_new - cls.T_m) <= 100)[
                0
            ]  # Adjust tolerance as needed
            if phase_change_indices.size > 0:
                boundary_indices[t_idx] = phase_change_indices[0]

        # print(f"Debug: solve_stefan_problem_enthalpy - T_arr shape: {T_arr.shape}, H_arr shape: {H_arr.shape}")
        return T_arr, H_arr, temp_mask_array, bound_mask_array, boundary_indices

    def update_gamma(self, cls, temp, dt_current):
        k_max = np.max([cls.calcThermalConductivity(t) for t in temp])
        c_max = np.max(cls.calcSpecificHeat(temp))
        alpha_max = k_max / (cls.rho * c_max)
        gamma = alpha_max * dt_current / cls.dx**2
        return gamma

    def initialize_enthalpy_temperature_arrays(self, x_arr, cls, t_steps):
        # Initialize the temperature array
        T = (
            np.ones((len(x_arr), t_steps)) * cls.T_a
        )  # Start all cells at ambient temperature
        T[0, :] = (
            cls.T_m + 10
        )  # Set the first spatial cell at the first time step to T_m + 10

        # Introduce a slight gradient or perturbation to the initial temperature array
        # for i in range(1, len(x_arr)):
        #     T[i, 0] = cls.T_a + np.random.uniform(0, 5)  # Add a small random perturbation

        # Initialize the enthalpy array
        H = self.initial_enthalpy(x_arr, cls, t_steps)
        return T, H

    def calculate_dt(self, cls, max_k=None, safety_factor=0.4, dt_multiplier=1.0):
        # Determine max_k based on the melting temperature if not provided
        if max_k is None:
            max_k = max(
                cls.calcThermalConductivity(cls.T_m),
                cls.calcThermalConductivity(cls.T_a),
            )

        max_alpha = max_k / (cls.rho * max(cls.c_solid, cls.c_liquid))
        max_dt = (safety_factor * cls.dx**2) / (max_alpha * 2)

        # Apply the multiplier to adjust dt
        calculated_dt = max_dt * dt_multiplier
        print(f"Calculated dt = {calculated_dt}")

        return float(calculated_dt)

    def update_enthalpy_temperature(self, H_current, cls, gamma, x_arr):
        dH = gamma * (np.roll(H_current, -1) - 2 * H_current + np.roll(H_current, 1))
        H_next = H_current + dH
        H_next[0] = H_current[0]  # Reapply left boundary condition
        H_next[-1] = H_current[-1]  # Reapply right boundary condition
        T_next = self.update_temperature(H_next, cls)
        return H_next, T_next

    def update_temperature(self, H_new, cls):
        T_new = np.zeros_like(H_new, dtype=np.float64)

        # Use the same epsilon as in calcEnthalpy2 by default
        epsilon = 0.01
        T_minus = cls.T_m - epsilon
        T_plus = cls.T_m + epsilon

        # Reference enthalpies at region boundaries (consistent units: J/m^3)
        H_minus = cls.rho * cls.c_solid * (T_minus - cls.T_a)
        H_at_Tm = cls.rho * cls.c_solid * (cls.T_m - cls.T_a) + cls.rho * cls.LH
        H_plus = H_at_Tm + cls.rho * cls.c_liquid * (T_plus - cls.T_m)

        for i, H in enumerate(H_new):
            if H <= H_minus:
                # Solid region
                T_new[i] = cls.T_a + H / (cls.rho * cls.c_solid)
            elif H >= H_plus:
                # Liquid region
                T_new[i] = cls.T_m + (H - H_at_Tm) / (cls.rho * cls.c_liquid)
            else:
                # Mushy region: invert the linear latent heat ramp used in calcEnthalpy2
                T_new[i] = T_minus + (H - H_minus) * (2 * epsilon) / (cls.rho * cls.LH)

            # Ensure temperature is not below ambient due to numerical noise
            if T_new[i] < cls.T_a:
                T_new[i] = cls.T_a

        return T_new

    def update_phase_mask(self, temperature_array, cls):
        tolerance = 10.0  # Wider tolerance around the melting point

        # Phase mask: 0 for solid, 1 for liquid, 2 for mushy zone
        phase_mask = np.select(
            [
                temperature_array < cls.T_m - tolerance,
                temperature_array > cls.T_m + tolerance,
            ],
            [0, 1],  # 0 for solid, 1 for liquid
            default=2,  # 2 for mushy zone (near melting point)
        )

        # Boundary mask: Detect sharp temperature gradients indicating a phase boundary
        gradient = np.gradient(temperature_array)
        gradient_threshold = (
            10  # Lower threshold for detecting smoother changes in temperature
        )
        boundary_mask = np.where(np.abs(gradient) > gradient_threshold, 1, 0)

        return phase_mask, boundary_mask

    def calculate_boundary_indices(
        self,
        x,
        x_max,
        dt,
        T=None,
        T_m=None,
        tolerance=100,
        mode="initial",
        atol=1e-8,
        rtol=1e-5,
    ):
        if mode == "initial":
            boundary_indices = {"condition1": [], "condition2": []}
            dt_indices = np.isclose(x[:, 1], dt, atol=atol, rtol=rtol)
            boundary_indices["condition1"] = np.where(
                dt_indices & np.isclose(x[:, 0], 0, atol=atol, rtol=rtol)
            )[0].tolist()
            boundary_indices["condition2"] = np.where(
                dt_indices & ~np.isclose(x[:, 0], x_max, atol=atol, rtol=rtol)
            )[0].tolist()
            return boundary_indices

        elif mode == "moving_boundary":
            if T is None or T_m is None:
                raise ValueError(
                    "Temperature array T and melting point T_m must be provided for 'moving_boundary' mode."
                )

            moving_boundary_indices = np.full(T.shape[1], -1, dtype=int)

            for n in range(T.shape[1]):
                phase_change_indices = np.where(np.abs(T[:, n] - T_m) <= tolerance)[0]
                if phase_change_indices.size > 0:
                    moving_boundary_indices[n] = phase_change_indices[0]

            return moving_boundary_indices

        else:
            raise ValueError(
                "Invalid mode. Choose between 'initial' and 'moving_boundary'."
            )

    def calculate_moving_boundary_indices(self, T_arr, T_m):
        boundary_indices = np.full(T_arr.shape[1], -1, dtype=int)
        for t_idx in range(T_arr.shape[1]):
            phase_change_indices = np.where(np.abs(T_arr[:, t_idx] - T_m) < 1e-2)[0]
            if phase_change_indices.size > 0:
                boundary_indices[t_idx] = phase_change_indices[0]
            else:
                boundary_indices[t_idx] = -1
            # print(f"Debug: Time step {t_idx} - phase_change_indices: {phase_change_indices}")
            # print(f"Debug: Time step {t_idx} - boundary_indices: {boundary_indices[t_idx]}")
        return boundary_indices

    def inverseEnthalpy2(self, H, cls):
        # Delegate to update_temperature to ensure a single consistent inversion implementation
        return self.update_temperature(H, cls)

    def initial_enthalpy(self, x_arr, cls, t_steps):
        H_arr = np.zeros((len(x_arr), t_steps), dtype=np.float64)
        for i, _ in enumerate(x_arr):
            T_initial = (
                cls.T_a if i != 0 else cls.T_m + 10
            )  # Initial temperature condition
            initial_H = cls.calcEnthalpy2(T_initial, cls)
            H_arr[i, :] = initial_H
        return H_arr

    def calcEnthalpy2(self, T, cls, epsilon=0.01, smoothing="h"):
        T_minus = cls.T_m - epsilon
        T_plus = cls.T_m + epsilon

        # Reference enthalpies for consistency
        H_minus = cls.rho * cls.c_solid * (T_minus - cls.T_a)
        H_at_Tm = cls.rho * cls.c_solid * (cls.T_m - cls.T_a) + cls.rho * cls.LH

        def smoothed_enthalpy(T_local):
            # Linear ramp of latent heat across [T_minus, T_plus]
            return H_minus + (cls.rho * cls.LH) * (T_local - T_minus) / (2 * epsilon)

        if smoothing in ("linear", "h"):
            H = np.where(
                T < T_minus,
                cls.rho * cls.c_solid * (T - cls.T_a),
                np.where(
                    T > T_plus,
                    H_at_Tm + cls.rho * cls.c_liquid * (T - cls.T_m),
                    smoothed_enthalpy(T),
                ),
            )
        elif smoothing == "erf":
            # Smooth transition between solid sensible and liquid sensible + latent
            eta = 0.5 * (1 + erf((T - cls.T_m) / (np.sqrt(2) * epsilon)))
            H_solid = cls.rho * cls.c_solid * (T - cls.T_a)
            H_liquid = H_at_Tm + cls.rho * cls.c_liquid * (T - cls.T_m)
            H = (1 - eta) * H_solid + eta * H_liquid
        else:
            raise ValueError(
                "Unknown smoothing method: choose 'erf', 'linear', or 'h'."
            )

        return H

    def calcEnthalpy(self, x_arr, t_max, cls):
        num_points = len(x_arr)
        T = np.full(num_points, cls.T_a)  # temperature array
        c = np.array([self.calcSpecificHeat(temp) for temp in T])
        H = c * T  # enthalpy array
        H = np.array(H, dtype=np.float64)
        c = np.array(c, dtype=np.float64)

        t_vals = np.linspace(cls.dt, t_max, num_points)  # Time values

        # Time evolution
        for t in range(len(t_vals) - 1):
            # Build the system of equations for the backward Euler method
            A = np.eye(num_points)
            b = np.copy(H)

            for i in range(1, num_points - 1):  # interior points
                k = self.calcThermalConductivity(T[i])
                A[i, i - 1] = -cls.dt * k / (cls.rho * cls.dx**2)
                A[i, i] += 2 * cls.dt * k / (cls.rho * cls.dx**2)
                A[i, i + 1] = -cls.dt * k / (cls.rho * cls.dx**2)

            # Solve the system of equations
            H_new = np.linalg.solve(A, b)

            # Calculate temperature from enthalpy
            for i in range(len(H_new)):
                if H_new[i] < c[i] * cls.T_m:
                    T[i] = H_new[i] / c[i]
                elif H_new[i] < c[i] * cls.T_m + cls.LH:
                    T[i] = cls.T_m
                else:
                    T[i] = (H_new[i] - cls.LH) / c[i]

            H = H_new

        return H, T

    def calcEnergySufficiency(self, H_vals):
        E_vals = H_vals  # Enthalpy is the energy in kJ

        P_settlement = 50  # Power requirement for the settlement in kW
        time_hours = (
            14.75 * 24
        )  # Half of a lunar day-night cycle (only day or night) in hours

        # Energy needed for the settlement over the adjusted lunar day-night cycle directly in kJ
        E_settlement = P_settlement * time_hours * 3.6e3  # Convert kWh to kJ

        # Total energy stored in the regolith over the adjusted cycle, assuming E_vals is already in kJ
        E_regolith_total = np.sum(
            E_vals
        )  # Directly use the values without further conversion

        # Formulate the result string
        result_string = f"Energy needed: {E_settlement:.0f} kJ, Energy calculated: {E_regolith_total:.0f} kJ. "
        if E_regolith_total >= E_settlement:
            result_string += "The thermal energy in the regolith is sufficient to power the settlement."
        else:
            result_string += "The thermal energy in the regolith is not sufficient to power the settlement."

        return result_string

    def analyticalSol(self, x_val, t_arr, cls):
        T_initial = cls.T_a
        T_final = cls.T_m
        T = np.full((len(x_val), len(t_arr)), T_initial, dtype=np.float64)
        T[0, :] = T_final

        for t_idx, t_val in enumerate(t_arr):
            if t_val > 0:
                alpha2 = self.alpha(cls.k, cls.c, cls.rho)
                x_term = x_val / (2 * np.sqrt(alpha2 * t_val))
                T[:, t_idx] = T_initial + (T_final - T_initial) * (1 - erf(x_term))
            else:
                T[:, t_idx] = T_initial

        return T


class customPCM(PCM):
    def __init__(
        self,
        k: float,
        c: float,
        rho: float,
        T_m: float,
        LH: float,
        ambient_temperature: Optional[float] = None,
        dx: float = 0.1,
    ) -> None:
        self.k = float(k)
        self.c = float(c)
        self.c_solid = float(c)
        self.c_liquid = float(c)
        self.rho = float(rho)
        self.T_m = float(T_m)
        self.T_a = (
            float(ambient_temperature)
            if ambient_temperature is not None
            else max(self.T_m - 50.0, 0.0)
        )
        self.LH = float(LH)
        self.dx = float(dx)
        self.alpha2 = self.alpha(self.k, self.c, self.rho)
        self.dt = float((0.5 * self.dx**2) / self.alpha2)
        self.dt = float(self.enforce_stability_dt(self.dt, safety_limit=0.49))
        self.lmbda = self.dt / (self.dx**2)
        self.cycle_duration = 1.0
        self.heat_source_max = 0.0
        self.x_max = self.dx
        print(f"alpha = {self.alpha2}")

    def calcThermalConductivity(self, temp: FloatArray) -> FloatArray:
        return self.k

    def calcSpecificHeat(self, temp: FloatArray) -> FloatArray:
        return self.c

    def generate_data(self, x_max: float, t_max: float) -> Tuple[np.ndarray, ...]:
        self.x_max = float(x_max)
        x_grid = np.arange(0.0, self.x_max, self.dx, dtype=float)
        if x_grid.size == 0:
            x_grid = np.array([0.0, self.dx], dtype=float)

        t_grid = np.arange(self.dt, float(t_max), self.dt, dtype=float)
        if t_grid.size == 0:
            t_grid = np.array([self.dt], dtype=float)

        X, T = np.meshgrid(x_grid, t_grid, indexing="ij")
        x_features = np.column_stack([X.ravel(), T.ravel()])

        y_T = np.full(x_features.shape[0], self.T_a, dtype=float)
        boundary_mask = x_features[:, 0] == 0.0
        y_T[boundary_mask] = self.T_m + 10.0

        y_B = np.zeros_like(y_T)
        x_boundary = x_features.copy()

        T_arr = np.full((x_grid.size, t_grid.size), self.T_a, dtype=float)
        H_arr = self.initial_enthalpy(x_grid, self, t_grid.size)

        return x_features, y_T, y_B, x_boundary, x_grid, t_grid, T_arr, H_arr

    def explicitNumerical(
        self,
        x_arr: np.ndarray,
        t_arr: np.ndarray,
        T_arr: np.ndarray,
        cls: "PCM",
        phase_mask_array: Optional[np.ndarray] = None,
        boundary_mask_array: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nx = len(x_arr)
        num_timesteps = len(t_arr)

        if phase_mask_array is None:
            phase_mask_array = np.zeros((nx, num_timesteps), dtype=int)
        if boundary_mask_array is None:
            boundary_mask_array = np.zeros((nx, num_timesteps), dtype=int)

        alpha = cls.alpha(self.k, self.c, self.rho)
        moving_boundary_indices = np.full(num_timesteps, -1, dtype=int)

        T_arr[:, 0] = self.T_a
        T_arr[0, :] = self.T_m + 10.0

        tolerance = 100.0
        T_minus = self.T_m - tolerance
        T_plus = self.T_m + tolerance

        for timestep in range(1, num_timesteps):
            T_old = T_arr[:, timestep - 1]
            diffusive_term = np.zeros_like(T_old)
            diffusive_term[1:-1] = T_old[2:] - 2 * T_old[1:-1] + T_old[:-2]

            T_new = T_old + (alpha * self.dt / self.dx**2) * diffusive_term
            T_new[0] = self.T_m + 10.0
            T_new[-1] = T_old[-1]

            T_arr[:, timestep] = T_new
            phase_mask_array[:, timestep] = np.where(
                T_new < T_minus, 0, np.where(T_new <= T_plus, 1, 2)
            )
            boundary_mask_array[:, timestep] = np.where(
                phase_mask_array[:, timestep] == 1, 1, 0
            )

        if np.all(moving_boundary_indices == -1):
            x_features = np.column_stack(
                [np.tile(x_arr, len(t_arr)), np.repeat(t_arr, len(x_arr))]
            )
            moving_boundary_indices = cast(
                np.ndarray,
                self.calculate_boundary_indices(
                    x_features,
                    x_arr[-1],
                    self.dt,
                    T=T_arr,
                    T_m=self.T_m,
                    mode="moving_boundary",
                ),
            )

        return T_arr, phase_mask_array, boundary_mask_array, moving_boundary_indices

    def implicitSol(
        self,
        x_arr: np.ndarray,
        t_arr: np.ndarray,
        T_arr: np.ndarray,
        H_arr: np.ndarray,
        cls: "PCM",
        phase_mask_array: Optional[np.ndarray] = None,
        boundary_mask_array: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_segments = len(x_arr)
        num_timesteps = len(t_arr)

        if phase_mask_array is None:
            phase_mask_array = np.zeros((num_segments, num_timesteps), dtype=int)
        if boundary_mask_array is None:
            boundary_mask_array = np.zeros((num_segments, num_timesteps), dtype=int)

        moving_boundary_indices = np.full(num_timesteps, -1, dtype=int)
        T_arr[:, 0] = self.T_a
        T_arr[0, :] = self.T_m + 10.0

        alpha_const = self.alpha(self.k, self.c, self.rho)
        lmbda_vals = np.full(
            num_segments, (self.dt / (self.dx**2)) * alpha_const, dtype=float
        )

        for time_step in range(1, num_timesteps):
            T_old = T_arr[:, time_step - 1]
            T_explicit = T_old

            try:
                if SCIPY_AVAILABLE:
                    from scipy.sparse import lil_matrix  # type: ignore
                    from scipy.sparse.linalg import splu  # type: ignore

                    A = lil_matrix((num_segments, num_segments))
                    A.setdiag(1 + 2 * lmbda_vals)
                    A.setdiag(-lmbda_vals[1:], -1)
                    A.setdiag(-lmbda_vals[:-1], 1)
                    A[0, :] = 0
                    A[0, 0] = 1
                    A[-1, -2] = -lmbda_vals[-1]
                    A[-1, -1] = 1 + lmbda_vals[-1]
                    A = A.tocsc()
                    lu = splu(A)
                    T_new = lu.solve(T_explicit)
                else:
                    A = np.diag(1 + 2 * lmbda_vals)
                    A += np.diag(-lmbda_vals[1:], k=-1)
                    A += np.diag(-lmbda_vals[:-1], k=1)
                    A[0, :] = 0.0
                    A[0, 0] = 1.0
                    A[-1, -2] = -lmbda_vals[-1]
                    A[-1, -1] = 1 + lmbda_vals[-1]
                    T_new = np.linalg.solve(A, T_explicit)

                H_new = self.calcEnthalpy2(T_new, self)
                T_arr[:, time_step] = T_new
                H_arr[:, time_step] = H_new

                phase_mask, boundary_mask = self.update_phase_mask(T_new, self)
                phase_mask_array[:, time_step] = phase_mask
                boundary_mask_array[:, time_step] = boundary_mask

                moving_boundary = cast(
                    np.ndarray,
                    self.calculate_boundary_indices(
                        x_arr,
                        self.x_max,
                        self.dt,
                        T=T_arr,
                        T_m=self.T_m,
                        mode="moving_boundary",
                        tolerance=100,
                    ),
                )
                moving_boundary_indices[time_step] = moving_boundary[time_step]
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        return (
            T_arr,
            H_arr,
            phase_mask_array,
            boundary_mask_array,
            moving_boundary_indices,
        )

    def heat_source_function(
        self,
        x: Union[float, np.ndarray],
        t: float,
        cycle_duration: Optional[float] = None,
        heat_source_max: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        if isinstance(x, np.ndarray):
            return np.zeros_like(x, dtype=float)
        return 0.0


class Regolith(PCM):
    T_m = 1373  # Melting temperature in Kelvin
    LH = 1429  # Latent heat in J/kg (kept for compatibility; see cp units below)
    rho = 1800  # Density in kg/m³ (corrected units)
    T_a = 253  # Ambient temperature in Kelvin
    cycle_duration = 708  # Lunar cycle duration

    def __init__(self):
        # Use a finer spatial grid to provide smoother phase-front tracking
        self.dx = 0.01  # Spatial step
        self._base_dx = self.dx

        # Calculate specific heats for solid and liquid states
        self.c_solid = float(self.calcSpecificHeat(float(Regolith.T_a)))
        self.c_liquid = float(self.calcSpecificHeat(float(Regolith.T_m)))
        self.c = self.c_liquid

        # Calculate thermal conductivity at melting temperature and diffusivity
        self.k = float(self.calcThermalConductivity(float(Regolith.T_m)))
        self.alpha2 = self.alpha(self.k, self.c, Regolith.rho)

        # Calculate dt using the calculate_dt method and enforce stability
        initial_dt = self.calculate_dt(cls=self, safety_factor=0.4, dt_multiplier=1.0)
        self._update_dt_with_stability(initial_dt)
        self._base_dt = self.dt

        # Set the solar incidence angle
        self.solar_incidence_angle = 45  # degrees

        # Define heat_source_max as an attribute for Regolith
        self.heat_source_max = 100

        print(f"Calculated dt = {self.dt}")
        print(f"alpha = {self.alpha2}")

    def _update_dt_with_stability(self, proposed_dt: float) -> None:
        self.dt = float(self.enforce_stability_dt(float(proposed_dt), safety_limit=0.49))
        self.lmbda = self.dt / (self.dx**2)

    def generate_data(self, x_max, t_max):
        self.x_max = x_max  # Save x_max as an instance attribute
        x_grid = np.arange(0, x_max, self.dx)

        # Ensure the simulated time span captures diffusion both per cell and across the domain.
        Fo_target_cell = 5.0
        Fo_target_domain = 0.05
        alpha_eff = max(self.alpha2, 1e-12)
        t_needed_cell = Fo_target_cell * (self.dx**2) / alpha_eff
        t_needed_domain_raw = Fo_target_domain * (max(x_max, self.dx) ** 2) / alpha_eff
        t_needed_domain = min(t_needed_domain_raw, 20.0 * t_needed_cell)
        t_sim = max(t_max, t_needed_cell, t_needed_domain)
        if t_sim > t_max:
            print(
                "Info: Extending simulation time for diffusion visibility: "
                f"t_sim={t_sim:.3f} s (input t_max={t_max})."
            )

        # Adjust dt if necessary to ensure at least one time step
        if t_sim <= self.dt:
            new_dt = max(t_sim / 50.0, 1e-6) if t_sim > 0 else self.dt
            if new_dt < self.dt:
                print(
                    f"Info: t_sim ({t_sim}) <= dt ({self.dt}). Reducing dt to {new_dt} for data generation."
                )
                self._update_dt_with_stability(new_dt)

        # Ensure a minimum number of time steps for smoother plots
        min_steps = 400
        if t_sim > 0 and (t_sim / self.dt) < min_steps:
            refined_dt = float(t_sim / min_steps)
            print(
                f"Info: Increasing time resolution to achieve ~{min_steps} steps: dt={refined_dt:.6g}"
            )
            self._update_dt_with_stability(refined_dt)
        else:
            self._update_dt_with_stability(self.dt)

        # Build time grid (include endpoint tolerance to avoid empty due to float rounding)
        t_grid = np.arange(self.dt, t_sim + 1e-12, self.dt)
        if len(t_grid) == 0:
            # Fallback: ensure at least one time step
            t_grid = np.array([self.dt], dtype=float)

        X, T = np.meshgrid(x_grid, t_grid, indexing="ij")
        x_features = np.column_stack([X.ravel(), T.ravel()])

        # Initialize temperature (y_T) and boundary (y_B) arrays
        y_T = np.full(
            x_features.shape[0], self.T_a, dtype=np.float64
        )  # Ambient temperature for all
        y_B = np.zeros_like(y_T, dtype=np.float64)  # Initialize boundary array as zeros
        # Provide the same (x, t) feature grid to the boundary sub-network so it can learn time-dependent boundaries.
        x_boundary = x_features.copy()

        # Set initial boundary condition at x=0 (e.g., higher temperature at the boundary)
        boundary_condition_indices = x_features[:, 0] == 0
        y_T[boundary_condition_indices] = self.T_m + 10.0  # Temperature at the boundary
        y_B[boundary_condition_indices] = 1.0  # Mark boundary location

        # Use provided methods to initialize temperature and enthalpy arrays
        T_arr, H_arr = self.initialize_enthalpy_temperature_arrays(
            x_grid, self, len(t_grid)
        )

        # Debugging prints to visualize initial conditions
        print(f"Initial T_arr shape: {T_arr.shape}")
        print(f"Initial H_arr shape: {H_arr.shape}")
        print(
            f"Initial boundary condition indices: {np.sum(boundary_condition_indices)}"
        )

        return x_features, y_T, y_B, x_boundary, x_grid, t_grid, T_arr, H_arr

    def explicitNumerical(
        self, x_arr, t_arr, T_arr, cls, phase_mask_array=None, boundary_mask_array=None
    ):
        nx = len(x_arr)
        num_timesteps = len(t_arr)

        if phase_mask_array is None or boundary_mask_array is None:
            phase_mask_array = np.zeros((nx, num_timesteps), dtype=int)
            boundary_mask_array = np.zeros((nx, num_timesteps), dtype=int)

        alpha = cls.alpha(cls.k, cls.c, cls.rho)
        max_temperature = 4000.0  # Maximum temperature allowed in the system
        moving_boundary_indices = np.full(num_timesteps, -1, dtype=int)

        # Initial conditions
        T_arr[:, 0] = cls.T_a
        T_arr[0, :] = cls.T_m + 10.0

        for timestep in range(1, num_timesteps):
            T_old = T_arr[:, timestep - 1]

            # Compute diffusive term with proper boundary handling
            diffusive_term = np.zeros_like(T_old)
            diffusive_term[1:-1] = T_old[2:] - 2 * T_old[1:-1] + T_old[:-2]
            # Apply boundary conditions
            diffusive_term[0] = 0
            diffusive_term[-1] = 0

            T_new = T_old + (alpha * cls.dt / cls.dx**2) * diffusive_term

            # Keep boundary at fixed temperature; no additional heat source during comparison
            # This ensures consistency with the analytical solution (pure diffusion)
            # heat_source disabled for parity

            T_new[-1] = T_old[-1]  # Keep the last cell's temperature constant

            # Clip temperatures to prevent numerical issues
            T_new = np.clip(T_new, cls.T_a, max_temperature)

            T_arr[:, timestep] = T_new

            # Update phase and boundary masks
            phase_mask, boundary_mask = self.update_phase_mask(T_new, cls)
            phase_mask_array[:, timestep] = phase_mask
            boundary_mask_array[:, timestep] = boundary_mask

            # Update moving boundary indices
            moving_boundary = cast(
                np.ndarray,
                self.calculate_boundary_indices(
                    x=x_arr,
                    x_max=x_arr[-1],
                    dt=cls.dt,
                    T=T_arr,
                    T_m=cls.T_m,
                    mode="moving_boundary",
                    tolerance=100,
                ),
            )
            moving_boundary_indices[timestep] = moving_boundary[timestep]

        return T_arr, phase_mask_array, boundary_mask_array, moving_boundary_indices

    def implicitSol(
        self,
        x_arr,
        t_arr,
        T_arr,
        H_arr,
        cls,
        phase_mask_array=None,
        boundary_mask_array=None,
    ):
        num_segments = len(x_arr)
        num_timesteps = len(t_arr)

        if phase_mask_array is None:
            phase_mask_array = np.zeros((num_segments, num_timesteps), dtype=int)
        if boundary_mask_array is None:
            boundary_mask_array = np.zeros((num_segments, num_timesteps), dtype=int)

        moving_boundary_indices = np.full(num_timesteps, -1, dtype=int)

        # Initial condition: Set the first time step
        T_arr[:, 0] = cls.T_a  # Ambient temperature for all spatial cells
        T_arr[0, :] = cls.T_m + 10.0  # Higher temperature at the first spatial cell

        for time_step in range(1, num_timesteps):
            T_old = T_arr[:, time_step - 1]

            # No external heat source to keep consistency with the analytical solution
            # Use pure diffusion implicit step (T_explicit equals previous temperature)
            T_explicit = T_old

            # Use constant thermal properties to maintain parity with analytical/explicit solutions
            alpha_const = cls.alpha(cls.k, cls.c, cls.rho)
            lmbda_vals = np.full(
                num_segments, (cls.dt / (cls.dx**2)) * alpha_const, dtype=float
            )

            # Construct tridiagonal matrix A for the implicit diffusion term
            try:
                if SCIPY_AVAILABLE:
                    A = lil_matrix((num_segments, num_segments))
                    A.setdiag(1 + 2 * lmbda_vals)
                    A.setdiag(-lmbda_vals[1:], -1)
                    A.setdiag(-lmbda_vals[:-1], 1)

                    # Apply boundary conditions
                    A[0, :] = 0
                    A[0, 0] = 1

                    # Do not fix the temperature at the rightmost cell, allowing it to evolve
                    A[-1, -2] = -lmbda_vals[-1]  # Adjust the last cell in the matrix A
                    A[-1, -1] = 1 + lmbda_vals[-1]

                    # Convert A to CSC format for efficient solving
                    A = A.tocsc()

                    # Solve the system using LU decomposition
                    lu = splu(A)
                    T_new = lu.solve(T_explicit)
                else:
                    # Dense NumPy fallback when SciPy is unavailable
                    A = np.diag(1 + 2 * lmbda_vals)
                    A += np.diag(-lmbda_vals[1:], k=-1)
                    A += np.diag(-lmbda_vals[:-1], k=1)

                    # Apply boundary conditions
                    A[0, :] = 0.0
                    A[0, 0] = 1.0
                    A[-1, -2] = -lmbda_vals[-1]
                    A[-1, -1] = 1 + lmbda_vals[-1]

                    T_new = np.linalg.solve(A, T_explicit)

                # Calculate new enthalpy from updated temperature
                H_new = cls.calcEnthalpy2(T_new, cls)
                T_arr[:, time_step] = T_new
                H_arr[:, time_step] = H_new

                # Compute masks
                phase_mask, boundary_mask = self.update_phase_mask(T_new, cls)
                phase_mask_array[:, time_step] = phase_mask
                boundary_mask_array[:, time_step] = boundary_mask

                moving_boundary_array = cast(
                    np.ndarray,
                    self.calculate_boundary_indices(
                        x_arr,
                        self.x_max,
                        cls.dt,
                        T=T_arr,
                        T_m=cls.T_m,
                        mode="moving_boundary",
                        tolerance=100,
                    ),
                )
                moving_boundary_indices[time_step] = moving_boundary_array[time_step]
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        return (
            T_arr,
            H_arr,
            phase_mask_array,
            boundary_mask_array,
            moving_boundary_indices,
        )

    # def calcThermalConductivity(self, temp):
    #     C1 = 1.281e-2
    #     C2 = 4.431e-4
    #     epsilon = 1e-6  # Small value to avoid zero division
    #
    #     if isinstance(temp, (list, np.ndarray)):
    #         k_granular = C1 + C2 * (np.array(temp, dtype=np.float64) + epsilon) ** (-3)
    #     else:
    #         k_granular = C1 + C2 * (float(temp) + epsilon) ** (-3)
    #
    #     k_molten_end = 2.5  # Thermal conductivity for molten regolith
    #
    #     if isinstance(temp, (list, np.ndarray)):
    #         k_final = np.where(temp < self.T_m, k_granular, k_molten_end)
    #     else:
    #         k_final = k_granular if temp < self.T_m else k_molten_end
    #
    #     return 1000.0 * k_final

    # def calcSpecificHeat(self, temp):
    #     specific_heat = -1848.5 + 1047.41 * np.log(temp)
    #     return specific_heat

    def calcThermalConductivity(self, temp: FloatArray) -> FloatArray:
        k_solid = 0.01  # W/m·K for solid regolith
        k_molten = 2.5  # W/m·K for molten regolith

        if isinstance(temp, (list, np.ndarray)):
            k_final = np.where(temp < self.T_m, k_solid, k_molten)
        else:
            k_final = k_solid if temp < self.T_m else k_molten

        return k_final

    def calcSpecificHeat(self, temp: FloatArray) -> FloatArray:
        # Specific heat capacity in J/(kg·K). Using realistic values for lunar regolith.
        c_solid = 800.0  # Solid regolith
        c_molten = 1200.0  # Molten regolith

        if isinstance(temp, (list, np.ndarray)):
            c_final = np.where(temp < self.T_m, c_solid, c_molten)
        else:
            c_final = c_solid if temp < self.T_m else c_molten

        return c_final

    def heat_source_function(
        self,
        x: Union[float, np.ndarray],
        t: float,
        cycle_duration: Optional[float] = None,
        heat_source_max: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        cd = (
            float(cycle_duration)
            if cycle_duration is not None
            else float(getattr(self, "cycle_duration", 1.0))
        )
        hs_max = (
            float(heat_source_max)
            if heat_source_max is not None
            else float(getattr(self, "heat_source_max", 0.0))
        )
        tube_radius = 0.1  # radius of the tube

        solar_constant = 1361  # W/m^2
        solar_flux = solar_constant * np.cos(
            np.deg2rad(self.solar_incidence_angle)
        )  # Adjusted for incidence angle

        # Simulate cyclic heat source variation over the lunar cycle
        cyclic_variation = (1 + np.sin(2 * np.pi * t / cd)) / 2

        # Calculate the heat flux from the tube with a sinusoidal time dependency
        distance_from_tube = np.abs(x)
        heat_flux_tube = (
            hs_max * cyclic_variation * np.exp(-distance_from_tube / tube_radius)
        )

        # Total heat flux is the sum of the tube heat flux and the solar flux
        total_heat_flux = heat_flux_tube + solar_flux

        return total_heat_flux


class Iron(PCM):
    T_m, LH, rho, T_a = 1810, 247000, 7870, 293
    c_solid = 0.449  # specific heat of solid iron in J/g°C
    c_liquid = 0.82  # specific heat of liquid iron in J/g°C
    cycle_duration = 86400  # seconds for a day
    heat_source_max = 1000  # example value in W/m^3

    def __init__(self):
        self.dx = 0.05  # Spatial discretization
        self.k = float(self.calcThermalConductivity(Iron.T_a))  # Thermal conductivity
        self.c = float(self.calcSpecificHeat(Iron.T_a))  # Specific heat
        self.alpha2 = self.alpha(self.k, self.c, Iron.rho)  # Thermal diffusivity

        # Stability condition for dt calculation
        self.dt = float((0.5 * self.dx**2) / self.alpha2)
        self.dt = float(self.enforce_stability_dt(self.dt, safety_limit=0.49))

        print("dt = ", self.dt)
        print(f"alpha = {self.alpha2}")

        self.lmbda = self.dt / (self.dx**2)  # Used for numerical methods

    def generate_data(self, x_max, t_max):
        self.x_max = x_max  # Save x_max as an instance attribute
        x_grid = np.arange(0, x_max, self.dx)
        t_grid = np.arange(self.dt, t_max, self.dt)  # Exclude the final time step
        # print(f"Debug from generate_data: x_grid shape = {x_grid.shape}")
        # print(f"Debug from generate_data: t_grid shape = {t_grid.shape}")

        X, T = np.meshgrid(x_grid, t_grid, indexing="ij")
        x_features = np.column_stack([X.ravel(), T.ravel()])

        y_T = np.full(x_features.shape[0], self.T_a, dtype=np.float64)
        boundary_condition_indices = x_features[:, 0] == 0
        y_T[boundary_condition_indices] = self.T_m + 10.0

        y_B = y_T.copy()
        x_boundary = x_features.copy()

        T_arr = np.full((len(x_grid), len(t_grid)), self.T_a)
        H_arr = self.initial_enthalpy(x_grid, self, len(t_grid))

        # print(f"Debug: x_features shape = {x_features.shape}")
        # print(f"Debug: y_T shape = {y_T.shape}, y_B shape = {y_B.shape}")

        return x_features, y_T, y_B, x_boundary, x_grid, t_grid, T_arr, H_arr

    def explicitNumerical(
        self, x_arr, t_arr, T_arr, cls, phase_mask_array=None, boundary_mask_array=None
    ):
        nx = len(x_arr)
        num_timesteps = len(t_arr)

        if phase_mask_array is None or boundary_mask_array is None:
            phase_mask_array = np.zeros((nx, num_timesteps), dtype=int)
            boundary_mask_array = np.zeros((nx, num_timesteps), dtype=int)

        alpha = cls.alpha(cls.k, cls.c, cls.rho)
        moving_boundary_indices = np.full(
            num_timesteps, -1, dtype=int
        )  # Initialize moving_boundary_indices

        T_minus = cls.T_m - 100  # Assuming the default tolerance of 100
        T_plus = cls.T_m + 100

        for timestep in range(1, num_timesteps):
            T_old = T_arr[:, timestep - 1]
            diffusive_term = np.roll(T_old, -1) - 2 * T_old + np.roll(T_old, 1)
            T_new = T_old + (alpha * cls.dt / cls.dx**2) * diffusive_term

            T_new[0] = cls.T_m + 10.0  # Fix the left boundary to T_m + 10.0
            T_new[-1] = T_old[-1]  # Maintain the last cell's temperature

            T_arr[:, timestep] = T_new

            # Directly compute the phase mask and boundary mask in the loop
            phase_mask_array[:, timestep] = np.where(
                T_new < T_minus, 0, np.where(T_new <= T_plus, 1, 2)
            )
            boundary_mask_array[:, timestep] = np.where(
                phase_mask_array[:, timestep] == 1, 1, 0
            )

        if np.all(
            moving_boundary_indices == -1
        ):  # Check if moving_boundary_indices was never updated
            x_features = np.column_stack(
                [np.tile(x_arr, len(t_arr)), np.repeat(t_arr, len(x_arr))]
            )
            moving_boundary_indices = cast(
                np.ndarray,
                self.calculate_boundary_indices(
                    x_features,
                    x_arr[-1],
                    cls.dt,
                    T=T_arr,
                    T_m=cls.T_m,
                    mode="moving_boundary",
                ),
            )

        return T_arr, phase_mask_array, boundary_mask_array, moving_boundary_indices

    def implicitSol(
        self,
        x_arr,
        t_arr,
        T_arr,
        H_arr,
        cls,
        phase_mask_array=None,
        boundary_mask_array=None,
    ):
        num_segments = len(x_arr)
        num_timesteps = len(t_arr)

        if phase_mask_array is None:
            phase_mask_array = np.zeros((num_segments, num_timesteps), dtype=int)
        if boundary_mask_array is None:
            boundary_mask_array = np.zeros((num_segments, num_timesteps), dtype=int)

        moving_boundary_indices = np.full(num_timesteps, -1, dtype=int)

        # Initial condition: Set the first time step
        T_arr[:, 0] = cls.T_a  # Ambient temperature for all spatial cells
        T_arr[0, :] = cls.T_m + 10.0  # Higher temperature at the first spatial cell

        for time_step in range(1, num_timesteps):
            T_old = T_arr[:, time_step - 1]

            # Heat source function specific to iron
            heat_source = cls.heat_source_function(x_arr, t_arr[time_step])

            # Explicit update for the nonlinear heat source term
            T_explicit = T_old + cls.dt * heat_source / (cls.rho * cls.c)

            # Calculate thermal properties
            k_vals = np.array(
                [cls.calcThermalConductivity(T_old[i]) for i in range(num_segments)]
            )
            c_vals = np.array(
                [cls.calcSpecificHeat(T_old[i]) for i in range(num_segments)]
            )
            rho = cls.rho
            alpha_vals = k_vals / (c_vals * rho)
            lmbda_vals = cls.dt / (cls.dx**2) * alpha_vals

            # Construct tridiagonal matrix A for the implicit diffusion term
            try:
                if SCIPY_AVAILABLE:
                    A = lil_matrix((num_segments, num_segments))
                    A.setdiag(1 + 2 * lmbda_vals)
                    A.setdiag(-lmbda_vals[1:], -1)
                    A.setdiag(-lmbda_vals[:-1], 1)

                    # Apply boundary conditions
                    A[0, :] = 0
                    A[0, 0] = 1

                    # Do not fix the temperature at the rightmost cell, allowing it to evolve
                    A[-1, -2] = -lmbda_vals[-1]  # Adjust the last cell in the matrix A
                    A[-1, -1] = 1 + lmbda_vals[-1]

                    # Convert A to CSC format for efficient solving
                    A = A.tocsc()

                    # Solve the system using LU decomposition
                    lu = splu(A)
                    T_new = lu.solve(T_explicit)
                else:
                    # Dense NumPy fallback when SciPy is unavailable
                    A = np.diag(1 + 2 * lmbda_vals)
                    A += np.diag(-lmbda_vals[1:], k=-1)
                    A += np.diag(-lmbda_vals[:-1], k=1)

                    # Apply boundary conditions
                    A[0, :] = 0.0
                    A[0, 0] = 1.0
                    A[-1, -2] = -lmbda_vals[-1]
                    A[-1, -1] = 1 + lmbda_vals[-1]

                    T_new = np.linalg.solve(A, T_explicit)

                # Calculate new enthalpy from updated temperature
                H_new = cls.calcEnthalpy2(T_new, cls)
                T_arr[:, time_step] = T_new
                H_arr[:, time_step] = H_new

                # Compute masks
                phase_mask, boundary_mask = self.update_phase_mask(T_new, cls)
                phase_mask_array[:, time_step] = phase_mask
                boundary_mask_array[:, time_step] = boundary_mask

                moving_boundary_array = cast(
                    np.ndarray,
                    self.calculate_boundary_indices(
                        x_arr,
                        self.x_max,
                        cls.dt,
                        T=T_arr,
                        T_m=cls.T_m,
                        mode="moving_boundary",
                        tolerance=100,
                    ),
                )
                moving_boundary_indices[time_step] = moving_boundary_array[time_step]
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        return (
            T_arr,
            H_arr,
            phase_mask_array,
            boundary_mask_array,
            moving_boundary_indices,
        )

    def calcThermalConductivity(self, temp: FloatArray) -> FloatArray:
        """Calculate the thermal conductivity of iron based on its phase (solid or liquid).

        Args:
            temp (float): Temperature in degrees Celsius.

        Returns:
            float: Thermal conductivity in W/m·K.
        """
        if temp < self.T_m:
            return 73  # Thermal conductivity for solid iron in W/m·K
        else:
            return 35  # Thermal conductivity for liquid iron in W/m·K

    def calcSpecificHeat(self, temp: FloatArray) -> FloatArray:
        return np.where(temp < Iron.T_m, Iron.c_solid, Iron.c_liquid)

    def heat_source_function(self, x, t):
        half_cycle = self.cycle_duration / 2
        if 0 <= t % self.cycle_duration < half_cycle:
            return self.heat_source_max
        else:
            return 0
