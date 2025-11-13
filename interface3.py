import math
import os

# NumPy compatibility shims (e.g., np.bool8 on NumPy 2)
import np_compat  # noqa: F401

import sys
import warnings
from typing import Dict, List, Optional, Tuple, Type, cast

from np_compat import check_numpy_health, friendly_numpy_error


import tkinter as tk

# import google
# import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
from tkinter import messagebox
import backend

# Map selectable material names to their backend implementations.
MATERIAL_CLASSES: Dict[str, Type[backend.PCM]] = {
    "Regolith": backend.Regolith,
    "Iron": backend.Iron,
}

PINN_TRAINING_EPOCHS = 50
PINN_TARGET_BATCHES_PER_EPOCH = 32
PINN_MAX_TRAIN_SAMPLES = 20000
BOUNDARY_HEAD_VIOLATION_THRESHOLD = 0.05

# Work around protobuf >=3.19 descriptor change by forcing Python implementation before importing TensorFlow/Keras
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
np.set_printoptions(precision=8, suppress=True)

# Silence noisy 'NumPy module was reloaded' warning sometimes emitted by IDE helpers
warnings.filterwarnings(
    "ignore", message="The NumPy module was reloaded", category=UserWarning
)


_ok, _npmsg = check_numpy_health()
if not _ok:
    try:
        import tkinter as _tk
        from tkinter import messagebox as _messagebox

        _root = _tk.Tk()
        _root.withdraw()
        _messagebox.showerror("NumPy import error", friendly_numpy_error(_npmsg))
    except Exception:
        print(friendly_numpy_error(_npmsg), file=sys.stderr)
    sys.exit(1)


class chooseInput(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Input Window")
        self.label = tk.Label(self, text="Choose mode:")
        self.label.grid(row=0, column=0)
        self.v = tk.IntVar()
        self.v.set(0)
        self.choices = [
            ("Input thermo-physical properties", 0),
            ("Choose between materials", 1),
        ]
        for string, val in self.choices:
            tk.Radiobutton(self, text=string, variable=self.v, value=val).grid(
                row=val + 1, column=0
            )

        # Create submit button
        submit_button = tk.Button(self, text="Submit", command=self.callInputMode)
        submit_button.grid(row=3, column=0, columnspan=2)

    def callInputMode(self):
        self.destroy()


class InputWindow0(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Input Window")
        # Create input fields
        x_label = tk.Label(self, text="Length of the bar:")
        x_label.grid(row=0, column=0)
        self.x_entry = tk.Entry(self)
        self.x_entry.grid(row=0, column=1)

        t_label = tk.Label(self, text="Final time:")
        t_label.grid(row=1, column=0)
        self.t_entry = tk.Entry(self)
        self.t_entry.grid(row=1, column=1)

        k_label = tk.Label(self, text="Conductivity:")
        k_label.grid(row=2, column=0)
        self.k_entry = tk.Entry(self)
        self.k_entry.grid(row=2, column=1)

        c_label = tk.Label(self, text="Specific Heat:")
        c_label.grid(row=3, column=0)
        self.c_entry = tk.Entry(self)
        self.c_entry.grid(row=3, column=1)

        rho_label = tk.Label(self, text="Density:")
        rho_label.grid(row=4, column=0)
        self.rho_entry = tk.Entry(self)
        self.rho_entry.grid(row=4, column=1)

        Tm_label = tk.Label(self, text="Melting Temperature:")
        Tm_label.grid(row=5, column=0)
        self.Tm_entry = tk.Entry(self)
        self.Tm_entry.grid(row=5, column=1)

        LH_label = tk.Label(self, text="Latent Heat:")
        LH_label.grid(row=6, column=0)
        self.LH_entry = tk.Entry(self)
        self.LH_entry.grid(row=6, column=1)
        # Create submit button
        submit_button = tk.Button(self, text="Submit", command=self.submit)
        submit_button.grid(row=7, column=1, columnspan=2)

    def submit(self):
        self.x_input = self.x_entry.get()
        self.t_input = self.t_entry.get()

        self.k_input = self.k_entry.get()
        self.c_input = self.c_entry.get()
        self.rho_input = self.rho_entry.get()

        self.Tm_input = self.Tm_entry.get()
        self.LH_input = self.LH_entry.get()
        x = float(self.x_input)
        t = float(self.t_input)
        k = float(self.k_input)
        c = float(self.c_input)
        rho = float(self.rho_input)
        Tm = float(self.Tm_input)
        LH = float(self.LH_input)

        # Close the window and return the input values
        self.quit()
        return x, t, k, c, rho, Tm, LH


class InputWindow1(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Input Window")
        # Create input fields
        x_label = tk.Label(self, text="Length of the bar:")
        x_label.grid(row=0, column=0)
        self.x_entry = tk.Entry(self)
        self.x_entry.grid(row=0, column=1)

        t_label = tk.Label(self, text="Final time:")
        t_label.grid(row=1, column=0)
        self.t_entry = tk.Entry(self)
        self.t_entry.grid(row=1, column=1)

        # Create material selection dropdown
        material_label = tk.Label(self, text="Material:")
        material_label.grid(row=2, column=0)
        self.material_var = tk.StringVar()
        self.material_var.set("Regolith")  # Set default material to iron
        material_options = ["Regolith", "Iron"]
        self.material_dropdown = tk.OptionMenu(
            self, self.material_var, *material_options
        )
        self.material_dropdown.grid(row=2, column=1)

        # Create submit button
        submit_button = tk.Button(self, text="Submit", command=self.submit)
        submit_button.grid(row=3, column=1, columnspan=2)

    def submit(self):
        self.x_input = self.x_entry.get()
        self.t_input = self.t_entry.get()
        self.material_input = self.material_var.get()
        x = float(self.x_input)
        t = float(self.t_input)

        # Close the window and return the input values
        self.quit()
        return x, t, self.material_input


class App:
    def __init__(self):
        self.choiceWindow = chooseInput()
        self.choiceWindow.wait_window()
        choice = self.choiceWindow.v.get()
        self.pcm: backend.PCM

        if choice == 0:
            self.input_window = InputWindow0()
            self.input_window.mainloop()
            self.L, self.t_max, self.k, self.c, self.rho, self.T_m, self.LH = (
                self.input_window.submit()
            )
            self.input_window.destroy()
            self.material = "Custom"
            self.pcm = backend.customPCM(self.k, self.c, self.rho, self.T_m, self.LH)
        else:
            self.input_window = InputWindow1()
            self.input_window.mainloop()
            self.L, self.t_max, self.material = self.input_window.submit()
            self.input_window.destroy()
            material_cls = MATERIAL_CLASSES.get(self.material)
            if material_cls is None:
                raise ValueError(f"Unsupported material: {self.material}")
            self.pcm = material_cls()

        print(f"Debug: Selected material = {self.material if choice else 'Custom'}")

        # Create the main window
        self.root = tk.Tk()
        self.root.wm_title("Embedding in Tk")
        self.root.configure(bg="white")
        self.root.title("Heat Conduction")
        self.root.grid_anchor("center")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Create a frame to hold the scale and the line plot
        self.frame = tk.Frame(self.root)
        self.frame.grid(
            row=0, column=0, columnspan=3, sticky="n"
        )  # Span the frame across columns

        # Create the scale widget
        self.time_scale = tk.Scale(
            self.root,
            from_=0,
            to=self.t_max,
            resolution=self.pcm.dt,
            orient=tk.HORIZONTAL,
            command=self.update_time,
        )
        self.time_scale.grid(row=0, column=0, sticky="nsew")

        # Create a variable for the dropdown selection
        self.solution_type = tk.StringVar(self.root)
        self.solution_type.set("Analytical")  # default value

        # Create the dropdown menu for the main solution type selection
        solution_menu = tk.OptionMenu(
            self.frame,
            self.solution_type,
            "Analytical",
            "Implicit",
            "Numerical",
            "PINN",
        )
        solution_menu.grid(row=0, column=1, sticky="nsew")
        self.solution_type.trace("w", self.handle_solution_type_change)
        # Initialize gold standard variables
        self.gold_standard = None
        self.gold_standard_temp_array: Optional[np.ndarray] = None
        self.root.focus_force()
        self.moving_boundary_indices: Dict[str, Optional[np.ndarray]] = {}
        self.temp_mask_arrays: Dict[str, Optional[np.ndarray]] = {
            "Numerical": None,
            "Analytical": None,
            "Implicit": None,
        }
        self.bound_mask_arrays: Dict[str, Optional[np.ndarray]] = {
            "Numerical": None,
            "Analytical": None,
            "Implicit": None,
        }
        self.moving_boundary_positions: Dict[str, Optional[np.ndarray]] = {
            "Numerical": None,
            "Analytical": None,
            "Implicit": None,
        }
        self.moving_boundary_positions: Dict[str, Optional[np.ndarray]] = {
            "Numerical": None,
            "Analytical": None,
            "Implicit": None,
        }
        self.temperature_solutions: Dict[str, Optional[np.ndarray]] = {
            "Numerical": None,
            "Analytical": None,
            "Implicit": None,
        }
        self.temp_mask_array: Optional[np.ndarray] = None
        self.bound_mask_array: Optional[np.ndarray] = None
        self.x_input_full: Optional[np.ndarray] = None
        self.x_boundary_input_full: Optional[np.ndarray] = None
        self.T_arr_to_display: Optional[np.ndarray] = None
        self.moving_boundary_indices_to_display: Optional[np.ndarray] = None
        self.T_arr_numerical: Optional[np.ndarray] = None
        self.T_arr_analytical: Optional[np.ndarray] = None
        self.T_arr_implicit: Optional[np.ndarray] = None
        self.H_arr_final: Optional[np.ndarray] = None
        self.boundary_indices: Optional[np.ndarray] = None
        self.moving_boundary_locations: Optional[np.ndarray] = None
        self.indices: List[int] = []
        self.raw_loss_values: List[float] = []
        self.gold_standard_boundary_per_time: Optional[np.ndarray] = None

        # Create the 2D line plot
        self.fig1, self.ax1 = plt.subplots(figsize=(4, 3), dpi=100)
        self.ax1.set_title("T(E) Line Plot")
        self.ax1.set_xlabel("E")
        self.ax1.set_ylabel("T")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.root)
        self.canvas1.draw()
        self.fig2 = Figure(figsize=(4, 3), dpi=100)
        self.ax2 = cast(Axes3D, self.fig2.add_subplot(111, projection="3d"))
        self.ax2.set_title("3D Surface Plot")
        self.ax2.set_xlabel("x")
        self.ax2.set_ylabel("t")
        self.ax2.set_zlabel("T")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.root)

        self.fig3 = Figure(figsize=(5, 5), dpi=100)
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.grid(True)
        self.ax3.set_xlabel("x [m]")
        self.ax3.set_ylabel("T [C]")
        self.ax3.set_title("T(x)")
        self.canvas3 = FigureCanvasTkAgg(
            self.fig3, master=self.root
        )  # A tk.DrawingArea.
        self.canvas3.draw()
        self.canvas3.get_tk_widget().grid(row=1, column=1, sticky="nsew")
        (self.line1,) = self.ax1.plot(
            [], [], "r-", label="Line Plot 1"
        )  # Notice the comma
        (self.line3,) = self.ax3.plot(
            [], [], "g-", label="Line Plot 3"
        )  # Notice the comma

        self.ax3.legend()

        # Configure the frame to give all extra space to the line plot
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        self.canvas1.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.canvas2.get_tk_widget().grid(row=1, column=2, sticky="nsew")

        # Add this to the App's __init__ method to create the label
        self.energy_label = tk.Label(
            self.root, text="", bg="white", wraplength=500, justify="center"
        )
        self.energy_label.grid(row=2, column=0, columnspan=3, pady=10)

        # Configure the frame to give all extra space to the line plot
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, weight=1)

        # Configure the main window to distribute extra space among the widgets
        self.root.grid_rowconfigure(0, weight=1)  # Row for the time scale
        self.root.grid_rowconfigure(1, weight=3)  # Row for the plots
        self.root.grid_rowconfigure(2, weight=1)  # Row for the energy sufficiency label

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        self.fig_PINN = Figure(figsize=(15, 5), dpi=100)
        self.moving_boundary_locations = None
        self.indices = []

        self.calcAll()
        # Sync the time slider with the actual simulated time span (may differ from user input if adjusted internally)
        try:
            if hasattr(self, "t_grid") and len(self.t_grid) > 0:
                self.time_scale.config(
                    to=float(self.t_grid[-1]), resolution=float(self.pcm.dt)
                )
        except Exception as e:
            print(f"Warning: Failed to sync time slider with simulation time grid: {e}")
        self.update_solution_type()
        self.update_plots()

    # Call the energy calculation method and update the label with the result string
    def update_energy_label(self):
        # Example: Assuming self.H_arr_final contains the enthalpy values needed for the energy calculation
        if self.H_arr_final is not None:
            result_string = self.pcm.calcEnergySufficiency(self.H_arr_final)
            self.energy_label.config(text=result_string)
        else:
            self.energy_label.config(
                text="No enthalpy data available for energy calculation."
            )

    def handle_gold_standard_selection(self, selected_solution):
        self.gold_standard = selected_solution
        self.set_gold_standard_temp_array(self.gold_standard)

        if self.gold_standard_temp_array is None:
            print(
                f"Error: Selected solution '{selected_solution}' does not have a valid temperature array."
            )
            return

        self.prepare_PINN_model_and_train()

    def set_gold_standard_temp_array(self, selected_solution):
        self.gold_standard_temp_array = self.temperature_solutions.get(
            selected_solution
        )
        if self.gold_standard_temp_array is None:
            print(
                f"Error: No temperature array found for the selected solution '{selected_solution}'."
            )
            return

        self.temp_mask_array = self.temp_mask_arrays.get(selected_solution)
        self.bound_mask_array = self.bound_mask_arrays.get(selected_solution)
        self.moving_boundary_indices_to_display = self.moving_boundary_indices.get(
            selected_solution
        )

    def prompt_for_gold_standard(self):
        self.gold_standard_window = GoldStandardSelectionWindow(
            self.root, self.handle_gold_standard_selection
        )
        self.gold_standard_window.mainloop()

    def handle_solution_type_change(self, *args):
        selected_solution = self.solution_type.get()
        if selected_solution == "PINN":
            # Prompt the user to select the gold standard solution type
            self.prompt_for_gold_standard()
        else:
            # Handle other solution types normally
            self.update_solution_type()
            self.update_plots()

    def calcAll(self):
        print("Debug: Starting calcAll")

        # Initialize attributes
        self.moving_boundary_indices: Dict[str, Optional[np.ndarray]] = {}
        self.temp_mask_arrays: Dict[str, Optional[np.ndarray]] = {
            "Numerical": None,
            "Analytical": None,
            "Implicit": None,
        }
        self.bound_mask_arrays: Dict[str, Optional[np.ndarray]] = {
            "Numerical": None,
            "Analytical": None,
            "Implicit": None,
        }
        self.temperature_solutions: Dict[str, Optional[np.ndarray]] = {
            "Numerical": None,
            "Analytical": None,
            "Implicit": None,
        }
        self.temp_mask_array: Optional[np.ndarray] = None
        self.bound_mask_array: Optional[np.ndarray] = None
        self.T_arr_to_display: Optional[np.ndarray] = None
        self.moving_boundary_indices_to_display: Optional[np.ndarray] = None
        self.T_arr_numerical: Optional[np.ndarray] = None
        self.T_arr_analytical: Optional[np.ndarray] = None
        self.T_arr_implicit: Optional[np.ndarray] = None
        self.H_arr_final: Optional[np.ndarray] = None

        # Generate data
        print("Debug: Generating data")
        (
            self.x,
            self.y_T,
            self.y_B,
            self.x_boundary,
            self.x_grid,
            self.t_grid,
            self.T_arr,
            self.H_arr,
        ) = self.pcm.generate_data(self.L, self.t_max)
        nx = len(self.x_grid)
        nt = len(self.t_grid)

        print(f"Debug: nx = {nx}, nt = {nt}")

        self.initial_data_T = (self.x, self.y_T)
        self.initial_data_B = (self.x_boundary, self.y_B)

        # Numerical solution
        (
            self.T_arr_numerical,
            _numerical_phase_mask_series,
            _numerical_boundary_mask_series,
            moving_boundary_indices_numerical,
        ) = self.pcm.explicitNumerical(
            self.x_grid, self.t_grid, self.T_arr.copy(), self.pcm
        )

        phase_mask_final, boundary_mask_final = self.pcm.update_phase_mask(
            self.T_arr_numerical[:, -1], self.pcm
        )
        self.temp_mask_arrays["Numerical"] = phase_mask_final
        self.bound_mask_arrays["Numerical"] = boundary_mask_final

        # Check for empty mask arrays
        if np.all(phase_mask_final == 0) or np.all(boundary_mask_final == 0):
            print(
                "Warning: Numerical solution mask arrays are empty. Check the temperature evolution and masking logic."
            )

        self.temperature_solutions["Numerical"] = self.T_arr_numerical

        # Analytical solution
        print("Debug: Calculating analytical solution")
        self.T_arr_analytical = self.pcm.analyticalSol(
            self.x_grid, self.t_grid, self.pcm
        )
        analytical_phase_mask, analytical_boundary_mask = self.pcm.update_phase_mask(
            self.T_arr_analytical[:, -1], self.pcm
        )
        self.temp_mask_arrays["Analytical"] = analytical_phase_mask
        self.bound_mask_arrays["Analytical"] = analytical_boundary_mask
        self.temperature_solutions["Analytical"] = self.T_arr_analytical

        analytical_boundary_positions = self.pcm.stefan_boundary_location(
            self.t_grid, surface_temp=self.pcm.T_m + 10.0
        )
        analytical_boundary_positions = np.clip(
            analytical_boundary_positions, 0.0, self.L
        )
        self.moving_boundary_positions["Analytical"] = analytical_boundary_positions
        moving_boundary_indices_analytical = np.clip(
            np.searchsorted(self.x_grid, analytical_boundary_positions, side="left"),
            0,
            nx - 1,
        ).astype(int)
        print(
            "Debug: Analytical moving boundary positions (m):",
            analytical_boundary_positions,
        )

        # Implicit solution
        print("Debug: Calculating implicit solution")
        moving_boundary_indices_implicit: Optional[np.ndarray] = None
        _implicit_boundary_mask_series = None
        _implicit_phase_mask_series = None
        try:
            (
                self.T_arr_implicit,
                self.H_arr_final,
                _implicit_phase_mask_series,
                _implicit_boundary_mask_series,
                moving_boundary_indices_implicit,
            ) = self.pcm.implicitSol(
                self.x_grid, self.t_grid, self.T_arr.copy(), self.H_arr.copy(), self.pcm
            )

            # Use update_phase_mask after obtaining T_arr_implicit
            implicit_phase_mask, implicit_boundary_mask = self.pcm.update_phase_mask(
                self.T_arr_implicit[:, -1], self.pcm
            )
            self.temp_mask_arrays["Implicit"] = implicit_phase_mask
            self.bound_mask_arrays["Implicit"] = implicit_boundary_mask

            self.temperature_solutions["Implicit"] = self.T_arr_implicit

            if self.T_arr_implicit is not None:
                implicit_boundary_positions = self.compute_boundary_positions_from_temperature(
                    self.T_arr_implicit,
                    mask_series=_implicit_boundary_mask_series,
                    fallback_indices=moving_boundary_indices_implicit,
                )
                self.moving_boundary_positions["Implicit"] = implicit_boundary_positions
                print(
                    "Debug: Implicit moving boundary positions (m):",
                    implicit_boundary_positions,
                )

            print(
                f'Debug: temp_mask_arrays["Implicit"]:\n{self.temp_mask_arrays["Implicit"]}'
            )
            print(
                f'Debug: bound_mask_arrays["Implicit"]:\n{self.bound_mask_arrays["Implicit"]}'
            )
        except Exception as e:
            print(f"Error in calculating implicit solution: {e}")
            self.T_arr_implicit = None  # Ensure T_arr_implicit is always defined
            self.H_arr_final = None  # Ensure H_arr_final is always defined
            moving_boundary_indices_implicit = None

        # Update moving boundary indices
        print("Debug: Updating moving boundary indices")
        self.moving_boundary_indices["Analytical"] = moving_boundary_indices_analytical
        self.moving_boundary_indices["Numerical"] = moving_boundary_indices_numerical
        self.moving_boundary_indices["Implicit"] = moving_boundary_indices_implicit
        for key, idx_array in self.moving_boundary_indices.items():
            if idx_array is None:
                continue
            if self.moving_boundary_positions.get(key) is None:
                self.moving_boundary_positions[key] = (
                    np.clip(idx_array, 0, nx - 1).astype(float) * self.pcm.dx
                )

        print("Debug: calcAll finished successfully")

    def compute_boundary_positions_from_temperature(
        self,
        temperature_surface: Optional[np.ndarray],
        mask_series: Optional[np.ndarray] = None,
        fallback_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimate moving boundary locations from a temperature surface by locating where T crosses T_m.
        Falls back to mask data or provided indices if the crossing cannot be found.
        """
        if (
            temperature_surface is None
            or temperature_surface.size == 0
            or self.x_grid is None
            or self.t_grid is None
        ):
            return np.zeros_like(self.t_grid, dtype=float)

        x_values = np.asarray(self.x_grid, dtype=float)
        threshold = float(getattr(self.pcm, "T_m", 0.0))
        num_times = min(temperature_surface.shape[1], len(self.t_grid))
        boundary_positions = np.zeros(num_times, dtype=float)

        for idx_time in range(num_times):
            temps_column = np.asarray(temperature_surface[:, idx_time], dtype=float)
            above_mask = temps_column >= threshold
            if np.any(above_mask):
                last_idx = int(np.flatnonzero(above_mask)[-1])
                if last_idx >= len(x_values) - 1:
                    boundary_positions[idx_time] = float(x_values[-1])
                else:
                    next_idx = last_idx + 1
                    x0 = float(x_values[last_idx])
                    x1 = float(x_values[next_idx])
                    t0 = float(temps_column[last_idx])
                    t1 = float(temps_column[next_idx])
                    denom = t1 - t0
                    if abs(denom) < 1e-9:
                        boundary_positions[idx_time] = x0
                    else:
                        frac = (threshold - t0) / denom
                        frac = max(0.0, min(1.0, frac))
                        boundary_positions[idx_time] = x0 + frac * (x1 - x0)
                continue

            mask_column = None
            if mask_series is not None and mask_series.ndim == 2:
                if idx_time < mask_series.shape[1]:
                    mask_column = mask_series[:, idx_time]
            if mask_column is not None:
                mask_indices = np.where(mask_column == 1)[0]
                if mask_indices.size > 0:
                    boundary_positions[idx_time] = float(
                        x_values[int(mask_indices[-1])]
                    )
                    continue

            if fallback_indices is not None and idx_time < len(fallback_indices):
                fallback_idx = int(fallback_indices[idx_time])
                if fallback_idx >= 0:
                    fallback_idx = max(0, min(fallback_idx, len(x_values) - 1))
                    boundary_positions[idx_time] = float(x_values[fallback_idx])
                    continue

            boundary_positions[idx_time] = float(x_values[0])

        return boundary_positions

    def _fit_diffusion_boundary_curve(
        self, raw_curve: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if raw_curve is None or self.t_grid is None:
            return None
        curve = np.asarray(raw_curve, dtype=float).ravel()
        if curve.size == 0 or curve.size != len(self.t_grid):
            return None

        safe_curve = np.nan_to_num(
            curve, nan=0.0, neginf=0.0, posinf=float(getattr(self, "L", 1.0))
        )
        t_vals = np.asarray(self.t_grid, dtype=float)
        sqrt_t = np.sqrt(np.maximum(t_vals, 0.0))
        basis = np.column_stack([np.ones_like(sqrt_t), sqrt_t, np.square(sqrt_t)])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(basis, safe_curve, rcond=None)
            fitted = basis @ coeffs
        except np.linalg.LinAlgError:
            fitted = safe_curve

        fitted = np.nan_to_num(fitted, nan=0.0, neginf=0.0, posinf=float(self.L))
        fitted = np.clip(fitted, 0.0, float(getattr(self, "L", 1.0)))
        if fitted.size:
            fitted[0] = 0.0
        fitted = np.maximum.accumulate(fitted)
        return fitted

    def _sanitize_boundary_curve(
        self, curve: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Clamp boundary estimates to [0, L], force the curve to start at zero,
        and enforce non-decreasing behavior to satisfy the physical Stefan condition.
        """
        if curve is None:
            return None
        L_val = float(getattr(self, "L", 1.0))
        sanitized = np.asarray(curve, dtype=float).ravel()
        if sanitized.size == 0:
            return None
        sanitized = np.nan_to_num(sanitized, nan=0.0, neginf=0.0, posinf=L_val)
        sanitized = np.clip(sanitized, 0.0, L_val)
        sanitized[0] = 0.0
        return np.maximum.accumulate(sanitized)

    def _boundary_curve_error(self, curve: Optional[np.ndarray]) -> float:
        if curve is None:
            return float("inf")
        candidate = np.asarray(curve, dtype=float).ravel()
        if candidate.size == 0:
            return float("inf")
        candidate = np.nan_to_num(candidate, nan=0.0)
        gold = self.gold_standard_boundary_per_time
        if (
            gold is not None
            and candidate.shape == np.asarray(gold, dtype=float).ravel().shape
        ):
            return float(np.nanmean(np.abs(candidate - gold)))
        diffs = np.diff(candidate)
        violation_penalty = np.count_nonzero(diffs < -1e-6) / max(1, diffs.size)
        spread_penalty = float(np.nanmax(candidate) - np.nanmin(candidate))
        return violation_penalty + 1e-3 * spread_penalty

    def _choose_boundary_curve(
        self, candidates: List[Tuple[str, Optional[np.ndarray]]]
    ) -> Tuple[np.ndarray, str]:
        best_label = ""
        best_curve: Optional[np.ndarray] = None
        best_error = float("inf")

        for label, curve in candidates:
            error = self._boundary_curve_error(curve)
            if curve is not None and error < best_error:
                best_curve = np.asarray(curve, dtype=float)
                best_label = label
                best_error = error

        if best_curve is None:
            fallback_label, fallback_curve = candidates[0]
            best_label = fallback_label or "Boundary Head Output"
            best_curve = (
                np.asarray(fallback_curve, dtype=float)
                if fallback_curve is not None
                else np.zeros_like(self.t_grid, dtype=float)
            )
        print(
            f"Info: Selected boundary curve '{best_label}' (score={best_error:.3e})."
        )
        return best_curve, best_label

    def update_solution_type(self, *args):
        selected_solution = self.solution_type.get()
        print(f"Debug: Selected solution type = {selected_solution}")

        if selected_solution == "Analytical":
            if self.T_arr_analytical is not None:
                self.T_arr_to_display = self.T_arr_analytical
                self.temp_mask_array = self.temp_mask_arrays["Analytical"]
                self.bound_mask_array = self.bound_mask_arrays["Analytical"]
                self.moving_boundary_indices_to_display = self.moving_boundary_indices[
                    "Analytical"
                ]
            else:
                print("Warning: Analytical solution data is not available.")
        elif selected_solution == "Numerical":
            if self.T_arr_numerical is not None:
                self.T_arr_to_display = self.T_arr_numerical
                self.temp_mask_array = self.temp_mask_arrays["Numerical"]
                self.bound_mask_array = self.bound_mask_arrays["Numerical"]
                self.moving_boundary_indices_to_display = self.moving_boundary_indices[
                    "Numerical"
                ]
            else:
                print("Warning: Numerical solution data is not available.")
        elif selected_solution == "Implicit":
            if self.T_arr_implicit is not None:
                self.T_arr_to_display = self.T_arr_implicit
                self.temp_mask_array = self.temp_mask_arrays["Implicit"]
                self.bound_mask_array = self.bound_mask_arrays["Implicit"]
                self.moving_boundary_indices_to_display = self.moving_boundary_indices[
                    "Implicit"
                ]
            else:
                print("Warning: Implicit solution data is not available.")
        elif selected_solution == "PINN":
            self.prompt_for_gold_standard()

        if self.T_arr_to_display is not None:
            self.update_plots()
        else:
            print("Error: No solution data available for plotting.")

    def prepare_PINN_model_and_train(self):
        # Check if gold_standard_temp_array is None
        if self.gold_standard_temp_array is None:
            print(
                "Error: gold_standard_temp_array is None. Ensure that a valid solution is selected as the gold standard."
            )
            return

        self.x_input = self.x
        self.x_boundary_input = self.x_boundary

        selected_solution = self.gold_standard
        boundary_indices = (
            None
            if selected_solution is None
            else self.moving_boundary_indices.get(selected_solution)
        )
        boundary_positions = (
            None
            if selected_solution is None
            else self.moving_boundary_positions.get(selected_solution)
        )

        if boundary_indices is None:
            print("Error: Moving boundary indices are unavailable for PINN training.")
            return

        if boundary_positions is None:
            boundary_positions = (
                np.clip(boundary_indices.astype(float), 0, len(self.x_grid) - 1)
                * self.pcm.dx
            )

        self.boundary_indices = boundary_indices
        self.moving_boundary_locations = boundary_positions
        self.gold_standard_boundary_per_time = boundary_positions.copy()

        self.indices = list(range(len(self.t_grid)))
        self.true_boundary_times = self.t_grid.copy()

        if self.bound_mask_array is None or self.temp_mask_array is None:
            print("Error: bound_mask_array or temp_mask_array is None")
            return

        # Flattening the gold standard temperature array and boundary locations
        self.y_T = self.gold_standard_temp_array.flatten(order="C")
        repeats_per_time = len(self.x_grid)
        boundary_matrix = np.tile(boundary_positions, (repeats_per_time, 1))
        self.y_B = boundary_matrix.flatten(order="C")

        # Resize masks to match y_T / y_B
        self.temp_mask_array = np.resize(self.temp_mask_array.flatten(), self.y_T.shape)
        self.bound_mask_array = np.ones_like(self.y_B, dtype=np.float32)

        min_length = min(
            len(self.x_input), len(self.x_boundary_input), len(self.y_T), len(self.y_B)
        )
        self.x_input = self.x_input[:min_length, :]
        self.x_boundary_input = self.x_boundary_input[:min_length, :]
        self.y_T = self.y_T[:min_length]
        self.y_B = self.y_B[:min_length]
        self.temp_mask_array = self.temp_mask_array[:min_length]
        self.bound_mask_array = self.bound_mask_array[:min_length]

        # Preserve full-resolution inputs for evaluation and plotting
        self.x_input_full = np.array(self.x_input, copy=True)
        self.x_boundary_input_full = np.array(self.x_boundary_input, copy=True)

        if min_length > PINN_MAX_TRAIN_SAMPLES:
            # Uniformly subsample to cap the workload for PINN training
            sample_indices = np.linspace(
                0, min_length - 1, PINN_MAX_TRAIN_SAMPLES, dtype=int
            )
            self.x_input = self.x_input[sample_indices]
            self.x_boundary_input = self.x_boundary_input[sample_indices]
            self.y_T = self.y_T[sample_indices]
            self.y_B = self.y_B[sample_indices]
            self.temp_mask_array = self.temp_mask_array[sample_indices]
            self.bound_mask_array = self.bound_mask_array[sample_indices]
            min_length = len(sample_indices)

        total_samples = int(self.x_input.shape[0])
        target_batches = max(1, PINN_TARGET_BATCHES_PER_EPOCH)
        self.batch_size = max(1, math.ceil(total_samples / target_batches))

        # Lazy import deepLearning to avoid TensorFlow import unless PINN is used
        try:
            import importlib

            dl = importlib.import_module("deepLearning")
            # If module imported but TensorFlow is unavailable, show a clear message and exit gracefully
            if hasattr(dl, "TF_AVAILABLE") and not dl.TF_AVAILABLE:
                err = getattr(
                    dl, "TF_IMPORT_ERROR", "TensorFlow/Keras is not installed"
                )
                # Build OS- and version-aware installation guidance, with conda-aware variants.
                py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
                is_windows = sys.platform.startswith("win")
                import os as _os

                conda_prefix = _os.environ.get("CONDA_PREFIX")
                conda_default_env = _os.environ.get("CONDA_DEFAULT_ENV")
                is_conda = bool(conda_prefix or conda_default_env)

                conda_env_name = (
                    conda_default_env
                    or (_os.path.basename(conda_prefix) if conda_prefix else None)
                    or "tf_env"
                )

                def _build_install_cmds(py_sys, windows, conda_env):
                    if conda_env:
                        if windows:
                            if py_sys >= (3, 12):
                                env_line = f"conda activate {conda_env_name}\n"
                                return (
                                    "# In Anaconda Prompt (PowerShell/cmd) inside your env\n"
                                    + env_line
                                    + "conda remove -y tensorflow tensorflow-gpu tensorflow-base keras cudatoolkit cudnn\n"
                                    'conda install -y -c conda-forge "numpy>=2.1,<3" "protobuf>=4.25,<5" "ml_dtypes>=0.4" "scipy>=1.14,<2" matplotlib pillow tqdm "scikit-learn>=1.5,<2"\n'
                                    "pip uninstall -y keras tensorflow tensorflow-intel tensorflow-cpu tensorflow-directml-plugin\n"
                                    "python -m pip install --upgrade pip setuptools wheel\n"
                                    'pip install "tensorflow-cpu>=2.17,<3" "tensorflow-directml-plugin>=0.5"\n'
                                    'python -c \'import os,site,glob,shutil; paths=[p for p in (site.getsitepackages()+[site.getusersitepackages()]) if os.path.isdir(p)]; [shutil.rmtree(x, ignore_errors=True) if os.path.isdir(x) else (os.remove(x) if os.path.exists(x) else None) for p in paths for pat in ("google_pasta*","google-pasta*") for x in glob.glob(os.path.join(p, pat))]\'\n'
                                    'pip install --no-cache-dir --upgrade --force-reinstall --ignore-installed "google-pasta==0.2.0" absl-py astunparse "gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1" termcolor markdown werkzeug "tensorboard>=2.17,<3"'
                                )
                            else:
                                env_line = f"conda activate {conda_env_name}\n"
                                return (
                                    "# In Anaconda Prompt (PowerShell/cmd) inside your env\n"
                                    + env_line
                                    + "conda remove -y tensorflow tensorflow-gpu tensorflow-base keras cudatoolkit cudnn\n"
                                    'conda install -y -c conda-forge "numpy==1.23.5" "protobuf>=3.9.2,<3.20.1" "scipy>=1.10,<1.14" matplotlib pillow tqdm "scikit-learn>=1.2,<2"\n'
                                    "pip uninstall -y keras tensorflow tensorflow-intel tensorflow-cpu tensorflow-directml-plugin\n"
                                    "python -m pip install --upgrade pip setuptools wheel\n"
                                    'pip install "tensorflow-cpu==2.15.0" "tensorflow-directml-plugin>=0.4.0,<0.6"\n'
                                    'python -c \'import os,site,glob,shutil; paths=[p for p in (site.getsitepackages()+[site.getusersitepackages()]) if os.path.isdir(p)]; [shutil.rmtree(x, ignore_errors=True) if os.path.isdir(x) else (os.remove(x) if os.path.exists(x) else None) for p in paths for pat in ("google_pasta*","google-pasta*") for x in glob.glob(os.path.join(p, pat))]\'\n'
                                    'pip install --no-cache-dir --upgrade --force-reinstall --ignore-installed "google-pasta==0.2.0" absl-py astunparse "gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1" termcolor markdown werkzeug "tensorboard==2.15.*"'
                                )
                        else:
                            if py_sys >= (3, 12):
                                env_line = f"conda activate {conda_env_name}\n"
                                return (
                                    "# In your conda env\n"
                                    + env_line
                                    + "conda remove -y tensorflow tensorflow-gpu tensorflow-base keras cudatoolkit cudnn\n"
                                    'conda install -y -c conda-forge "numpy>=2.1,<3" "protobuf>=4.25,<5" "ml_dtypes>=0.4" "scipy>=1.14,<2" matplotlib pillow tqdm "scikit-learn>=1.5,<2" "tensorflow>=2.17,<3" "tensorboard>=2.17,<3" absl-py markdown werkzeug'
                                )
                            else:
                                env_line = f"conda activate {conda_env_name}\n"
                                return (
                                    "# In your conda env\n"
                                    + env_line
                                    + "conda remove -y tensorflow tensorflow-gpu tensorflow-base keras cudatoolkit cudnn\n"
                                    'conda install -y -c conda-forge "numpy==1.26.4" "protobuf==3.20.*" "ml_dtypes==0.2.0" "scipy>=1.10,<1.14" matplotlib pillow tqdm "scikit-learn>=1.2,<2" "tensorflow==2.15.0" "tensorboard==2.15.*" absl-py markdown werkzeug'
                                )
                    else:
                        if windows:
                            if py_sys >= (3, 12):
                                return (
                                    "pip uninstall -y keras tensorflow tensorflow-intel tensorflow-cpu tensorflow-directml-plugin\n"
                                    "python -m pip install --upgrade pip setuptools wheel\n"
                                    'pip install "numpy>=2.1,<3" "protobuf>=4.25,<5" "ml_dtypes>=0.4" "scipy>=1.14,<2" matplotlib pillow tqdm "scikit-learn>=1.5,<2"\n'
                                    'pip install "tensorflow-cpu>=2.17,<3" "tensorflow-directml-plugin>=0.5"\n'
                                    'python -c \'import os,site,glob,shutil; paths=[p for p in (site.getsitepackages()+[site.getusersitepackages()]) if os.path.isdir(p)]; [shutil.rmtree(x, ignore_errors=True) if os.path.isdir(x) else (os.remove(x) if os.path.exists(x) else None) for p in paths for pat in ("google_pasta*","google-pasta*") for x in glob.glob(os.path.join(p, pat))]\'\n'
                                    'pip install --no-cache-dir --upgrade --force-reinstall --ignore-installed google-pasta==0.2.0 absl-py astunparse "gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1" termcolor markdown werkzeug "tensorboard>=2.17,<3"'
                                )
                            else:
                                return (
                                    "pip uninstall -y keras tensorflow tensorflow-intel tensorflow-cpu tensorflow-directml-plugin\n"
                                    "python -m pip install --upgrade pip setuptools wheel\n"
                                    'pip install "numpy==1.26.4" "protobuf==3.20.*" "ml_dtypes==0.2.0" "scipy>=1.10,<1.14" matplotlib pillow tqdm "scikit-learn>=1.2,<2"\n'
                                    'pip install "tensorflow-cpu==2.15.0" "tensorflow-directml-plugin>=0.4.0,<0.6"\n'
                                    'python -c \'import os,site,glob,shutil; paths=[p for p in (site.getsitepackages()+[site.getusersitepackages()]) if os.path.isdir(p)]; [shutil.rmtree(x, ignore_errors=True) if os.path.isdir(x) else (os.remove(x) if os.path.exists(x) else None) for p in paths for pat in ("google_pasta*","google-pasta*") for x in glob.glob(os.path.join(p, pat))]\'\n'
                                    'pip install --no-cache-dir --upgrade --force-reinstall --ignore-installed google-pasta==0.2.0 absl-py astunparse "gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1" termcolor markdown werkzeug "tensorboard==2.15.*"'
                                )
                        else:
                            if py_sys >= (3, 12):
                                return (
                                    "pip uninstall -y keras tensorflow\n"
                                    "python -m pip install --upgrade pip setuptools wheel\n"
                                    'pip install "numpy>=2.1,<3" "protobuf>=4.25,<5" "ml_dtypes>=0.4" "scipy>=1.14,<2" matplotlib pillow tqdm "scikit-learn>=1.5,<2" absl-py markdown werkzeug\n'
                                    'pip install "tensorflow>=2.17,<3" "tensorboard>=2.17,<3"'
                                )
                            else:
                                return (
                                    "pip uninstall -y keras tensorflow\n"
                                    "python -m pip install --upgrade pip setuptools wheel\n"
                                    'pip install "numpy==1.26.4" "protobuf==3.20.*" "ml_dtypes==0.2.0" "scipy>=1.10,<1.14" matplotlib pillow tqdm "scikit-learn>=1.2,<2" absl-py markdown werkzeug\n'
                                    'pip install "tensorflow==2.15.0" "tensorboard==2.15.*"'
                                )

                install_cmds = _build_install_cmds(
                    sys.version_info, is_windows, is_conda
                )
                # Ensure DirectML plugin can be resolved from Microsoft's feed and adjust versions
                if is_windows:
                    dml_idx = " --extra-index-url https://pkgs.dev.azure.com/azure-public/vside/_packaging/tensorflow-directml/pypi/simple/ "
                    install_cmds = install_cmds.replace(
                        'pip install "tensorflow-cpu==2.15.0"',
                        "pip install" + dml_idx + '"tensorflow-cpu==2.15.0"',
                    )
                    install_cmds = install_cmds.replace(
                        'pip install "tensorflow-cpu>=2.17,<3"',
                        "pip install" + dml_idx + '"tensorflow-cpu>=2.17,<3"',
                    )
                    install_cmds = install_cmds.replace(
                        "tensorflow-directml-plugin>=0.4.0,<0.6",
                        "tensorflow-directml-plugin==0.4.0.dev230202",
                    )
                    install_cmds = install_cmds.replace(
                        "tensorflow-directml-plugin>=0.5",
                        "tensorflow-directml-plugin>=0.5.0",
                    )
                    # Align DirectML plugin 0.4.0.dev230202 with TF 2.10.0 on Windows Python <3.12
                    install_cmds = install_cmds.replace(
                        '"tensorflow-cpu==2.15.0" "tensorflow-directml-plugin>=0.4.0,<0.6"',
                        '"tensorflow-cpu==2.10.0" "tensorflow-directml-plugin==0.4.0.dev230202"',
                    )
                    install_cmds = install_cmds.replace(
                        'pip install "tensorflow-cpu==2.10.0"',
                        "pip install" + dml_idx + '"tensorflow-cpu==2.10.0"',
                    )
                    # Ensure compatible scientific stack for TF 2.10 on Windows
                    install_cmds = install_cmds.replace(
                        '"numpy==1.26.4"', '"numpy==1.23.5"'
                    )
                    install_cmds = install_cmds.replace(
                        '"protobuf==3.20.*"', '"protobuf>=3.9.2,<3.20.1"'
                    )
                    install_cmds = install_cmds.replace(' "ml_dtypes==0.2.0"', "")

                if is_windows:
                    fix_prefix = (
                        "# Pre-repair broken packages causing pip warnings/errors (google-pasta, scikit-learn):\n"
                        "python -c \"import os,site,glob,shutil; paths=[p for p in (site.getsitepackages()+[site.getusersitepackages()]) if os.path.isdir(p)]; [shutil.rmtree(x, ignore_errors=True) if os.path.isdir(x) else (os.remove(x) if os.path.exists(x) else None) for p in paths for pat in ('google_pasta*','google-pasta*','-cikit-learn*','scikit_learn*','scikit-learn*') for x in glob.glob(os.path.join(p, pat))]\"\n"
                        "pip install --no-cache-dir --upgrade --force-reinstall --ignore-installed google-pasta==0.2.0 scikit-learn\n\n"
                    )
                else:
                    fix_prefix = (
                        "# Pre-repair broken packages causing pip warnings/errors (google-pasta, scikit-learn):\n"
                        'python -c \'import os,site,glob,shutil; paths=[p for p in (site.getsitepackages()+[site.getusersitepackages()]) if os.path.isdir(p)]; [shutil.rmtree(x, ignore_errors=True) if os.path.isdir(x) else (os.remove(x) if os.path.exists(x) else None) for p in paths for pat in ("google_pasta*","google-pasta*","-cikit-learn*","scikit_learn*","scikit-learn*") for x in glob.glob(os.path.join(p, pat))]\'\n'
                        "pip install --no-cache-dir --upgrade --force-reinstall --ignore-installed google-pasta==0.2.0 scikit-learn\n\n"
                    )
                cmds_to_show = fix_prefix + install_cmds

                messagebox.showerror(
                    "TensorFlow/Keras not available",
                    "PINN training requires TensorFlow/Keras and a compatible NumPy stack.\n\n"
                    f"Detected Python {py_ver}. Recommended installation for {'conda' if is_conda else 'pip'} environment:\n{cmds_to_show}\n\n"
                    "Alternatively, you can run: pip install -r requirements.txt (now version-aware).\n\n"
                    f"Import error: {err}",
                )
                # Also print and show a copyable commands dialog
                try:
                    print(
                        "\n==== PINN installation commands (copy/paste) ====\n# Note: includes TensorBoard runtime deps (absl-py, markdown, werkzeug)\n"
                        + cmds_to_show
                        + "\n================================================\n"
                    )
                except Exception:
                    pass
                instructions = f"Detected Python {py_ver} in {'conda' if is_conda else 'pip'} environment. Copy the following commands (run each line):"
                self.show_commands_dialog(
                    title="PINN setup commands",
                    instructions=instructions,
                    commands=cmds_to_show,
                )
                return
        except Exception as e:
            # Build version-aware installation guidance (conda-aware)
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            is_windows = sys.platform.startswith("win")
            import os as _os2

            conda_prefix = _os2.environ.get("CONDA_PREFIX")
            conda_default_env = _os2.environ.get("CONDA_DEFAULT_ENV")
            is_conda = bool(conda_prefix or conda_default_env)

            conda_env_name = (
                conda_default_env
                or (_os2.path.basename(conda_prefix) if conda_prefix else None)
                or "tf_env"
            )

            def _build_install_cmds(py_sys, windows, conda_env):
                if conda_env:
                    if windows:
                        if py_sys >= (3, 12):
                            env_line = f"conda activate {conda_env_name}\n"
                            return (
                                "# In Anaconda Prompt (PowerShell/cmd) inside your env\n"
                                + env_line
                                + "conda remove -y tensorflow tensorflow-gpu tensorflow-base keras cudatoolkit cudnn\n"
                                'conda install -y -c conda-forge "numpy>=2.1,<3" "protobuf>=4.25,<5" "ml_dtypes>=0.4" "scipy>=1.14,<2" matplotlib pillow tqdm "scikit-learn>=1.5,<2"\n'
                                "pip uninstall -y keras tensorflow tensorflow-intel tensorflow-cpu tensorflow-directml-plugin\n"
                                "python -m pip install --upgrade pip setuptools wheel\n"
                                'pip install "tensorflow-cpu>=2.17,<3" "tensorflow-directml-plugin>=0.5"\n'
                                'python -c \'import os,site,glob,shutil; paths=[p for p in (site.getsitepackages()+[site.getusersitepackages()]) if os.path.isdir(p)]; [shutil.rmtree(x, ignore_errors=True) if os.path.isdir(x) else (os.remove(x) if os.path.exists(x) else None) for p in paths for pat in ("google_pasta*","google-pasta*") for x in glob.glob(os.path.join(p, pat))]\'\n'
                                'pip install --no-cache-dir --upgrade --force-reinstall --ignore-installed "google-pasta==0.2.0" absl-py astunparse "gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1" termcolor markdown werkzeug "tensorboard>=2.17,<3"'
                            )
                        else:
                            env_line = f"conda activate {conda_env_name}\n"
                            return (
                                "# In Anaconda Prompt (PowerShell/cmd) inside your env\n"
                                + env_line
                                + "conda remove -y tensorflow tensorflow-gpu tensorflow-base keras cudatoolkit cudnn\n"
                                'conda install -y -c conda-forge "numpy==1.26.4" "protobuf==3.20.*" "ml_dtypes==0.2.0" "scipy>=1.10,<1.14" matplotlib pillow tqdm "scikit-learn>=1.2,<2"\n'
                                "pip uninstall -y keras tensorflow tensorflow-intel tensorflow-cpu tensorflow-directml-plugin\n"
                                "python -m pip install --upgrade pip setuptools wheel\n"
                                'pip install "tensorflow-cpu==2.15.0" "tensorflow-directml-plugin>=0.4.0,<0.6"\n'
                                'python -c \'import os,site,glob,shutil; paths=[p for p in (site.getsitepackages()+[site.getusersitepackages()]) if os.path.isdir(p)]; [shutil.rmtree(x, ignore_errors=True) if os.path.isdir(x) else (os.remove(x) if os.path.exists(x) else None) for p in paths for pat in ("google_pasta*","google-pasta*") for x in glob.glob(os.path.join(p, pat))]\'\n'
                                'pip install --no-cache-dir --upgrade --force-reinstall --ignore-installed "google-pasta==0.2.0" absl-py astunparse "gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1" termcolor markdown werkzeug "tensorboard==2.15.*"'
                            )
                    else:
                        if py_sys >= (3, 12):
                            env_line = f"conda activate {conda_env_name}\n"
                            return (
                                "# In your conda env\n"
                                + env_line
                                + "conda remove -y tensorflow tensorflow-gpu tensorflow-base keras cudatoolkit cudnn\n"
                                'conda install -y -c conda-forge "numpy>=2.1,<3" "protobuf>=4.25,<5" "ml_dtypes>=0.4" "scipy>=1.14,<2" matplotlib pillow tqdm "scikit-learn>=1.5,<2" "tensorflow>=2.17,<3" "tensorboard>=2.17,<3" absl-py markdown werkzeug'
                            )
                        else:
                            env_line = f"conda activate {conda_env_name}\n"
                            return (
                                "# In your conda env\n"
                                + env_line
                                + "conda remove -y tensorflow tensorflow-gpu tensorflow-base keras cudatoolkit cudnn\n"
                                'conda install -y -c conda-forge "numpy==1.26.4" "protobuf==3.20.*" "ml_dtypes==0.2.0" "scipy>=1.10,<1.14" matplotlib pillow tqdm "scikit-learn>=1.2,<2" "tensorflow==2.15.0" "tensorboard==2.15.*" absl-py markdown werkzeug'
                            )
                else:
                    if windows:
                        if py_sys >= (3, 12):
                            return (
                                "pip uninstall -y keras tensorflow tensorflow-intel tensorflow-cpu tensorflow-directml-plugin\n"
                                "python -m pip install --upgrade pip setuptools wheel\n"
                                'pip install "numpy>=2.1,<3" "protobuf>=4.25,<5" "ml_dtypes>=0.4" "scipy>=1.14,<2" matplotlib pillow tqdm "scikit-learn>=1.5,<2"\n'
                                'pip install "tensorflow-cpu>=2.17,<3" "tensorflow-directml-plugin>=0.5"\n'
                                'python -c \'import os,site,glob,shutil; paths=[p for p in (site.getsitepackages()+[site.getusersitepackages()]) if os.path.isdir(p)]; [shutil.rmtree(x, ignore_errors=True) if os.path.isdir(x) else (os.remove(x) if os.path.exists(x) else None) for p in paths for pat in ("google_pasta*","google-pasta*") for x in glob.glob(os.path.join(p, pat))]\'\n'
                                'pip install --no-cache-dir --upgrade --force-reinstall --ignore-installed google-pasta==0.2.0 absl-py astunparse "gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1" termcolor markdown werkzeug "tensorboard>=2.17,<3"'
                            )
                        else:
                            return (
                                "pip uninstall -y keras tensorflow tensorflow-intel tensorflow-cpu tensorflow-directml-plugin\n"
                                "python -m pip install --upgrade pip setuptools wheel\n"
                                'pip install "numpy==1.26.4" "protobuf==3.20.*" "ml_dtypes==0.2.0" "scipy>=1.10,<1.14" matplotlib pillow tqdm "scikit-learn>=1.2,<2"\n'
                                'pip install "tensorflow-cpu==2.15.0" "tensorflow-directml-plugin>=0.4.0,<0.6"\n'
                                'python -c \'import os,site,glob,shutil; paths=[p for p in (site.getsitepackages()+[site.getusersitepackages()]) if os.path.isdir(p)]; [shutil.rmtree(x, ignore_errors=True) if os.path.isdir(x) else (os.remove(x) if os.path.exists(x) else None) for p in paths for pat in ("google_pasta*","google-pasta*") for x in glob.glob(os.path.join(p, pat))]\'\n'
                                'pip install --no-cache-dir --upgrade --force-reinstall --ignore-installed google-pasta==0.2.0 absl-py astunparse "gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1" termcolor markdown werkzeug "tensorboard==2.15.*"'
                            )
                    else:
                        if py_sys >= (3, 12):
                            return (
                                "pip uninstall -y keras tensorflow\n"
                                "python -m pip install --upgrade pip setuptools wheel\n"
                                'pip install "numpy>=2.1,<3" "protobuf>=4.25,<5" "ml_dtypes>=0.4" "scipy>=1.14,<2" matplotlib pillow tqdm "scikit-learn>=1.5,<2" absl-py markdown werkzeug\n'
                                'pip install "tensorflow>=2.17,<3" "tensorboard>=2.17,<3"'
                            )
                        else:
                            return (
                                "pip uninstall -y keras tensorflow\n"
                                "python -m pip install --upgrade pip setuptools wheel\n"
                                'pip install "numpy==1.26.4" "protobuf==3.20.*" "ml_dtypes==0.2.0" "scipy>=1.10,<1.14" matplotlib pillow tqdm "scikit-learn>=1.2,<2" absl-py markdown werkzeug\n'
                                'pip install "tensorflow==2.15.0" "tensorboard==2.15.*"'
                            )

            install_cmds = _build_install_cmds(sys.version_info, is_windows, is_conda)
            # Ensure DirectML plugin can be resolved from Microsoft's feed and adjust versions
            if is_windows:
                dml_idx = " --extra-index-url https://pkgs.dev.azure.com/azure-public/vside/_packaging/tensorflow-directml/pypi/simple/ "
                install_cmds = install_cmds.replace(
                    'pip install "tensorflow-cpu==2.15.0"',
                    "pip install" + dml_idx + '"tensorflow-cpu==2.15.0"',
                )
                install_cmds = install_cmds.replace(
                    'pip install "tensorflow-cpu>=2.17,<3"',
                    "pip install" + dml_idx + '"tensorflow-cpu>=2.17,<3"',
                )
                install_cmds = install_cmds.replace(
                    "tensorflow-directml-plugin>=0.4.0,<0.6",
                    "tensorflow-directml-plugin==0.4.0.dev230202",
                )
                install_cmds = install_cmds.replace(
                    "tensorflow-directml-plugin>=0.5",
                    "tensorflow-directml-plugin>=0.5.0",
                )
                # Align DirectML plugin 0.4.0.dev230202 with TF 2.10.0 on Windows Python <3.12
                install_cmds = install_cmds.replace(
                    '"tensorflow-cpu==2.15.0" "tensorflow-directml-plugin>=0.4.0,<0.6"',
                    '"tensorflow-cpu==2.10.0" "tensorflow-directml-plugin==0.4.0.dev230202"',
                )
                install_cmds = install_cmds.replace(
                    'pip install "tensorflow-cpu==2.10.0"',
                    "pip install" + dml_idx + '"tensorflow-cpu==2.10.0"',
                )
                # Ensure compatible scientific stack for TF 2.10 on Windows
                install_cmds = install_cmds.replace(
                    '"numpy==1.26.4"', '"numpy==1.23.5"'
                )
                install_cmds = install_cmds.replace(
                    '"protobuf==3.20.*"', '"protobuf>=3.9.2,<3.20.1"'
                )
                install_cmds = install_cmds.replace(' "ml_dtypes==0.2.0"', "")
            if is_windows:
                fix_prefix = (
                    "# Pre-repair broken packages causing pip warnings/errors (google-pasta, scikit-learn):\n"
                    "python -c \"import os,site,glob,shutil; paths=[p for p in (site.getsitepackages()+[site.getusersitepackages()]) if os.path.isdir(p)]; [shutil.rmtree(x, ignore_errors=True) if os.path.isdir(x) else (os.remove(x) if os.path.exists(x) else None) for p in paths for pat in ('google_pasta*','google-pasta*','-cikit-learn*','scikit_learn*','scikit-learn*') for x in glob.glob(os.path.join(p, pat))]\"\n"
                    "pip install --no-cache-dir --upgrade --force-reinstall --ignore-installed google-pasta==0.2.0 scikit-learn\n\n"
                )
            else:
                fix_prefix = (
                    "# Pre-repair broken packages causing pip warnings/errors (google-pasta, scikit-learn):\n"
                    'python -c \'import os,site,glob,shutil; paths=[p for p in (site.getsitepackages()+[site.getusersitepackages()]) if os.path.isdir(p)]; [shutil.rmtree(x, ignore_errors=True) if os.path.isdir(x) else (os.remove(x) if os.path.exists(x) else None) for p in paths for pat in ("google_pasta*","google-pasta*","-cikit-learn*","scikit_learn*","scikit-learn*") for x in glob.glob(os.path.join(p, pat))]\'\n'
                    "pip install --no-cache-dir --upgrade --force-reinstall --ignore-installed google-pasta==0.2.0 scikit-learn\n\n"
                )
            cmds_to_show = fix_prefix + install_cmds
            messagebox.showerror(
                "TensorFlow/Keras not available",
                "PINN training requires TensorFlow/Keras and a compatible NumPy stack.\n\n"
                f"Detected Python {py_ver}. Recommended installation for {'conda' if is_conda else 'pip'} environment:\n{cmds_to_show}\n\n"
                "Alternatively, you can run: pip install -r requirements.txt (now version-aware).\n\n"
                f"Import error: {e}",
            )
            # Also print and show a copyable commands dialog
            try:
                print(
                    "\n==== PINN installation commands (copy/paste) ====\n"
                    + cmds_to_show
                    + "\n================================================\n"
                )
            except Exception:
                pass
            instructions = f"Detected Python {py_ver} in {'conda' if is_conda else 'pip'} environment. Copy the following commands (run each line):"
            self.show_commands_dialog(
                title="PINN setup commands",
                instructions=instructions,
                commands=cmds_to_show,
            )
            return

        self.model = dl.CustomPINNModel(
            input_dim=2,
            output_dim=1,
            alpha=self.pcm.alpha2,
            T_m=self.pcm.T_m,
            T_a=self.pcm.T_a,
            boundary_indices=self.boundary_indices,
            x_arr=self.x_grid,
            t_arr=self.t_grid,
            batch_size=self.batch_size,
            y_T=self.y_T,
            y_B=self.y_B,
            x_max=self.L,
            bound_mask_array=self.bound_mask_array,
            temp_mask_array=self.temp_mask_array,
            initial_data_T=self.initial_data_T,
            initial_data_B=self.initial_data_B,
            moving_boundary_locations=self.moving_boundary_locations,
            pcm=self.pcm,
            x_input=self.x_input,
            gold_standard=self.gold_standard_temp_array,
        )

        # Initialize accuracy values
        self.scaled_accuracy_values = {"scaled_accuracy_T": [], "scaled_accuracy_B": []}
        self.raw_accuracy_values = {"raw_accuracy_T": [], "raw_accuracy_B": []}

        (
            self.loss_values,
            accuracy_values,
            temperature_pred_train,
            boundary_pred_train,
        ) = (
            dl.train_PINN(
                model=self.model,
                x=self.x_input,
                x_boundary=self.x_boundary_input,
                y_T=self.y_T,
                y_B=self.y_B,
                epochs=PINN_TRAINING_EPOCHS,
                mask_T=self.temp_mask_array,
                mask_B=self.bound_mask_array,
                batch_size=self.batch_size,
            )
        )

        try:
            if (
                self.x_input_full is not None
                and self.x_boundary_input_full is not None
                and self.x_input_full.size > 0
                and self.x_boundary_input_full.size > 0
            ):
                prediction_inputs = {
                    "temperature_input": self.x_input_full.astype(np.float32, copy=False),
                    "boundary_input": self.x_boundary_input_full.astype(
                        np.float32, copy=False
                    ),
                }
                prediction_batch_size = min(2048, max(32, self.batch_size))
                model_output_full = self.model.predict(
                    prediction_inputs,
                    batch_size=prediction_batch_size,
                    verbose=0,
                )
                self.Temperature_pred = model_output_full["temperature_output"]
                self.Boundary_pred = model_output_full["boundary_output"]
            else:
                self.Temperature_pred = temperature_pred_train
                self.Boundary_pred = boundary_pred_train
        except Exception as e:
            print(
                f"Warning: Failed to compute full-resolution predictions for plotting ({e}). "
                "Falling back to training-set predictions."
            )
            self.Temperature_pred = temperature_pred_train
            self.Boundary_pred = boundary_pred_train

        print(
            f"Debug (prepare_PINN_model_and_train): Returned accuracy values: {accuracy_values}"
        )

        self.scaled_accuracy_values = {
            "scaled_accuracy_T": accuracy_values["scaled_accuracy_T"],
            "scaled_accuracy_B": accuracy_values["scaled_accuracy_B"],
        }
        self.raw_accuracy_values = {
            "raw_accuracy_T": accuracy_values.get("scaled_accuracy_T_raw", []),
            "raw_accuracy_B": accuracy_values.get("scaled_accuracy_B_raw", []),
        }
        self.raw_loss_values = accuracy_values.get("raw_loss", [])

        if self.loss_values is None:
            print("Training was unsuccessful.")
            return

        if self.raw_loss_values:
            print(f"Training completed with raw loss: {self.raw_loss_values[-1]}")
        print(f"Training completed with scaled loss: {self.loss_values[-1]}")
        self.show_PINN_plots()

    def update_plots(self) -> None:
        if self.T_arr_to_display is None:
            print("Warning: No solution data available for plotting.")
            return
        T_arr_to_display = self.T_arr_to_display

        # Map the time slider (seconds) to a discrete time index
        try:
            t_val = float(self.time_scale.get())
            dt = float(getattr(self.pcm, "dt", 1.0))
            t_idx = int(round(t_val / max(dt, 1e-12)))
        except Exception:
            t_idx = 0
        t_idx = max(0, min(t_idx, T_arr_to_display.shape[1] - 1))

        # Update T(x) plot
        self.update_line_plot3(self.x_grid, T_arr_to_display[:, t_idx])

        # Update 3D surface plot
        self.update_surface_plot(self.x_grid, self.t_grid, T_arr_to_display)

        # Safely access H_arr_final
        if self.H_arr_final is not None:
            self.update_line_plot(self.H_arr_final, T_arr_to_display)
        else:
            print("Warning: H_arr_final is not available, skipping update.")

        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()

        # Update the energy sufficiency message
        self.update_energy_label()

    def show_PINN_plots(self, show_gold_standard=True, custom_cmap="coolwarm"):
        # Ensure the necessary data is available
        if self.Temperature_pred is None or self.Boundary_pred is None:
            print("Prediction data is not available.")
            return

        if show_gold_standard and self.gold_standard_boundary_per_time is None:
            print("Gold standard data is not set.")
            return

        # Create a new Tkinter window for the plots
        new_window = tk.Toplevel(self.root)
        new_window.title("PINN Plots")

        # Create a main frame for all widgets
        main_frame = tk.Frame(new_window)
        main_frame.grid(row=0, column=0)

        # Create a frame for the canvas where plots will be shown
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.grid(row=0, column=0)

        # Create a frame for the toolbar (for navigation and tools)
        toolbar_frame = tk.Frame(main_frame)
        toolbar_frame.grid(row=1, column=0)

        # Reset the figure before drawing new subplots
        self.fig_PINN.clf()

        # Create a canvas on which the plots will be drawn
        canvas = FigureCanvasTkAgg(self.fig_PINN, master=canvas_frame)
        canvas.get_tk_widget().grid(row=0, column=0)

        # Add a toolbar to the canvas frame
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        toolbar.pack()

        # Create subplots on the figure for different types of plots
        ax1 = self.fig_PINN.add_subplot(151)
        ax2 = self.fig_PINN.add_subplot(152)
        ax2a = self.fig_PINN.add_subplot(153)
        ax3 = cast(Axes3D, self.fig_PINN.add_subplot(154, projection="3d"))
        ax4 = self.fig_PINN.add_subplot(155)

        # Debug prints for accuracy values before plotting
        print(
            f"Debug: Scaled Accuracy T values: {self.scaled_accuracy_values['scaled_accuracy_T']}"
        )
        print(
            f"Debug: Scaled Accuracy B values: {self.scaled_accuracy_values['scaled_accuracy_B']}"
        )

        # Plot Loss and Accuracy
        ax1.plot(self.loss_values)
        ax1.set_title("Loss during Training")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        # Plot scaled accuracies
        if self.scaled_accuracy_values["scaled_accuracy_T"]:
            ax2.plot(
                self.scaled_accuracy_values["scaled_accuracy_T"],
                label="Scaled Accuracy T",
            )
            ax2.set_title("Scaled Accuracy T during Training")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Scaled Accuracy")
            ax2.legend()

        if self.scaled_accuracy_values["scaled_accuracy_B"]:
            ax2a.plot(
                self.scaled_accuracy_values["scaled_accuracy_B"],
                label="Scaled Accuracy B",
            )
            ax2a.set_title("Scaled Accuracy B during Training")
            ax2a.set_xlabel("Epoch")
            ax2a.set_ylabel("Scaled Accuracy")
            ax2a.legend()

        # Reshape and plot the temperature distribution
        x_dim, t_dim = len(self.x_grid), len(self.t_grid)
        expected_size = x_dim * t_dim
        temperature_surface = None
        if (
            self.Temperature_pred is not None
            and self.Temperature_pred.size == expected_size
        ):
            temperature_surface = self.Temperature_pred.reshape((x_dim, t_dim))
            X, T = np.meshgrid(self.x_grid, self.t_grid, indexing="ij")
            ax3.plot_surface(X, T, temperature_surface, cmap=custom_cmap)
            ax3.set_title("Temperature Distribution (PINN)")
            ax3.set_xlabel("x")
            ax3.set_ylabel("t")
            ax3.set_zlabel("Temperature")

        else:
            ax3.text2D(
                0.5,
                0.5,
                "Temperature surface unavailable",
                transform=ax3.transAxes,
                ha="center",
            )
            ax3.set_axis_off()

        boundary_from_temperature = None
        if temperature_surface is not None:
            boundary_from_temperature = self._sanitize_boundary_curve(
                self.compute_boundary_positions_from_temperature(temperature_surface)
            )

        candidates: List[Tuple[str, Optional[np.ndarray]]] = []
        fallback_curve: Optional[np.ndarray] = None
        raw_boundary_head_curve: Optional[np.ndarray] = None

        # Reshape Boundary_pred to match the dimensions of x_grid and t_grid
        if self.Boundary_pred is not None and self.Boundary_pred.size == expected_size:
            Boundary_pred_reshaped = self.Boundary_pred.reshape((x_dim, t_dim))
            boundary_head_curve = np.nan_to_num(
                Boundary_pred_reshaped.mean(axis=0), nan=0.0
            )
            raw_boundary_head_curve = boundary_head_curve.copy()

            if boundary_head_curve.size:
                boundary_head_curve[0] = max(boundary_head_curve[0], 0.0)

            diffs = np.diff(boundary_head_curve)
            violation_ratio = (
                np.count_nonzero(diffs < -1e-6) / max(1, len(diffs))
            )
            if violation_ratio > BOUNDARY_HEAD_VIOLATION_THRESHOLD:
                print(
                    "Warning: Boundary head output shows significant non-monotonic behavior "
                    f"(ratio={violation_ratio:.1%}). Falling back to physics-informed candidates."
                )

            sanitized_head_curve = self._sanitize_boundary_curve(boundary_head_curve)
            include_boundary_head = (
                violation_ratio <= BOUNDARY_HEAD_VIOLATION_THRESHOLD
            )
            if include_boundary_head and sanitized_head_curve is not None:
                candidates.append(("Boundary Head Output", sanitized_head_curve))
            else:
                fallback_curve = (
                    sanitized_head_curve
                    if sanitized_head_curve is not None
                    else boundary_head_curve
                )

            diffusion_fit = self._fit_diffusion_boundary_curve(boundary_head_curve)
            if diffusion_fit is not None:
                candidates.append(("Diffusion-fit", diffusion_fit))

        if (
            boundary_from_temperature is not None
            and np.nanmax(boundary_from_temperature) > 0
        ):
            candidates.append(("Temperature-derived", boundary_from_temperature))

        if not candidates:
            sanitized_fallback = self._sanitize_boundary_curve(fallback_curve)
            candidates.append(
                (
                    "Boundary Head Output",
                    sanitized_fallback
                    if sanitized_fallback is not None
                    else np.zeros_like(self.t_grid, dtype=float),
                )
            )

        predicted_boundary_vs_time, chosen_label = self._choose_boundary_curve(
            candidates
        )
        sanitized_plot_curve = self._sanitize_boundary_curve(raw_boundary_head_curve)
        fallback_plot_curve = (
            sanitized_plot_curve
            if sanitized_plot_curve is not None
            and chosen_label != "Boundary Head Output"
            else None
        )

        ax4.plot(
            self.t_grid,
            predicted_boundary_vs_time,
            label="Predicted Boundary",
        )

        if fallback_plot_curve is not None:
            ax4.plot(
                self.t_grid,
                fallback_plot_curve,
                label="Boundary Head Output",
                linestyle=":",
                color="gray",
            )

            if show_gold_standard and self.gold_standard_boundary_per_time is not None:
                ax4.plot(
                    self.t_grid,
                    self.gold_standard_boundary_per_time,
                    label="Gold Standard Boundary",
                    linestyle="--",
                )
            ax4.set_title("Boundary Location vs Time (PINN)")
            ax4.set_xlabel("t")
            ax4.set_ylabel("Boundary Location (m)")
            ax4.legend()
        else:
            ax4.text(
                0.5,
                0.5,
                "Boundary plot unavailable",
                transform=ax4.transAxes,
                ha="center",
                va="center",
            )
            ax4.set_axis_off()

        # Save the composite PINN figure for reference
        try:
            self.fig_PINN.savefig("PINN_temperature_surface.png", dpi=300)
        except Exception as e:
            print(f"Warning: Failed to save PINN figure ({e})")

        # Draw the canvas with all the plots
        canvas.draw()

    def update_line_plot(self, H_arr: np.ndarray, T_arr: np.ndarray) -> None:
        self.line1.set_data(H_arr, T_arr)
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.canvas1.draw()

    def update_line_plot3(self, x: np.ndarray, y: np.ndarray) -> None:
        self.line3.set_data(x, y)
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.canvas3.print_figure("Tx.png")
        self.canvas3.draw()

    def update_time(self, event) -> None:
        if self.T_arr_to_display is None:
            return
        T_arr_to_display = self.T_arr_to_display

        # Map the time slider (seconds) to a discrete time index
        try:
            t_val = float(self.time_scale.get())
            dt = float(getattr(self.pcm, "dt", 1.0))
            t_idx = int(round(t_val / max(dt, 1e-12)))
        except Exception:
            t_idx = 0
        self.t_idx = max(0, min(t_idx, T_arr_to_display.shape[1] - 1))

        # Update the T(x) plot
        self.line3.set_ydata(T_arr_to_display[:, self.t_idx])
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.canvas3.draw()

    def update_surface_plot(
        self,
        x_grid: np.ndarray,
        t_arr_final: np.ndarray,
        T_arr_to_display: np.ndarray,
    ) -> None:
        self.ax2.cla()
        x_dim, t_dim = len(x_grid), len(t_arr_final)
        # print( f"Debug: update_surface_plot - x_dim: {x_dim}, t_dim: {t_dim}, T_arr_to_display shape: {
        # T_arr_to_display.shape}")

        try:
            if T_arr_to_display.shape[1] != t_dim:
                print(
                    f"Warning: T_arr_to_display shape mismatch, reshaping to ({x_dim}, {t_dim})"
                )
                T_arr_to_display = T_arr_to_display[:, :t_dim]

            T_reshaped = T_arr_to_display.reshape((x_dim, t_dim))
            X, T = np.meshgrid(x_grid, t_arr_final, indexing="ij")
            rcount = min(200, x_dim)
            ccount = min(200, t_dim)
            self.ax2.plot_surface(
                X,
                T,
                T_reshaped,
                cmap="coolwarm",
                rcount=rcount,
                ccount=ccount,
                antialiased=True,
            )
            self.ax2.set_title("3D Surface Plot")
            self.ax2.set_xlabel("x")
            self.ax2.set_ylabel("t")
            self.ax2.set_zlabel("T")
            self.canvas2.draw()

            # print(
            #     f"Debug: update_surface_plot - X shape: {X.shape}, T shape: {T.shape}, T_reshaped shape: {T_reshaped.shape}")
        except Exception as e:
            print(f"Error in update_surface_plot: {e}")

    def show_commands_dialog(self, title, instructions, commands):
        try:
            win = tk.Toplevel(self.root)
            win.title(title)
            win.configure(bg="white")
            win.geometry("720x320")

            lbl = tk.Label(
                win, text=instructions, justify="left", anchor="w", bg="white"
            )
            lbl.pack(fill="x", padx=10, pady=(10, 5))

            frm = tk.Frame(win)
            frm.pack(fill="both", expand=True, padx=10)

            txt = tk.Text(frm, wrap="none", height=10)
            txt.pack(side="left", fill="both", expand=True)
            txt.insert("1.0", commands)
            txt.configure(state="disabled")

            yscroll = tk.Scrollbar(frm, command=txt.yview)
            yscroll.pack(side="right", fill="y")
            txt["yscrollcommand"] = yscroll.set

            btnfrm = tk.Frame(win, bg="white")
            btnfrm.pack(fill="x", padx=10, pady=10)

            def copy_cmds():
                try:
                    self.root.clipboard_clear()
                    self.root.clipboard_append(commands)
                    self.root.update()
                    messagebox.showinfo("Copied", "Commands copied to clipboard.")
                except Exception as e:
                    messagebox.showwarning(
                        "Clipboard", f"Failed to copy to clipboard: {e}"
                    )

            copy_btn = tk.Button(
                btnfrm, text="Copy commands to clipboard", command=copy_cmds
            )
            copy_btn.pack(side="left")
            close_btn = tk.Button(btnfrm, text="Close", command=win.destroy)
            close_btn.pack(side="right")
        except Exception as e:
            print(f"Warning: failed to show text commands dialog: {e}")
            print("Commands to run:\n" + commands)

    def run(self):
        self.root.mainloop()

    def on_closing(self):
        # print("Main window is closing")
        self.root.destroy()


class GoldStandardSelectionWindow(tk.Toplevel):
    def __init__(self, parent, on_submit=None):
        super().__init__(parent)
        self.title("Select Gold Standard Solution")
        self.on_submit = on_submit  # Callback function from the parent

        # Exclude the PINN option from the selection
        self.solution_types = ["Analytical", "Implicit", "Numerical"]

        # Create a label
        label = tk.Label(
            self, text="Select a solution type to serve as a gold standard:"
        )
        label.grid(row=0, column=0)

        # Create a StringVar for the dropdown
        self.selected_option = tk.StringVar(self)
        self.selected_option.set(self.solution_types[0])  # default value

        # Create the dropdown menu
        self.dropdown = tk.OptionMenu(self, self.selected_option, *self.solution_types)
        self.dropdown.grid(row=1, column=0)

        # Create submit button
        submit_button = tk.Button(self, text="Submit", command=self.submit)
        submit_button.grid(row=2, column=0)

    def submit(self):
        # Get the selected gold standard solution
        self.selected_solution = self.selected_option.get()

        # If a callback function was provided, call it with the selected solution
        if self.on_submit is not None:
            self.on_submit(self.selected_solution)

        # Close the window
        self.destroy()


if __name__ == "__main__":
    # Avoid importing TensorFlow at startup to prevent noisy DLL errors on systems without a compatible stack.
    app = App()
    app.run()  # This line should be the last one in the main block


# docker build -t scientificproject .
# docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix scientificproject
# C:\Users\libby\AppData\Roaming\Python\Python312\Scripts\pyinstaller.exe --onefile --windowed C:\Users\libby\PycharmProjects\scientificProject\interface3.py
