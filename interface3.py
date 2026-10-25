import sys

# Keep legacy imports and App available to existing callers and regression tests.
# Executing this file opens the modern workspace; --legacy retains the old GUI.
if __name__ == "__main__" and "--legacy" not in sys.argv:
    try:
        from magnetic_torus.app import main as torus_main
    except ImportError as error:
        print(f"Magnetic Torus desktop dependency is unavailable: {error}\n"
              "Activate conda environment torch-gpu, then run:\n"
              "python -m pip install -r requirements-desktop.txt\n"
              "python interface3.py\n"
              "The original application remains available with --legacy.", file=sys.stderr)
        raise SystemExit(1) from error
    raise SystemExit(torus_main())

import math
import os

# NumPy compatibility shims (e.g., np.bool8 on NumPy 2)
import np_compat  # noqa: F401

import sys
import warnings
from typing import Dict, List, Optional, Tuple, Type, cast

from np_compat import check_numpy_health, friendly_numpy_error


import tkinter as tk
from tkinter import font as tkfont

# import google
# import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
from tkinter import messagebox, ttk
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
PLOT_ENTHALPY_SCALE = 1e-3  # Convert specific enthalpy from J/kg to kJ/kg for T(E).

THEME = {
    "bg": "#f3f6fa",
    "surface": "#ffffff",
    "surface_alt": "#eef3f8",
    "border": "#d8e0ea",
    "text": "#172033",
    "muted": "#5b677a",
    "primary": "#2563eb",
    "primary_hover": "#1d4ed8",
    "danger": "#dc2626",
}

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


class RoundedFrame(tk.Frame):
    """Canvas-backed rounded panel used for the dashboard's main cards."""

    def __init__(
        self,
        master,
        *,
        radius: int = 18,
        padding: int = 22,
        fill: str = THEME["surface"],
        outline: str = THEME["border"],
        bg: str = THEME["bg"],
        **kwargs,
    ):
        super().__init__(master, bg=bg, highlightthickness=0, bd=0, **kwargs)
        self.radius = radius
        self.padding = padding
        self.fill = fill
        self.outline = outline

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.content = tk.Frame(self.canvas, bg=fill, highlightthickness=0, bd=0)
        self.content_window = self.canvas.create_window(
            (padding, padding), window=self.content, anchor="nw"
        )
        self.shape_id = None
        self.canvas.bind("<Configure>", self._draw)

    def _draw(self, event=None) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        radius = min(self.radius, width // 2, height // 2)
        self.canvas.delete("rounded_panel")
        self._create_rounded_rect(
            1,
            1,
            width - 1,
            height - 1,
            radius,
            fill=self.fill,
            outline=self.outline,
            width=1,
            tags="rounded_panel",
        )
        inner_width = max(1, width - (2 * self.padding))
        inner_height = max(1, height - (2 * self.padding))
        self.canvas.coords(self.content_window, self.padding, self.padding)
        self.canvas.itemconfigure(
            self.content_window, width=inner_width, height=inner_height
        )
        self.canvas.tag_lower("rounded_panel")

    def _create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius,
            y1,
            x2 - radius,
            y1,
            x2,
            y1,
            x2,
            y1 + radius,
            x2,
            y2 - radius,
            x2,
            y2,
            x2 - radius,
            y2,
            x1 + radius,
            y2,
            x1,
            y2,
            x1,
            y2 - radius,
            x1,
            y1 + radius,
            x1,
            y1,
        ]
        return self.canvas.create_polygon(points, smooth=True, splinesteps=16, **kwargs)


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.wm_title("Heat Conduction Dashboard")
        self.root.configure(bg=THEME["bg"])
        self.root.title("Heat Conduction Dashboard")
        self.root.geometry("1360x900")
        self.root.minsize(1180, 760)
        self.root.grid_anchor("center")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.configure_theme()

        self.pcm: Optional[backend.PCM] = None
        self.L = 0.0
        self.t_max = 0.0
        self.material = "Regolith"
        self.simulation_ready = False

        self.solution_type = tk.StringVar(self.root)
        self.solution_type.set("Analytical")
        self.gold_standard = None
        self.gold_standard_temp_array: Optional[np.ndarray] = None

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
        self.temperature_solutions: Dict[str, Optional[np.ndarray]] = {
            "Numerical": None,
            "Analytical": None,
            "Implicit": None,
        }
        self.temp_mask_array: Optional[np.ndarray] = None
        self.bound_mask_array: Optional[np.ndarray] = None
        self.x: Optional[np.ndarray] = None
        self.y_T: Optional[np.ndarray] = None
        self.y_B: Optional[np.ndarray] = None
        self.x_boundary: Optional[np.ndarray] = None
        self.x_grid: Optional[np.ndarray] = None
        self.t_grid: Optional[np.ndarray] = None
        self.T_arr: Optional[np.ndarray] = None
        self.H_arr: Optional[np.ndarray] = None
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
        self.true_boundary_times: Optional[np.ndarray] = None
        self.Temperature_pred: Optional[np.ndarray] = None
        self.Boundary_pred: Optional[np.ndarray] = None
        self.loss_values: List[float] = []
        self.scaled_accuracy_values = {"scaled_accuracy_T": [], "scaled_accuracy_B": []}
        self.raw_accuracy_values = {"raw_accuracy_T": [], "raw_accuracy_B": []}
        self.model = None
        self.batch_size = 1
        self.t_idx = 0

        self.animation_running = False
        self.animation_after_id: Optional[str] = None
        self.animation_interval_ms = 60
        self.animation_frame_step = 10
        self.surface_time_slice_artist = None
        self.terrain_x_values: Optional[np.ndarray] = None
        self.terrain_y_values: Optional[np.ndarray] = None
        self.terrain_profile_values: Optional[np.ndarray] = None
        self.terrain_surface_values: Optional[np.ndarray] = None
        self.last_commands = ""

        self.build_dashboard()

    def configure_theme(self) -> None:
        self.font_family = "Segoe UI"
        self.body_font = tkfont.Font(family=self.font_family, size=12)
        self.small_font = tkfont.Font(family=self.font_family, size=10)
        self.label_font = tkfont.Font(family=self.font_family, size=12)
        self.heading_font = tkfont.Font(family=self.font_family, size=18, weight="bold")
        self.section_font = tkfont.Font(family=self.font_family, size=13, weight="bold")
        self.button_font = tkfont.Font(family=self.font_family, size=12, weight="bold")

        self.root.option_add("*Font", self.body_font)
        self.root.option_add("*TCombobox*Listbox.font", self.body_font)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", font=self.body_font)
        style.configure("App.TFrame", background=THEME["bg"])
        style.configure("Card.TFrame", background=THEME["surface"])
        style.configure(
            "TLabel",
            background=THEME["bg"],
            foreground=THEME["text"],
            font=self.body_font,
        )
        style.configure(
            "Card.TLabel",
            background=THEME["surface"],
            foreground=THEME["text"],
            font=self.body_font,
        )
        style.configure(
            "Muted.Card.TLabel",
            background=THEME["surface"],
            foreground=THEME["muted"],
            font=self.small_font,
        )
        style.configure(
            "Header.TLabel",
            background=THEME["bg"],
            foreground=THEME["text"],
            font=self.heading_font,
        )
        style.configure(
            "Section.Card.TLabel",
            background=THEME["surface"],
            foreground=THEME["text"],
            font=self.section_font,
        )
        style.configure(
            "Status.TLabel",
            background=THEME["surface"],
            foreground=THEME["muted"],
            padding=(14, 8),
            font=self.small_font,
        )
        style.configure(
            "TNotebook",
            background=THEME["bg"],
            borderwidth=0,
            tabmargins=(20, 8, 20, 0),
        )
        style.configure(
            "TNotebook.Tab",
            background=THEME["surface_alt"],
            foreground=THEME["muted"],
            padding=(22, 11),
            font=self.button_font,
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", THEME["surface"]), ("active", "#e7edf6")],
            foreground=[("selected", THEME["primary"]), ("active", THEME["text"])],
            padding=[("selected", (22, 11)), ("active", (22, 11))],
        )
        style.configure(
            "TButton",
            background=THEME["surface_alt"],
            foreground=THEME["text"],
            borderwidth=0,
            focusthickness=0,
            focuscolor=THEME["surface_alt"],
            padding=(16, 9),
            font=self.button_font,
        )
        style.map(
            "TButton",
            background=[
                ("disabled", "#e5e7eb"),
                ("pressed", "#d6e2f2"),
                ("active", "#dde8f7"),
            ],
            foreground=[("disabled", "#9ca3af")],
        )
        style.configure(
            "Primary.TButton",
            background=THEME["primary"],
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            focuscolor=THEME["primary"],
            padding=(18, 11),
            font=self.button_font,
        )
        style.map(
            "Primary.TButton",
            background=[
                ("disabled", "#93c5fd"),
                ("pressed", "#1e40af"),
                ("active", THEME["primary_hover"]),
            ],
            foreground=[("disabled", "#eff6ff"), ("active", "#ffffff")],
        )
        style.configure(
            "TEntry",
            fieldbackground="#f8fafc",
            background="#f8fafc",
            foreground=THEME["text"],
            bordercolor=THEME["border"],
            lightcolor=THEME["border"],
            darkcolor=THEME["border"],
            padding=(10, 8),
            insertcolor=THEME["text"],
            font=self.body_font,
        )
        style.configure(
            "TCombobox",
            fieldbackground="#f8fafc",
            background="#f8fafc",
            foreground=THEME["text"],
            bordercolor=THEME["border"],
            arrowcolor=THEME["primary"],
            padding=(10, 8),
            font=self.body_font,
        )
        style.configure(
            "TRadiobutton",
            background=THEME["surface"],
            foreground=THEME["text"],
            font=self.body_font,
            padding=(4, 4),
        )
        style.map(
            "TRadiobutton",
            background=[("active", THEME["surface"])],
            foreground=[("disabled", "#9ca3af")],
        )

        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
            }
        )

    def build_dashboard(self) -> None:
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_columnconfigure(0, weight=1)

        header = ttk.Frame(self.root, style="App.TFrame", padding=(24, 18, 24, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ttk.Label(
            header, text="Heat Conduction Dashboard", style="Header.TLabel"
        ).grid(row=0, column=0, sticky="w")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 10))

        self.setup_tab = ttk.Frame(self.notebook, style="App.TFrame")
        self.simulation_tab = ttk.Frame(self.notebook, style="App.TFrame")
        self.pinn_tab = ttk.Frame(self.notebook, style="App.TFrame")
        self.messages_tab = ttk.Frame(self.notebook, style="App.TFrame")

        self.notebook.add(self.setup_tab, text="Setup")
        self.notebook.add(self.simulation_tab, text="Simulation")
        self.notebook.add(self.pinn_tab, text="PINN")
        self.notebook.add(self.messages_tab, text="Messages")

        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            style="Status.TLabel",
        )
        status_bar.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 14))

        self.build_setup_tab()
        self.build_simulation_tab()
        self.build_pinn_tab()
        self.build_messages_tab()
        self._set_simulation_controls_enabled(False)
        self._set_pinn_controls_enabled(False)
        self.notebook.select(self.setup_tab)

    def build_setup_tab(self) -> None:
        self.setup_tab.grid_rowconfigure(0, weight=1)
        self.setup_tab.grid_columnconfigure(0, weight=1)

        content = ttk.Frame(self.setup_tab, style="App.TFrame", padding=(26, 22, 26, 26))
        content.grid(row=0, column=0, sticky="nsew")
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)

        setup_card = RoundedFrame(content, radius=22, padding=28)
        setup_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        panel = setup_card.content
        panel.grid_columnconfigure(1, weight=1)

        custom_card = RoundedFrame(content, radius=22, padding=28)
        custom_card.grid(row=0, column=1, sticky="nsew", padx=(14, 0))
        custom_panel = custom_card.content
        custom_panel.grid_columnconfigure(1, weight=1)

        self.setup_mode_var = tk.IntVar(value=1)
        ttk.Label(
            panel, text="Simulation Setup", style="Section.Card.TLabel"
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 18))

        ttk.Label(panel, text="Mode", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Radiobutton(
            panel,
            text="Choose material",
            variable=self.setup_mode_var,
            value=1,
            command=self.update_setup_mode,
        ).grid(row=1, column=1, sticky="w", padx=(16, 0), pady=(0, 8))
        ttk.Radiobutton(
            panel,
            text="Custom properties",
            variable=self.setup_mode_var,
            value=0,
            command=self.update_setup_mode,
        ).grid(row=1, column=2, sticky="w", padx=(16, 0), pady=(0, 8))

        self.setup_x_var = tk.StringVar(value="0.2")
        self.setup_t_var = tk.StringVar(value="2")
        self.setup_material_var = tk.StringVar(value="Regolith")
        self.setup_k_var = tk.StringVar(value="1.0")
        self.setup_c_var = tk.StringVar(value="800.0")
        self.setup_rho_var = tk.StringVar(value="1800.0")
        self.setup_tm_var = tk.StringVar(value="1373.0")
        self.setup_lh_var = tk.StringVar(value="1429.0")

        fields = (
            ("Length of the bar", self.setup_x_var),
            ("Final time", self.setup_t_var),
        )
        ttk.Label(panel, text="Domain", style="Section.Card.TLabel").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(24, 10)
        )
        for row, (label, var) in enumerate(fields, start=3):
            ttk.Label(panel, text=label, style="Card.TLabel").grid(
                row=row, column=0, sticky="w", pady=8
            )
            ttk.Entry(panel, textvariable=var, width=22).grid(
                row=row, column=1, columnspan=2, sticky="ew", padx=(16, 0), pady=8
            )

        ttk.Label(panel, text="Material", style="Section.Card.TLabel").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(28, 10)
        )
        self.material_frame = ttk.Frame(panel, style="Card.TFrame")
        self.material_frame.grid(row=6, column=0, columnspan=3, sticky="ew")
        self.material_frame.grid_columnconfigure(0, weight=1)
        self.material_selector = ttk.Combobox(
            self.material_frame,
            textvariable=self.setup_material_var,
            values=list(MATERIAL_CLASSES.keys()),
            state="readonly",
            width=24,
        )
        self.material_selector.grid(row=0, column=0, sticky="ew")

        ttk.Button(
            panel,
            text="Run Simulation",
            command=self.submit_setup_form,
            style="Primary.TButton",
        ).grid(row=7, column=0, columnspan=3, sticky="ew", pady=(36, 0))

        ttk.Label(
            custom_panel, text="Custom Properties", style="Section.Card.TLabel"
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 18))
        self.custom_frame = ttk.Frame(custom_panel, style="Card.TFrame")
        self.custom_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.custom_frame.grid_columnconfigure(1, weight=1)
        custom_fields = (
            ("Conductivity", self.setup_k_var),
            ("Specific Heat", self.setup_c_var),
            ("Density", self.setup_rho_var),
            ("Melting Temperature", self.setup_tm_var),
            ("Latent Heat", self.setup_lh_var),
        )
        for row, (label, var) in enumerate(custom_fields):
            ttk.Label(self.custom_frame, text=label, style="Card.TLabel").grid(
                row=row, column=0, sticky="w", pady=8
            )
            ttk.Entry(self.custom_frame, textvariable=var, width=20).grid(
                row=row, column=1, sticky="ew", padx=(16, 0), pady=8
            )

        self.update_setup_mode()

    def update_setup_mode(self) -> None:
        custom_state = "normal" if self.setup_mode_var.get() == 0 else "disabled"
        material_state = "readonly" if self.setup_mode_var.get() == 1 else "disabled"
        for child in self.custom_frame.winfo_children():
            if isinstance(child, ttk.Entry):
                child.configure(state=custom_state)
        for child in self.material_frame.winfo_children():
            if isinstance(child, ttk.Combobox):
                child.configure(state=material_state)

    def submit_setup_form(self) -> None:
        try:
            self.L = float(self.setup_x_var.get())
            self.t_max = float(self.setup_t_var.get())
            if self.L <= 0 or self.t_max <= 0:
                raise ValueError("Length and final time must be positive.")

            if self.setup_mode_var.get() == 0:
                self.k = float(self.setup_k_var.get())
                self.c = float(self.setup_c_var.get())
                self.rho = float(self.setup_rho_var.get())
                self.T_m = float(self.setup_tm_var.get())
                self.LH = float(self.setup_lh_var.get())
                self.material = "Custom"
                self.pcm = backend.customPCM(self.k, self.c, self.rho, self.T_m, self.LH)
            else:
                self.material = self.setup_material_var.get()
                material_cls = MATERIAL_CLASSES.get(self.material)
                if material_cls is None:
                    raise ValueError(f"Unsupported material: {self.material}")
                self.pcm = material_cls()
        except Exception as e:
            self.set_status(f"Setup error: {e}", error=True)
            self.notebook.select(self.messages_tab)
            return

        print(f"Debug: Selected material = {self.material}")
        self.stop_animation()
        self.simulation_ready = True
        self._set_simulation_controls_enabled(True)
        self._set_pinn_controls_enabled(True)
        self.calcAll()
        try:
            if self.t_grid is not None and len(self.t_grid) > 0 and self.pcm is not None:
                self.time_scale.config(
                    to=float(self.t_grid[-1]), resolution=float(self.pcm.dt)
                )
                self.time_scale.set(float(self.t_grid[0]))
        except Exception as e:
            self.set_status(f"Warning: failed to sync time slider: {e}", error=True)

        self.solution_type.set("Analytical")
        self.update_solution_type()
        self.update_plots()
        self.notebook.select(self.simulation_tab)
        self.set_status(f"Simulation ready: {self.material}")

    def build_simulation_tab(self) -> None:
        self.simulation_tab.grid_rowconfigure(1, weight=1)
        self.simulation_tab.grid_columnconfigure(0, weight=1)

        controls_card = RoundedFrame(self.simulation_tab, radius=20, padding=18)
        controls_card.configure(height=108)
        controls_card.grid_propagate(False)
        controls_card.grid(row=0, column=0, sticky="ew", padx=26, pady=(22, 12))
        controls = controls_card.content
        controls.columnconfigure(3, weight=1)

        ttk.Label(controls, text="Solution", style="Card.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.solution_menu = ttk.Combobox(
            controls,
            textvariable=self.solution_type,
            values=("Analytical", "Implicit", "Numerical", "PINN"),
            state="readonly",
            width=16,
        )
        self.solution_menu.grid(row=0, column=1, sticky="w", padx=(10, 18))
        self.solution_type.trace_add("write", self.handle_solution_type_change)

        ttk.Label(controls, text="Time", style="Card.TLabel").grid(
            row=0, column=2, sticky="e"
        )
        self.time_scale = tk.Scale(
            controls,
            from_=0,
            to=1,
            resolution=1,
            orient=tk.HORIZONTAL,
            command=self.update_time,
            length=420,
            bg=THEME["surface"],
            fg=THEME["text"],
            troughcolor=THEME["surface_alt"],
            activebackground=THEME["primary"],
            highlightthickness=0,
            bd=0,
            font=self.small_font,
        )
        self.time_scale.grid(row=0, column=3, sticky="ew", padx=(10, 14))

        self.step_back_button = ttk.Button(
            controls, text="Step Back", command=lambda: self.step_animation(-1)
        )
        self.step_back_button.grid(row=0, column=4, sticky="ew", padx=2)
        self.play_button = ttk.Button(
            controls, text="Play", command=self.toggle_animation
        )
        self.play_button.grid(row=0, column=5, sticky="ew", padx=2)
        self.step_forward_button = ttk.Button(
            controls, text="Step Forward", command=lambda: self.step_animation(1)
        )
        self.step_forward_button.grid(row=0, column=6, sticky="ew", padx=2)

        self.animation_speed_var = tk.IntVar(value=self.animation_interval_ms)
        ttk.Label(controls, text="Speed", style="Card.TLabel").grid(
            row=0, column=7, sticky="e", padx=(14, 0)
        )
        self.animation_speed_scale = tk.Scale(
            controls,
            from_=50,
            to=1000,
            resolution=50,
            orient=tk.HORIZONTAL,
            variable=self.animation_speed_var,
            length=150,
            bg=THEME["surface"],
            fg=THEME["text"],
            troughcolor=THEME["surface_alt"],
            activebackground=THEME["primary"],
            highlightthickness=0,
            bd=0,
            font=self.small_font,
        )
        self.animation_speed_scale.grid(row=0, column=8, sticky="ew", padx=(10, 0))

        plots_frame = ttk.Frame(
            self.simulation_tab, style="App.TFrame", padding=(26, 0, 26, 10)
        )
        plots_frame.grid(row=1, column=0, sticky="nsew")
        for col in range(3):
            plots_frame.grid_columnconfigure(col, weight=1)
        plots_frame.grid_rowconfigure(0, weight=1)

        plot_cards = []
        for col in range(3):
            card = RoundedFrame(plots_frame, radius=20, padding=12)
            card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 0 if col == 2 else 8))
            card.content.grid_rowconfigure(0, weight=1)
            card.content.grid_columnconfigure(0, weight=1)
            plot_cards.append(card)

        self.fig1, self.ax1 = plt.subplots(figsize=(4, 3), dpi=100)
        self.ax1.set_title("T(E) Line Plot")
        self.ax1.set_xlabel("E (kJ/kg)")
        self.ax1.set_ylabel("T")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, plot_cards[0].content)
        self.canvas1.draw()

        self.fig3 = Figure(figsize=(5, 5), dpi=100)
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.grid(True)
        self.ax3.set_xlabel("x [m]")
        self.ax3.set_ylabel("T [C]")
        self.ax3.set_title("T(x)")
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=plot_cards[1].content)
        self.canvas3.draw()

        self.fig2 = Figure(figsize=(4, 3), dpi=100)
        self.ax2 = cast(Axes3D, self.fig2.add_subplot(111, projection="3d"))
        self.ax2.set_title("Thermal Terrain Snapshot")
        self.ax2.set_xlabel("x [m]")
        self.ax2.set_ylabel("y [m]")
        self.ax2.set_zlabel("Temperature [C]")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, plot_cards[2].content)
        self.canvas2.draw()

        for canvas in (self.canvas1, self.canvas3, self.canvas2):
            canvas.get_tk_widget().configure(
                bg=THEME["surface"], highlightthickness=0, bd=0
            )
        self.canvas1.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas3.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas2.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        (self.line1,) = self.ax1.plot([], [], "r-", label="Line Plot 1")
        (self.line3,) = self.ax3.plot([], [], "g-", label="Line Plot 3")
        self.ax3.legend()

        self.energy_label = ttk.Label(
            self.simulation_tab,
            text="",
            anchor="center",
            wraplength=1100,
            style="Status.TLabel",
        )
        self.energy_label.grid(row=2, column=0, sticky="ew", padx=26, pady=(0, 18))

    def build_pinn_tab(self) -> None:
        self.pinn_tab.grid_rowconfigure(1, weight=1)
        self.pinn_tab.grid_columnconfigure(0, weight=1)

        controls_card = RoundedFrame(self.pinn_tab, radius=20, padding=18)
        controls_card.configure(height=108)
        controls_card.grid_propagate(False)
        controls_card.grid(row=0, column=0, sticky="ew", padx=26, pady=(22, 12))
        controls = controls_card.content
        controls.columnconfigure(3, weight=1)

        ttk.Label(controls, text="Gold Standard", style="Card.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.gold_standard_var = tk.StringVar(value="Analytical")
        self.gold_standard_menu = ttk.Combobox(
            controls,
            textvariable=self.gold_standard_var,
            values=("Analytical", "Implicit", "Numerical"),
            state="readonly",
            width=16,
        )
        self.gold_standard_menu.grid(row=0, column=1, sticky="w", padx=(10, 12))
        self.train_pinn_button = ttk.Button(
            controls,
            text="Train PINN",
            command=lambda: self.handle_gold_standard_selection(
                self.gold_standard_var.get()
            ),
            style="Primary.TButton",
        )
        self.train_pinn_button.grid(row=0, column=2, sticky="w", padx=(8, 0))

        self.fig_PINN = Figure(figsize=(15, 5), dpi=100)
        self.pinn_canvas_card = RoundedFrame(self.pinn_tab, radius=20, padding=12)
        self.pinn_canvas_card.grid(
            row=1, column=0, sticky="nsew", padx=26, pady=(0, 10)
        )
        self.pinn_canvas_frame = self.pinn_canvas_card.content
        self.pinn_canvas_frame.grid_rowconfigure(0, weight=1)
        self.pinn_canvas_frame.grid_columnconfigure(0, weight=1)
        self.pinn_canvas = FigureCanvasTkAgg(self.fig_PINN, master=self.pinn_canvas_frame)
        self.pinn_canvas.get_tk_widget().configure(
            bg=THEME["surface"], highlightthickness=0, bd=0
        )
        self.pinn_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.pinn_toolbar_frame = ttk.Frame(
            self.pinn_tab, style="App.TFrame", padding=(26, 0, 26, 14)
        )
        self.pinn_toolbar_frame.grid(row=2, column=0, sticky="ew")
        self.pinn_toolbar = NavigationToolbar2Tk(self.pinn_canvas, self.pinn_toolbar_frame)
        self.pinn_toolbar.update()

    def build_messages_tab(self) -> None:
        self.messages_tab.grid_rowconfigure(1, weight=1)
        self.messages_tab.grid_columnconfigure(0, weight=1)

        button_card = RoundedFrame(self.messages_tab, radius=20, padding=18)
        button_card.configure(height=108)
        button_card.grid_propagate(False)
        button_card.grid(row=0, column=0, sticky="ew", padx=26, pady=(22, 12))
        button_bar = button_card.content
        button_bar.grid_columnconfigure(1, weight=1)
        ttk.Label(button_bar, text="Messages", style="Section.Card.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.copy_commands_button = ttk.Button(
            button_bar,
            text="Copy Commands",
            command=self.copy_last_commands,
        )
        self.copy_commands_button.grid(row=0, column=1, sticky="e")

        text_card = RoundedFrame(self.messages_tab, radius=20, padding=14)
        text_card.grid(row=1, column=0, sticky="nsew", padx=26, pady=(0, 18))
        text_frame = text_card.content
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        self.messages_text = tk.Text(
            text_frame,
            wrap="word",
            height=12,
            bg="#f8fafc",
            fg=THEME["text"],
            insertbackground=THEME["text"],
            relief="flat",
            bd=0,
            padx=16,
            pady=14,
            font=self.body_font,
        )
        self.messages_text.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(text_frame, command=self.messages_text.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.messages_text["yscrollcommand"] = yscroll.set
        self.messages_text.configure(state="disabled")

    def _set_simulation_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            getattr(self, "solution_menu", None),
            getattr(self, "time_scale", None),
            getattr(self, "step_back_button", None),
            getattr(self, "play_button", None),
            getattr(self, "step_forward_button", None),
            getattr(self, "animation_speed_scale", None),
        ):
            if widget is not None:
                try:
                    state = (
                        "readonly"
                        if enabled and isinstance(widget, ttk.Combobox)
                        else "normal"
                        if enabled
                        else "disabled"
                    )
                    widget.configure(state=state)
                except tk.TclError:
                    pass

    def _set_pinn_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            getattr(self, "gold_standard_menu", None),
            getattr(self, "train_pinn_button", None),
        ):
            if widget is not None:
                try:
                    state = (
                        "readonly"
                        if enabled and isinstance(widget, ttk.Combobox)
                        else "normal"
                        if enabled
                        else "disabled"
                    )
                    widget.configure(state=state)
                except tk.TclError:
                    pass

    def set_status(self, message: str, error: bool = False) -> None:
        prefix = "Error: " if error and not message.startswith("Error") else ""
        text = prefix + message
        self.status_var.set(text)
        try:
            self.messages_text.configure(state="normal")
            self.messages_text.insert("end", text + "\n")
            self.messages_text.see("end")
            self.messages_text.configure(state="disabled")
        except Exception:
            print(text)

    def show_commands_panel(self, title, instructions, commands):
        self.last_commands = commands
        content = f"{title}\n\n{instructions}\n\n{commands}\n"
        self.messages_text.configure(state="normal")
        self.messages_text.delete("1.0", "end")
        self.messages_text.insert("1.0", content)
        self.messages_text.configure(state="disabled")
        self.notebook.select(self.messages_tab)
        self.set_status(title)

    def copy_last_commands(self) -> None:
        if not self.last_commands:
            self.set_status("No commands available to copy.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.last_commands)
        self.root.update()
        self.set_status("Commands copied to clipboard.")

    # Call the energy calculation method and update the label with the result string
    def update_energy_label(self):
        # Example: Assuming self.H_arr_final contains the enthalpy values needed for the energy calculation
        if self.H_arr_final is not None and self.pcm is not None:
            # Use the simulated time span (seconds) to set the load duration in hours
            try:
                sim_hours = float(self.t_grid[-1]) / 3600.0 if self.t_grid is not None else None
            except Exception:
                sim_hours = None

            result_string = self.pcm.calcEnergySufficiency(
                self.H_arr_final,
                settlement_hours=sim_hours,
            )
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
        self.show_gold_standard_controls()

    def handle_solution_type_change(self, *args):
        if not self.simulation_ready:
            return
        selected_solution = self.solution_type.get()
        if selected_solution == "PINN":
            self.show_gold_standard_controls()
        else:
            self.hide_gold_standard_controls()
            self.update_solution_type()
            self.update_plots()

    def show_gold_standard_controls(self):
        self.notebook.select(self.pinn_tab)
        self.set_status("Select a gold standard and train the PINN.")

    def hide_gold_standard_controls(self):
        return

    def calcAll(self):
        print("Debug: Starting calcAll")
        pcm = self.pcm
        if pcm is None:
            raise RuntimeError("Select a material before calculating the simulation.")

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
        ) = pcm.generate_data(self.L, self.t_max)
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
        ) = pcm.explicitNumerical(
            self.x_grid, self.t_grid, self.T_arr.copy(), pcm
        )

        phase_mask_final, boundary_mask_final = pcm.update_phase_mask(
            self.T_arr_numerical[:, -1], pcm
        )
        self.temp_mask_arrays["Numerical"] = phase_mask_final
        self.bound_mask_arrays["Numerical"] = boundary_mask_final

        # Check for empty mask arrays
        if np.all(phase_mask_final == 0) or np.all(boundary_mask_final == 0):
            print(
                "Warning: Numerical solution mask arrays are empty. Check the temperature evolution and masking logic."
            )

        self.temperature_solutions["Numerical"] = self.T_arr_numerical

        if self.T_arr_numerical is not None:
            numerical_boundary_positions = self.compute_boundary_positions_from_temperature(
                self.T_arr_numerical,
                mask_series=_numerical_boundary_mask_series,
                fallback_indices=moving_boundary_indices_numerical,
            )
            self.moving_boundary_positions["Numerical"] = numerical_boundary_positions

        # Analytical solution
        print("Debug: Calculating analytical solution")
        self.T_arr_analytical = pcm.analyticalSol(
            self.x_grid, self.t_grid, pcm
        )
        analytical_phase_mask, analytical_boundary_mask = pcm.update_phase_mask(
            self.T_arr_analytical[:, -1], pcm
        )
        self.temp_mask_arrays["Analytical"] = analytical_phase_mask
        self.bound_mask_arrays["Analytical"] = analytical_boundary_mask
        self.temperature_solutions["Analytical"] = self.T_arr_analytical

        analytical_boundary_positions = pcm.stefan_boundary_location(
            self.t_grid, surface_temp=pcm.T_m + 10.0
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
            ) = pcm.implicitSol(
                self.x_grid, self.t_grid, self.T_arr.copy(), self.H_arr.copy(), pcm
            )

            # Use update_phase_mask after obtaining T_arr_implicit
            implicit_phase_mask, implicit_boundary_mask = pcm.update_phase_mask(
                self.T_arr_implicit[:, -1], pcm
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
                    np.clip(idx_array, 0, nx - 1).astype(float) * pcm.dx
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
        if not self.simulation_ready:
            return
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
            self.show_gold_standard_controls()
            return

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

        if (
            self.pcm is None
            or self.x is None
            or self.x_boundary is None
            or self.x_grid is None
            or self.t_grid is None
        ):
            print("Error: Run a simulation before preparing PINN training data.")
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
        sanitized_gold = self._sanitize_boundary_curve(boundary_positions)
        self.gold_standard_boundary_per_time = (
            sanitized_gold if sanitized_gold is not None else boundary_positions.copy()
        )

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

                self.set_status(
                    "TensorFlow/Keras not available. See Messages tab for setup commands.",
                    error=True,
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
            self.set_status(
                "TensorFlow/Keras not available. See Messages tab for setup commands.",
                error=True,
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
                sample_count = prediction_inputs["temperature_input"].shape[0]
                print(
                    f"Generating PINN predictions for visualization "
                    f"({sample_count} samples, batch size {prediction_batch_size})...",
                    flush=True,
                )
                model_output_full = self.model.predict(
                    prediction_inputs,
                    batch_size=prediction_batch_size,
                    verbose=1,
                )
                print("Full-resolution predictions complete.", flush=True)
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
        if (
            self.T_arr_to_display is None
            or self.x_grid is None
            or self.t_grid is None
        ):
            print("Warning: No solution data available for plotting.")
            return
        T_arr_to_display = self.T_arr_to_display

        enthalpy_curve, temperature_curve = self.build_enthalpy_temperature_curve(
            T_arr_to_display
        )

        self.t_idx = max(
            0,
            min(self.time_value_to_index(), T_arr_to_display.shape[1] - 1),
        )

        # Update T(x) plot
        self.update_line_plot3(self.x_grid, T_arr_to_display[:, self.t_idx])

        # Update 3D surface plot
        self.update_surface_plot(self.x_grid, self.t_grid, T_arr_to_display)

        # Update T(E) plot using a monotonic enthalpy-temperature relation with a latent-heat plateau
        self.update_line_plot(enthalpy_curve, temperature_curve)

        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()

        # Update the energy sufficiency message
        self.update_energy_label()

    def time_value_to_index(self, value=None) -> int:
        try:
            t_val = float(self.time_scale.get() if value is None else value)
        except Exception:
            t_val = 0.0

        if self.t_grid is not None and len(self.t_grid) > 0:
            t_values = np.asarray(self.t_grid, dtype=float)
            finite_mask = np.isfinite(t_values)
            if finite_mask.any():
                valid_indices = np.flatnonzero(finite_mask)
                nearest_valid = int(np.argmin(np.abs(t_values[finite_mask] - t_val)))
                return int(valid_indices[nearest_valid])

        try:
            dt = float(getattr(self.pcm, "dt", 1.0))
            max_idx = self.T_arr_to_display.shape[1] - 1 if self.T_arr_to_display is not None else 0
            return max(0, min(int(round(t_val / max(dt, 1e-12))), max_idx))
        except Exception:
            return 0

    def build_enthalpy_temperature_curve(
        self, temperature_surface: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return specific enthalpy (kJ/kg) and temperature with E=0 at solid melting.

        In J/kg, E=c_solid*(T-Tm) below melting, E spans [0, LH]
        at Tm, and E=LH+c_liquid*(T-Tm) above melting.
        """
        tm = float(getattr(self.pcm, "T_m", 0.0))
        c_solid = max(float(getattr(self.pcm, "c_solid", getattr(self.pcm, "c", 1.0))), 1e-12)
        c_liquid = max(float(getattr(self.pcm, "c_liquid", getattr(self.pcm, "c", 1.0))), 1e-12)
        latent_heat = max(float(getattr(self.pcm, "LH", 0.0)), 0.0)

        # Iron stores heat capacities in J/(g K); the other materials use J/(kg K).
        # Normalize the graph without changing the simulation's material properties.
        if isinstance(self.pcm, backend.Iron):
            c_solid *= 1000.0
            c_liquid *= 1000.0

        temperatures = np.asarray(temperature_surface, dtype=float)
        finite_temperatures = temperatures[np.isfinite(temperatures)]
        if finite_temperatures.size:
            t_min = float(np.min(finite_temperatures))
            t_max = float(np.max(finite_temperatures))
        else:
            t_min = tm
            t_max = tm

        # This material curve must show both phases even if the simulation only
        # reaches one of them. A tiny extension above Tm hides the liquid branch.
        branch_temperature_span = max(0.25 * (t_max - t_min), 25.0)
        t_min = min(t_min, tm - branch_temperature_span)
        t_max = max(t_max, tm + branch_temperature_span)

        solid_enthalpy = np.linspace(
            c_solid * (t_min - tm),
            0.0,
            120,
            endpoint=True,
        )
        solid_temps = tm + solid_enthalpy / c_solid

        plateau_points = 60 if latent_heat > 0 else 1
        plateau_enthalpy = np.linspace(0.0, latent_heat, plateau_points)
        plateau_temps = np.full_like(plateau_enthalpy, tm)

        liquid_enthalpy = np.linspace(
            latent_heat,
            latent_heat + c_liquid * (t_max - tm),
            180,
            endpoint=True,
        )
        liquid_temps = tm + (liquid_enthalpy - latent_heat) / c_liquid

        enthalpy_curve = np.concatenate(
            [solid_enthalpy, plateau_enthalpy[1:], liquid_enthalpy[1:]]
        )
        temperature_curve = np.concatenate(
            [solid_temps, plateau_temps[1:], liquid_temps[1:]]
        )
        return enthalpy_curve * PLOT_ENTHALPY_SCALE, temperature_curve

    def show_PINN_plots(self, show_gold_standard=True, custom_cmap="coolwarm"):
        self.render_pinn_results(show_gold_standard, custom_cmap)

    def render_pinn_results(self, show_gold_standard=True, custom_cmap="coolwarm"):
        # Ensure the necessary data is available
        if self.Temperature_pred is None or self.Boundary_pred is None:
            print("Prediction data is not available.")
            self.set_status("Prediction data is not available.", error=True)
            return

        if self.x_grid is None or self.t_grid is None:
            self.set_status("Simulation grids are not available. Run a simulation first.", error=True)
            return

        if show_gold_standard and self.gold_standard_boundary_per_time is None:
            print("Gold standard data is not set.")
            self.set_status("Gold standard data is not set.", error=True)
            return

        # Reset the figure before drawing new subplots
        self.fig_PINN.clf()
        self.notebook.select(self.pinn_tab)

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

        # Save the composite PINN figure for reference
        try:
            self.fig_PINN.savefig("PINN_temperature_surface.png", dpi=300)
        except Exception as e:
            print(f"Warning: Failed to save PINN figure ({e})")

        # Draw the canvas with all the plots
        self.pinn_canvas.draw()
        self.set_status("PINN plots updated.")

    def update_line_plot(
        self, enthalpy_values: np.ndarray, temperature_values: np.ndarray
    ) -> None:
        enthalpy_values = np.asarray(enthalpy_values, dtype=float).ravel()
        temperature_values = np.asarray(temperature_values, dtype=float).ravel()

        if enthalpy_values.size != temperature_values.size:
            min_len = min(enthalpy_values.size, temperature_values.size)
            enthalpy_values = enthalpy_values[:min_len]
            temperature_values = temperature_values[:min_len]

        sort_idx = np.argsort(enthalpy_values)
        enthalpy_sorted = enthalpy_values[sort_idx]
        temperature_sorted = temperature_values[sort_idx]

        self.line1.set_data(enthalpy_sorted, temperature_sorted)
        self.ax1.relim()
        self.ax1.autoscale_view()

        if enthalpy_sorted.size:
            x_min = float(np.nanmin(enthalpy_sorted))
            x_max = float(np.nanmax(enthalpy_sorted))
            x_span = x_max - x_min
            x_pad = 0.1 * (x_span if x_span > 0 else max(abs(x_min), abs(x_max), 1.0))
            self.ax1.set_xlim(x_min - x_pad, x_max + x_pad)

        if temperature_sorted.size:
            y_min = float(np.nanmin(temperature_sorted))
            y_max = float(np.nanmax(temperature_sorted))
            y_span = y_max - y_min
            y_pad = 0.1 * (y_span if y_span > 0 else max(abs(y_min), abs(y_max), 1.0))

            tm = float(getattr(self.pcm, "T_m", 0.0))
            above_tm_margin = max(0.05 * max(tm, 1.0), 25.0)
            y_upper = max(y_max + y_pad, tm + above_tm_margin)
            self.ax1.set_ylim(y_min - y_pad, y_upper)

        self.ax1.xaxis.set_major_locator(MaxNLocator(nbins=6))
        self.ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
        self.canvas1.draw()

    def update_line_plot3(self, x: np.ndarray, y: np.ndarray) -> None:
        self.line3.set_data(x, y)
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.canvas3.print_figure("Tx.png")
        self.canvas3.draw()

    def update_time(self, event) -> None:
        if self.T_arr_to_display is None or self.x_grid is None:
            return
        T_arr_to_display = self.T_arr_to_display

        self.t_idx = max(
            0,
            min(self.time_value_to_index(event), T_arr_to_display.shape[1] - 1),
        )

        # Update the T(x) plot
        self.line3.set_xdata(self.x_grid)
        self.line3.set_ydata(T_arr_to_display[:, self.t_idx])
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.canvas3.draw()
        if self.x_grid is not None and self.t_grid is not None:
            self.update_surface_plot(self.x_grid, self.t_grid, T_arr_to_display)

    def update_surface_time_slice(self) -> None:
        if (
            self.T_arr_to_display is None
            or self.x_grid is None
        ):
            return

        try:
            if self.surface_time_slice_artist is not None:
                self.surface_time_slice_artist.remove()
        except Exception:
            pass
        self.surface_time_slice_artist = None

        t_idx = max(0, min(int(getattr(self, "t_idx", 0)), self.T_arr_to_display.shape[1] - 1))
        terrain_x_values = getattr(self, "terrain_x_values", None)
        terrain_y_values = getattr(self, "terrain_y_values", None)
        terrain_profile_values = getattr(self, "terrain_profile_values", None)
        x_values = (
            np.asarray(terrain_x_values, dtype=float)
            if terrain_x_values is not None
            else np.asarray(self.x_grid, dtype=float)
        )
        terrain_center = (
            np.asarray(terrain_y_values, dtype=float)
            if terrain_y_values is not None
            else np.zeros_like(x_values, dtype=float)
        )
        z_values = (
            np.asarray(terrain_profile_values, dtype=float)
            if terrain_profile_values is not None
            else np.asarray(self.T_arr_to_display[:, t_idx], dtype=float)
        )
        (self.surface_time_slice_artist,) = self.ax2.plot(
            x_values,
            terrain_center,
            z_values,
            color="black",
            linewidth=3,
            label="Current terrain ridge",
        )
        try:
            self.ax2.legend(loc="upper right")
        except Exception:
            pass
        self.canvas2.draw_idle()

    def toggle_animation(self) -> None:
        if not self.simulation_ready or self.T_arr_to_display is None:
            return
        if self.animation_running:
            self.stop_animation()
            return
        self.animation_running = True
        self.play_button.configure(text="Pause")
        self.advance_animation_frame()

    def advance_animation_frame(self) -> None:
        if not self.animation_running or self.T_arr_to_display is None:
            return
        next_idx = int(getattr(self, "t_idx", 0)) + int(self.animation_frame_step)
        if next_idx >= self.T_arr_to_display.shape[1]:
            next_idx = 0
        self.t_idx = next_idx
        if self.t_grid is not None and len(self.t_grid) > next_idx:
            self.time_scale.set(float(self.t_grid[next_idx]))
        interval = max(50, int(self.animation_speed_var.get()))
        self.animation_after_id = self.root.after(interval, self.advance_animation_frame)

    def step_animation(self, delta: int) -> None:
        if not self.simulation_ready or self.T_arr_to_display is None:
            return
        self.stop_animation()
        next_idx = int(getattr(self, "t_idx", 0)) + int(delta)
        next_idx = max(0, min(next_idx, self.T_arr_to_display.shape[1] - 1))
        self.t_idx = next_idx
        if self.t_grid is not None and len(self.t_grid) > next_idx:
            self.time_scale.set(float(self.t_grid[next_idx]))
        else:
            self.update_time(None)

    def stop_animation(self) -> None:
        if self.animation_after_id is not None:
            try:
                self.root.after_cancel(self.animation_after_id)
            except Exception:
                pass
        self.animation_after_id = None
        self.animation_running = False
        try:
            self.play_button.configure(text="Play")
        except Exception:
            pass

    def update_surface_plot(
        self,
        x_grid: np.ndarray,
        t_arr_final: np.ndarray,
        T_arr_to_display: np.ndarray,
    ) -> None:
        self.ax2.cla()
        self.surface_time_slice_artist = None
        x_dim, t_dim = len(x_grid), len(t_arr_final)
        # print( f"Debug: update_surface_plot - x_dim: {x_dim}, t_dim: {t_dim}, T_arr_to_display shape: {
        # T_arr_to_display.shape}")

        try:
            if x_dim < 2 or t_dim < 1:
                return

            if T_arr_to_display.shape[1] != t_dim:
                print(
                    f"Warning: T_arr_to_display shape mismatch, reshaping to ({x_dim}, {t_dim})"
                )
                T_arr_to_display = T_arr_to_display[:, :t_dim]

            T_reshaped = T_arr_to_display.reshape((x_dim, t_dim))
            finite_t = T_reshaped[np.isfinite(T_reshaped)]
            if not finite_t.size:
                return

            temp_min = float(np.nanmin(finite_t))
            temp_max = float(np.nanmax(finite_t))
            temp_span = temp_max - temp_min
            if temp_span <= 0:
                temp_span = max(abs(temp_min), 1.0)

            current_t_idx = max(
                0,
                min(int(getattr(self, "t_idx", 0)), T_reshaped.shape[1] - 1),
            )
            profile = np.asarray(T_reshaped[:, current_t_idx], dtype=float)
            profile = np.nan_to_num(profile, nan=temp_min, posinf=temp_max, neginf=temp_min)

            x_values = np.asarray(x_grid, dtype=float)
            profile_min = float(np.nanmin(profile))
            activity_threshold = max(1.0, 0.01 * temp_span)
            active_indices = np.flatnonzero(np.abs(profile - profile_min) > activity_threshold)
            window_start = 0
            window_end = x_dim - 1
            if active_indices.size:
                active_start = int(active_indices[0])
                active_end = int(active_indices[-1])
                active_count = active_end - active_start + 1
                pad = max(8, int(0.35 * active_count))
                window_start = max(0, active_start - 2)
                window_end = min(x_dim - 1, active_end + pad)

                min_points = min(x_dim, max(18, min(70, active_count * 3)))
                if (window_end - window_start + 1) < min_points:
                    missing = min_points - (window_end - window_start + 1)
                    window_start = max(0, window_start - missing // 3)
                    window_end = min(x_dim - 1, window_end + missing - missing // 3)

                if (window_end - window_start + 1) >= int(0.9 * x_dim):
                    window_start = 0
                    window_end = x_dim - 1

            x_plot = x_values[window_start : window_end + 1]
            profile_plot = profile[window_start : window_end + 1]
            self.terrain_x_values = x_plot
            terrain_is_focused = len(x_plot) < len(x_values)

            x_span = float(np.nanmax(x_plot) - np.nanmin(x_plot))
            dx = abs(float(getattr(self.pcm, "dx", 0.01))) if self.pcm is not None else 0.01
            y_extent = max(x_span, dx * 8.0)
            y_values = np.linspace(0.0, y_extent, 41)
            X, Y = np.meshgrid(x_plot, y_values, indexing="ij")

            heat_profile = np.clip(profile_plot - profile_min, 0.0, None)
            heat_max = float(np.nanmax(heat_profile)) if heat_profile.size else 0.0
            domain_depth = max(abs(float(x_values[-1] - x_values[0])), dx)
            front_index = int(active_indices[-1]) if active_indices.size else window_start
            front_x = float(x_values[min(front_index, len(x_values) - 1)])
            front_x = float(np.clip(front_x, np.nanmin(x_plot), np.nanmax(x_plot)))
            front_depth = abs(front_x - float(x_values[0]))
            progress = current_t_idx / max(T_reshaped.shape[1] - 1, 1)
            thermal_progress = float(
                np.clip(
                    max(
                        2.0 * np.sqrt(max(front_depth, 0.0) / domain_depth),
                        0.75 * np.sqrt(progress),
                    ),
                    0.0,
                    1.0,
                )
            )

            if heat_max > 1e-9:
                heat_norm = np.clip(heat_profile / heat_max, 0.0, 1.0)
                heat_shape = np.power(heat_norm, 0.55)
                heat_weight_sum = float(np.sum(heat_profile))
                if heat_weight_sum > 0:
                    centroid_x = float(np.sum(x_plot * heat_profile) / heat_weight_sum)
                else:
                    centroid_x = float(x_plot[0])

                peak_x = 0.55 * centroid_x + 0.45 * front_x
                peak_x = float(np.clip(peak_x, np.nanmin(x_plot), np.nanmax(x_plot)))
                peak_y = y_extent * (0.10 + 0.62 * thermal_progress)
                peak_y = float(np.clip(peak_y, 0.08 * y_extent, 0.82 * y_extent))

                sigma_x = max(dx * 3.0, x_span * (0.12 + 0.28 * thermal_progress))
                sigma_y = max(dx * 3.0, y_extent * (0.10 + 0.32 * thermal_progress))
                x_mound = np.exp(-0.5 * ((X - peak_x) / sigma_x) ** 2)
                y_mound = np.exp(-0.5 * ((Y - peak_y) / sigma_y) ** 2)
                terrain_weight = x_mound * y_mound * (0.45 + 0.55 * heat_shape[:, None])
                weight_max = float(np.nanmax(terrain_weight))
                if weight_max > 0:
                    terrain_weight = terrain_weight / weight_max

                mountain_strength = 0.35 + 0.65 * min(1.0, 1.35 * thermal_progress)
                Z = profile_min + heat_max * mountain_strength * terrain_weight
                Z = np.clip(Z, temp_min, temp_max)

                ridge_col = int(np.argmin(np.abs(y_values - peak_y)))
                self.terrain_y_values = np.full_like(x_plot, y_values[ridge_col], dtype=float)
                self.terrain_profile_values = Z[:, ridge_col]
            else:
                Z = np.full_like(X, profile_min, dtype=float)
                self.terrain_y_values = np.zeros_like(x_plot, dtype=float)
                self.terrain_profile_values = np.full_like(x_plot, profile_min, dtype=float)

            self.terrain_surface_values = Z

            rcount = min(200, len(x_plot))
            ccount = min(80, len(y_values))
            self.ax2.plot_surface(
                X,
                Y,
                Z,
                cmap="coolwarm",
                vmin=temp_min,
                vmax=temp_max,
                rcount=rcount,
                ccount=ccount,
                antialiased=True,
                alpha=0.96,
            )

            time_label = ""
            if self.t_grid is not None and len(self.t_grid) > current_t_idx:
                time_label = f" (t={float(self.t_grid[current_t_idx]):.3g}s)"
            self.ax2.set_title(f"Thermal Terrain Snapshot{time_label}")
            self.ax2.set_xlabel("x [m] (thermal region)" if terrain_is_focused else "x [m]")
            self.ax2.set_ylabel("y [m]")
            self.ax2.set_zlabel("Temperature [C]")
            self.ax2.set_xlim(float(np.nanmin(x_plot)), float(np.nanmax(x_plot)))
            self.ax2.set_ylim(0.0, y_extent)
            z_pad = 0.05 * temp_span
            self.ax2.set_zlim(temp_min - z_pad, temp_max + z_pad)
            try:
                self.ax2.set_box_aspect((1.4, 0.8, 0.9))
                self.ax2.view_init(elev=28, azim=-125)
            except Exception:
                pass
            self.update_surface_time_slice()
            self.canvas2.draw()

            # print(
            #     f"Debug: update_surface_plot - X shape: {X.shape}, T shape: {T.shape}, T_reshaped shape: {T_reshaped.shape}")
        except Exception as e:
            print(f"Error in update_surface_plot: {e}")

    def show_commands_dialog(self, title, instructions, commands):
        try:
            self.show_commands_panel(title, instructions, commands)
        except Exception as e:
            print(f"Warning: failed to show text commands panel: {e}")
            print("Commands to run:\n" + commands)

    def run(self):
        self.root.mainloop()

    def on_closing(self):
        # print("Main window is closing")
        self.root.destroy()


if __name__ == "__main__":
    # Avoid importing TensorFlow at startup to prevent noisy DLL errors on systems without a compatible stack.
    app = App()
    app.run()  # This line should be the last one in the main block


# docker build -t scientificproject .
# docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix scientificproject
# C:\Users\libby\AppData\Roaming\Python\Python312\Scripts\pyinstaller.exe --onefile --windowed C:\Users\libby\PycharmProjects\scientificProject\interface3.py
