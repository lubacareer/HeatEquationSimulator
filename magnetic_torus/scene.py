"""Persistent VTK actors for geometry, computed fields and independent particle replay."""
from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
import PyQt6.QtCore as QtCore
import PyQt6.QtWidgets as QtWidgets
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from vtkmodules.vtkCommonCore import reference

from .models import DoseResult, RunResult, ScenarioConfig


_TEMPERATURE_CMAP = LinearSegmentedColormap.from_list(
    "visible_inferno", colormaps["inferno"](np.linspace(.25, .98, 256)))

QUANTITIES = {
    "Temperature": ("temperature", "K", _TEMPERATURE_CMAP),
    "Liquid fraction": ("liquid_fraction", "fraction", "cividis"),
    "Specific enthalpy": ("enthalpy", "kJ/kg", "viridis"),
    "Flow speed": ("velocity", "m/s", "viridis"),
    "Magnetic strength": ("magnetic_field", "T", "plasma"),
    "Pressure": ("pressure", "Pa", "coolwarm"),
}


def scalar_values(result: RunResult, quantity: str, index: int, fields=None) -> np.ndarray | None:
    key = QUANTITIES[quantity][0]
    values = (fields if fields is not None else result.fields).get(key)
    if values is None:
        return None
    values = np.asarray(values[index], dtype=float)
    if values.ndim == 2:
        values = np.linalg.norm(values, axis=-1)
    return values / 1000.0 if key == "enthalpy" else values


def conductor_mask(result: RunResult) -> np.ndarray:
    supplied = result.metadata.get("conductor_mask")
    if supplied is not None and np.asarray(supplied).shape == (len(result.points),):
        return np.asarray(supplied, dtype=bool)
    radius = np.hypot(result.points[:, 0], result.points[:, 1])
    return (radius - result.config.major_radius) ** 2 + result.points[:, 2] ** 2 <= result.config.minor_radius ** 2 * 1.001


class SceneWidget(QtWidgets.QWidget):
    point_selected = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(auto_update=False)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#102234", top="#20394b")
        self.plotter.add_axes(color="#cbd9e4", line_width=2)
        self.plotter.enable_anti_aliasing("fxaa")
        self.result: RunResult | None = None
        self.config = ScenarioConfig()
        self.index = 0
        self.quantity = "Temperature"
        self.opacity = 1.0
        self.cutaway = True
        self.slice_enabled = False
        self.slice_position = 0.0
        self.show_field = False
        self.show_flow = False
        self.show_tracers = False
        self.visual_detail = "Medium"
        self._tracer_cache = {}
        self.show_phase = False
        self.show_particles = True
        self._grid: pv.UnstructuredGrid | None = None
        self._surface: pv.PolyData | None = None
        self._global_grid: Any = None
        self._mask: np.ndarray | None = None
        self._tree: KDTree | None = None
        self._mask_indices: np.ndarray | None = None
        self._actor = None
        self._fixed_clim: tuple[float, float] | None = None
        self._field_cache = {}
        self._flow_cache = {}
        self._display_fields = None
        self._grid_operator = None
        self._surface_operator = None
        self._text_actors = {}
        self._text_values = {}
        self._quantity_clims = {}
        self._mapper_style = None
        self._draft_geometry_key = None
        self._difference = False
        self.trajectory_data: dict[str, Any] | None = None
        self.dose: DoseResult | None = None
        self.preview(self.config, reset_camera=True)
        self.plotter.enable_point_picking(callback=self._pick, show_message=False, show_point=False,
                                         left_clicking=False, picker="point")

    def _remove(self, name):
        self.plotter.remove_actor(name, reset_camera=False, render=False)
        self._text_actors.pop(name, None)
        self._text_values.pop(name, None)

    def _set_text(self, name, text, position="upper_left", **style):
        """Keep text actors and font textures alive; render once per animation frame."""
        if name == "caption":
            text = textwrap.fill(text, width=88)
        actor = self._text_actors.get(name)
        if actor is None:
            # vtkCornerAnnotation repeatedly fits its font to the viewport when
            # the physical-time string changes (about 40 ms on the reference
            # AMD GPU). A fixed-font TextActor keeps dynamic labels inexpensive.
            positions = {"upper_left": (.02, .97), "upper_right": (.98, .97),
                         "lower_left": (.02, .15), "lower_right": (.98, .16),
                         "upper_edge": (.5, .99), "lower_edge": (.5, .02)}
            actor = self.plotter.add_text(text, position=positions.get(position, (.02, .97)),
                                          viewport=True, name=name, render=False, **style)
            prop = actor.GetTextProperty()
            prop.SetFontSize(round(style.get("font_size", 11) * 1.6))
            if "right" in position:
                prop.SetJustificationToRight()
            elif "edge" in position:
                prop.SetJustificationToCentered()
            else:
                prop.SetJustificationToLeft()
            if "upper" in position:
                prop.SetVerticalJustificationToTop()
            else:
                prop.SetVerticalJustificationToBottom()
            self._text_actors[name] = actor
        elif self._text_values.get(name) != text:
            actor.SetInput(text)
        self._text_values[name] = text
        return actor

    def _clear_actors(self):
        self.plotter.clear()
        self._text_actors.clear()
        self._text_values.clear()
        self._mapper_style = None
        self._draft_geometry_key = None

    def preview_geometry(self, config: ScenarioConfig):
        """A wireframe draft never replaces accepted numerical fields or camera."""
        if self.result is None:
            self.preview(config)
            return
        def geometry(value):
            return (value.major_radius, value.minor_radius, value.habitat_radius,
                    value.vessel_thickness, value.habitat_thickness, value.ground_enabled)
        key = geometry(config)
        if key == self._draft_geometry_key:
            return
        self._draft_geometry_key = key
        for name in ("draft_geometry", "draft_habitat", "draft_note"):
            self._remove(name)
        if key != geometry(self.result.config):
            torus = pv.ParametricTorus(ringradius=config.major_radius,
                                      crosssectionradius=config.minor_radius,
                                      u_res=48, v_res=20)
            if self.cutaway:
                torus = torus.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=True)
            self.plotter.add_mesh(torus, color="#b6f0ef", style="wireframe", opacity=.45,
                                  line_width=1, name="draft_geometry", render=False, reset_camera=False)
            self.plotter.add_mesh(pv.Sphere(radius=config.habitat_radius, theta_resolution=20, phi_resolution=12),
                                  color="#ffe0a1", style="wireframe", opacity=.3,
                                  name="draft_habitat", render=False, reset_camera=False)
            self._set_text("draft_note", "Draft geometry preview · Run to compute", position="upper_edge",
                           font_size=9, color="#b6f0ef")
        self.plotter.render()

    def preview(self, config: ScenarioConfig, reset_camera=False):
        camera = self.plotter.camera_position
        self.result = None
        self.config = config
        self._grid = None
        self._surface = None
        self._grid_operator = None
        self._surface_operator = None
        self.trajectory_data = None
        self.dose = None
        self._field_cache.clear()
        self._flow_cache.clear()
        self._tracer_cache.clear()
        self._clear_actors()
        self.plotter.add_axes(color="#cbd9e4")
        torus = pv.ParametricTorus(ringradius=config.major_radius, crosssectionradius=config.minor_radius,
                                  u_res=100, v_res=36)
        if self.cutaway:
            torus = torus.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=True)
        self.plotter.add_mesh(torus, color="#65b4c7", smooth_shading=True, specular=0.35,
                              opacity=self.opacity, name="reservoir")
        self._habitat(config)
        self._set_text("title", "GEOMETRY PREVIEW", position="upper_left", font_size=12, color="#9ddce8")
        self._set_text("caption", "Edit parameters → Run simulation\nRight-click a computed reservoir point to inspect it",
                       position="lower_left", font_size=9, color="#c5d4df")
        if reset_camera:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
        else:
            self.plotter.camera_position = camera
        self.plotter.render()

    def _habitat(self, config):
        sphere = pv.Sphere(radius=config.habitat_radius, theta_resolution=32, phi_resolution=24)
        self.plotter.add_mesh(sphere, color="#e2d3b7", opacity=.28, smooth_shading=True,
                              name="habitat", show_edges=False)
        outline = pv.Circle(radius=config.major_radius, resolution=120)
        self.plotter.add_mesh(outline, color="#436476", line_width=1, name="centerline")
        if config.ground_enabled:
            plane = pv.Plane(center=(0, 0, -config.habitat_radius - config.habitat_thickness),
                             direction=(0, 0, 1), i_size=3 * config.major_radius, j_size=3 * config.major_radius)
            self.plotter.add_mesh(plane, color="#776c60", opacity=.18, name="ground")

    def set_result(self, result: RunResult, quantity="Temperature", index=0, reset_camera=False):
        camera = self.plotter.camera_position
        if self.result is None or self.result.run_id != result.run_id:
            self.trajectory_data = None
            self.dose = None
        self.result = result
        self.config = result.config
        self.quantity = quantity
        self.index = min(index, len(result.times) - 1)
        self._mask = conductor_mask(result)
        self._mask_indices = np.flatnonzero(self._mask)
        self._tree = KDTree(result.points[self._mask]) if self._mask.any() else None
        self._field_cache.clear()
        self._flow_cache.clear()
        self._tracer_cache.clear()
        self._quantity_clims.clear()
        self._grid_operator = None
        self._surface_operator = None
        cells = np.asarray(result.metadata.get("cells", []), dtype=np.int64)
        self._global_grid = None
        self._grid = None
        if cells.ndim == 2 and cells.shape[1] == 4 and cells.size:
            vtk_cells = np.column_stack([np.full(len(cells), 4), cells]).ravel()
            self._global_grid = pv.UnstructuredGrid(vtk_cells, np.full(len(cells), pv.CellType.TETRA), result.points)
            self._global_grid.point_data["source_id"] = np.arange(len(result.points))
            conductor_cells = result.metadata.get("conductor_cells")
            if conductor_cells is not None:
                selected = np.asarray(conductor_cells)
                if selected.ndim == 2:
                    supplied = {tuple(sorted(row)) for row in selected}
                    selected = np.fromiter((tuple(sorted(row)) in supplied for row in cells), dtype=bool, count=len(cells))
            else:
                selected = np.all(self._mask[cells], axis=1)
            self._grid = self._global_grid.extract_cells(selected)
            if self._grid is None:
                raise ValueError("The accepted conductor mesh could not be extracted.")
            source = np.asarray(self._grid.point_data["source_id"], dtype=int)
            self._grid_operator = csr_matrix((np.ones(len(source)), (np.arange(len(source)), source)),
                                             shape=(len(source), len(result.points)))
        elif result.metadata.get("structured_shape") is not None:
            nr, nt, nphi = (int(x) for x in result.metadata["structured_shape"])
            radii = np.r_[0., (np.arange(nr) + .5) * self.config.minor_radius / nr, self.config.minor_radius]
            theta = (np.arange(nt + 1) + .5) * 2 * np.pi / nt
            phi = (np.arange(nphi + 1) + .5) * 2 * np.pi / nphi
            rr, tt, pp = np.meshgrid(radii, theta, phi, indexing="ij")
            grid = pv.StructuredGrid((self.config.major_radius + rr * np.cos(tt)) * np.cos(pp),
                                     (self.config.major_radius + rr * np.cos(tt)) * np.sin(pp), rr * np.sin(tt))
            if self._tree is not None:
                _, nearest = self._tree.query(grid.points)
                grid.point_data["source_id"] = self._mask_indices[nearest]
                self._grid = grid.cast_to_unstructured_grid()
                rows, cols, weights = [], [], []
                for k in range(nphi + 1):
                    for j in range(nt + 1):
                        for radial in range(nr + 2):
                            row = radial + (nr + 2) * (j + (nt + 1) * k)
                            if radial == 0:
                                # theta is undefined on the tube centreline.
                                # Average its surrounding first-shell FV cells.
                                for angular in range(nt):
                                    rows.append(row); cols.append(angular * nphi + k % nphi); weights.append(1 / nt)
                            else:
                                source_radial = min(radial - 1, nr - 1)
                                rows.append(row); cols.append((source_radial * nt + j % nt) * nphi + k % nphi); weights.append(1.)
                self._grid_operator = csr_matrix((weights, (rows, cols)), shape=(grid.n_points, len(result.points)))
        self._clear_actors()
        self.plotter.add_axes(color="#cbd9e4")
        self._habitat(result.config)
        self._fixed_clim = None
        self.rebuild_surface()
        if reset_camera:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
        else:
            self.plotter.camera_position = camera
        self.plotter.render()

    def _surface_interpolator(self, surface):
        """Precompute true cell basis weights at clipped surface vertices.

        VTK interpolates integer arrays too. Interpolating a node ID and then
        converting it to an integer selects an unrelated node, so display data
        must be sampled with FE/structured-cell basis functions instead.
        """
        if self._grid is None or self._grid_operator is None:
            return None
        nearest_distance, nearest = KDTree(self._grid.points).query(surface.points)
        nearest_distance, nearest = np.asarray(nearest_distance), np.asarray(nearest, dtype=np.intp)
        tolerance = max(self.config.major_radius, 1.) * 1.e-9
        rows, cols, weights = [], [], []
        unresolved = np.flatnonzero(nearest_distance > tolerance)
        containing = self._grid.find_containing_cell(surface.points[unresolved]) if len(unresolved) else []
        resolved = np.flatnonzero(nearest_distance <= tolerance)
        rows.extend(resolved.tolist()); cols.extend(nearest[resolved].tolist()); weights.extend(np.ones(len(resolved)))
        for row, cell_id in zip(unresolved, np.asarray(containing).ravel()):
            if cell_id < 0:
                cell_id = int(np.asarray(self._grid.find_closest_cell(surface.points[row], return_closest_point=False)).reshape(-1)[0])
            cell = self._grid.GetCell(int(cell_id))
            basis_weights = [0.] * cell.GetNumberOfPoints()
            closest, local = [0.] * 3, [0.] * 3
            # VTK requires mutable reference outputs; its generated annotations
            # describe the unwrapped C++ scalar types instead.
            evaluate_position: Callable[..., int] = cell.EvaluatePosition
            evaluate_position(surface.points[row], closest, reference(0), local, reference(0.), basis_weights)
            basis_weights = np.asarray(basis_weights)
            if not np.all(np.isfinite(basis_weights)) or abs(basis_weights.sum() - 1) > 1.e-6:
                raise ValueError("Cannot interpolate a cutaway point in the accepted simulation mesh.")
            for local_index, weight in enumerate(basis_weights):
                if abs(weight) > 1.e-14:
                    rows.append(int(row)); cols.append(cell.GetPointId(local_index)); weights.append(float(weight))
        local_operator = csr_matrix((weights, (rows, cols)), shape=(surface.n_points, self._grid.n_points))
        return local_operator @ self._grid_operator

    def rebuild_surface(self):
        if self.result is None:
            self.preview(self.config)
            return
        if self._grid is not None and self._grid.n_cells:
            surface = self._grid.extract_surface(algorithm="dataset_surface")
        else:
            surface = pv.ParametricTorus(ringradius=self.config.major_radius, crosssectionradius=self.config.minor_radius,
                                        u_res=96, v_res=32)
            if self._tree is not None and self._mask_indices is not None:
                _, nearest = self._tree.query(surface.points)
                surface.point_data["source_id"] = self._mask_indices[nearest]
        if self.cutaway:
            if self._grid is not None and self._grid.n_cells:
                surface = self._grid.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=True).extract_surface(algorithm="dataset_surface")
            else:
                surface = surface.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=True)
        self._surface = surface
        self._surface_operator = self._surface_interpolator(surface)
        self._remove("reservoir")
        try:
            self.plotter.remove_scalar_bar()
        except (ValueError, KeyError, IndexError, StopIteration):
            pass
        values = scalar_values(self.result, self.quantity, self.index)
        if values is not None and surface.n_points:
            ids = np.asarray(surface.point_data.get("source_id", []), dtype=int)
            if len(ids) != surface.n_points:
                if self._tree is None or self._mask_indices is None:
                    return
                _, nearest = self._tree.query(surface.points)
                ids = self._mask_indices[nearest]
            self._surface.point_data["source_id"] = ids
            self._surface.point_data["display"] = self._surface_operator @ values if self._surface_operator is not None else values[ids]
            self._actor = self.plotter.add_mesh(surface, scalars="display", cmap=QUANTITIES[self.quantity][2],
                                                clim=self.color_range(), opacity=self.opacity, smooth_shading=False,
                                                name="reservoir", scalar_bar_args={"title": f"{self.quantity} [{QUANTITIES[self.quantity][1]}]",
                                                "color": "#d8e6ef", "vertical": False, "position_x": .32,
                                                "position_y": .05, "width": .60, "height": .08,
                                                "title_font_size": 11, "label_font_size": 10}, render=False)
            self._mapper_style = None
        else:
            self._actor = self.plotter.add_mesh(surface, color="#65b4c7", opacity=self.opacity,
                                                name="reservoir", render=False)
        self.update_index(self.index)

    def color_range(self):
        if self._fixed_clim is not None:
            return self._fixed_clim
        if self.quantity in self._quantity_clims:
            return self._quantity_clims[self.quantity]
        if self.result is None or self._mask is None:
            return (0.0, 1.0)
        key = QUANTITIES[self.quantity][0]
        values = self.result.fields.get(key)
        if values is None:
            return (0.0, 1.0)
        values = np.asarray(values)[:, self._mask]
        if values.ndim == 3:
            values = np.linalg.norm(values, axis=-1)
        if key == "enthalpy":
            values = values / 1000
        finite = values[np.isfinite(values)]
        if not finite.size:
            return (0.0, 1.0)
        low, high = float(finite.min()), float(finite.max())
        if key == "liquid_fraction":
            return (0.0, 1.0)
        if high - low < 1e-12:
            high = low + max(abs(low) * .01, 1e-6)
        self._quantity_clims[self.quantity] = (low, high)
        return (low, high)

    def update_index(self, index, fields=None, difference=False, source_label=None):
        if self.result is None or self._mask is None:
            return
        self.index = max(0, min(int(index), len(self.result.times) - 1))
        if fields is not self._display_fields:
            self._field_cache.clear()
            self._tracer_cache.clear()
        self._display_fields = fields
        self._difference = difference
        values = scalar_values(self.result, self.quantity, self.index, fields)
        if values is not None and self._surface is not None and "source_id" in self._surface.point_data:
            ids = np.asarray(self._surface.point_data["source_id"], dtype=int)
            self._surface.point_data["display"] = self._surface_operator @ values if self._surface_operator is not None else values[ids]
            if self._actor is not None:
                if difference and fields is not None:
                    key = QUANTITIES[self.quantity][0]
                    all_values = np.asarray(fields[key])[:, self._mask]
                    bound = float(np.max(np.abs(all_values))) if all_values.ndim == 2 else float(np.max(np.linalg.norm(all_values, axis=-1)))
                    if key == "enthalpy":
                        bound /= 1000
                    bound = max(bound, 1e-12)
                    self._actor.mapper.scalar_range = (-bound, bound) if all_values.ndim == 2 else (0, bound)
                    cmap = "coolwarm" if all_values.ndim == 2 else "magma"
                else:
                    self._actor.mapper.scalar_range = self.color_range()
                    cmap = QUANTITIES[self.quantity][2]
                style = (self.quantity, difference, id(cmap) if not isinstance(cmap, str) else cmap)
                if self._mapper_style != style:
                    self._actor.mapper.lookup_table.cmap = cmap
                    self._mapper_style = style
                try:
                    self.plotter.scalar_bar.SetTitle(f"{'Error: ' if difference else ''}{self.quantity} [{QUANTITIES[self.quantity][1]}]")
                except AttributeError:
                    pass
        source = "PINN" if fields is not None else "Numerical"
        if difference:
            source = "PINN − numerical"
        if source_label:
            source = source_label
        self._set_text("title", f"{self.config.name}\n{self.quantity} · t = {self.result.times[self.index]:.4g} s · {source}",
                       position="upper_left", font_size=11, color="#e1edf5")
        backend = str(self.result.metadata.get("backend", self.config.solver))
        validation = str(self.result.metadata.get("validation", self.result.metadata.get("validation_status", "Unvalidated")))
        self._set_text("caption", f"{backend} · {validation}\nRun {self.result.run_id[:8]} · config {self.config.config_hash[:8]}", position="lower_left", font_size=9,
                       color="#a7becb")
        self._update_slice()
        self._update_phase()
        self._update_vectors()
        self._update_tracers()
        self._analysis_visibility()
        self.plotter.render()

    def _analysis_visibility(self):
        if self.result is None:
            return
        for data, names in ((self.trajectory_data, ("particles", "particle_paths", "baseline_paths", "particle_time")),
                            (self.dose, ("dose", "dose_caption"))):
            if data is None:
                continue
            index = int(data["snapshot_index"]) if isinstance(data, dict) else data.snapshot_index
            for name in names:
                actor = self.plotter.actors.get(name)
                if actor is not None:
                    actor.SetVisibility(index == self.index and not self._difference)
        notes = []
        if self.dose is not None:
            time = self.result.times[self.dose.snapshot_index]
            source = self.dose.metadata.get("field_source", "numerical")
            notes.append(f"Dose: frozen {source} field at t={time:.4g} s")
            if self.dose.snapshot_index != self.index:
                notes.append("Dose layer hidden at the current thermal/MHD time")
        if self.trajectory_data is not None:
            time = self.result.times[int(self.trajectory_data["snapshot_index"])]
            metadata = self.trajectory_data.get("metadata", {})
            notes.append(f"Particle paths: frozen {metadata.get('selected_source', 'numerical')} field at t={time:.4g} s")
        if notes:
            self._set_text("frozen_note", "\n".join(notes), position="lower_right", font_size=8, color="#d6d3c0")
        else:
            self._remove("frozen_note")

    def _update_slice(self):
        self._remove("slice")
        if not self.slice_enabled or self._grid is None or self.result is None:
            return
        values = scalar_values(self.result, self.quantity, self.index, self._display_fields)
        if values is None:
            return
        self._grid.point_data["slice_value"] = self._grid_operator @ values if self._grid_operator is not None else values[
            np.asarray(self._grid.point_data["source_id"], dtype=int)]
        surface = self._grid.slice(normal=(1, 0, 0), origin=(self.slice_position, 0, 0))
        if surface.n_points:
            original = (self._display_fields if self._display_fields is not None else self.result.fields)[QUANTITIES[self.quantity][0]]
            cmap = ("coolwarm" if original.ndim == 2 else "magma") if self._difference else QUANTITIES[self.quantity][2]
            limits = tuple(self._actor.mapper.scalar_range) if self._actor is not None else self.color_range()
            self.plotter.add_mesh(surface, scalars="slice_value", cmap=cmap,
                                  clim=limits, show_scalar_bar=False, name="slice", render=False)

    def _update_phase(self):
        self._remove("phase")
        if not self.show_phase or self._grid is None or self.result is None or self._difference:
            return
        values = (self._display_fields if self._display_fields is not None else self.result.fields).get("liquid_fraction")
        if values is None:
            return
        self._grid.point_data["phase"] = self._grid_operator @ values[self.index] if self._grid_operator is not None else values[self.index][np.asarray(self._grid.point_data["source_id"], dtype=int)]
        if np.min(self._grid["phase"]) < .5 < np.max(self._grid["phase"]):
            contour = self._grid.contour([.5], scalars="phase")
            if contour.n_points:
                self.plotter.add_mesh(contour, color="#ffec9a", opacity=.85, name="phase", render=False)

    def _update_vectors(self):
        for name in ("field_lines", "flow_vectors"):
            self._remove(name)
        if self.result is None:
            return
        if self._difference:
            return
        display_fields = self._display_fields if self._display_fields is not None else self.result.fields
        if self.show_field and self._global_grid is None and self.result.metadata.get("imposed_coil") is not None:
            from .physics import coil_magnetic_field
            extent = (self.config.major_radius + self.config.minor_radius) * 2.5
            grid = pv.ImageData(dimensions=(25, 25, 25), spacing=(2 * extent / 24,) * 3, origin=(-extent,) * 3)
            grid.point_data["B"] = coil_magnetic_field(grid.points, self.config)
            self._global_grid = grid
        if self.show_field and self._global_grid is not None and "magnetic_field" in display_fields:
            if self.index not in self._field_cache:
                if self._global_grid.n_points == len(self.result.points):
                    self._global_grid.point_data["B"] = display_fields["magnetic_field"][self.index]
                if np.linalg.norm(np.asarray(self._global_grid.point_data["B"]), axis=1).max() > 1e-14:
                    try:
                        lines = self._global_grid.streamlines(vectors="B", source_center=(self.config.major_radius, 0, 0),
                                                             source_radius=self.config.minor_radius * 1.5, n_points={"Low": 8, "Medium": 16, "High": 32}[self.visual_detail],
                                                             integration_direction="both", max_length=8 * self.config.major_radius)
                        self._field_cache[self.index] = lines
                    except (ValueError, RuntimeError):
                        self._field_cache[self.index] = None
                else:
                    self._field_cache[self.index] = None
            lines = self._field_cache[self.index]
            if lines is not None and lines.n_points:
                self.plotter.add_mesh(lines, color="#70ead9", line_width=2, name="field_lines", render=False)
        if self.show_flow and "velocity" in display_fields and self._mask is not None and self._mask_indices is not None and self._mask.any():
            count = {"Low": 40, "Medium": 80, "High": 160}[self.visual_detail]
            selected = self._mask_indices[::max(1, len(self._mask_indices) // count)]
            velocity = display_fields["velocity"][self.index, selected]
            speed = np.linalg.norm(velocity, axis=1)
            active = speed > 1e-10
            if active.any():
                arrows = pv.PolyData(self.result.points[selected][active])
                arrows["velocity"] = velocity[active]
                arrows["speed"] = speed[active]
                glyphs = arrows.glyph(orient="velocity", scale="speed", factor=self.config.minor_radius * .5 / max(speed.max(), 1e-12))
                self.plotter.add_mesh(glyphs, color="#bbe8ff", name="flow_vectors", render=False)

    def _update_tracers(self):
        self._remove("flow_tracers")
        if (not self.show_tracers or self._difference or self.result is None or self._tree is None
                or self._mask_indices is None or not len(self._mask_indices)):
            return
        fields = self._display_fields if self._display_fields is not None else self.result.fields
        velocity = fields.get("velocity")
        if velocity is None or not np.any(velocity):
            return
        count = {"Low": 24, "Medium": 48, "High": 96}[self.visual_detail]
        if 0 not in self._tracer_cache:
            selected = self._mask_indices[np.linspace(0, len(self._mask_indices) - 1, count, dtype=int)]
            self._tracer_cache[0] = self.result.points[selected].copy()
        previous = max(i for i in self._tracer_cache if i <= self.index)
        positions = self._tracer_cache[previous].copy()
        for i in range(previous + 1, self.index + 1):
            dt = float(self.result.times[i] - self.result.times[i - 1])
            maximum = float(np.max(np.linalg.norm(velocity[i, self._mask_indices], axis=-1)))
            substeps = max(1, int(np.ceil(dt * maximum / max(self.config.minor_radius * .1, 1e-8))))
            if substeps > 256:
                return  # Do not display under-resolved pathlines.
            for substep in range(substeps):
                fraction = (substep + .5) / substeps
                distance, nearest = self._tree.query(positions, k=min(4, len(self._mask_indices)))
                distance, nearest = np.asarray(distance), np.asarray(nearest)
                if distance.ndim == 1:
                    distance, nearest = distance[:, None], nearest[:, None]
                weights = 1 / np.maximum(distance, 1e-10) ** 2
                weights /= weights.sum(axis=1, keepdims=True)
                ids = self._mask_indices[nearest]
                flow = (1 - fraction) * velocity[i - 1, ids] + fraction * velocity[i, ids]
                updated = positions + dt / substeps * np.sum(flow * weights[:, :, None], axis=1)
                radius = np.hypot(updated[:, 0], updated[:, 1])
                inside = (radius - self.config.major_radius) ** 2 + updated[:, 2] ** 2 < self.config.minor_radius ** 2
                positions[inside] = updated[inside]
            self._tracer_cache[i] = positions.copy()
        particles = pv.PolyData(positions)
        self.plotter.add_mesh(particles, color="#e5fbff", point_size=6, render_points_as_spheres=True,
                              name="flow_tracers", render=False)

    def set_trajectories(self, data):
        self.trajectory_data = data
        self.draw_particles(0.0)

    def draw_particles(self, fraction: float):
        for name in ("particles", "particle_paths", "baseline_paths"):
            self._remove(name)
        data = self.trajectory_data
        if data is None or not self.show_particles:
            self._remove("particle_time")
            self.plotter.render()
            return
        paths = np.asarray(data["paths"])
        baseline = np.asarray(data["baseline_paths"])
        # Separate particle-flight clock, in seconds, never the MHD playback clock.
        length = max(2, min(paths.shape[1], int(fraction * (paths.shape[1] - 1)) + 2))
        for key, array, color in (("particle_paths", paths, "#ffb66b"), ("baseline_paths", baseline, "#657f92")):
            selected = array[:min(len(array), 120), :length]
            points = selected.reshape(-1, 3)
            counts = np.full((len(selected), 1), length)
            ids = np.arange(len(points)).reshape(len(selected), length)
            poly = pv.PolyData(points)
            poly.lines = np.concatenate([counts, ids], axis=1).ravel()
            self.plotter.add_mesh(poly, color=color, opacity=.85 if key == "particle_paths" else .3,
                                  line_width=2 if key == "particle_paths" else 1, name=key, render=False)
        dots = pv.PolyData(paths[:120, length - 1])
        self.plotter.add_mesh(dots, color="#ffddac", point_size=7, render_points_as_spheres=True,
                              name="particles", render=False)
        times = np.asarray(data["times"])
        elapsed = float(np.max(times[:, length - 1]))
        self.plotter.add_text(f"Particle replay · max flight time {elapsed:.3g} s\nOrange: field on   Gray: field off",
                              position="upper_right", font_size=9, color="#ffddb3", name="particle_time")
        self._analysis_visibility()
        self.plotter.render()

    def set_dose(self, dose: DoseResult | None):
        self.dose = dose
        self._remove("dose")
        self._remove("dose_caption")
        if dose is None:
            return
        voxel_path = dose.metadata.get("voxel_dose_path")
        if voxel_path and Path(voxel_path).is_file():
            with np.load(voxel_path, allow_pickle=False) as data:
                points = pv.PolyData(data["points"])
                points["dose"] = data["dose_gy"]
            positive = points["dose"][points["dose"] > 0]
            if positive.size:
                self.plotter.add_mesh(points, scalars="dose", cmap="magma", log_scale=True,
                                      clim=(float(positive.min()), float(positive.max()) * 1.0001),
                                      point_size=6, render_points_as_spheres=True,
                                      name="dose", scalar_bar_args={"title": "Absorbed dose [Gy]", "color": "white"}, render=False)
        elif dose.target == "water" and len(dose.dose_gy):
            self.plotter.add_text(f"Water target mean: {dose.dose_gy[0]:.3g} ± {dose.standard_error_gy[0]:.2g} Gy\nSpatial dose map not provided by this run",
                                  position="upper_right", font_size=10, color="#ffdbb9", name="dose_caption")
        self._analysis_visibility()
        self.plotter.render()

    def _pick(self, point):
        if self.result is None or self._tree is None or self._mask_indices is None:
            return
        distance, local = self._tree.query(np.asarray(point))
        index = int(self._mask_indices[int(local)])
        self.point_selected.emit(index)
        self.plotter.add_mesh(pv.Sphere(radius=self.config.minor_radius * .035, center=self.result.points[index]),
                              color="#ffffff", name="probe", render=False)
        self.plotter.render()

    def camera_preset(self, kind):
        if kind == "Top":
            self.plotter.view_xy()
        elif kind == "Side":
            self.plotter.view_xz()
        else:
            self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.camera.zoom(1.3)

    def close_scene(self):
        self.plotter.close()
