"""Inspect a source-bound PINN pressure fit without changing saved runs.

All-node maps and the previously examined validation samples are exploratory;
they must not select training weights or qualify a model.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def heldout_mask(times: int, points: int, seed: int) -> np.ndarray:
    """Match the seeded point-time split used by the original trainer."""
    ids = np.arange(times * points, dtype=np.int64)
    np.random.default_rng(seed).shuffle(ids)
    count = min(len(ids) - 1, max(1, len(ids) // 5))
    mask = np.zeros(len(ids), dtype=bool)
    mask[ids[:count]] = True
    return mask.reshape(times, points)


def coordinate_mapping(reference: np.ndarray, refined: np.ndarray, scale: float) -> np.ndarray:
    if reference.shape != refined.shape:
        raise ValueError("The refined mesh has a different number of points.")
    if np.array_equal(reference, refined):
        return np.arange(len(reference))
    from scipy.spatial import cKDTree
    distance, indices = cKDTree(refined).query(reference)
    if np.max(distance) > max(1e-12, scale * 1e-10) or len(np.unique(indices)) != len(reference):
        raise ValueError("The refined mesh cannot be matched one-to-one to the original coordinates.")
    return indices


def lorentz_statistics(source, index: int) -> tuple[dict[str, float], np.ndarray]:
    """Reconstruct J×B at the FE quadrature points using the accepted weak operator."""
    with np.load(BytesIO(source.artifact_bytes("weak_operator_path")), allow_pickle=False) as data:
        cells = np.asarray(data["cells"], dtype=int)
        gradients = np.asarray(data["gradients"])
        volumes = np.asarray(data["volumes"])
        conductor = np.asarray(data["cell_labels"]) == 1
    cells, gradients, volumes = cells[conductor], gradients[conductor], volumes[conductor]
    config = source.config
    a, a0 = source.fields["vector_potential"][index], source.fields["vector_potential"][index - 1]
    velocity, temperature = source.fields["velocity"][index], source.fields["temperature"][index]
    potential = source.fields["electric_potential"][index]
    basis = np.full((4, 4), .1381966011250105) + np.eye(4) * (.5854101966249685 - .1381966011250105)
    grad_a = np.einsum("cai,caj->cij", a[cells], gradients)
    magnetic = np.stack((grad_a[:, 2, 1] - grad_a[:, 1, 2], grad_a[:, 0, 2] - grad_a[:, 2, 0],
                         grad_a[:, 1, 0] - grad_a[:, 0, 1]), axis=1)
    grad_phi = np.einsum("ca,caj->cj", potential[cells], gradients)
    uq = np.einsum("qa,cai->cqi", basis, velocity[cells])
    at = np.einsum("qa,cai->cqi", basis, (a[cells] - a0[cells]) / (source.times[index] - source.times[index - 1]))
    tq = np.einsum("qa,ca->cq", basis, temperature[cells])
    fraction = np.clip((tq - config.melting_temperature) / config.transition_width + .5, 0, 1)
    sigma = config.electrical_conductivity_solid * (1 - fraction) + config.electrical_conductivity_liquid * fraction
    current = sigma[..., None] * (-at - grad_phi[:, None, :] + np.cross(uq, magnetic[:, None, :]))
    magnitude = np.linalg.norm(np.cross(current, magnetic[:, None, :]), axis=-1)
    cell_mean = magnitude.mean(axis=1)
    node_sum = np.zeros(len(source.points))
    node_weight = np.zeros(len(source.points))
    np.add.at(node_sum, cells.ravel(), np.repeat(cell_mean * volumes / 4, 4))
    np.add.at(node_weight, cells.ravel(), np.repeat(volumes / 4, 4))
    nodal = np.divide(node_sum, node_weight, out=np.zeros_like(node_sum), where=node_weight > 0)
    return ({"volume_weighted_mean_n_per_m3": float(np.average(cell_mean, weights=volumes)),
             "maximum_quadrature_n_per_m3": float(magnitude.max())}, nodal)


def exact_time(times: np.ndarray, value: float) -> int:
    matches = np.flatnonzero(np.isclose(times, value, rtol=0, atol=1e-12))
    if len(matches) != 1:
        raise ValueError(f"Time {value:g} s is not exactly one accepted snapshot; no interpolation was used.")
    return int(matches[0])


def check_pressure_anchors(source, refined, mapping: np.ndarray) -> None:
    """Compare FE pressure only when its fixed gauge is at the same mesh nodes."""
    def anchors(result):
        nodes = result.metadata.get("pinn_physics", {}).get("bc_nodes", {}).get("1", ())
        return np.unique(np.asarray(nodes, dtype=int))

    before, after = anchors(source), anchors(refined)
    if not len(before) or np.any(before < 0) or np.any(before >= len(mapping)):
        raise ValueError("The original source lacks a valid pressure anchor.")
    if not np.array_equal(np.sort(mapping[before]), after):
        raise ValueError("The refined pressure anchor differs; a raw pressure comparison would change gauge.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pinn", type=Path, required=True)
    parser.add_argument("--refined", type=Path)
    parser.add_argument("--time", type=float, default=.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error("Choose an empty output directory to preserve prior evidence.")
    directory.mkdir(parents=True, exist_ok=True)
    inputs = directory / "inputs"
    inputs.mkdir()
    from magnetic_torus.learning.fields import validate_pinn_source
    from magnetic_torus.store import RunStore

    paths = {"source": args.source.resolve(strict=True), "pinn": args.pinn.resolve(strict=True)}
    if args.refined:
        paths["refined"] = args.refined.resolve(strict=True)
    hashes = {name: digest(path) for name, path in paths.items()}
    copies = {}
    for name, path in paths.items():
        copies[name] = inputs / f"{name}.h5"
        shutil.copy2(path, copies[name])
    source, pinn = RunStore.load_run(copies["source"]), RunStore.load_pinn(copies["pinn"])
    validate_pinn_source(pinn, source)
    if "pressure" not in pinn.fields or "pressure" not in source.fields:
        raise ValueError("Both source and PINN must contain pressure fields.")
    conductor = np.asarray(source.metadata["conductor_mask"], dtype=bool)
    if conductor.shape != (len(source.points),) or not conductor.any():
        raise ValueError("The source has no valid conductor mask.")
    pressure_scale = float(pinn.metadata["validation_field_scales"]["pressure"])
    if not np.isfinite(pressure_scale) or pressure_scale <= 0:
        raise ValueError("The saved pressure validation scale must be positive and finite.")
    pressure = np.asarray(source.fields["pressure"])
    error = np.asarray(pinn.fields["pressure"]) - pressure
    heldout = heldout_mask(len(source.times), len(source.points), source.config.seed)
    radial = np.hypot(source.points[:, 0], source.points[:, 1])
    tube_radius = np.hypot(radial - source.config.major_radius, source.points[:, 2])
    wall_distance = source.config.minor_radius - tube_radius
    selected = exact_time(source.times, args.time)
    statistics = []
    selected_force = None
    total_squared = float(np.square(error[heldout & conductor[None, :] & (np.arange(len(source.times))[:, None] > 0)]).sum())
    for ti, time in enumerate(source.times):
        valid = heldout[ti] & conductor if ti else np.zeros(len(conductor), dtype=bool)
        row = {"time_s": float(time), "reference_pressure_mean_pa": float(pressure[ti, conductor].mean()),
               "velocity_speed_mean_m_s": float(np.linalg.norm(source.fields["velocity"][ti, conductor], axis=1).mean()),
               "liquid_fraction_min": float(source.fields["liquid_fraction"][ti, conductor].min()),
               "liquid_fraction_max": float(source.fields["liquid_fraction"][ti, conductor].max()),
               "heldout_count": int(valid.sum()), "heldout_squared_error_fraction":
                   float(np.square(error[ti, valid]).sum() / total_squared) if total_squared else 0.}
        if valid.any():
            row["heldout_pressure_nrmse"] = float(np.sqrt(np.mean(error[ti, valid] ** 2)) / pressure_scale)
        if ti:
            row["previous_accepted_time_s"] = float(source.times[ti - 1])
            row["pressure_change_rms_pa"] = float(np.sqrt(np.mean((pressure[ti, conductor] - pressure[ti - 1, conductor]) ** 2)))
            row["velocity_change_rms_m_s"] = float(np.sqrt(np.mean((source.fields["velocity"][ti, conductor] - source.fields["velocity"][ti - 1, conductor]) ** 2)))
            row["lorentz_force"], nodal_force = lorentz_statistics(source, ti)
            if ti == selected:
                selected_force = nodal_force
        statistics.append(row)
    selected_heldout = heldout[selected] & conductor if selected else np.zeros(len(conductor), dtype=bool)
    groups = []
    for name, member in (("inner_half", tube_radius < .5 * source.config.minor_radius),
                         ("middle", (tube_radius >= .5 * source.config.minor_radius) & (tube_radius < .8 * source.config.minor_radius)),
                         ("outer", tube_radius >= .8 * source.config.minor_radius)):
        region = selected_heldout & member
        groups.append({"region": name, "heldout_count": int(region.sum()),
                       "heldout_fraction": float(region.sum() / selected_heldout.sum()) if selected_heldout.any() else 0.,
                       "heldout_pressure_nrmse": float(np.sqrt(np.mean(error[selected, region] ** 2)) / pressure_scale) if region.any() else None,
                       "squared_error_fraction_at_selected_time": float(np.square(error[selected, region]).sum() / np.square(error[selected, selected_heldout]).sum())
                           if selected_heldout.any() and np.square(error[selected, selected_heldout]).sum() else 0.})
    comparison = None
    if "refined" in copies:
        refined = RunStore.load_run(copies["refined"])
        if source.config.seed != refined.config.seed and source.config.mode != "driven" and source.config.seed_field != 0:
            raise ValueError("Changing the seed changes the imposed dynamo field; the sources are not physically equivalent.")
        original_config, refined_config = source.config.to_dict(), refined.config.to_dict()
        for key in ("time_steps", "seed"):
            original_config.pop(key)
            refined_config.pop(key)
        if original_config != refined_config:
            raise ValueError("The refined source changes parameters other than time_steps and seed.")
        mapping = coordinate_mapping(source.points, refined.points, source.config.major_radius)
        check_pressure_anchors(source, refined, mapping)
        refined_mask = np.asarray(refined.metadata["conductor_mask"], dtype=bool)[mapping]
        if not np.array_equal(conductor, refined_mask):
            raise ValueError("Conductor masks differ after mesh coordinate matching.")
        ri = exact_time(refined.times, args.time)
        dp = refined.fields["pressure"][ri, mapping][conductor] - pressure[selected, conductor]
        dv = refined.fields["velocity"][ri, mapping][conductor] - source.fields["velocity"][selected, conductor]
        comparison = {"refined_run_id": refined.run_id, "refined_config_hash": refined.config.config_hash,
                      "accepted_time_s": float(refined.times[ri]), "interpolated": False, "pressure_anchor_match": True,
                      "pressure_difference_rms_pa": float(np.sqrt(np.mean(dp ** 2))),
                      "velocity_difference_rms_m_s": float(np.sqrt(np.mean(dv ** 2))),
                      "liquid_fraction_max_abs_difference": float(np.max(np.abs(refined.fields["liquid_fraction"][ri, mapping][conductor] - source.fields["liquid_fraction"][selected, conductor])))}
    np.savez_compressed(directory / "pressure_error.npz", times=source.times, points=source.points,
                        error_pa=error, conductor_mask=conductor, heldout_mask=heldout,
                        wall_distance_m=wall_distance, lorentz_force_n_per_m3=selected_force if selected_force is not None else np.full(len(source.points), np.nan))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    points, values = source.points[conductor], error[selected, conductor]
    limit = max(float(np.max(np.abs(values))), np.finfo(float).eps)
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    for position, quantity, label, cmap, vmin, vmax in (
        (1, values, "Signed pressure error [Pa]", "coolwarm", -limit, limit),
        (2, np.abs(values), "Absolute pressure error [Pa]", "viridis", 0., limit),
    ):
        ax = fig.add_subplot(1, 2, position, projection="3d")
        artist = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=quantity, cmap=cmap,
                            vmin=vmin, vmax=vmax, s=8, depthshade=False)
        ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title=label)
        ax.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-12))
        ax.zaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=8, pad=1)
        fig.colorbar(artist, ax=ax, shrink=.7, pad=.08, label="Pa")
    fig.suptitle(f"Exploratory conductor nodes · accepted t={source.times[selected]:.6g} s · source {source.run_id}")
    fig.savefig(directory / "pressure_error_maps.png", dpi=160)
    plt.close(fig)
    unchanged = {name: digest(path) == hashes[name] for name, path in paths.items()}
    report = {"source_run_id": source.run_id, "pinn_run_id": pinn.run_id, "source_config_hash": source.config.config_hash,
              "source_files_sha256": hashes, "inputs_unchanged": unchanged, "exploratory": True,
              "selection_warning": "Previously inspected held-out labels must not select training weights or hyperparameters.",
              "time_s": float(source.times[selected]), "interpolated": False, "pressure_scale_pa": pressure_scale,
              "lorentz_force_available": selected_force is not None,
              "wall_distance_m": {"minimum": float(wall_distance[conductor].min()), "maximum": float(wall_distance[conductor].max())},
              "radial_groups": groups, "radial_group_definition": "Analytic tube radius; descriptive nodal statistics, not a volume-weighted boundary-layer test.",
              "per_accepted_time": statistics, "refined_comparison": comparison,
              "maps": "pressure_error_maps.png", "raw_arrays": "pressure_error.npz"}
    if not all(unchanged.values()):
        raise RuntimeError("A selected input changed during diagnostics.")
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str((directory / "report.json").resolve()), "time_s": report["time_s"],
                      "refined_comparison": comparison}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
