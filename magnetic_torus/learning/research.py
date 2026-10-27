"""Explicit, bounded PINN hypothesis experiments; never selected by the desktop.

Training may use adaptive sampling, causal weights or a different architecture.
All candidates return to the unchanged source operator and qualification gates.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

import numpy as np

from ..models import PINNResult, Progress, RunResult, SimulationCancelled
from . import training as base
from .fields import _cancelled, validate_pinn_source
from .validation import validate_prediction


class ResearchPilotTimeout(SimulationCancelled):
    """A pressure pilot expired before completing its independent audit."""

    def __init__(self, steps: int, elapsed_s: float):
        super().__init__("The pressure research deadline expired before qualification.")
        self.steps = steps
        self.elapsed_s = elapsed_s


@dataclass(frozen=True)
class ResearchOptions:
    strategy: str = "uniform"
    architecture: str = "mlp"
    adam_steps: int = 800
    lbfgs_steps: int = 20
    seconds: float = 180.
    balance_labels: bool = False
    causal_epsilon: float = 1.
    causal_floor: float = .05

    def validate(self) -> ResearchOptions:
        if self.strategy not in {"uniform", "front", "causal", "front-causal", "temporal-pressure"}:
            raise ValueError("Unknown research sampling strategy.")
        if self.architecture not in {"mlp", "film", "temporal-phi", "method5", "pressure-fourier"}:
            raise ValueError("Unknown research architecture.")
        if self.strategy == "temporal-pressure" and self.architecture != "method5":
            raise ValueError("Temporal pressure sampling requires the method5 research architecture.")
        if (self.architecture in {"method5", "pressure-fourier"}
                and (self.strategy not in {"uniform", "temporal-pressure"} or self.balance_labels)):
            raise ValueError("Method-5 pressure pilots retain the original joint objective.")
        if (not isinstance(self.adam_steps, int) or isinstance(self.adam_steps, bool)
                or not isinstance(self.lbfgs_steps, int) or isinstance(self.lbfgs_steps, bool)
                or self.adam_steps < 0 or self.lbfgs_steps < 0
                or not 1 <= self.adam_steps + self.lbfgs_steps <= 10000):
            raise ValueError("Choose between 1 and 10000 explicit research steps.")
        if not np.isfinite(self.seconds) or not 1 <= self.seconds <= 3600:
            raise ValueError("Research training deadlines must be between 1 and 3600 seconds.")
        if (not np.isfinite(self.causal_epsilon) or self.causal_epsilon <= 0
                or not np.isfinite(self.causal_floor) or not 0 < self.causal_floor <= 1):
            raise ValueError("Causal epsilon must be positive and the floor in (0,1].")
        return self


def _label_gradient_norms(components: dict[str, Any], parameters: tuple[Any, ...], torch: Any) -> dict[str, float]:
    values = {}
    for name, loss in components.items():
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        square = sum((value.detach().square().sum() for value in gradients if value is not None),
                     torch.zeros((), device=loss.device, dtype=loss.dtype))
        values[name] = float(square.sqrt())
    return values


def run_research(source: RunResult, options: ResearchOptions, progress: Progress,
                 cancel: threading.Event, output_dir: Path,
                 checkpoint: Path | None = None) -> tuple[RunResult, PINNResult, dict[str, Any]]:
    """Run a declared experiment on a new, explicitly attributed source copy.

    No accepted source, old checkpoint, normal history or workspace support
    policy is modified. Warm starts are explicit and validated against their
    original source before their weights are used on the attributed copy.
    """
    options.validate()
    start = time.monotonic()
    deadline = start + options.seconds
    _cancelled(cancel)
    from .curriculum import CausalWeights, LabelGradientBalancer, TimeLabelSampler
    from .corrections import (CORRECTION_ARCHITECTURE, create_correction_model,
                              create_pressure_fourier_model, restore_model_from_checkpoint)
    from .research_scaling import fit_training_weak_scales
    from .sampling import FrontResidualSampler

    thermal = source.config.solver == "thermal_benchmark"
    method5_pilot = options.architecture in {"method5", "pressure-fourier"}
    if method5_pilot and checkpoint is None:
        raise ValueError("Method-5 pressure pilots require an explicit source-bound pretrained checkpoint.")
    if options.architecture != "mlp" and thermal:
        raise ValueError("The experimental time architectures require a coupled FE source.")
    if "front" in options.strategy and not thermal:
        raise ValueError("Front collocation requires the native conservative thermal operator.")
    if options.architecture == "temporal-phi" and checkpoint is None:
        raise ValueError("Temporal phi experiments require an explicitly selected source checkpoint.")
    source_copy = replace(source, run_id="research-source-" + uuid4().hex[:12],
        config=replace(source.config, pinn_epochs=options.adam_steps + options.lbfgs_steps).validate(),
        metadata={**source.metadata, "reused_fields_from": source.run_id,
                  "research_source_copy": True})
    data = base._prepare_data(source_copy, cancel)
    original_weak = data.weak
    training_scales = None
    if not thermal:
        training_scales = fit_training_weak_scales(source_copy, data.train_ids)
        def training_weak(config, reference, fields, indices):
            return original_weak(config, reference, fields, indices,
                                 normalization_scales=training_scales)
        data = replace(data, weak=training_weak)
    run = base._prepare_training(data, progress, cancel, output_dir)
    field, torch = run.field, run.field.torch
    if checkpoint is not None:
        if options.architecture == "film":
            raise ValueError("FiLM is a fresh architecture; do not attach another architecture's checkpoint.")
        checkpoint_bytes = checkpoint.read_bytes()
        payload = torch.load(BytesIO(checkpoint_bytes), weights_only=True, map_location=field.device)
        field.model = restore_model_from_checkpoint(payload, source, str(field.device))
        if options.architecture == "mlp" and not isinstance(field.model, torch.nn.Sequential):
            raise ValueError("MLP research warm starts require a legacy Sequential checkpoint; inspect other architectures with their source-bound loader.")
        cloned_sequential_core = False
        if method5_pilot:
            if isinstance(field.model, torch.nn.Sequential):
                field.model = create_correction_model(field.model, ("elliptic", "thermofluid"), torch=torch)
                cloned_sequential_core = True
            elif (not hasattr(field.model, "architecture_metadata")
                    or field.model.architecture_metadata().get("kind") != CORRECTION_ARCHITECTURE):
                raise ValueError("Method-5 pressure pilots require a source-bound core or correction checkpoint.")
        if (tuple(payload.get("field_names", ())) != data.scaling.names
                or tuple(payload.get("field_widths", ())) != data.scaling.widths
                or not np.array_equal(payload.get("field_offsets"), data.scaling.offsets)
                or not np.array_equal(payload.get("field_scales"), data.scaling.scales)):
            raise ValueError("Warm-start normalization differs from this training split; use a fresh model.")
        run.metadata["research_warm_start"] = {"checkpoint_sha256": sha256(checkpoint_bytes).hexdigest(),
            "source_run_id": source.run_id, "source_config_hash": source.config.config_hash,
            "checkpoint_epoch": int(payload["epoch"]), "explicit": True,
            "cloned_sequential_core": cloned_sequential_core}
    if options.architecture == "film":
        from .modulation import create_film_model
        if 2 * data.point_count * data.output_width * 3 > 2_000_000:
            raise base.UnsupportedPINNConfiguration("The FiLM FE pair exceeds its bounded architecture memory limit.")
        devices = [torch.cuda.current_device()] if field.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(data.config.seed)
            field.model = create_film_model(torch=torch).to(field.device)
    elif options.architecture == "temporal-phi":
        from .gauge import create_temporal_gauge_model
        normalized_times = (source_copy.times - field.start_time) / field.time_scale
        field.model = create_temporal_gauge_model(field.model, normalized_times, torch=torch)
    elif options.architecture == "pressure-fourier":
        field.model = create_pressure_fourier_model(field.model, torch=torch)
    if hasattr(field.model, "architecture_metadata"):
        run.metadata["architecture"] = field.model.architecture_metadata()
    parameters = tuple(parameter for parameter in field.model.parameters() if parameter.requires_grad)
    run.optimizer = torch.optim.Adam(parameters, lr=1e-3)
    labels = TimeLabelSampler(run.losses, seed=data.config.seed ^ 0xA11CE)
    if options.strategy == "temporal-pressure":
        from .pressure_pilots import PressureTemporalSampler
        pressure_sampler = PressureTemporalSampler(run.losses)
    else:
        pressure_sampler = None
    time_rng = np.random.default_rng(data.config.seed ^ 0x71AE)
    sampler = FrontResidualSampler(data.config.seed ^ 0xF20A7) if "front" in options.strategy else None
    causal = CausalWeights(data.snapshot_count, epsilon=options.causal_epsilon,
                           floor=options.causal_floor) if "causal" in options.strategy else None
    balancer = LabelGradientBalancer(cancel=cancel) if options.balance_labels else None
    fixed_centers = np.unique(np.linspace(0, data.point_count - 1,
        min(field.maximum_batch, data.point_count), dtype=np.int64))
    panels = base._audit_intervals(data)
    options_record = {name: getattr(options, name) for name in options.__dataclass_fields__}
    run.metadata.update(research=True, research_method_version=1, research_options=options_record,
        research_options_sha256=sha256(json.dumps(options_record, sort_keys=True).encode()).hexdigest(),
        originating_run_id=source.run_id, originating_config_hash=source.config.config_hash,
        requested_training_steps=source_copy.config.pinn_epochs,
        optimizer={"name": "explicit research Adam + original raw L-BFGS",
                   "adam": {"steps": options.adam_steps}, "lbfgs": {"steps": options.lbfgs_steps, "closure_evaluation_cap": 8}},
        research_training_weak_scales=training_scales,
        research_weak_normalization="training labels only; original oracle restored for final qualification",
        research_label_sampling=("method-5 free-DOF stratified training labels" if method5_pilot else
            "same-accepted-time stratified free-DOF training labels; separate seeded RNG"),
        research_physics_evaluations={"adam": 0, "causal_bootstrap": 0},
        research_completed=False)
    initial_validation = run.losses.validation_loss()
    initial_norms = None
    last_validation = initial_validation
    timed_out = False
    stopped = None
    completed = False
    # The timer includes preparation. Final diagnosis/publication is reported
    # separately and cannot turn an interrupted experiment into qualification.
    timer = threading.Timer(max(.001, options.seconds - (time.monotonic() - start)), cancel.set)
    timer.daemon = True
    timer.start()

    def check_cancel():
        if time.monotonic() >= deadline:
            cancel.set()
        _cancelled(cancel)

    def record(terms, stage, *, validation=False, time_weight=1.):
        nonlocal last_validation
        epoch = len(run.history["epoch"]) + 1
        for key in ("loss", "data_loss", "physics_loss", "boundary_loss"):
            run.history[key].append(float(terms[key].detach()))
        run.history["epoch"].append(float(epoch))
        run.history["validation_loss"].append(last_validation)
        run.metadata.update(actual_training_steps=epoch, training_epochs=epoch,
                            last_physics_evaluated_epoch=epoch)
        stages = run.metadata["training_stages"]
        if not stages or stages[-1]["stage"] != stage:
            stages.append({"stage": stage, "first_step": epoch, "last_step": epoch, "steps": 1})
        else:
            stages[-1].update(last_step=epoch, steps=stages[-1]["steps"] + 1)
        if validation:
            last_validation = run.losses.validation_loss()
            run.history["validation_loss"][-1] = last_validation
        if epoch % 100 == 0 or validation:
            progress(.9 * epoch / source_copy.config.pinn_epochs,
                f"Research {stage} {epoch}/{source_copy.config.pinn_epochs}; raw physics={float(terms['physics_loss'].detach()):.6g}; time weight={time_weight:.4g}")

    try:
        if causal is not None:
            for ti in range(1, data.snapshot_count):
                check_cancel()
                ids = data.train_ids[data.train_ids // data.point_count == ti][:field.maximum_batch]
                label_loss = run.losses.sampled_loss(ids) if len(ids) else torch.zeros((), **field.kw)
                physics, boundary = (run.losses.thermal_physics(ti, fixed_centers) if thermal
                                     else run.losses.coupled_physics(ti))
                causal.update(ti, float((label_loss + physics + boundary).detach()))
                run.metadata["research_physics_evaluations"]["causal_bootstrap"] += 1
        warmup = min(100, options.adam_steps // 8) if thermal and options.adam_steps >= 50 else 0
        run.metadata["research_native_warmup_steps"] = warmup
        run.metadata["research_native_ramp_steps"] = warmup
        for step in range(options.adam_steps):
            check_cancel()
            field.model.train()
            run.optimizer.zero_grad(set_to_none=True)
            ti = int(time_rng.integers(1, data.snapshot_count))
            components = {} if method5_pilot else labels.components(ti)
            if initial_norms is None and components:
                initial_norms = _label_gradient_norms(components, parameters, torch)
            label_loss = (pressure_sampler.sampled_loss() if pressure_sampler is not None else
                          run.losses.sampled_training_loss() if method5_pilot else
                          balancer.combine(components, parameters, step) if balancer is not None else
                          torch.stack(tuple(components.values())).mean() if components else
                          parameters[0].sum() * 0.)
            if thermal:
                if sampler is None:
                    centers = time_rng.choice(data.point_count, min(field.maximum_batch, data.point_count), replace=False)
                    physics, boundary = run.losses.thermal_physics(ti, centers)
                else:
                    # Keep the baseline's time/center RNG sequence unchanged.
                    time_rng.choice(data.point_count, min(field.maximum_batch, data.point_count), replace=False)
                    physics, boundary = sampler.loss(run.losses, ti, step)
            else:
                physics, boundary = run.losses.coupled_physics(ti)
            run.metadata["research_physics_evaluations"]["adam"] += 1
            raw_joint = label_loss + physics + boundary
            time_weight = causal.weight(ti) if causal is not None else 1.
            if causal is not None:
                causal.update(ti, float(raw_joint.detach()))
            ramp = 0. if step < warmup else (step - warmup + 1) / warmup if warmup and step < 2 * warmup else 1.
            objective = time_weight * (label_loss + ramp * (physics + boundary))
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("The research objective became non-finite.")
            check_cancel()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
            check_cancel()
            run.optimizer.step()
            run.metadata["optimizer_steps_completed"]["adam"] += 1
            run.metadata["function_evaluations"]["adam"] += 1
            record({"loss": raw_joint, "data_loss": label_loss, "physics_loss": physics, "boundary_loss": boundary},
                   "Adam", validation=step + 1 == options.adam_steps, time_weight=time_weight)
        if options.lbfgs_steps:
            run.optimizer = torch.optim.LBFGS(parameters, lr=1., max_iter=1, max_eval=8,
                history_size=20, line_search_fn="strong_wolfe")
        for step in range(options.lbfgs_steps):
            check_cancel()
            previous_outcomes = sum(run.metadata["refinement_step_outcomes"].values())
            try:
                terms, _ = base._refinement_iteration(run, panels, fixed_centers, parameters)
            except SimulationCancelled:
                if sum(run.metadata["refinement_step_outcomes"].values()) > previous_outcomes:
                    run.metadata["optimizer_steps_completed"]["lbfgs"] += 1
                    previous = {name: torch.as_tensor(run.history[name][-1] if run.history[name] else 0., **field.kw)
                                for name in ("loss", "data_loss", "physics_loss", "boundary_loss")}
                    record(previous, "unweighted L-BFGS")
                    run.metadata["research_interrupted_after_accepted_step"] = True
                raise
            run.metadata["optimizer_steps_completed"]["lbfgs"] += 1
            record(terms, "unweighted L-BFGS", validation=step + 1 == options.lbfgs_steps)
        check_cancel()
        completed = True
    except SimulationCancelled as error:
        timed_out = time.monotonic() - start >= options.seconds
        stopped = str(error)
    finally:
        if not method5_pilot or not completed:
            timer.cancel()
    training_seconds = time.monotonic() - start
    _cancelled_after_training = cancel.is_set()
    if method5_pilot and not completed:
        if timed_out:
            raise ResearchPilotTimeout(len(run.history["epoch"]), training_seconds)
        raise SimulationCancelled(stopped or "Pressure research was cancelled.")
    if not method5_pilot:
        cancel.clear()

    def finish():
        if method5_pilot:
            # Selection uses the same FE equations normalized on training labels.
            selection_terms = base._deterministic_objective(run, panels, fixed_centers, backward=False)
            diagnostics = base._training_diagnostics(run, panels)
            run.metadata["research_selection"] = {
                "source": "training labels and training-normalized raw equations only",
                "pressure_normalized_rmse": diagnostics["fields"].get("pressure"),
                "maximum_residual_rms": max(diagnostics["residuals"].values(), default=0.),
                "physics_loss": float(selection_terms["physics_loss"].detach()),
                "boundary_loss": float(selection_terms["boundary_loss"].detach()),
                "diagnostics": diagnostics,
                "eligible": completed,
            }
        # Restore the original source oracle for final qualification.
        field.data = replace(field.data, weak=original_weak)
        run.metadata.update(research_completed=completed, research_training_seconds=training_seconds,
                            research_initial_label_gradient_norms=initial_norms,
                            research_label_sampler=labels.metadata())
        if pressure_sampler is not None:
            run.metadata["research_pressure_sampler"] = pressure_sampler.metadata()
        if sampler is not None:
            run.metadata["research_front_sampler"] = sampler.metadata()
        if causal is not None:
            run.metadata["research_causal"] = causal.metadata()
        if balancer is not None:
            run.metadata["research_label_balancer"] = balancer.metadata()
        if not run.history["epoch"]:
            run.history = {key: [0.] for key in run.history}
            run.metadata.update(actual_training_steps=0, training_epochs=0)
        terms = base._deterministic_objective(run, panels, fixed_centers, backward=False)
        run.metadata["function_evaluations"]["final_objective"] += 1
        for key, value in terms.items():
            run.history[key][-1] = float(value.detach())
        run.history["validation_loss"][-1] = run.losses.validation_loss()
        predicted = base._predict_fields(run, progress, cancel)
        metrics = validate_prediction(run, predicted, initial_validation, progress, cancel)
        scientific_status = run.metadata["validation_status"]
        if not completed:
            run.metadata["validation_status"] = "failed"
        run.metadata["research_scientific_gate_status"] = scientific_status
        candidate = base._assemble_result(run, predicted, metrics, progress, cancel)
        validate_pinn_source(candidate, source_copy)
        if method5_pilot and cancel.is_set():
            # A deadline or interruption during atomic checkpoint publication
            # must not leave a qualified pilot checkpoint behind.
            if run.checkpoint_path is not None:
                run.checkpoint_path.unlink(missing_ok=True)
            _cancelled(cancel)
        if _cancelled_after_training:
            cancel.set()
        evidence = {"completed": completed, "timed_out": timed_out, "stop_reason": stopped,
            "scientific_gate_status": scientific_status,
            "status": "PASS" if completed and scientific_status == "passed" else "FAIL" if completed else "TIMEOUT" if timed_out else "INTERRUPTED",
            "training_seconds": training_seconds, "elapsed_s": time.monotonic() - start}
        if method5_pilot:
            evidence["training_only_selection"] = run.metadata["research_selection"]
        return source_copy, candidate, evidence

    try:
        return finish()
    except SimulationCancelled as error:
        if method5_pilot and time.monotonic() >= deadline:
            raise ResearchPilotTimeout(len(run.history["epoch"]), time.monotonic() - start) from error
        raise
    finally:
        timer.cancel()
