"""Fixed training-only continuation and frozen-core correction regressions."""

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from magnetic_torus.learning import training
from magnetic_torus.learning.corrections import create_correction_model
from magnetic_torus.models import SimulationCancelled
from test_torus_pinn import uniform_heating_run
import test_pinn_batching as batching
import test_pinn_conditioning as conditioning

SUITE_CAPABILITIES = ("torch_cpu",)


@unittest.skipIf(torch is None, "PyTorch is required")
class PINNRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def prepare(self, *, output=None):
        cancel = threading.Event()
        with patch.dict(os.environ, {"MAGNETIC_TORUS_PINN_DEVICE": "cpu"}):
            data = training._prepare_data(uniform_heating_run(pinn_epochs=50), cancel)
            return training._prepare_training(data, lambda *_: None, cancel, output)

    def test_native_warmup_is_fixed_and_coupled_primary_weight_is_unchanged(self):
        values = [training._physics_weight(i, 1000, thermal=True) for i in range(201)]
        self.assertEqual(values[:100], [0.0] * 100)
        np.testing.assert_allclose(values[100:200], np.arange(1, 101) / 100)
        self.assertEqual(values[200], 1.0)
        self.assertEqual(training._physics_weight(0, 49, thermal=True), 1.0)
        self.assertTrue(
            all(
                training._physics_weight(i, 1000, thermal=False) == 1.0
                for i in range(200)
            )
        )

    def test_primary_draws_do_not_change_when_native_physics_is_warmed_up(self):
        runs = [self.prepare(), self.prepare()]
        with patch.object(training, "_physics_weight", return_value=1.0):
            training._train_model(runs[0], lambda *_: None, runs[0].losses.cancel)
        training._train_model(runs[1], lambda *_: None, runs[1].losses.cancel)
        self.assertEqual(
            runs[0].field.data.rng.bit_generator.state,
            runs[1].field.data.rng.bit_generator.state,
        )
        self.assertFalse(runs[1].metadata["conditioning_activation"]["active_groups"])

    def test_audit_panels_include_native_charge_transition_neighbors(self):
        times = np.linspace(0.0, 1.0, 32) ** 1.2
        data = SimpleNamespace(
            snapshot_count=len(times),
            thermal=True,
            result=SimpleNamespace(times=times),
            config=SimpleNamespace(charge_fraction=0.41, duration=1.0),
        )
        indices = training._audit_intervals(data)
        transition = int(np.searchsorted(times, 0.41))
        self.assertTrue({transition - 1, transition, transition + 1} <= set(indices))
        self.assertTrue(
            set(np.linspace(1, len(times) - 1, 10, dtype=int)) <= set(indices)
        )
        self.assertTrue(np.all(np.diff(indices) > 0))
        data.thermal = False
        self.assertEqual(len(training._audit_intervals(data)), 10)

    def test_activation_uses_strict_training_target_and_named_equations(self):
        data = SimpleNamespace(thermal=False)
        empty = {"fields": {}, "residuals": {}}
        self.assertEqual(training._correction_groups(data, empty), ())
        self.assertEqual(
            training._correction_groups(
                data, {"fields": {"pressure": 0.05}, "residuals": {}}
            ),
            (),
        )
        self.assertEqual(
            training._correction_groups(
                data,
                {
                    "fields": {"electric_potential": 0.050001},
                    "residuals": {"continuity": 0.06},
                },
            ),
            ("elliptic", "thermofluid"),
        )
        self.assertEqual(
            training._correction_groups(
                data, {"fields": {}, "residuals": {"momentum": 0.06, "enthalpy": 0.1}}
            ),
            ("elliptic", "thermofluid"),
        )
        data.thermal = True
        self.assertEqual(
            training._correction_groups(
                data, {"fields": {"enthalpy": 0.06}, "residuals": {}}
            ),
            ("thermal",),
        )

    def test_training_diagnostics_ignore_changed_heldout_labels(self):
        source = batching.label_fixture(20)
        first = batching.prepared_labels(source)
        h = np.array(source.fields["enthalpy"], copy=True)
        ti, pi = np.divmod(first.field.data.validation_ids, len(source.points))
        h[ti[ti > 0], pi[ti > 0]] += 1e6
        second = batching.prepared_labels(
            replace(source, fields={**source.fields, "enthalpy": h})
        )
        before = deepcopy(first.field.data.rng.bit_generator.state)
        values = [
            training._training_diagnostics(
                run, training._audit_intervals(run.field.data)
            )
            for run in (first, second)
        ]
        self.assertEqual(values[0], values[1])
        self.assertEqual(before, first.field.data.rng.bit_generator.state)

    def test_heldout_pressure_and_potential_cannot_change_optimized_corrections(self):
        # Synthetic label/residual fixture only: no FE solve or physics claim.
        def prepare(changed_heldout):
            with torch.random.fork_rng():
                torch.manual_seed(42)
                field = conditioning.PINNConditioningTests().coupled_field()
            source = field.data.result
            config = replace(source.config, pinn_epochs=50, pinn_batch_size=8)
            fields = {
                name: np.zeros_like(value) for name, value in source.fields.items()
            }
            fields["temperature"][:] = config.melting_temperature
            for name, amplitude in (("pressure", 0.2), ("electric_potential", 0.3)):
                fields[name][1:] = amplitude
            heldout = np.array([7, 13], dtype=np.int64)
            train = np.setdiff1d(np.arange(15), heldout)
            if changed_heldout:
                ti, pi = np.divmod(heldout, 5)
                fields["pressure"][ti, pi] += 10.0
                fields["electric_potential"][ti, pi] -= 20.0
            source = replace(source, config=config, fields=fields)
            reference = field.data.scaling.references(source)
            cache = training._fixed_training_cache(
                source, field.data.scaling, reference, train, 64 * 1024**2
            )

            def weak(_config, _result, values, _times):
                zero = values["pressure"].sum() * 0.0
                return {
                    name: zero.reshape(1)
                    for name in (
                        "enthalpy",
                        "momentum",
                        "continuity",
                        "induction",
                        "charge",
                        "gauge",
                    )
                }

            field.data = replace(
                field.data,
                result=source,
                reference=reference,
                train_ids=train,
                validation_ids=heldout,
                cache=cache,
                rng=np.random.default_rng(42),
                weak=weak,
            )
            with torch.no_grad():
                field.model[-1].weight.zero_()
                field.model[-1].bias.zero_()
                field.model[-1].bias[10:] = 1.0
            field.maximum_batch = 8
            cancel = threading.Event()
            losses = training.TrainingLosses(
                field, np.ones((5, 12), dtype=bool), cancel
            )
            with patch.object(
                training, "_operator_fingerprint", return_value="synthetic-fixture"
            ):
                metadata = training._training_metadata(field.data, field, None)
            initial_loss = float(losses.stream_training_loss(backward=False))
            history = {
                "epoch": [50.0],
                "loss": [initial_loss],
                "data_loss": [initial_loss],
                "physics_loss": [0.0],
                "boundary_loss": [0.0],
                "validation_loss": [losses.validation_loss()],
            }
            optimizer = torch.optim.Adam(field.model.parameters(), lr=1e-3)
            return training.TrainingRun(
                field,
                losses,
                optimizer,
                None,
                "synthetic-correction-fixture",
                None,
                metadata,
                history,
            )

        runs = [prepare(False), prepare(True)]
        np.testing.assert_array_equal(
            runs[0].field.data.cache.targets, runs[1].field.data.cache.targets
        )
        schedules = []
        for run in runs:
            schedule = []

            def record(objective, stage, *, stage_end=False, detail=""):
                epoch = int(run.history["epoch"][-1]) + 1
                for key, value in objective.items():
                    run.history[key].append(float(value.detach()))
                run.history["epoch"].append(float(epoch))
                # Deliberately monitor changed held-out labels on EVERY step;
                # the optimizer and activation must remain independent of them.
                run.history["validation_loss"].append(run.losses.validation_loss())
                run.metadata["heldout_evaluated_epoch"] = epoch
                run.metadata["actual_training_steps"] = epoch
                run.metadata["last_physics_evaluated_epoch"] = epoch
                schedule.append((stage, stage_end, detail))

            groups = training._train_corrections(
                run,
                training._audit_intervals(run.field.data),
                np.arange(5),
                record,
                run.losses.cancel,
            )
            self.assertEqual(groups, ("elliptic",))
            self.assertEqual(
                run.metadata["additional_optimizer_steps_completed"],
                {"adam": 100, "lbfgs": 2},
            )
            schedules.append(schedule)
        self.assertEqual(schedules[0], schedules[1])
        self.assertEqual(len(schedules[0]), 102)
        self.assertEqual(
            runs[0].metadata["conditioning_activation"],
            runs[1].metadata["conditioning_activation"],
        )
        self.assertEqual(
            runs[0].metadata["function_evaluations"],
            runs[1].metadata["function_evaluations"],
        )
        self.assertEqual(
            runs[0].field.data.rng.bit_generator.state,
            runs[1].field.data.rng.bit_generator.state,
        )
        self.assertEqual(runs[0].history["loss"], runs[1].history["loss"])
        self.assertNotEqual(
            runs[0].history["validation_loss"], runs[1].history["validation_loss"]
        )
        states = [run.field.model.state_dict() for run in runs]
        self.assertEqual(states[0].keys(), states[1].keys())
        for name in states[0]:
            self.assertTrue(torch.equal(states[0][name], states[1][name]), name)

    def test_extra_native_stage_records_actual_steps_raw_audit_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.prepare(output=Path(directory))
            messages = []
            with patch.object(
                training,
                "_training_diagnostics",
                return_value={"fields": {"enthalpy": 0.06}, "residuals": {}},
            ):
                initial = training._train_model(
                    run, lambda _, text: messages.append(text), run.losses.cancel
                )
            prediction = training._predict_fields(
                run, lambda *_: None, run.losses.cancel
            )
            result = training._assemble_result(
                run, prediction, {"epochs": 152.0}, lambda *_: None, run.losses.cancel
            )
            saved = torch.load(
                result.metadata["checkpoint_path"],
                map_location="cpu",
                weights_only=True,
            )
        self.assertEqual(initial, 0.0)
        self.assertEqual(len(run.history["epoch"]), 152)
        self.assertEqual(run.metadata["requested_training_steps"], 50)
        self.assertEqual(run.metadata["actual_training_steps"], 152)
        self.assertEqual(run.metadata["training_epochs"], 152)
        self.assertEqual(saved["epoch"], 152)
        self.assertEqual(saved["source_run_id"], run.field.data.result.run_id)
        self.assertEqual(
            run.metadata["optimizer_steps_completed"], {"adam": 40, "lbfgs": 10}
        )
        self.assertEqual(
            run.metadata["additional_optimizer_steps_completed"],
            {"adam": 100, "lbfgs": 2},
        )
        self.assertEqual(run.metadata["last_physics_evaluated_epoch"], 152)
        self.assertEqual(run.metadata["function_evaluations"]["adam"], 140)
        self.assertTrue(
            all(
                f"physics@{epoch}" in text
                for epoch, text in enumerate(messages[50:150], 51)
            )
        )
        self.assertEqual(
            sum(stage["steps"] for stage in run.metadata["training_stages"]), 152
        )
        self.assertEqual(run.history["physics_loss"][-1], 0.0)

    def test_cancelled_extra_physics_evaluation_cannot_apply_an_update(self):
        run = self.prepare()
        original = run.losses.thermal_physics

        def cancel_additional(*args, **kwargs):
            value = original(*args, **kwargs)
            if len(run.history["epoch"]) >= 50:
                run.losses.cancel.set()
            return value

        with (
            patch.object(
                training,
                "_training_diagnostics",
                return_value={"fields": {"enthalpy": 0.06}, "residuals": {}},
            ),
            patch.object(run.losses, "thermal_physics", side_effect=cancel_additional),
        ):
            with self.assertRaises(SimulationCancelled):
                training._train_model(run, lambda *_: None, run.losses.cancel)
        self.assertEqual(
            run.metadata["additional_optimizer_steps_completed"]["adam"], 0
        )
        self.assertEqual(len(run.history["epoch"]), 50)

    def test_frozen_core_weak_components_keep_parameter_graph_without_input_gradients(
        self,
    ):
        # Differentiability fixture, not a numerical FE solution.
        field = conditioning.PINNConditioningTests().coupled_field()
        field.model = create_correction_model(field.model, ("elliptic",), torch=torch)

        def weak(_config, _result, values, _times):
            return {
                "enthalpy": values["enthalpy"],
                "momentum": values["velocity"],
                "continuity": values["velocity"].sum(-1),
                "induction": values["magnetic_field"],
                "charge": values["electric_potential"],
                "gauge": values["vector_potential"],
            }

        field.data = replace(field.data, weak=weak)
        losses = training.TrainingLosses(
            field, np.ones((len(field.data.result.points), 12), bool), threading.Event()
        )
        inputs = []
        hook = field.model.core.register_forward_pre_hook(
            lambda _module, args: inputs.append(args[0])
        )
        components = losses.coupled_components(1)
        hook.remove()
        self.assertTrue(all(value.requires_grad for value in components.values()))
        self.assertTrue(all(not value.requires_grad for value in inputs))
        self.assertEqual([len(value) for value in inputs], [2 * field.data.point_count])
        sum(value.square().mean() for value in components.values()).backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in field.model.core.parameters())
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in field.model.branch_parameters()
            )
        )

    def test_additional_adam_uses_full_joint_loss_and_nonzero_equation_gradients(self):
        run = self.prepare()
        for key in run.history:
            run.history[key] = [50.0 if key == "epoch" else 0.0]
        parameter = run.field.model[-1].bias
        observed = {}

        def record(objective, _stage, **_kwargs):
            observed.update(
                {key: float(value.detach()) for key, value in objective.items()}
            )
            observed["gradient"] = float(parameter.grad)
            observed["parameter"] = float(parameter.detach())
            raise SimulationCancelled("stop fixture after one accepted update")

        with (
            patch.object(
                training,
                "_training_diagnostics",
                return_value={"fields": {"enthalpy": 0.06}, "residuals": {}},
            ),
            patch.object(
                run.losses,
                "sampled_training_loss",
                side_effect=lambda: (parameter.sum() + 1).square(),
            ) as labels,
            patch.object(
                run.losses,
                "thermal_physics",
                side_effect=lambda: (
                    (parameter.sum() - 2).square(),
                    (parameter.sum() - 3).square(),
                ),
            ),
        ):
            with self.assertRaises(SimulationCancelled):
                training._train_corrections(
                    run,
                    training._audit_intervals(run.field.data),
                    np.arange(2),
                    record,
                    run.losses.cancel,
                )
        labels.assert_called_once_with()
        self.assertEqual(
            {
                key: observed[key]
                for key in ("loss", "data_loss", "physics_loss", "boundary_loss")
            },
            {"loss": 14.0, "data_loss": 1.0, "physics_loss": 4.0, "boundary_loss": 9.0},
        )
        self.assertAlmostEqual(observed["gradient"], -8.0, places=5)
        self.assertGreater(observed["parameter"], 0.0)
        self.assertEqual(
            run.metadata["additional_optimizer_steps_completed"]["adam"], 1
        )

    def test_thermal_physics_4096_forwards_preserve_loss_and_parameter_gradients(self):
        # Synthetic finite labels isolate batch equivalence, not physical accuracy.
        runs = [batching.prepared_labels(), batching.prepared_labels()]
        values, gradients, calls = [], [], []
        for limit, run in zip((256, 4096), runs):
            rows = []
            hook = run.field.model.register_forward_pre_hook(
                lambda _module, args: rows.append(len(args[0]))
            )
            with patch.object(training, "_FULL_LABEL_BATCH_SIZE", limit):
                physics, boundary = run.losses.thermal_physics(
                    4, np.arange(run.field.data.point_count)
                )
                loss = physics + boundary
                loss.backward()
            hook.remove()
            values.append(loss.detach())
            gradients.append(
                torch.cat(
                    [
                        parameter.grad.flatten()
                        for parameter in run.field.model.parameters()
                    ]
                )
            )
            calls.append(rows)
        torch.testing.assert_close(values[0], values[1], rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-11, atol=1e-12)
        self.assertLessEqual(max(calls[0]), 256)
        self.assertEqual(calls[1], [2600, 1300])
        self.assertTrue(all(count <= 4096 for count in calls[1]))

    def test_trained_wrapper_preserves_initial_state_and_each_dirichlet_mask(self):
        # Hard-constraint integration fixture, not an FE solve or qualification.
        field = conditioning.PINNConditioningTests().coupled_field()
        field.coupled_free[0, 0] = 0.0
        field.coupled_free[1, 7:10] = 0.0
        field.coupled_free[2, 10] = 0.0
        field.coupled_free[3, 11] = 0.0
        field.model = create_correction_model(
            field.model, ("elliptic", "thermofluid"), torch=torch
        )
        optimizer = torch.optim.SGD(field.model.branch_parameters(), lr=0.01)
        pi = np.arange(field.data.point_count)
        ti = np.ones(len(pi), dtype=np.int64)
        loss = field.forward(field.input_at(ti, pi), pi, ti).square().mean()
        loss.backward()
        optimizer.step()
        times = np.repeat(np.arange(field.data.snapshot_count), len(pi))
        points = np.tile(pi, field.data.snapshot_count)
        with torch.no_grad():
            actual = field.forward(field.input_at(times, points), points, times)
        actual = actual.numpy().reshape(field.data.snapshot_count, len(pi), 12)
        dynamical_columns = [*range(10), 11]
        np.testing.assert_array_equal(
            actual[0][:, dynamical_columns],
            field.coupled_trial[0][:, dynamical_columns],
        )
        # Pressure remains an elliptic unknown at t=0 outside its own mask.
        self.assertGreater(abs(actual[0, 4, 10]), 0.0)
        for node, columns in ((0, [0, 1, 2, 3]), (1, [7, 8, 9]), (2, [10]), (3, [11])):
            np.testing.assert_array_equal(
                actual[:, node][:, columns], field.coupled_trial[:, node][:, columns]
            )

    def test_correction_memory_guard_accounts_for_each_additional_trunk(self):
        data = SimpleNamespace(point_count=40000, output_width=12)
        training._validate_correction_memory(data, ("elliptic",))
        with self.assertRaises(training.UnsupportedPINNConfiguration):
            training._validate_correction_memory(data, ("elliptic", "thermofluid"))


if __name__ == "__main__":
    unittest.main()
