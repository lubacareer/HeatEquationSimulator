"""Small CPU-only correction and restricted checkpoint compatibility checks."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from magnetic_torus.models import RunResult, ScenarioConfig
from magnetic_torus.artifacts import FieldArtifact
from magnetic_torus.learning.corrections import (
    create_correction_model,
    restore_model_from_checkpoint,
)
from magnetic_torus.learning.fields import FORMAT_VERSION, _make_network

try:
    import torch
except ImportError:
    torch = None

SUITE_CAPABILITIES = ("torch_cpu",)


def source_run(*, thermal=False):
    config = replace(
        ScenarioConfig(), solver="thermal_benchmark" if thermal else "fenics"
    )
    return RunResult(
        "correction-source", config, np.array([0.0, 0.1]), np.zeros((2, 3)), {}, {}, {}
    )


def checkpoint_for(model, source, architecture):
    return {
        "format_version": FORMAT_VERSION,
        "source_run_id": source.run_id,
        "config_hash": source.config.config_hash,
        "model_state": model.state_dict(),
        "metadata": {
            "source_run_id": source.run_id,
            "source_config_hash": source.config.config_hash,
            "source_physics_hash": source.config.physics_hash,
            "source_snapshot_times_s": source.times.tolist(),
            "architecture": architecture,
        },
    }


@unittest.skipIf(torch is None, "PyTorch not installed")
class CorrectionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(123)
        self.core = _make_network(torch, 12, 21).double()
        self.inputs = torch.randn(11, 21, dtype=torch.float64)

    def test_clones_preserve_the_initial_function_and_original_model_without_random_draws(
        self,
    ):
        original = deepcopy(self.core.state_dict())
        state = torch.random.get_rng_state().clone()
        model = create_correction_model(
            self.core, ("elliptic", "thermofluid"), torch=torch
        )
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        torch.testing.assert_close(
            model(self.inputs), self.core(self.inputs), rtol=1e-14, atol=1e-15
        )
        for name, value in self.core.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]))
            self.assertTrue(torch.equal(model.core.state_dict()[name], value))
        self.assertTrue(
            all(parameter.requires_grad for parameter in self.core.parameters())
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.core.parameters())
        )
        for branch, rows in (
            (model.elliptic, [10, 11]),
            (model.thermofluid, [0, 1, 2, 3]),
        ):
            self.assertEqual(branch[0].weight.shape, (64, 53))
            self.assertTrue(torch.equal(branch[0].weight[:, :21], self.core[0].weight))
            self.assertEqual(torch.count_nonzero(branch[0].weight[:, 21:]).item(), 0)
            self.assertTrue(torch.equal(branch[2].weight, self.core[2].weight))
            self.assertTrue(torch.equal(branch[4].weight, self.core[4].weight))
            self.assertTrue(torch.equal(branch[-1].weight, self.core[-1].weight[rows]))
        branch_ids = {id(parameter) for parameter in model.branch_parameters()}
        self.assertTrue(branch_ids)
        self.assertFalse(
            branch_ids & {id(parameter) for parameter in model.core.parameters()}
        )

    def test_training_selected_branches_preserves_every_inactive_output_and_core_weight(
        self,
    ):
        for groups, active, inactive in (
            (("elliptic",), [10, 11], list(range(10))),
            (("thermofluid",), list(range(4)), list(range(4, 12))),
            (
                ("elliptic", "thermofluid"),
                list(range(4)) + [10, 11],
                list(range(4, 10)),
            ),
        ):
            with self.subTest(groups=groups):
                model = create_correction_model(self.core, groups, torch=torch)
                before = model(self.inputs).detach().clone()
                original = deepcopy(model.core.state_dict())
                optimizer = torch.optim.SGD(model.branch_parameters(), lr=0.01)
                optimizer.zero_grad()
                model(self.inputs)[:, active].square().sum().backward()
                self.assertTrue(
                    all(parameter.grad is None for parameter in model.core.parameters())
                )
                optimizer.step()
                after = model(self.inputs).detach()
                self.assertTrue(torch.equal(after[:, inactive], before[:, inactive]))
                self.assertFalse(torch.equal(after[:, active], before[:, active]))
                for name, value in model.core.state_dict().items():
                    self.assertTrue(torch.equal(value, original[name]))

    def test_spatial_enrichment_keeps_input_derivatives_and_uses_declared_geometry_features(
        self,
    ):
        model = create_correction_model(self.core, ("elliptic",), torch=torch)
        inputs = self.inputs.clone().requires_grad_(True)
        captured = []
        handle = model.elliptic[0].register_forward_pre_hook(
            lambda _module, args: captured.append(args[0])
        )
        prediction = model(inputs)
        handle.remove()
        enriched = captured[0]
        torch.testing.assert_close(enriched[:, :21], inputs, rtol=0, atol=0)
        coords = torch.stack(
            (
                inputs[:, 3] * inputs[:, 4],
                inputs[:, 3] * inputs[:, 5],
                inputs[:, 6],
                inputs[:, 7],
            ),
            dim=1,
        )
        phase = torch.pi * coords[:, :, None] * inputs.new_tensor([1, 2, 4, 8])
        torch.testing.assert_close(
            enriched[:, 21:37], phase.sin().flatten(1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            enriched[:, 37:], phase.cos().flatten(1), rtol=0, atol=0
        )
        actual = torch.autograd.grad(
            prediction[:, 4:10].sum(), inputs, retain_graph=True
        )[0]
        expected = torch.autograd.grad(self.core(inputs)[:, 4:10].sum(), inputs)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        active_gradient = torch.autograd.grad(prediction[:, 10:].sum(), inputs)[0]
        self.assertTrue(torch.isfinite(active_gradient).all())
        self.assertGreater(float(active_gradient.abs().sum()), 0)

    def test_known_legacy_and_correction_checkpoints_restore_from_restricted_files(
        self,
    ):
        for thermal in (False, True):
            source = source_run(thermal=thermal)
            core = _make_network(
                torch, 1 if thermal else 12, 4 if thermal else 21
            ).double()
            models = [core]
            if not thermal:
                models.append(
                    create_correction_model(
                        core, ("elliptic", "thermofluid"), torch=torch
                    )
                )
            for model in models:
                with (
                    self.subTest(thermal=thermal, model=type(model).__name__),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    architecture = (
                        model.architecture_metadata()
                        if hasattr(model, "architecture_metadata")
                        else {
                            "widths": [
                                4 if thermal else 21,
                                64,
                                64,
                                64,
                                1 if thermal else 12,
                            ],
                            "activation": "tanh",
                            "dtype": "float64",
                        }
                    )
                    path = Path(directory) / "checkpoint.pt"
                    torch.save(checkpoint_for(model, source, architecture), path)
                    state = torch.random.get_rng_state().clone()
                    with patch.object(torch, "load", wraps=torch.load) as load:
                        restored = restore_model_from_checkpoint(path, source)
                    self.assertTrue(load.call_args.kwargs["weights_only"])
                    self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
                    inputs = self.inputs if not thermal else self.inputs[:, :4]
                    torch.testing.assert_close(
                        restored(inputs), model(inputs), rtol=0, atol=0
                    )
                    if not thermal and model is not core:
                        self.assertEqual(restored.active_groups, model.active_groups)
                        self.assertTrue(
                            all(not p.requires_grad for p in restored.core.parameters())
                        )

    def test_restore_rejects_misattribution_unknown_architecture_and_malformed_weights(
        self,
    ):
        source = source_run()
        model = create_correction_model(self.core, ("elliptic",), torch=torch)
        checkpoint = checkpoint_for(model, source, model.architecture_metadata())
        for mutate in (
            lambda payload: payload.update(source_run_id="different"),
            lambda payload: payload.update(config_hash="different"),
            lambda payload: payload.update(format_version=999),
            lambda payload: payload["metadata"].update(source_physics_hash="different"),
            lambda payload: payload["metadata"].update(
                source_snapshot_times_s=[0.0, 99.0]
            ),
            lambda payload: payload["metadata"]["architecture"].update(
                kind="untrusted.module.Model"
            ),
            lambda payload: payload["metadata"]["architecture"].update(version=999),
            lambda payload: payload["metadata"]["architecture"].update(
                branch_input_width=55
            ),
            lambda payload: payload["metadata"]["architecture"].update(
                active_groups=["unknown"]
            ),
            lambda payload: payload["model_state"].pop("core.0.weight"),
            lambda payload: payload["model_state"]["core.0.weight"].fill_(float("nan")),
        ):
            payload = deepcopy(checkpoint)
            mutate(payload)
            with (
                self.subTest(payload=payload["metadata"]["architecture"]),
                self.assertRaises(ValueError),
            ):
                restore_model_from_checkpoint(payload, source)
        with self.assertRaises(ValueError):
            restore_model_from_checkpoint(
                checkpoint, replace(source, run_id="other-source")
            )

    def test_restore_checks_the_original_operator_artifact_when_provenance_is_recorded(
        self,
    ):
        from magnetic_torus.learning.fields import _operator_fingerprint

        with tempfile.TemporaryDirectory() as directory:
            operator = Path(directory) / "weak-operator.npz"
            operator.write_bytes(b"accepted original operator bytes")
            source = replace(
                source_run(),
                artifacts={
                    "weak_operator_path": FieldArtifact.capture(
                        operator, root=directory
                    )
                },
            )
            architecture = {
                "widths": [21, 64, 64, 64, 12],
                "activation": "tanh",
                "dtype": "float64",
            }
            checkpoint = checkpoint_for(self.core, source, architecture)
            checkpoint["metadata"]["source_operator_sha256"] = _operator_fingerprint(
                None, source
            )
            restored = restore_model_from_checkpoint(checkpoint, source)
            torch.testing.assert_close(
                restored(self.inputs), self.core(self.inputs), rtol=0, atol=0
            )
            operator.write_bytes(b"different operator bytes")
            altered = replace(
                source,
                artifacts={
                    "weak_operator_path": FieldArtifact.capture(
                        operator, root=directory
                    )
                },
            )
            with self.assertRaisesRegex(ValueError, "physical operator"):
                restore_model_from_checkpoint(checkpoint, altered)

    def test_invalid_groups_and_core_shapes_are_explicit_errors(self):
        for groups in (("unknown",), ("elliptic", "elliptic"), ["elliptic"], ({},)):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                create_correction_model(self.core, groups, torch=torch)
        with self.assertRaises(ValueError):
            create_correction_model(_make_network(torch, 1), ("elliptic",), torch=torch)
        with self.assertRaises(ValueError):
            create_correction_model(self.core, (), torch=torch)(self.inputs[:, :20])


class LazyCorrectionImportTests(unittest.TestCase):
    def test_import_does_not_load_torch(self):
        script = "import sys; sys.modules['torch']=None; import magnetic_torus.learning.corrections; assert sys.modules['torch'] is None"
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
