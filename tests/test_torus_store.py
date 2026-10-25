from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import json
from hashlib import sha256

import numpy as np

from magnetic_torus.models import ScenarioConfig, RunResult, DoseResult, PINNResult, preset
from magnetic_torus.store import RunStore


class ContractStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = RunStore(self.temporary.name)
        self.config = preset("Thermal benchmark")

    def result(self):
        return RunResult("test-run", self.config, np.array([0., 1.]), np.array([[.3, 0, 0]]),
                         {"temperature": np.array([[1800.], [1810.]])}, {"energy_j": np.array([0., 4.])},
                         {"nested": {"values": [1, 2]}, "validation_status": "test"})

    def test_validation_rejects_nonphysical_and_noninteger_parameters(self):
        for change in ({"major_radius": float("nan")}, {"efficiency": 1.1}, {"density": -1},
                       {"particle_z": 1.5}, {"particle_a": True}, {"habitat_radius": 4}, {"seed": -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.config, **change).validate()

    def test_observer_settings_reuse_physics_but_not_full_configuration_identity(self):
        changed = replace(self.config, particle_energy_mev_nucleon=400, target="female", pinn_epochs=100)
        self.assertEqual(self.config.physics_hash, changed.physics_hash)
        self.assertNotEqual(self.config.config_hash, changed.config_hash)
        self.assertNotEqual(self.config.physics_hash, replace(self.config, latent_heat=10000).physics_hash)

    def test_legacy_scenario_identity_survives_transport_schema_extension(self):
        config = replace(self.config, schema_version=1)
        previous = config.to_dict()
        for key in ("name", "preset", "transport_physics_list", "production_cut_m"):
            previous.pop(key)
        previous_hash = sha256(json.dumps(previous, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(config.config_hash, previous_hash)
        changed = replace(config, transport_physics_list="QGSP_BIC_HP")
        self.assertNotEqual(changed.config_hash, previous_hash)
        self.assertEqual(changed.physics_hash, config.physics_hash)

    def test_arrays_and_nested_metadata_are_immutable(self):
        result = self.result()
        with self.assertRaises(ValueError):
            result.fields["temperature"][0, 0] = 2
        with self.assertRaises(TypeError):
            result.metadata["nested"]["x"] = 1
        with self.assertRaises(TypeError):
            result.metadata["nested"]["values"][0] = 1

    def test_atomic_hdf_roundtrip_and_history(self):
        result = self.result()
        path = self.store.save_run(result)
        loaded = self.store.load_run(path)
        np.testing.assert_array_equal(loaded.fields["temperature"], result.fields["temperature"])
        self.assertEqual(loaded.config.config_hash, self.config.config_hash)
        self.assertEqual(self.store.list_runs()[0]["validation"], "test")
        self.assertFalse(list(path.parent.glob("*.partial")))

    def test_export_contains_independent_field_artifacts(self):
        artifact = Path(self.temporary.name) / "original-exterior.npz"
        np.savez(artifact, points=np.arange(12).reshape(4, 3))
        result = replace(self.result(), metadata={"exterior_field_path": str(artifact)})
        path = self.store.save_run(result)
        destination = Path(self.temporary.name) / "export" / "result.h5"
        destination.parent.mkdir()
        destination.write_bytes(path.read_bytes())
        loaded = self.store.load_run(destination)
        extracted = Path(loaded.metadata["exterior_field_path"])
        self.assertEqual(extracted.parent, destination.parent / "artifacts")
        self.assertEqual(extracted.read_bytes(), artifact.read_bytes())

    def test_unaccepted_field_data_cannot_be_published(self):
        with self.assertRaises(ValueError):
            replace(self.result(), fields={"temperature": np.array([[np.nan], [1.]])})

    def test_dose_shape_errors_rejected(self):
        with self.assertRaises(ValueError):
            DoseResult("dose", "test-run", "hash", 0, "water", ("water",), np.array([1., 2.]), np.array([1.]), 5)

    def test_pinn_and_dose_artifact_roundtrips(self):
        result = self.result()
        self.store.save_run(result)
        model = PINNResult("model", result.run_id, self.config.config_hash, result.fields,
                           {"loss": np.array([1., .1])}, {"heldout": .02}, {"device": "cpu"})
        self.assertEqual(self.store.load_pinn(self.store.save_pinn(model)).source_run_id, result.run_id)
        dose = DoseResult("dose", result.run_id, self.config.config_hash, 0, "water", ("Water",),
                          np.array([1e-9]), np.array([1e-11]), 100)
        np.testing.assert_array_equal(self.store.load_dose(self.store.save_dose(dose)).dose_gy, dose.dose_gy)


if __name__ == "__main__":
    unittest.main()
