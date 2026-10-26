"""Fast coverage-proof checks; actual scientific cases run in isolated children."""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import unittest

from simulation_suite.scientific_catalog import (
    BORIS_FACTORS,
    DOSE_FACTORS,
    PRESETS,
    matrix_coverage,
    pair_coverage,
    pairwise_rows,
    scientific_cases,
)


class ScientificMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = scientific_cases()
        cls.by_id = {case.id: case for case in cls.cases}

    def test_case_graph_is_unique_closed_and_acyclic(self):
        self.assertEqual(len(self.by_id), len(self.cases))
        complete = set()
        pending = dict(self.by_id)
        while pending:
            ready = [
                name
                for name, case in pending.items()
                if set(case.depends_on) <= complete
            ]
            self.assertTrue(
                ready, f"Missing prerequisites or dependency cycle: {sorted(pending)}"
            )
            for name in ready:
                complete.add(name)
                pending.pop(name)

    def test_every_shipped_preset_runs_with_unchanged_configuration(self):
        actual = {
            case.parameters["preset"]
            for case in self.cases
            if case.parameters.get("recipe") == "preset"
        }
        self.assertEqual(actual, set(PRESETS))
        for case in self.cases:
            if case.parameters.get("recipe") == "preset":
                self.assertEqual(set(case.parameters), {"action", "recipe", "preset"})

    def test_twelve_fe_phase_mode_closure_combinations_are_explicit(self):
        rows = [
            case.parameters
            for case in self.cases
            if case.parameters.get("recipe") == "fe_matrix"
        ]
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            {(row["phase"], row["mode"], row["closure"]) for row in rows},
            set(
                product(
                    ("solid", "mushy", "liquid"), ("driven", "dynamo"), (False, True)
                )
            ),
        )
        for name in (
            "fe.melting",
            "fe.freezing",
            "fe.stationary",
            "fe.references",
            "native.melting",
            "native.freezing",
        ):
            self.assertIn(name, self.by_id)

    def test_boris_and_geant4_cover_every_declared_feasible_pair(self):
        for prefix, factors in (
            ("boris.pair.", BORIS_FACTORS),
            ("dose.pair.", DOSE_FACTORS),
        ):
            with self.subTest(prefix=prefix):
                rows = [
                    {key: case.parameters[key] for key in factors}
                    for case in self.cases
                    if case.id.startswith(prefix)
                ]
                proof = pair_coverage(factors, rows)
                self.assertTrue(proof["complete"], proof)
                self.assertGreater(proof["required_pairs"], 80)
                self.assertEqual(proof["missing_pairs"], [])
                self.assertLess(len(rows), len(list(product(*factors.values()))))
                self.assertEqual(rows, pairwise_rows(factors))

    def test_pairwise_proof_detects_missing_levels_and_pairs(self):
        factors = {"species": ("proton", "alpha"), "ground": (False, True)}
        proof = pair_coverage(factors, [{"species": "proton", "ground": False}])
        self.assertFalse(proof["complete"])
        self.assertEqual(len(proof["missing_pairs"]), 3)
        self.assertEqual(
            proof["missing_levels"], {"species": ["alpha"], "ground": [True]}
        )

    def test_source_ground_and_pinn_identity_have_dependencies(self):
        for case in self.cases:
            if case.parameters["action"] not in {"dose", "boris"}:
                continue
            field = case.parameters["field"]
            ground = "ground" if case.parameters["ground"] else "vacuum"
            if case.parameters["action"] == "dose" or field in {"driven", "dynamo"}:
                self.assertIn(
                    f"fe.transport.{'zero' if field == 'pinn' else field}.{ground}",
                    case.depends_on,
                )
            if field == "pinn":
                self.assertIn(f"pinn.transport.zero.{ground}", case.depends_on)
            if case.parameters["action"] == "dose":
                self.assertIn("geant4", case.capabilities)
                if case.parameters["target"] != "water":
                    self.assertIn("icrp110", case.capabilities)

    def test_cpu_cuda_thermal_coupled_and_unchanged_gates_are_required(self):
        for problem, device in product(("thermal", "stationary"), ("cpu", "cuda")):
            case = self.by_id[f"pinn.{problem}.{device}"]
            self.assertEqual(case.parameters["device"], device)
            self.assertIn(f"torch_{device}", case.capabilities)
            self.assertEqual(case.kind, "scientific")
        for name in ("dose.water-precision", "dose.fluence-linearity"):
            self.assertEqual(self.by_id[name].kind, "scientific")
        for case in self.cases:
            self.assertNotIn("threshold", case.parameters)

    def test_invalid_cases_require_only_their_real_execution_boundary(self):
        self.assertFalse(self.by_id["invalid.native_dose"].depends_on)
        self.assertFalse(self.by_id["invalid.small_anatomy"].depends_on)
        self.assertEqual(
            self.by_id["invalid.small_anatomy"].capabilities, ("geant4", "icrp110")
        )
        self.assertIn("invalid.failed_pinn", self.by_id)

    def test_cases_have_finite_budgets_and_fingerprint_sources(self):
        for case in self.cases:
            self.assertGreater(case.timeout, 0)
            self.assertLessEqual(case.timeout, 3600)
            self.assertIn("--request", case.command)
            self.assertIn("--result", case.command)
            self.assertTrue(case.dependencies)
            for source in case.dependencies:
                self.assertTrue(
                    any(
                        path.is_file()
                        for path in Path(__file__).resolve().parents[1].glob(source)
                    ),
                    source,
                )
            json.dumps(case.to_dict(), allow_nan=False)

    def test_only_lunar_preset_budget_is_extended_and_precision_keeps_its_deadline(
        self,
    ):
        for name in PRESETS:
            case = self.by_id["preset." + name.lower().replace(" ", "-")]
            self.assertEqual(
                case.timeout, 3600.0 if name == "Lunar habitat" else 1800.0
            )
            self.assertEqual(
                case.parameters, {"action": "solve", "preset": name, "recipe": "preset"}
            )
        precision = self.by_id["dose.water-precision"]
        self.assertEqual(precision.parameters["maximum_histories"], 524288)
        self.assertEqual(precision.timeout, 3600.0)
        self.assertEqual(precision.kind, "scientific")

    def test_report_declares_finite_scope_and_unqualified_research(self):
        proof = matrix_coverage()
        self.assertTrue(proof["finite_scope"])
        self.assertEqual(proof["case_count"], len(self.cases))
        self.assertEqual(
            proof["fe_phase_mode_closure"],
            {"required": 12, "covered": 12, "missing": []},
        )
        self.assertTrue(proof["complete"])
        self.assertTrue(proof["boris"]["complete"])
        self.assertTrue(proof["geant4"]["complete"])
        self.assertTrue(proof["not_qualified_by_this_matrix"])

    def test_report_proves_actual_executable_cases_not_generator_intention(self):
        cases = [
            case for case in self.cases if case.parameters.get("species") != "iron"
        ]
        proof = matrix_coverage(cases)
        self.assertFalse(proof["complete"])
        self.assertIn("iron", proof["boris"]["missing_levels"]["species"])
        self.assertIn("iron", proof["geant4"]["missing_levels"]["species"])
        cases = [
            case
            for case in self.cases
            if case.id
            not in {"fe.mushy.dynamo.vms", "preset.iron", "pinn.thermal.cuda"}
        ]
        proof = matrix_coverage(cases)
        self.assertFalse(proof["complete"])
        self.assertEqual(proof["missing_presets"], ["Iron"])
        self.assertEqual(
            proof["fe_phase_mode_closure"]["missing"], [("mushy", "dynamo", True)]
        )
        self.assertEqual(
            proof["missing_pinn_device_problem_cases"], ["pinn.thermal.cuda"]
        )


if __name__ == "__main__":
    unittest.main()
