"""Nonfinite scientific evidence must survive strict structured publication."""

from contextlib import redirect_stdout
from io import StringIO
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

from simulation_suite import scientific


def strict_loads(text):
    def invalid_constant(token):
        raise AssertionError(f"Nonstandard JSON constant: {token}")

    return json.loads(text, parse_constant=invalid_constant)


class ScientificSerializationTests(unittest.TestCase):
    def test_nested_native_nonfinite_values_are_distinct_and_finite_data_is_preserved(
        self,
    ):
        original = {
            "status": "FAIL",
            "metrics": {"nested": [float("nan"), (float("inf"), -float("inf"))]},
            "finite": [None, True, False, 7, -3.25, -0.0, "NaN", "normal"],
        }
        converted = scientific._json_safe(original)
        self.assertEqual(
            converted["metrics"]["nested"], ["NaN", ["Infinity", "-Infinity"]]
        )
        self.assertEqual(converted["finite"], original["finite"])
        self.assertEqual(math.copysign(1.0, converted["finite"][5]), -1.0)
        self.assertTrue(math.isnan(original["metrics"]["nested"][0]))
        self.assertIsInstance(original["metrics"]["nested"][1], tuple)
        self.assertEqual(
            strict_loads(json.dumps(converted, allow_nan=False)), converted
        )

    def test_numpy_arrays_scalars_and_default_converted_values_are_recursively_sanitized(
        self,
    ):
        original = {
            "array": np.array([[np.nan, np.inf], [-np.inf, 1.25]]),
            "scalars": [
                np.float32(np.nan),
                np.float64(np.inf),
                np.float32(-np.inf),
                np.float32(0.5),
                np.int64(8),
                np.bool_(True),
            ],
            "mapping": MappingProxyType({"deep": {"value": np.float64(np.nan)}}),
            "path": Path("scientific-result.json"),
        }
        converted = scientific._json_safe(original)
        self.assertEqual(converted["array"], [["NaN", "Infinity"], ["-Infinity", 1.25]])
        self.assertEqual(
            converted["scalars"], ["NaN", "Infinity", "-Infinity", 0.5, 8, True]
        )
        self.assertEqual(converted["mapping"], {"deep": {"value": "NaN"}})
        self.assertEqual(converted["path"], str(original["path"]))
        self.assertTrue(np.isnan(original["array"][0, 0]))
        self.assertEqual(
            strict_loads(json.dumps(converted, allow_nan=False)), converted
        )

    def invoke_main(self, directory, handler):
        request = directory / "request.json"
        output = directory / "result.json"
        request.write_text(
            json.dumps(
                {
                    "case_id": "serialization.probe",
                    "case_dir": str(directory),
                    "parameters": {"action": "serialization_probe"},
                }
            ),
            encoding="utf-8",
        )
        output.write_text('{"status": "previous evidence"}', encoding="utf-8")
        replace = scientific.os.replace
        published = []

        def publish(temporary, destination):
            self.assertEqual(
                strict_loads(output.read_text(encoding="utf-8")),
                {"status": "previous evidence"},
            )
            published.append(strict_loads(Path(temporary).read_text(encoding="utf-8")))
            return replace(temporary, destination)

        with (
            patch.dict(scientific.HANDLERS, {"serialization_probe": handler}),
            patch.dict(os.environ, {}, clear=False),
            patch.object(scientific.os, "replace", side_effect=publish),
            redirect_stdout(StringIO()) as stdout,
        ):
            code = scientific.main(["--request", str(request), "--result", str(output)])
        self.assertEqual(len(published), 1)
        self.assertFalse(output.with_suffix(".partial").exists())
        result = strict_loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result, published[0])
        summary = strict_loads(stdout.getvalue().splitlines()[-1])
        self.assertEqual(summary["status"], result["status"])
        return code, result

    def test_failed_scientific_checks_publish_strict_json_and_keep_failure_exit_status(
        self,
    ):
        def handler(_request, checks, _progress, _cancel):
            checks.add(
                "nonfinite_accuracy",
                False,
                value=np.float64(np.nan),
                limit=0.1,
                nested={"conservation": [np.float32(np.inf), -float("inf")]},
            )
            return {}, {
                "loss": np.float32(np.nan),
                "history": np.array([1.0, np.inf, -np.inf]),
                "finite": 4.0,
            }

        with tempfile.TemporaryDirectory() as temporary:
            code, result = self.invoke_main(Path(temporary), handler)
        self.assertEqual(code, 3)
        self.assertEqual(
            (result["status"], result["execution"], result["qualification"]),
            ("FAIL", "PASS", "FAIL"),
        )
        self.assertEqual(result["reason"], "nonfinite_accuracy")
        self.assertEqual(
            result["metrics"],
            {"loss": "NaN", "history": [1.0, "Infinity", "-Infinity"], "finite": 4.0},
        )
        self.assertEqual(result["checks"][0]["value"], "NaN")
        self.assertEqual(
            result["checks"][0]["nested"], {"conservation": ["Infinity", "-Infinity"]}
        )
        self.assertFalse(result["checks"][0]["passed"])

    def test_handler_errors_also_preserve_nonfinite_check_evidence(self):
        def handler(_request, checks, _progress, _cancel):
            checks.add("nonfinite_field", False, values=np.array([[np.nan, np.inf]]))
            raise RuntimeError("diagnostic probe failed")

        with tempfile.TemporaryDirectory() as temporary:
            code, result = self.invoke_main(Path(temporary), handler)
        self.assertEqual(code, 1)
        self.assertEqual((result["status"], result["execution"]), ("ERROR", "FAIL"))
        self.assertEqual(result["reason"], "diagnostic probe failed")
        self.assertEqual(result["checks"][0]["values"], [["NaN", "Infinity"]])


if __name__ == "__main__":
    unittest.main()
