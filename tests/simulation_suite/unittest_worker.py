"""Run one unittest module in its own scientific/runtime process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import unittest


class CoverageResult(unittest.TextTestResult):
    """Distinguish whole-test skips from skipped subtests in execution counts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._started_tests = set()
        self.whole_test_skips = 0

    def startTest(self, test):
        self._started_tests.add(id(test))
        super().startTest(test)

    def stopTest(self, test):
        self._started_tests.discard(id(test))
        super().stopTest(test)

    def addSkip(self, test, reason):
        if id(test) in self._started_tests:
            self.whole_test_skips += 1
        super().addSkip(test, reason)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    parameters = request["parameters"]
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / "tests")]
    from simulation_suite.host_skips import is_host_inapplicable
    if parameters.get("test_id"):
        tests = unittest.defaultTestLoader.loadTestsFromName(parameters["test_id"])
    else:
        tests = unittest.defaultTestLoader.discover(
            str(root / "tests"), pattern=parameters["pattern"]
        )
        excluded = set(parameters.get("exclude_ids", ()))

        def remaining(suite):
            for item in suite:
                if isinstance(item, unittest.TestSuite):
                    yield from remaining(item)
                elif item.id() not in excluded:
                    yield item

        tests = unittest.TestSuite(remaining(tests))
    result = unittest.TextTestRunner(verbosity=2, resultclass=CoverageResult).run(tests)
    skipped = [
        {"test": test.id(), "reason": reason,
         "classification": "NOT_APPLICABLE" if is_host_inapplicable(test.id(), reason, os_name=os.name) else "MISSING"}
        for test, reason in result.skipped
    ]
    strict = request.get("profile") in {"features", "full"}
    executed = result.testsRun - result.whole_test_skips
    missing = strict and (any(skip["classification"] != "NOT_APPLICABLE" for skip in skipped) or not executed)
    status = (
        "PASS" if result.wasSuccessful() and not missing and result.testsRun else "FAIL"
    )
    data = {
        "status": status,
        "execution": "PASS" if result.wasSuccessful() and result.testsRun else "FAIL",
        "qualification": "NOT_APPLICABLE",
        "reason": "",
        "metrics": {
            "tests_run": result.testsRun,
            "tests_executed": executed,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
        },
        "skipped": skipped,
        "failures": [
            {"test": test.id(), "traceback": trace}
            for test, trace in result.failures + result.errors
        ],
        "artifacts": {},
    }
    if not result.testsRun:
        data["reason"] = "No tests were discovered; this module is not covered."
    elif missing and not executed:
        data["reason"] = "No applicable tests were executed; this module is not covered."
    elif missing:
        data["reason"] = "Required profile tests were skipped; see skipped records."
    elif not result.wasSuccessful():
        data["reason"] = "Unittest failures/errors; see the case log and tracebacks."
    args.result.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
