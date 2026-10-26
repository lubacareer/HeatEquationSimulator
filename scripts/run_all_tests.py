"""One command for the full modern-workspace regression and simulation matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import threading

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tests"), str(ROOT)]


def main(argv=None) -> int:
    from simulation_suite.catalog import all_cases, dependency_order
    from simulation_suite.reporting import write_reports
    from simulation_suite.runner import execute, select_cases, software_versions

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "fast"), default="full")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case ID or glob; required source cases are included",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume a prior run directory, validating completed evidence",
    )
    parser.add_argument("--output", type=Path, help="New private run directory")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the selected matrix without probing or running workers",
    )
    parser.add_argument(
        "--preflight", action="store_true", help="Check installed capabilities only"
    )
    args = parser.parse_args(argv)
    if args.resume and args.output:
        parser.error("--resume and --output are mutually exclusive")
    try:
        cases, coverage = all_cases(ROOT, profile=args.profile)
        full_count = len(cases)
        cases = dependency_order(select_cases(cases, args.case))
    except ValueError as error:
        parser.error(str(error))
    if args.list:
        for case in cases:
            print(
                f"{case.id:65} {case.timeout:6g}s {','.join(case.capabilities) or 'stdlib/core'}"
            )
        print(
            json.dumps(
                {
                    "selected_cases": len(cases),
                    "catalog_cases": full_count,
                    "coverage": coverage,
                },
                indent=2,
            )
        )
        return 0
    previous = None
    if args.resume:
        run_dir = args.resume.resolve()
        try:
            previous = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            parser.error(f"Cannot resume report: {error}")
        if previous.get("profile") != args.profile:
            parser.error("Resume profile differs from the recorded run")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = (args.output or ROOT / ".cache" / "test-suite" / stamp).resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            parser.error("--output must be empty; use --resume for an existing report")
    run_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def interrupt(_signum, _frame):
        print(
            "\nCancelling the owned case; preserving logs and completed evidence…",
            flush=True,
        )
        stop.set()

    prior = signal.signal(signal.SIGINT, interrupt)
    try:
        from simulation_suite.capabilities import detect_capabilities

        initial = {
            "schema_version": 1,
            "run_dir": str(run_dir),
            "profile": args.profile,
            "coverage": coverage,
            "manifest": [case.to_dict() for case in cases],
            "software": software_versions(),
            "results": [],
            "capabilities": {},
            "summary": {
                "status": "PREFLIGHT",
                "full_coverage": False,
                "selection_passed": False,
            },
        }
        # Preserve resume evidence until it has been fingerprint-checked by execute.
        if previous is None:
            write_reports(run_dir, initial)
        print(
            "Checking installed capabilities in bounded isolated processes…", flush=True
        )
        try:
            capabilities = detect_capabilities()
        except Exception as error:
            initial.update(
                preflight_error=str(error),
                results=[
                    {
                        "id": "preflight",
                        "status": "ERROR",
                        "execution": "FAIL",
                        "qualification": "NOT_APPLICABLE",
                        "reason": str(error),
                    }
                ],
            )
            initial["summary"]["status"] = "FAIL"
            write_reports(
                run_dir / "preflight-failure" if previous else run_dir, initial
            )
            print(f"Preflight failed: {error}. Report directory: {run_dir}", flush=True)
            return 1
        for name, record in capabilities.items():
            print(
                f"  {name}: {'AVAILABLE' if record['available'] else 'UNAVAILABLE'} — {record['reason']}",
                flush=True,
            )
        if args.preflight:
            initial["capabilities"] = capabilities
            initial["summary"]["status"] = "PREFLIGHT_ONLY"
            write_reports(run_dir / "preflight" if previous else run_dir, initial)
            print(f"Capability report: {run_dir}")
            return (
                130
                if stop.is_set()
                else (
                    0
                    if all(record["available"] for record in capabilities.values())
                    else 1
                )
            )
        report = execute(
            cases,
            root=ROOT,
            run_dir=run_dir,
            capabilities=capabilities,
            coverage=coverage,
            profile=args.profile,
            previous=previous,
            full_selection=len(cases) == full_count,
            stop_event=stop,
        )
    finally:
        signal.signal(signal.SIGINT, prior)
    latest = ROOT / ".cache" / "test-suite" / "latest"
    write_reports(latest, report)
    summary = report["summary"]
    print(f"\n{summary['status']}: {summary['counts']}")
    print(f"Full coverage passed: {summary['full_coverage']}")
    print(f"JSON report: {run_dir / 'report.json'}")
    print(f"HTML report: {run_dir / 'report.html'}")
    print(
        f'Resume: python -B scripts/run_all_tests.py --profile {args.profile} --resume "{run_dir}"'
    )
    if stop.is_set():
        return 130
    return 0 if summary["selection_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
