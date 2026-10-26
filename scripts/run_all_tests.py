"""One-hour representative feature checks; use --profile full for the full matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tests"), str(ROOT)]
PROFILES = ("features", "full", "fast")


def resolve_profile(requested: str | None, previous: dict | None) -> str:
    """Infer old resume profiles without silently converting their evidence."""
    saved = previous.get("profile") if previous is not None else None
    if previous is not None and saved not in PROFILES:
        raise ValueError("Resume report has an unknown profile")
    if requested is not None and saved is not None and requested != saved:
        raise ValueError("Resume profile differs from the recorded run")
    profile = requested or saved or "features"
    if profile not in PROFILES:
        raise ValueError("Unknown test profile")
    return profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default=None)
    parser.add_argument("--case", action="append", default=[],
                        help="Case ID or glob; required source cases are included")
    parser.add_argument("--resume", type=Path,
                        help="Resume evidence using its recorded profile by default")
    parser.add_argument("--output", type=Path, help="New private run directory")
    parser.add_argument("--list", action="store_true",
                        help="List the selected catalog without probing or running workers")
    parser.add_argument("--preflight", action="store_true",
                        help="Check required installed capabilities only")
    parser.add_argument("--_suite-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _print_report(report: dict, run_dir: Path, profile: str) -> None:
    summary = report["summary"]
    print(f"\n{summary['status']}: {summary.get('counts', {})}")
    print(f"Feature coverage passed: {summary.get('feature_coverage', False)}")
    print(f"Full coverage passed: {summary.get('full_coverage', False)}")
    print(f"JSON report: {run_dir / 'report.json'}")
    print(f"HTML report: {run_dir / 'report.html'}")
    print(f'Resume: python -B scripts/run_all_tests.py --profile {profile} --resume "{run_dir}"')


def _stopped_report(run_dir, previous, profile, cases, coverage, budget, reason):
    """Keep unvalidated prior evidence when a deadline stops preparation."""
    from simulation_suite.reporting import mark_stopped, write_reports

    old = {} if previous is None else {
        item["id"]: item
        for item in [*previous.get("retained_results", []), *previous.get("results", [])]
        if "id" in item
    }
    report = {
        "schema_version": 1, "run_dir": str(run_dir), "profile": profile,
        "coverage": coverage, "manifest": [case.to_dict() for case in cases],
        "results": [], "retained_results": list(old.values()),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    mark_stopped(report, reason=reason)
    if budget is not None:
        budget.request_stop(reason)
        report["invocation_budget"] = budget.report_fields()
    write_reports(run_dir, report)
    return report


def _execute(args, run_dir, previous, stop, budget):
    from simulation_suite.budget import BudgetExpired, InvocationInterrupted
    from simulation_suite.catalog import all_cases, dependency_order
    from simulation_suite.capabilities import detect_capabilities
    from simulation_suite.reporting import write_reports
    from simulation_suite.runner import execute, select_cases, software_versions

    cases, coverage = [], {}
    try:
        if budget is not None:
            budget.poll()
        catalog, coverage = all_cases(ROOT, profile=args.profile)
        cases = dependency_order(select_cases(catalog, args.case))
        if budget is not None:
            budget.poll()
        initial = {
            "schema_version": 1, "run_dir": str(run_dir), "profile": args.profile,
            "coverage": coverage, "manifest": [case.to_dict() for case in cases],
            "software": software_versions(), "results": [], "capabilities": {},
            "summary": {"status": "PREFLIGHT", "full_coverage": False,
                        "feature_coverage": False, "selection_passed": False},
        }
        if previous is None:
            write_reports(run_dir, initial)
        print("Checking required capabilities in bounded isolated processes…", flush=True)
        try:
            capabilities = detect_capabilities(budget=budget, stop_event=stop)
        except (BudgetExpired, InvocationInterrupted):
            raise
        except Exception as error:
            initial.update(preflight_error=str(error), results=[{
                "id": "preflight", "status": "ERROR", "execution": "FAIL",
                "qualification": "NOT_APPLICABLE", "reason": str(error),
            }])
            initial["summary"]["status"] = "FAIL"
            write_reports(run_dir / "preflight-failure" if previous else run_dir, initial)
            print(f"Preflight failed: {error}. Report directory: {run_dir}", flush=True)
            return 1
        required = {name for case in cases for name in case.capabilities}
        for name, record in capabilities.items():
            print(f"  {name}: {'AVAILABLE' if record['available'] else 'UNAVAILABLE'} — {record['reason']}", flush=True)
        if args.preflight:
            initial["capabilities"] = capabilities
            initial["summary"]["status"] = "PREFLIGHT_ONLY"
            write_reports(run_dir / "preflight" if previous else run_dir, initial)
            print(f"Capability report: {run_dir}")
            if budget is not None:
                budget.poll()
            return 130 if stop.is_set() else (0 if all(
                capabilities.get(name, {}).get("available") for name in required) else 1)
        report = execute(
            cases, root=ROOT, run_dir=run_dir, capabilities=capabilities,
            coverage=coverage, profile=args.profile, previous=previous,
            full_selection=len(cases) == len(catalog), stop_event=stop, budget=budget,
        )
    except (BudgetExpired, InvocationInterrupted) as error:
        report = _stopped_report(run_dir, previous, args.profile, cases, coverage,
                                 budget, error.reason)
    latest = ROOT / ".cache" / "test-suite" / "latest"
    write_reports(latest, report)
    _print_report(report, run_dir, args.profile)
    if report.get("budget_exhausted"):
        return 124
    if stop.is_set() or report.get("interrupted"):
        return 130
    return 0 if report["summary"]["selection_passed"] else 1


def main(argv=None) -> int:
    started_at = time.monotonic()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = _parser()
    args = parser.parse_args(arguments)
    if args.resume and args.output:
        parser.error("--resume and --output are mutually exclusive")
    previous = None
    if args.resume:
        run_dir = args.resume.resolve()
        try:
            previous = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            parser.error(f"Cannot resume report: {error}")
        if not isinstance(previous, dict):
            parser.error("Cannot resume report: expected a JSON object")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = (args.output or ROOT / ".cache" / "test-suite" / stamp).resolve()
    try:
        args.profile = resolve_profile(args.profile, previous)
    except ValueError as error:
        parser.error(str(error))
    if args.list:
        from simulation_suite.catalog import all_cases, dependency_order
        from simulation_suite.runner import select_cases
        try:
            catalog, coverage = all_cases(ROOT, profile=args.profile)
            cases = dependency_order(select_cases(catalog, args.case))
        except ValueError as error:
            parser.error(str(error))
        for case in cases:
            print(f"{case.id:65} {case.timeout:6g}s {','.join(case.capabilities) or 'stdlib/core'}")
        print(json.dumps({"selected_cases": len(cases), "catalog_cases": len(catalog),
                          "coverage": coverage}, indent=2))
        return 0
    if (not args._suite_worker and previous is None
            and run_dir.exists() and any(run_dir.iterdir())):
        parser.error("--output must be empty; use --resume for an existing report")
    stop = threading.Event()

    def interrupt(_signum, _frame):
        print("\nCancelling owned work; preserving completed evidence…", flush=True)
        stop.set()

    prior = signal.signal(signal.SIGINT, interrupt)
    try:
        from simulation_suite.budget import InvocationBudget, supervise
        if args.profile == "features" and not args._suite_worker:
            command = [sys.executable, "-B", str(Path(__file__).resolve()),
                       "--profile", args.profile, "--_suite-worker"]
            command += ["--resume" if args.resume else "--output", str(run_dir)]
            for pattern in args.case:
                command += ["--case", pattern]
            if args.preflight:
                command.append("--preflight")
            code = supervise(command, run_dir=run_dir, started_at=started_at,
                             stop_event=stop, cwd=ROOT)
            if code in (124, 130):
                status = "BUDGET_EXHAUSTED" if code == 124 else "INTERRUPTED"
                print(f"{status}: unfinished work cannot pass; completed evidence is retained.")
                print(f"JSON report: {run_dir / 'report.json'}")
                print(f'Resume: python -B scripts/run_all_tests.py --profile features --resume "{run_dir}"')
            return code
        budget = InvocationBudget.from_env(stop_event=stop) if args._suite_worker else None
        if args._suite_worker and (args.profile != "features" or budget is None):
            parser.error("The private suite worker requires its supervisor budget")
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            return _execute(args, run_dir, previous, stop, budget)
        except ValueError as error:
            parser.error(str(error))
    finally:
        signal.signal(signal.SIGINT, prior)


if __name__ == "__main__":
    raise SystemExit(main())
