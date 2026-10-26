"""Durable, escaped reports for both completed and interrupted test runs."""

from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path


def summarize(results: list[dict], *, full_coverage: bool) -> dict:
    counts = dict(Counter(item.get("status", "ERROR") for item in results))
    passed = bool(results) and all(
        item.get("status") == "PASS"
        and item.get("qualification") != "FAIL"
        and item.get("execution") == "PASS"
        and (item.get("kind") != "scientific" or item.get("qualification") == "PASS")
        for item in results
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "counts": counts,
        "selected_cases": len(results),
        "full_coverage": bool(full_coverage and passed),
        "selection_passed": passed,
    }


def render_html(report: dict) -> str:
    def esc(value):
        return html.escape(str(value), quote=True)

    rows = []
    for result in report.get("results", []):
        metrics = json.dumps(result.get("metrics", {}), indent=2, ensure_ascii=False)
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{esc(value)}</td>"
                for value in (
                    result.get("id", ""),
                    result.get("status", "NOT_RUN"),
                    result.get("execution", "NOT_RUN"),
                    result.get("qualification", "NOT_APPLICABLE"),
                    result.get("elapsed_s", ""),
                    result.get("reason", ""),
                )
            )
            + f"<td><details><summary>Metrics</summary><pre>{esc(metrics)}</pre></details>"
            + f"<code>{esc(result.get('log_path', ''))}</code></td></tr>"
        )
    summary = report.get("summary", {})
    failures = [
        item["id"] for item in report.get("results", []) if item.get("status") != "PASS"
    ]
    return """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Magnetic Torus test report</title><style>
body{font:15px system-ui;margin:2rem;color:#203547}table{border-collapse:collapse;width:100%}
td,th{padding:.5rem;border:1px solid #cad5de;text-align:left;vertical-align:top}
pre{white-space:pre-wrap}code{overflow-wrap:anywhere}th{background:#eef3f6}
</style><h1>Magnetic Torus test report</h1>""" + (
        f"<p>Status: <strong>{esc(summary.get('status', 'RUNNING'))}</strong>. "
        f"Full coverage passed: {esc(summary.get('full_coverage', False))}. "
        f"Profile: {esc(report.get('profile', ''))}.</p>"
        f"<p>Run: <code>{esc(report.get('run_dir', ''))}</code>. Commit: "
        f"<code>{esc(report.get('revision', ''))}</code>.</p>"
        f"<p>Non-passing cases: {esc(', '.join(failures) or 'none recorded')}.</p>"
        "<p>Execution and scientific qualification are separate. A completed solver or failed-fit "
        "rejection test does not qualify a physical model.</p>"
        "<table><thead><tr><th>Case</th><th>Status</th><th>Execution</th><th>Qualification</th>"
        "<th>Seconds</th><th>Reason</th><th>Evidence</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table><h2>Capabilities</h2><pre>"
        + esc(json.dumps(report.get("capabilities", {}), indent=2, ensure_ascii=False))
        + "</pre><h2>Coverage matrix</h2><pre>"
        + esc(json.dumps(report.get("coverage", {}), indent=2, ensure_ascii=False))
        + "</pre></html>"
    )


def write_reports(run_dir: Path, report: dict) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path, html_path = run_dir / "report.json", run_dir / "report.html"
    for path, content in (
        (json_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n"),
        (html_path, render_html(report)),
    ):
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    return json_path, html_path
