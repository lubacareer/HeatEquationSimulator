# Reviewed subprocess audit exceptions

The rule-qualified `nosemgrep` annotations apply only to the marked calls and
rules. They record an injection audit completed on 2026-11-05. Other rules and
call sites retain their existing coverage.

## Environment diagnostics

`magnetic_torus.diagnostics.inspect_environment` builds its command map internally.
The Python code strings are constants and the native interpreter is
`sys.executable`. WSL commands use the shared `wsl_command` helper's `--exec`
argument forwarding. The configured Geant4 worker path is one operand of the
fixed `/usr/bin/test -x` command; it is not interpreted as shell code.

This assumes an operator-controlled Python environment, WSL executable and distro
configuration. Adding commands or changing the helper, executable selection or
caller trust requires another audit.

## Geant4 worker teardown

`magnetic_torus.transport.geant4._stop_worker` sends the fixed `/usr/bin/kill -TERM`
command through WSL `--exec`. The PID is ASCII decimal, at most ten characters,
between 2 and 2,147,483,647, and converted to canonical decimal text before being
passed as one argument. Zero, init, negative, malformed and oversized values do
not reach the signal command; local wrapper cleanup still runs.

The worker writes `getpid()` to `worker.pid` in its UUID-scoped transport job.
That backend and job directory must remain operator-controlled. Numeric
validation does not establish process ownership or prevent PID reuse, and it
does not make an attacker-writable PID file trustworthy. Changing the PID source,
job-directory trust, WSL distro selection or callers requires another audit.

## Worker and local research launches

These reviewed sites invoke argument lists without `shell=True`. Paths and other
arguments are passed as operands, not joined into a shell command. The audit
assumes a trusted local operator controls the process environment, `PATH`,
Python executable and module search path, checkout and scripts, WSL executable
and selected distribution, installed workers, and all callers. In particular,
selecting an arbitrary executable through an environment override is intentional:
the application does not establish that the selected executable is trustworthy.
These assumptions do not make a compromised environment or untrusted caller safe.

| Marked call | Command and operands | Scope of the audit |
|---|---|---|
| `magnetic_torus/numerics/fenics.py` | Configured FE interpreter and fixed FE worker, request and artifact paths. | `MAGNETIC_TORUS_FE_PYTHON` selects the executable; Windows uses WSL `--exec`. |
| `magnetic_torus/numerics/validation.py` | Same interpreter and FE worker with the fixed benchmark option and output path. | Same interpreter and WSL trust boundary. |
| `scripts/compare_fe_operator.py` | Same interpreter and FE worker with request, output and probe paths. | Same interpreter and WSL trust boundary. |
| `magnetic_torus/transport/geant4.py` worker launch | Selected Geant4 worker and one request path. | `TORUS_GEANT4_WORKER`, local `PATH`, or the configured default selects executable code; Windows uses WSL `--exec`. |
| `magnetic_torus/pinn_validation.py` | `sys.executable` launches the fixed `magnetic_torus.worker` module with a request path. | The Python installation, module search path and request are operator controlled. |
| `scripts/pinn_pressure_diagnostics.py` | `sys.executable` runs the fixed residual script with copied source and result paths. | Local input/output paths are operands; the checkout and Python environment are trusted. |
| `scripts/pinn_refine_source.py` | `sys.executable` runs the resolved current script's fixed `verify` subcommand with saved source and output paths. | The checkout and Python environment are trusted. |
| `scripts/profile_pinn_fe_vjp.py` | `sys.executable` relaunches the fixed current script with parsed paths, counts and a strategy. | Local CLI inputs and executable/script are trusted; strategies are parsed choices. |
| `scripts/verify_pinn_fe_vjp.py` | `sys.executable` relaunches the fixed current script with parsed paths, numeric settings and device. | Local CLI inputs and executable/script are trusted; device is a parsed choice. |
| `scripts/run_checks.py` | `sys.executable` runs fixed unittest discovery for each local `test_*.py` name. | The local test tree, Python environment and checkout are trusted. |
| `scripts/research_supervisor.py` | Runs its caller's command list under a deadline. | This function accepts arbitrary commands; the current production caller `pinn_research.py` builds `sys.executable`, its own fixed script and locally parsed arguments. It is not safe to expose to an untrusted caller. |

Re-audit the affected site if a new caller supplies commands, executable selection
or argument construction changes, a launch gains shell interpretation, or any
of the stated environment, checkout, worker or caller trust boundaries changes.

Existing `test_fe_launch_configuration.py` covers selected/default interpreters
and both Linux and Windows command construction; `test_wsl_command.py` tests
literal hostile arguments through actual WSL when available.
`test_fe_operator_compare.py` checks probe integrity with a mocked worker,
research supervisor tests exercise deadlines and owned-process cleanup, and
Geant4 transport integration uses the selected worker command. These tests do
not establish hostile-path coverage for every local diagnostic launcher.

## Verification

The hostile-argument WSL regression exercises literal argument forwarding on
actual WSL. The PID regressions reject malformed, zero/init, Unicode, overflow
and 5,000-digit inputs, and retain valid canonical PID controls. Signal calls are
mocked in the PID tests. The zero/init regression failed before the guard fix and
passed afterward; the complete isolated test profile also passed. Independent
reviews assessed both the guard and these narrowly qualified audit exceptions.
