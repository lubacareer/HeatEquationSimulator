# Reviewed subprocess audit exceptions

The two qualified `nosemgrep` annotations apply to
`python.lang.security.audit.dangerous-subprocess-use-audit` at the marked calls.
They record an injection audit completed on 2026-11-05. Other rules and call sites
retain their existing coverage.

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

## Verification

The hostile-argument WSL regression exercises literal argument forwarding on
actual WSL. The PID regressions reject malformed, zero/init, Unicode, overflow
and 5,000-digit inputs, and retain valid canonical PID controls. Signal calls are
mocked in the PID tests. The zero/init regression failed before the guard fix and
passed afterward; the complete isolated test profile also passed. Independent
reviews assessed both the guard and these narrowly qualified audit exceptions.
