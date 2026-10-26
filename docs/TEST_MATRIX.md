# Representative and exhaustive test suites

Run from the repository root in a graphical Windows session:

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py
```

The default `features` profile runs one fixed configuration per major feature, plus short regressions. No enabling flags or extra environment switches are required. It checks installed workers/datasets but never installs dependencies. Required missing capabilities and skips prevent success. Keep the graphical session available for Qt/VTK, exports, and playback measurement.

| Profile | Purpose | Runtime policy |
| --- | --- | --- |
| `features` (fresh-run default) | Representative end-to-end feature coverage with strict accuracy gates | 55-minute work deadline including preflight, five-minute cleanup reserve, one-hour supervisor backstop |
| `full` | Original exhaustive finite matrix and broader qualification cases | Multi-hour explicit opt-in |
| `fast` | Development regressions with declared integration skips | No scientific coverage claim |

## Representative feature coverage

| Feature | One selected configuration |
| --- | --- |
| Configuration, persistence, failures | Short regression modules; duplicate expensive worker scenarios are excluded from this profile rather than skip-passed |
| Thermal | Iron, resolution 4, 8 s / 32 steps, 11 MW charge/discharge at fraction 0.5, emissivity 0.1, initial temperature 1 K below melting; assert melting, freezing, radiation and energy balance |
| Driven MHD | Small liquid driven/laminar case, resolution 3, 0.05 s / 3 steps |
| Dynamo/VMS | Small mushy dynamo/VMS case with accepted seed-removal event |
| Native PINN | CPU, short solid-heating source, resolution 4, 2 s / 12 steps, 1,000 training steps |
| Coupled PINN | CUDA, stationary accepted zero-field source, 250 training steps; this validates static equilibrium only |
| Trajectories | Proton, 100 MeV/nucleon, paired field-on/off and independent relativistic oracle |
| Water dose | Shared accepted zero-field map, 100 MeV proton beam, exactly 524,288 histories once, nonzero dose and measured relative standard error ≤5% |
| Anatomical dose | Male ICRP110 target, mixed proton/alpha/iron spectrum, nonzero field, ground enabled, 128 histories, Shielding + EM option 4; actual organ uncertainties are reported |
| Desktop/export/performance | One real desktop journey including decoded MP4 resize/export; unchanged resolution-8 five × 60-frame benchmark and 30 FPS gate |

All selected scenarios use seed 42. The stationary zero-field source is shared by coupled training and water dose; the driven source is shared by trajectories and anatomical dose. Analytic reference and rejection checks remain. Every required feature must have its declared provider in the executable manifest. A passing subset cannot claim complete representative coverage.

The feature profile is not the full pairwise matrix. It does not qualify every preset, female anatomy, all option combinations, phase-changing PINNs, or dynamic coupled PINNs. Those cases and known failures remain in `full`; no acceptance limit is weakened for a retained scientific case. Expected representative time is approximately 25–40 minutes on the reference workstation, pending a complete measured run. Reaching the water history count alone does not pass its precision gate.

Full-label PINN refinement uses bounded 4,096-row forward/loss batches independently of Adam's unchanged sampling batches. Training-only normalized labels and coordinate transforms may use a capped 64 MiB host cache, with streaming fallback; held-out labels never fit this cache. Validation totals accumulate on the device before transfer. These optimizations preserve float64, masks, equations, random draws, optimizer safeguards, and all accuracy/conservation checks.

The deadline starts before catalog/preflight preparation. At 55 minutes no new work is started, the active job is cancelled, and completed/unvisited evidence is retained. The supervisor enforces owned-process cleanup within the remaining five minutes. Budget exhaustion is explicitly unsuccessful (`124`), with pending case IDs separate from cached evidence; Ctrl+C remains `130`. A future resume gets a fresh budget. No unrelated process is terminated.

Run the exhaustive matrix, or list it without workers, explicitly:

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --profile full
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --profile full --list
```

`--resume` without `--profile` inherits the report's recorded profile. Old `full` reports therefore remain exhaustive and multi-hour; they are never rewritten as representative reports. An explicit mismatched profile is rejected. Start a fresh default invocation to use the one-hour feature profile.

The initial 125-case run completed on 2026-10-26 with 94 passes, 16 failures, five errors, one timeout, and nine blocked cases. Its JSON/HTML reports and case logs remain under `.cache/test-suite/20261026T065717.373686Z/`. Fixes require validation reruns; completion does not establish that the matrix passes. Scientific results and open qualification gates are documented in [VALIDATION.md](VALIDATION.md).

## Exhaustive finite coverage (`full`)

The catalog contains 96 scientific/simulation cases plus isolated regression modules, two isolated FE regression methods, a desktop journey, and a performance benchmark. Counts update automatically as regression modules are added. Inspect the executable catalog without starting workers:

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --profile full --list
```

| Area | Cases and oracles |
| --- | --- |
| Configuration | Every shipped preset; editor roundtrips and repeated edits; supported enum/bool choices; declared numeric boundaries, nonfinite and invalid inputs; 40 seeded custom configurations; independent geometry volume, immutable snapshots, and physical/observer identity checks. |
| Thermal | Solid, liquid, melting/freezing, radiation, zero latent heat, charge endpoints/roundoff, efficiency endpoints, and material presets. Independent sharp enthalpy law, source/loss integrals, torus volume weights, energy balance, and Stefan-front reference. |
| MHD | All 12 solid/mushy/liquid × driven/dynamo × laminar/VMS combinations; melting/freezing, stationary state, as-shipped presets, and eight transport field sources. Check accepted timelines, boundary velocities, seed removal, magnetic-energy decomposition, pumping work, matching differentiable weak equations, Hartmann flow, and magnetic diffusion. |
| PINNs | CPU/CUDA thermal and stationary coupled fits, dynamic coupled fits for all FE presets and selected transition/turbulent sources, and transport source models. Retain held-out accuracy, conservation, validation loss, training-only normalization, source identity, cancellation, checkpoints, reproducibility, and rejected-model safeguards. |
| Boris diagnostics | 13 deterministic interaction rows covering all 89 pairs across proton/alpha/iron, 10/100/1000 MeV/nucleon, beam/isotropic/spectrum, ground on/off, and zero/imposed/driven/dynamo fields. Independent relativistic momentum and gyro-radius oracles; identical incident primaries for field comparisons. |
| Geant4 | 19 deterministic interaction rows covering all 186 pairs across those particle/source/ground choices, five field sources including a qualified PINN, both supported physics lists, and water/ICRP110 male/female targets. Additional mixed spectrum, fluence linearity, and water precision cases. Check masses, source binding, secondary contributions, uncertainty accounting, and exact zero-field pairing. |
| Desktop | Every view and Numerical/PINN/error/comparison source, compatible/incompatible comparisons, synchronized probes/scrubbing, camera retention, overlays/sections, looping/stepping/end frames, separate thermal/particle timing, edit/recalculate, cancellation, failure preservation, and stale-result isolation. Synthetic display fixtures are labelled and do not qualify trained fields. |
| Persistence/export | Run history and independent artifact roundtrips; malformed/stale/missing artifacts; atomic publication and artifact security; PNG/SVG/JSON/HDF5; fixed-size MP4 decoding during window resize; export cancellation and filesystem failures. |
| Performance | Resolution-8 reference, five repetitions of 60 frames including Qt event processing, approximately 30 FPS required. Report dimensions, overlays, detail, per-repeat timings, and median FPS. |

The pairwise generator has stable tie-breaking. Coverage is checked against the actual executable cases, including missing enum levels, missing pairs, presets, all 12 MHD triples, and CPU/CUDA problem combinations. Removing required rows fails the coverage proof. Numeric values outside the listed discrete radiation energies receive validation/boundary tests; this is not an exhaustive test of every continuous scenario or every multiway interaction.

Comparison selection and result reactivation require accepted snapshots from both runs in the shared time interval. Loading a new active run revalidates the retained comparison, clears it when invalid, and clamps playback before display. The MP4 smoke check records the resize frame count and requires a later encoded frame whose raw screenshot dimensions differ from both the initial capture and the encoder dimensions, proving that resampling was exercised. It also decodes every frame at the expected fixed dimensions.

Functional transport rows use small samples to exercise interactions and report their actual uncertainty. They do not claim 5% dose precision. The separate water precision gate increases histories through 262,144 to a maximum of 524,288 and requires a nonzero score with relative standard error ≤5%; zero hits and insufficient precision remain failures. Anatomical rows test execution/scoring across targets, not independent anatomical dosimetry qualification. Coupled PINN accuracy remains a required scientific gate even if later functional rejection tests pass.

FE equation agreement retains the `1e-7` maximum residual gate for assembled equations, boundary conditions, and magnetic consistency. The Coulomb divergence diagnostic is measured separately using the existing normalized RMS limit of `0.1`; its maximum is also recorded. The div-div penalty is already included in the induction equation. Treating this diagnostic as an independently assembled zero residual was an incorrect test oracle; the numerical formulation is unchanged.

Dependent training and transport require passing source cases: execution must pass, and scientific parents must also pass qualification. A failed or incomplete prerequisite makes its dependents `BLOCKED` before starting work or reusing a cached result. Negative tests construct their rejected inputs from passing prerequisites, so expected rejection coverage remains independent.

Published turbulent MHD Taylor–Green adaptation, full coupled manufactured-solution convergence, exterior-domain refinement, subgrid total-energy validation, certified material data, and independent radiation/anatomical reference qualification remain broader research gates. Passing this matrix does not demonstrate habitat shielding efficacy or a self-sustaining dynamo.

## Requirements and runtime

Preflight checks the real Qt/VTK render/readback, PyTorch CPU and CUDA kernels, FEniCSx/Gmsh/PETSc imports, Geant4 worker startup/physics datasets, and all official male/female anatomical slice headers. Setup instructions and pinned environments are in [README.md](../README.md).

`MAGNETIC_TORUS_FE_PYTHON` selects the FE interpreter for preflight, solves, and benchmarks; it defaults to `/usr/bin/python3`. On Windows, supply the interpreter's Linux path inside the selected WSL distribution.

The exhaustive `full` profile requires several hours, potentially longer on slower hardware or with cold compilation caches. The unsuccessful initial run took about 93 minutes; changed training and transport methods require fresh timing. Its scientific cases' declared timeout budget totals about 22.1 hours, before regression/desktop limits and bounded cleanup. Individual limits are visible with `--profile full --list`: most small FE cases allow 10 minutes, shipped presets 30 minutes except Lunar habitat at 60 minutes, dynamic PINNs 40 minutes, transport interactions 15 minutes, and water precision 60 minutes. The full scenarios' duration, geometry, and scientific acceptance limits are unchanged. A timeout is an unsuccessful outcome, not a skipped pass.

Only one heavy job runs at a time. SciPy and PyTorch stay in separate native processes where required; mixed FE/PyTorch regression methods have their own invocations. Stochastic batches retain the scenario's configured limit (capped at 4,096); deterministic full-label refinement uses its independent 4,096-row cap. The runner never sets `KMP_DUPLICATE_LIB_OK`.

PINN parameterization version 2 uses a geometry/source-derived bulk-energy trial for native thermal fields, accepted-first-step correction ramps for coupled fields, and training-only free-DOF label conditioning. Fewer than 50 requested steps retain Adam; larger budgets allocate 80% to Adam and the remainder to deterministic L-BFGS refinement. Training labels are streamed in bounded batches and fixed physics panels remain independent of held-out labels. Method version 3 retains the strict eight-closure limit, but can recover an already-evaluated, finite Armijo-decreasing training trial after budget exhaustion and restart L-BFGS curvature. Rejected searches restore the previous state, and bounded recovery prevents indefinite learning-rate halving. No held-out labels select recovery steps. Checkpoints record attempts, actual updates, retries, evaluations, and method versions. Validation masks, training-only normalization, raw residual scales, and all qualification limits are unchanged; old saved prediction fields remain inspectable.

Training progress identifies Adam batch losses versus deterministic refinement losses and gives the epoch of the latest held-out evaluation. Held-out loss is measured every ten epochs and retained between checks. Repeated rounded values alone do not establish a stall; step/recovery diagnostics distinguish unchanged parameters from applied updates. A stationary optimizer is not evidence that the scientific qualification gates passed.

Finite-map radiation transport validates physical-world and primary containment before enabling its field path. Nonzero fields use a propagation-step cap derived from actual accepted-mesh clearance; out-of-map queries remain fatal and report their coordinates. Exactly zero maps use ordinary transportation for both paired ensembles after the same validation, with no approximate-zero threshold. Dose metadata records the map clearance, integration mode, step limit, and actual field-query diagnostics.

Press Ctrl+C to request cancellation and preserve evidence. The runner writes cooperative cancellation files, allows bounded worker cleanup, then stops its owned processes. Windows Job Objects contain native descendants; WSL fallback targets only same-user commands carrying that case's private path. No broad process-name killing is used.

## Reports and reruns

Each invocation creates `.cache/test-suite/<UTC timestamp>/` with:

- `report.json` and escaped `report.html`, updated after each case, including execution status separately from scientific qualification, versions/capabilities, configuration/seed/metrics, coverage, and artifact hashes.
- `cases/<case ID>/attempt-0001/` containing the exact request, live `output.log`, structured worker result, runner result, and case-specific fields/checkpoints/exports. Reruns create another attempt.
- `.cache/test-suite/latest/report.json` and `report.html`, copies of the latest finished report. Resume the original timestamp directory so its artifacts remain available.

Non-finite scientific evidence is recorded as the explicit strings `NaN`, `Infinity`, or `-Infinity`, including nested metrics and failed checks. Structured results remain strict JSON; these representations do not change scientific pass/fail decisions.

The terminal prints the exact report directory and resume command. To rerun only selected cases (including their prerequisites):

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --profile full --case "pinn.*"
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --profile full --case "dose.pair.01"
```

To resume the full run after interruption or fixes:

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --resume ".cache/test-suite/<original run>"
```

Reuse requires a previous passing execution and qualification, matching inputs/source dependencies/software and worker fingerprints, and intact hashed artifacts inside that run. Failed, interrupted, missing, modified, or stale evidence reruns. Changed numerical sources invalidate downstream analyses. Do not move evidence directories or edit their reports. The report supplies data only; executable commands come from the current catalog.

Unvisited cached records remain in each saved report during resume. Cancellation can still validate and reuse passing cached evidence without starting processes; the interrupted invocation remains unsuccessful even if every recorded case was reused successfully.

Use `--case` together with `--resume` for a targeted retry in the same evidence directory. Its summary covers only that selection; resume without a filter afterward to restore the profile's complete report. `feature_coverage` requires the entire representative manifest and passing required gates; `full_coverage` is reserved for the complete exhaustive matrix. A passing subset or `fast` never claims either. `--preflight` records required capabilities without running cases.

Exit code 0 means all selected required cases and gates passed; exit 1 means a failure, missing capability, timeout, or coverage deficit; exit 124 means invocation budget exhaustion; exit 130 means interruption. Test history, temporary files, and workers are directed into the private run directory. Normal saved simulations and application history are untouched. Artifacts are retained for diagnosis; exports happen only inside test evidence.

After a full run, share `report.json` and the logs for non-passing cases. Scientific accuracy failures will be investigated without weakening thresholds, followed by focused reruns and a complete resume.

## Rerun the 2026-10-26 failure groups

The saved run above already contains the required JSON and per-case text logs. These commands use the same evidence directory; accepted cached results are reused only if current code, inputs, runtime identity, and artifacts still match. Changed fingerprints can require previously passing cases to execute again. Each targeted report is partial; the full resume at the end restores coverage of the complete catalog.

Start with the corrected dynamo checks, trajectories, transport interactions, and display benchmark:

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --resume ".cache/test-suite/20261026T065717.373686Z" --case "fe.*dynamo*" --case "boris.*" --case "dose.pair.*" --case "dose.mixed-spectrum" --case "dose.fluence-linearity" --case "desktop.performance"
```

Run the canonical PINN cases, lunar source, and full water-precision gate when the longer runtime is available. This includes the unchanged accuracy/conservation limits; the short native pilot remains unqualified and is not substituted for these cases:

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --resume ".cache/test-suite/20261026T065717.373686Z" --case "pinn.*" --case "preset.lunar-habitat" --case "dose.water-precision"
```

Finally, resume every required case and gate:

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --resume ".cache/test-suite/20261026T065717.373686Z"
```
