# Full test suite

Run from the repository root in a graphical Windows session:

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py
```

No enabling flags or extra environment switches are required. The default `full` profile runs every case serially, including installed WSL workers and CPU/CUDA training. It does not install dependencies. Missing capabilities are explicit `UNAVAILABLE` outcomes and prevent success. Keep the graphical session available for real Qt/VTK rendering, exports, and playback measurement.

The complete run is pending. Short implementation checks verify orchestration, coverage proofs, desktop behavior, native thermal invariants, and analytic trajectories; they do not establish that the full matrix passes. Existing failed dynamic coupled PINN fits and open physics qualification gates remain documented in [VALIDATION.md](VALIDATION.md).

## Declared finite coverage

The catalog contains 96 scientific/simulation cases plus isolated regression modules, two isolated FE regression methods, a desktop journey, and a performance benchmark. Counts update automatically as regression modules are added. Inspect the executable catalog without starting workers:

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --list
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

Functional transport rows use small samples to exercise interactions and report their actual uncertainty. They do not claim 5% dose precision. The separate water precision gate increases histories up to 262,144 and requires a nonzero score with relative standard error ≤5%; zero hits fail. Anatomical rows test execution/scoring across targets, not independent anatomical dosimetry qualification. Coupled PINN accuracy remains a required scientific gate even if later functional rejection tests pass.

Dependent training and transport require passing source cases: execution must pass, and scientific parents must also pass qualification. A failed or incomplete prerequisite makes its dependents `BLOCKED` before starting work or reusing a cached result. Negative tests construct their rejected inputs from passing prerequisites, so expected rejection coverage remains independent.

Published turbulent MHD Taylor–Green adaptation, full coupled manufactured-solution convergence, exterior-domain refinement, subgrid total-energy validation, certified material data, and independent radiation/anatomical reference qualification remain broader research gates. Passing this matrix does not demonstrate habitat shielding efficacy or a self-sustaining dynamo.

## Requirements and runtime

Preflight checks the real Qt/VTK render/readback, PyTorch CPU and CUDA kernels, FEniCSx/Gmsh/PETSc imports, Geant4 worker startup/physics datasets, and all official male/female anatomical slice headers. Setup instructions and pinned environments are in [README.md](../README.md).

`MAGNETIC_TORUS_FE_PYTHON` selects the FE interpreter for preflight, solves, and benchmarks; it defaults to `/usr/bin/python3`. On Windows, supply the interpreter's Linux path inside the selected WSL distribution.

Expect several hours, potentially longer on slower hardware or with cold compilation caches; there is no completed full-run timing yet. The scientific cases' declared timeout budget totals about 21.6 hours, before regression/desktop limits and bounded cleanup. Individual limits are visible with `--list`: most small FE cases allow 10 minutes, shipped presets 30 minutes, dynamic PINNs 40 minutes, transport interactions 15 minutes, and water precision 60 minutes. A timeout is an unsuccessful outcome, not a skipped pass.

Only one heavy job runs at a time. SciPy and PyTorch stay in separate native processes where required; mixed FE/PyTorch regression methods have their own invocations. CUDA batches remain bounded by each scenario's training configuration. The runner never sets `KMP_DUPLICATE_LIB_OK`.

Press Ctrl+C to request cancellation and preserve evidence. The runner writes cooperative cancellation files, allows bounded worker cleanup, then stops its owned processes. Windows Job Objects contain native descendants; WSL fallback targets only same-user commands carrying that case's private path. No broad process-name killing is used.

## Reports and reruns

Each invocation creates `.cache/test-suite/<UTC timestamp>/` with:

- `report.json` and escaped `report.html`, updated after each case, including execution status separately from scientific qualification, versions/capabilities, configuration/seed/metrics, coverage, and artifact hashes.
- `cases/<case ID>/attempt-0001/` containing the exact request, live `output.log`, structured worker result, runner result, and case-specific fields/checkpoints/exports. Reruns create another attempt.
- `.cache/test-suite/latest/report.json` and `report.html`, copies of the latest finished report. Resume the original timestamp directory so its artifacts remain available.

Non-finite scientific evidence is recorded as the explicit strings `NaN`, `Infinity`, or `-Infinity`, including nested metrics and failed checks. Structured results remain strict JSON; these representations do not change scientific pass/fail decisions.

The terminal prints the exact report directory and resume command. To rerun only selected cases (including their prerequisites):

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --case "pinn.*"
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --case "dose.pair.01"
```

To resume the full run after interruption or fixes:

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --resume ".cache/test-suite/<original run>"
```

Reuse requires a previous passing execution and qualification, matching inputs/source dependencies/software and worker fingerprints, and intact hashed artifacts inside that run. Failed, interrupted, missing, modified, or stale evidence reruns. Changed numerical sources invalidate downstream analyses. Do not move evidence directories or edit their reports. The report supplies data only; executable commands come from the current catalog.

Unvisited cached records remain in each saved report during resume. Cancellation can still validate and reuse passing cached evidence without starting processes; the interrupted invocation remains unsuccessful even if every recorded case was reused successfully.

Use `--case` together with `--resume` for a targeted retry in the same evidence directory. Its summary explicitly covers only that selection; run a full `--resume` afterward to restore the full report. A passing subset or `--profile fast` never reports full coverage. `--preflight` records capabilities without running cases. `--profile fast` runs shorter isolated development regressions with explicit skips and is not the release gate.

Exit code 0 means all selected required cases and gates passed; exit 1 means a failure, missing capability, timeout, or coverage deficit; exit 130 means interruption. Test history, temporary files, and workers are directed into the private run directory. Normal saved simulations and application history are untouched. Artifacts are retained for diagnosis; exports happen only inside test evidence.

After a full run, share `report.json` and the logs for non-passing cases. Scientific accuracy failures will be investigated without weakening thresholds, followed by focused reruns and a complete resume.
