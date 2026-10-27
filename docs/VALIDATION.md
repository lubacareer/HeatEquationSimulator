# Validation record

Reference workstation: Windows, Python 3.12 `torch-gpu`, Radeon 780M rendering, RTX 4060 Laptop CUDA training, 16 GB host RAM; Ubuntu WSL DOLFINx 0.10.0. These are integration/reference checks, not a qualified shielding design.

Reproduce with `python -B scripts/run_checks.py`, `python -B scripts/desktop_smoke.py`, and `python -B -m magnetic_torus.physics --benchmark-report .cache/physics_validation.json --include-fenics`.

## Representative suite and batching optimization (2026-10-26)

At this stage, fresh `scripts/run_all_tests.py` invocations selected `features`: 52 cases, including 33 short regression modules and one fixed scientific configuration per major feature. The then-current `full` profile had 133 cases. Both executable coverage proofs passed independently. The current catalog counts are in [TEST_MATRIX.md](TEST_MATRIX.md). Representative coverage excludes all-preset/option-pair qualification, female anatomy, and phase-changing/dynamic coupled PINN qualification. All retained scientific gates, including measured water precision ≤5%, are unchanged.

There are 135 passing focused checks: eight batching/cache, 17 conditioning, 16 existing PINN, four normalization, seven PINN diagnostics, eight CLI/profile, nine representative catalog, 12 exhaustive matrix, 11 invocation budget, 24 existing runner, eight artifact security, seven atomic publication, and four FE-launch checks. Focused Ruff and whitespace checks pass. Tests exercise the 55-minute work cutoff/five-minute cleanup reserve with short/fake deadlines, including an actual 1.2-second supervisor run that terminates its own grandchild, returns 124, and preserves cached evidence. Required work left unfinished is never a passing result.

A bounded synthetic CPU benchmark measured only the full-label gradient calculation: 5,200 training labels, float64, one CPU thread, five repetitions. Median time decreased from 57.315 ms (uncached 256-row batches, 21 neural calls) to 29.918 ms (cached 4,096-row batches, two calls), a 1.916× speedup. Loss was identical, maximum absolute gradient difference was 9.72e-17, and the host cache occupied 124,840 bytes. Evidence is `.cache/pinn-stall-worktree/.cache/pinn-batching-benchmark.json`. This is not a measured speedup for an entire PINN fit or scientific qualification.

The exact conda entrypoint passed a real three-case selection (CLI regressions, melting/freezing/radiating thermal cycle, and short native source) with all six installed capabilities verified, followed by a successful inferred-profile targeted resume. Both correctly reported incomplete overall feature/full coverage; the resume retained unvisited evidence. Reports are in `.cache/pinn-stall-worktree/.cache/representative-cli-smoke-20261026/`. A complete representative run, its wall time, and anatomical transport remain pending. A later targeted 1,048,576-history water precision run passed as recorded below. The estimated 25–40 minutes is not a completed representative-profile measurement.

Development and these checks used an isolated checkout; the already-running full suite and its history were left untouched. Update the main working directory after it finishes, then start a fresh default command for the one-hour representative profile. Resuming an old `full` report intentionally retains exhaustive, multi-hour behavior.

## Full-suite failure investigation (2026-10-26)

The initial full run at `.cache/test-suite/20261026T065717.373686Z/` completed 125 cases in about 93 minutes: 94 PASS, 16 FAIL, five ERROR, one TIMEOUT, and nine BLOCKED. Original reports and all 116 recorded case logs are preserved. All isolated regression modules passed. A nonzero conda exit reflected failed test gates, rather than a failure to save the results.

- Eight dynamo FE cases failed only because the test applied the assembled-equation `1e-7` tolerance to the separate Coulomb divergence diagnostic. The solver uses a div-div penalty whose weak contribution is already checked in induction. Rechecking all eight saved accepted sources with the corrected oracle passes every equation, boundary, gradient, and gauge check: diagnostic RMS ranges from 0.00337025 to 0.00416658, below the existing physical RMS limit of 0.1; raw maxima of 0.0418273–0.0545488 remain reported. No FE equation or accepted field was changed. Regression tests still reject assembled defects above `1e-7` and excessive/nonfinite gauge diagnostics.
- Lunar habitat reached accepted time 53.43/60 s before its 1,800 s timeout. Only its execution deadline increases to 3,600 s; geometry, timesteps, material properties, duration, and physical checks remain unchanged. Completion still needs a real rerun.
- The water precision reference reached relative standard error 0.0674898 at 262,144 histories and honestly failed the 0.05 gate. The history sequence now permits 524,288 with the original 3,600 s deadline. Actual measured precision must pass; a larger history count alone is not qualification.
- Six PINN accuracy cases failed despite successful execution; stationary and zero-field CPU/CUDA cases passed. The baseline showed thermal trial-field overheating and diluted/suppressed coupled corrections. New training methods must be judged against unchanged held-out and conservation checks, with actual outcomes recorded below.
- Five high-energy transport cases terminated with `OutsideAcceptedMap`; the complete saved logs retain the fatal diagnostic even though the former generic error message showed only unrelated trailing initialization output. The physical world lies inside the accepted mesh, so bounded field-integration trials and exact-zero field handling require real transport checks.
- The original five-repeat desktop benchmark measured 16.016 FPS and emitted a Qt `DISPLAY1` warning. A later short diagnostic measured 37.13 FPS on the Radeon 780M and physical 1280 × 800 screen, suggesting a display-state contribution. This short diagnostic does not replace the five-repeat 30 FPS acceptance gate.

The unchanged resolution-8, five × 60-frame desktop benchmark subsequently passed at **42.046 FPS** (repeat medians 39.30, 41.11, 42.66, 48.22, and 42.05 FPS), including Qt event processing. Evidence is `.cache/playback-reproducibility/300dd8b51a94417d9502a3fec85195d3/benchmark.json`. It records an AMD Radeon 780M, OpenGL 3.2/driver 24.10.38, physical TL134ADXP01-0 screen at 165 Hz/DPR 2, raw scene 2012 × 668 pixels, and an active visible window. No application rendering optimization was needed. Benchmark and desktop capability fingerprints now record graphics/display identity; transient per-window visibility is evidence rather than a fingerprint input. The original failed run remains unchanged.

PINN method version 2 has 31 passing targeted checks (11 conditioning, 16 existing PINN, four training-only-normalization tests), including a version-independent eight-closure limit that rolls back weights, gradients, and optimizer state on exhaustion. The bounded native CPU pilot used 100 steps (80 Adam, 20 L-BFGS), completed in 130.25 s, and remains **unqualified**. Its temperature normalized RMSE is 0.163 versus 0.509 in the original 1,000-step run; enthalpy normalized RMSE is 0.667 versus 0.959; independent enthalpy residual RMS is 2.842 versus 29.16. Held-out loss is still 0.445 and liquid-fraction RMSE 0.295, so the unchanged accuracy and conservation gates correctly fail. It recorded 70 L-BFGS closures, three budget-limited rollbacks, and all evaluation counts. Evidence is `.cache/pinn-conditioning-native-100/summary.json`. These differently sized runs demonstrate conditioning improvements, not a qualified CPU/CUDA or coupled model; canonical training reruns remain required.

A real stationary coupled FE source passed a separate 50-step CPU pilot (40 Adam, 10 L-BFGS; 54.58 s total). Held-out loss was 6.60e-9, raw physics loss 1.11e-6, maximum field NRMSE 1.62e-4, and maximum independent conservation RMS 0.008325. Source/model/checkpoint associations were verified with a fresh training alias. Evidence is `.cache/pinn-conditioning-stationary-50/summary.json`. This qualifies that stationary pilot against the existing source-based gates, not the dynamic coupled matrix or the physical model independently.

The subsequent full run under `.cache/test-suite/20261026T133138.135583Z/` exposed a method-version-2 optimizer defect in the canonical 1,000-step native CPU case. It completed in 1,186.813 s but failed: held-out loss 0.735288, training loss 7.77721, physics loss 7.03861, and enthalpy residual RMS 2.83234. Its last 150 recorded epochs were exactly constant, with 40 budget-limited searches reducing the learning rate from 1 to `2^-40` (9.09495e-13) while the maximum gradient magnitude remained 1.45159. This is stagnation, not convergence or just three-significant-digit formatting. The held-out value is also intentionally repeated between ten-epoch evaluations. Original result, checkpoint, and history evidence remain unchanged.

Method version 3 replaces indefinite retry halving with bounded, training-only recovery and explicit update/retry/stall reporting. The original eight-closure guard still restores rejected search state; recovery may retain an already-evaluated Armijo-decreasing finite trial and reset curvature, without consulting held-out labels or adding evaluations. Parameterization and scientific acceptance limits are unchanged. The fixes were developed in a separate checkout while the live user suite was left unchanged, preventing later cases from silently importing changed code during the run. Canonical CPU/CUDA qualification requires a fresh replay after updating the checkout.

All 44 focused checks pass: 17 conditioning/recovery, 16 existing PINN, four training-only field-normalization, and seven diagnostic tests. A read-only checkpoint replay on one CPU thread used only five closures over two recovery attempts in 12.219 s: the first restarted the collapsed learning rate, and the second applied a maximum parameter change of 0.000306461, reducing the full training loss from 7.777209540462489 to 7.773546733641410. Original source/checkpoint hashes were unchanged. Evidence is `.cache/pinn-stall-worktree/.cache/checkpoint-recovery-probe.json`. This verifies restored optimizer motion, not scientific qualification; the native model remains unqualified pending canonical replay. The still-running suite recorded a 1,800 s timeout for its old-code CUDA counterpart.

After that run finishes and the checkout is updated, retry the native thermal pair with `python -B scripts/run_all_tests.py --resume ".cache/test-suite/20261026T133138.135583Z" --case "pinn.thermal.*"` in the `torch-gpu` environment. Keep using the saved report for all other required gates; a successful recovery step alone does not qualify the trained fields.

The rebuilt Geant4 worker passes nine new actual-binary/diagnostic tests and 16 existing radiation regressions. All five previously failing requests (`dose.pair.03`, `.06`, `.11`, `.16`, and `dose.mixed-spectrum`) now complete successfully, using immutable original field and primary files with redirected private outputs. Every original request, field, and primary hash is unchanged. Nonzero-field case 16 queried at most radius 6.48980 m inside certified support radius 6.84973 m, with step cap 0.0924321 m. Exactly zero maps use no magnetic field lookups and preserve bitwise-equal paired doses and standard errors. Evidence is `.cache/qualification-fixes/transport/guards-20261026-v1/report.json`; installed worker SHA-256 is `14e4a2f4082564208f34286a4ce8de8c21c268d38e22a666ae59c2d6a6f549f0`.

An 8,192-history zero-field water pilot completed in 16.657 s; its RSE 0.40809 is **not** a passing precision result. Halving the nonzero-field step cap changed the sampled dose by 0.986 combined standard errors and retained the same primary habitat/hardware counts. With only 128 histories and RSEs 0.508/0.699, dose convergence is statistically unresolved. The 524,288-history precision case and the full matrix must still pass their measured gates; neither timing projections nor successful execution qualify them.

## Structure and playback refactor (2026-10-25)

- Before/after comparisons preserve all 30 captured native thermal, diagnostic,
  source-sampling, and trajectory arrays exactly. PINN comparisons preserve 27
  native arrays/metrics and model/optimizer checkpoint tensors exactly; a saved
  FE case also preserves all 52 coupled arrays/metrics (1961 points, five snapshots).
- The isolated regression suite, real desktop tests, and actual desktop smoke
  workflow pass. Additional checks cover dependency isolation, headless session
  state, coalesced rendering, cached linear section interpolation, visible
  pre-run duration editing, and movie startup/cancellation when switching runs.
- Small actual stationary/driven FE and ground/water Geant4 integrations pass
  after moving their Python execution boundaries. These checks do not close the
  scientific qualification gates listed below.
- A fresh comparison on the reference workstation measured **44.93 FPS before**
  and **52.17 FPS after** (about **16.1% faster**), using the median of five
  60-frame repetitions including Qt event processing. Both use the resolution-8
  thermal reference (2048 points), a 1282×780 window, Medium detail, opacity 100%,
  and optional overlays off. This is separate from the earlier 31.77 FPS record;
  timings depend on the machine and workload.
- Thermal plot regressions preserve small changes on large baselines: the
  recorded 5 s Iron run adds 65 kJ to 65.1327 kWh initially stored and warms from
  1800 K to approximately 1800.0277 K. Heating stops at 3.25 s; the remaining
  plateau is expected with zero discharge/radiation loss. The default change
  view exposes this signal, labels the 1700 K discharge reference, and retains
  absolute totals as an option. Probe axes fit computed histories rather than
  an off-scale melting point. Compact-card checks cover complete axis labels and
  reference annotations; equations, inputs, and accepted fields are unchanged.
  With those clarified chart layouts, the latest five-run median is 50.82 FPS
  versus 52.17 FPS before the plot changes (within the 10% regression budget,
  and approximately 13.1% faster than the original 44.93 FPS baseline).

Reproduce paired timing using the same Python environment and graphical session:

```powershell
python -B scripts/benchmark_desktop.py --output .cache/playback-before.json
python -B scripts/benchmark_desktop.py --output .cache/playback-after.json --baseline .cache/playback-before.json
```

Run the first command before changes and the second after them. Historical
checkouts require their matching benchmark harness and dependencies; copying the
current script alone is insufficient. Legacy baseline JSON remains unchanged;
the new comparison report marks it with `baseline_environment_verified: false`
when its graphics-environment identity is absent. The current harness
rejects mismatched fixture sizes/counts and a slowdown greater than 10%.

## Coupled Iron PINN follow-up (2026-10-25)

The reported default Iron fit used 250 CUDA epochs, batch size 256 and seed 42,
against 43 accepted FE snapshots on 1961 points. A separate private diagnostic
changed only the epoch count to 1000 and completed in 161.7 seconds. It preserved
the user's saved runs and used the same equations, normalization, split and gates.

| Check | 250 epochs | 1000 epochs | Required |
| --- | ---: | ---: | ---: |
| Held-out normalized MSE | 0.1373 | 0.0514 | ≤0.01 |
| Velocity normalized RMSE | 0.2865 | 0.2744 | ≤0.1 |
| Pressure normalized RMSE | 0.4630 | 0.2419 | ≤0.1 |
| Electric-potential normalized RMSE | 0.7260 | 0.4179 | ≤0.1 |

Both fits correctly remain **failed**. Temperature RMSE improved from
5.42e-5 K to 1.68e-5 K; magnetic-field accuracy and all independently checked
normalized residuals meet their gates. Those passing checks do not override
the failed field-accuracy gates. Longer training improves this fit but does not
qualify the coupled PINN. The source is fully solid, with only 39 free velocity
nodes; training-label sampling and electrical-potential conditioning remain
specific investigation targets. Diagnostic evidence is in
`.cache/iron_pinn_diagnostic/report.json`.

## Checks exercised

- Sharp T(E) branches and latent-heat plateau; SI heat-capacity units; regularized constitutive continuity.
- Conservative charging/discharging and radiation loss; closed native thermal balance residual about 2.14e-6 J in the recorded run.
- Neumann Stefan similarity front: 32/64/128 cells yield relative front errors 0.2409% / 0.0147% / 0.0460%. Subcell front estimation is not monotonically convergent at the finest pair; the report retains those values rather than claiming an observed convergence order.
- Actual 3D conductor/vacuum FE stationary and heated/pumped/current-driven mixed-VMS solves. Accepted fields satisfy the exported differentiable weak operator; backpropagation is finite.
- Seed removal in a molten pumped torus reports decay, not artificial dynamo growth.
- FE Hartmann channel, Ha=4: velocity relative L2 errors 0.835% / 0.211% / 0.0528% at 16/32/64 elements. Magnetic diffusion eigenmode amplitude error 0.223% after 200 steps.
- Per-scenario PINN CPU/CUDA execution, strict source binding, conservative operator checks, held-out splits and checkpointing. A radiating native benchmark passed after 1000 epochs with temperature RMSE 0.0708 K and maximum interval energy defect 0.0199%. Coupled training can underfit flow/electrical fields and must be reported failed when it misses its gates.
- Relativistic Boris zero-field motion, gyro-radius and energy; true FE cell curl(A), out-of-map rejection, heavy-ion/spectrum validation, paired source identity, and stale model/dose rejection.
- Real Qt/VTK repeated edit/run, probe, comparison, invalid input, cancellation, failed workers, late-response isolation, image export, and decoding all 60 encoded MP4 frames. Playback performance is measured and reported by the smoke script; it must be evaluated after rendering changes.
- Fixed-font persistent annotations and queued chart repaints measured **31.77 FPS** including Qt event processing on the accepted native resolution-8 scene (2048 finite-volume points), Medium visual detail, opacity 100%, default overlays off, actual 1282×780 window, Radeon 780M. Scene-only field updates measured about 143 FPS. Additional layers, transparency, larger fields, and concurrent computation need their own workload measurements.
- Linear FE scalar interpolation at cutaway vertices and sections was checked against an analytic linear temperature field: maximum errors 5.68e-12 K and 2.27e-13 K. Clipped source-node identifiers are never treated as interpolated scientific values.
- Post-change isolated stationary and driven FE checks passed with the final stored-field energy diagnostics and memory controls. The driven check verifies imposed + induced + mutual energy equals total magnetic energy, accumulated pump work matches accepted-step power integration, and the actual MUMPS ICNTL(23) readback equals its configured RAM ceiling. The fast native module passes ten tests; its two optional FE tests run separately. The impossible native snapshot request is rejected before graph allocation.

The installed Ubuntu FE binary package versions, including Python, DOLFINx,
Gmsh, PETSc, MUMPS, NumPy and SciPy, are recorded in
`env/physics-ubuntu.lock.txt`. The final cap/energy checks are recorded in
`.cache/physics_postchange_validation.json`. Stored imposed-field energy is
reported separately from pumping; it does not represent the electrical
consumption of an ideal imposed-current circuit.

Runtime artifacts and machine-readable numerical reports live under `.cache/`. Test modules run in isolated processes to avoid the conda environment's SciPy/PyTorch OpenMP conflict. The optional full FE tests require `TORUS_TEST_FENICS=1`; they are exercised separately from the fast suite.

## Recorded PINN training outcomes

- Native radiating thermal source, 1000 epochs: passed the held-out field and conservation gates, temperature RMSE 0.0713 K and maximum interval energy defect 0.01997%. Report: `.cache/pinn_final_native/report.json`.
- Current-code stationary FE regression: passed. This does not validate evolving turbulent fields. Report: `.cache/pinn_stationary_regression/report.json`.
- Nontrivial driven/mushy mixed-VMS source, 2000 CUDA epochs: failed. Held-out mean normalized MSE 0.00817 and small equation residuals did not compensate for pressure normalized RMSE 0.158 and electric-potential normalized RMSE 0.140, above the unchanged 0.1 per-field gate. Report: `.cache/pinn_coupled_2000/report.json`.
- Nontrivial molten seed-removal mixed-VMS source, 1000 CUDA epochs: failed. Magnetic field, pressure, and potential normalized RMSE were 0.255, 0.139, and 0.325. Report: `.cache/pinn_dynamo_1000/report.json`.

Failed models remain inspectable and are labelled failed in the scene and validation charts. They cannot supply radiation fields. The general coupled PINN accuracy gate remains open.

## Full-suite follow-up (2026-10-26)

The original attempt-0001 full-profile run at
`.cache/test-suite/20261026T133138.135583Z/`, revision `47c87d7`, recorded 129
cases: 120 PASS, seven FAIL, and two TIMEOUT after 4 h 30 min 28 s. All numerical
solves, functional transport cases, and desktop checks passed. The five-repeat
desktop benchmark measured 48.13 FPS. The attempt-0001 artifacts remain
preserved. The 524,288-history water ensemble measured 5.08554% relative
standard error, correctly failing the 5% gate. A later 10-case method-5 resume
updated the report's selected-case summary; its results are recorded below.

Both current profiles now use one fixed 1,048,576-history ensemble with the
same seed, source, geometry, normalization, and physics list. Its projected
uncertainty is approximately 3.6% and paired runtime approximately 19 minutes,
based on the last ensemble's measured statistics and time. These are projections;
the actual new nonzero score must satisfy the unchanged measured uncertainty
gate. Earlier correlated attempts are not pooled, and zero-field pairing does
not supply twice as many independent histories.

Training method 5 adds a fixed native data warmup/physics ramp and training-only
correction activation. Coupled corrections freeze the original core and use
cloned thermofluid and/or pressure/potential branches with fixed spatial Fourier
features. The primary request remains separate from the additional bounded Adam
and joint L-BFGS budgets. Metadata and checkpoints record actual completed
steps, stages, evaluations, and architecture versions. Final validation still
uses the original held-out field, loss, conservation, and source-identity gates.
Cancelled or failed line searches now restore the pre-step weights, gradients,
and optimizer state before propagating the error.

The eight unsuccessful full-profile PINN cases had 3,600 s execution limits;
feature case limits and the one-hour representative supervisor remain unchanged.
Longer execution does not qualify an inaccurate model. The targeted five-case
method-5 outcomes are recorded below; Mushy and the other unsuccessful PINNs
remain unqualified. Later targeted Liquid Dynamo and enlarged water runs passed
their existing gates, without qualifying the complete profiles. See
[TEST_MATRIX.md](TEST_MATRIX.md#rerun-the-outstanding-pinn-and-water-precision-cases)
for the current targeted resume command and separate one-hour default.

Private checkpoint pilots preserved the original source, checkpoint, and report
hashes. They diagnose conditioning rather than replaying primary training:

| Source | Diagnostic stages | Time | Outcome |
| --- | --- | ---: | --- |
| Iron, old completed epoch-1000 checkpoint | Prototype method 4: 2,000 label-only Adam + 50 joint L-BFGS | 217.4 s | Per-source gates passed: pressure normalized RMSE 0.06561, potential 0.04689, held-out MSE 0.001088. This prototype is not the shipped method. |
| Lunar, old partial epoch-900 checkpoint | Prototype method 4: 2,000 label-only Adam + 50 joint L-BFGS | 335.2 s | Failed: enthalpy 0.3239, pressure 0.1197, potential 0.5487 normalized RMSE; continuity/charge also failed badly. |
| Native phase-changing source, old completed epoch-1000 checkpoint, CUDA pilot | Method 5: 2,000 original-joint-objective Adam + 50 joint L-BFGS | 184.3 s | Failed: enthalpy 0.8174, temperature 0.1554, liquid fraction 0.3639 normalized RMSE; enthalpy residual RMS 2.0097. |

The label-only correction policy was rejected after the Lunar pilot: fitting
pointwise labels without equation losses damaged continuity and charge.
Method 5 retains the original full-field sampled data, physics, and boundary
objective on every correction Adam step. Regression checks cover its nonzero
physics/boundary gradients, cancellation rollback, and held-out independence.
The first interrupted prototype attempts are also retained, but their
post-interruption trial weights are not qualification evidence. Current pilots
and notes are under `.cache/conditioning-pilots-20261026/`. Original checkpoints
did not contain NumPy generator state; these diagnostic stages start a fresh
seeded preparation stream and are not uninterrupted canonical fits.

Following the requested scope reduction, normal desktop training now excludes
phase-changing native thermal and time-varying coupled MHD scenarios. Pure-phase
native thermal and stationary coupled training remain available. A disabled
training button explains the limitation; numerical simulation and saved model
inspection remain available. These family checks do not guarantee that every
enabled custom fit will qualify. Failed models still fail the original gates
and cannot supply radiation fields. The full research cases remain executable
and unsuccessful outcomes are preserved; no failed qualification was converted
into a passing test to reduce the supported desktop scope.

Current method-5 checks confirm both retained training paths:

- The representative single-phase native source and its fresh 1,000-step CPU
  fit both passed through the normal suite runner. Fit time was 62.58 s;
  held-out MSE was `4.55e-8`, temperature RMSE `0.0002132 K`, normalized enthalpy
  residual RMS `0.0006017`, and relative interval energy defect `5.58e-5`.
  Evidence: `.cache/conditioning-feature-native-20261026/report.json`. This
  two-case selection correctly claims neither complete feature nor full coverage.
- A fresh 250-step CUDA fit on the saved accepted stationary FE source passed
  in 105.19 s: held-out MSE `1.49e-10`, maximum field normalized RMSE `2.77e-5`,
  maximum independently checked residual RMS `0.0005857`. Source binding and
  saved-result roundtrips passed. Evidence:
  `.cache/conditioning-stationary-20261026/report.json`.

The focused regression selection passed 196 tests, including 15 real Qt/VTK
desktop tests with desktop checks enabled, source eligibility, immutable-source
caching, unavailable-action guards, and saved passed/failed model inspection.
The real desktop smoke journey also passed with four runs, particle replay,
saved-model inspection, and all 60 MP4 frames decoded. Its raw renderer size
changed from 1938×400 to 1714×478 during export; smoke playback measured
35.64 FPS. This is the 256-point smoke fixture, not a replacement for the
resolution-8 five-repeat benchmark. Evidence:
`.cache/conditioning-desktop-20261026/output.log`.
At the time of these checks, the complete one-hour representative invocation and
enlarged water ensemble had not been run. The later targeted water run passed as
recorded below; these short checks do not replace a complete invocation.

## Canonical method-5 coupled PINN replay (2026-10-27)

The requested full-profile resume selected five accepted FE source cases and
their five dependent PINNs. Source fingerprints required recomputing all five
parents; all five source runs passed and the completed PINNs remained bound to
their recorded source run IDs. Method 5 used seed 42, CUDA, batch size 256, and
1,000 requested primary steps. Its training-only activation target remained
0.05. Completed fits used 3,050 total optimization steps: 800 primary Adam,
200 primary L-BFGS, 2,000 joint correction Adam, and 50 joint correction
L-BFGS. No thresholds or physics objectives changed.

| PINN case | Source run | Activated correction groups and triggering training diagnostics | Held-out MSE / physics MSE / boundary MSE | Maximum normalized field RMSE | Maximum independent residual RMS | Elapsed | Result |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| Iron | `training-0f29303d2efd4826a2b4816f2b6e8000` | Elliptic (pressure 0.2039, electric potential 0.2413) and thermofluid (velocity 0.0830); maximum activation residual 0.04202 (charge) | 0.0011653 / 0.0002213 / 3.31e-37 | 0.06801 (pressure) | 0.04202 (charge) | 1000.61 s | **PASS** |
| Regolith composite | `training-7c874db6c9b74a8fb754ea17924f5998` | Elliptic (pressure 0.1280, electric potential 0.06710); maximum activation residual 0.0008660 (charge) | 0.0009555 / 1.39e-7 / 8.84e-38 | 0.05955 (pressure) | 0.0008660 (charge) | 1039.98 s | **PASS** |
| Mushy Driven VMS | `training-6481dcff0aac45308dc25f33f5dce3e1` | Elliptic (pressure 0.1603, electric potential 0.1396) and thermofluid (velocity 0.06520); maximum activation residual 0.007600 (charge) | 0.0025108 / 8.91e-6 / 1.32e-37 | **0.11633 (pressure)** | 0.007600 (charge) | 862.98 s | **FAIL** |
| Liquid Dynamo VMS | No finalized PINN metadata | No correction activation recorded; primary L-BFGS reached epoch 862 before CUDA out-of-memory errors in backpropagation and rollback | Intermediate held-out MSE 0.010006 at epoch 860; no final loss audit | Not available | Not available | 3659.95 s | **TIMEOUT** |
| Freezing | `training-f7a7eac9a1ae454db67eecb95b45e7b5` | Elliptic (pressure 0.1138); maximum activation residual 0.08493 (momentum) | 0.00005525 / 0.0007200 / 2.41e-64 | 0.01773 (pressure) | 0.07298 (momentum) | 723.36 s | **PASS** |

The full training-only field and equation diagnostics, activation decisions,
source hashes, optimizer stages, and actual step/evaluation counts are in each
completed attempt's `training/*/metadata.json`; final checks are in its
`result.json` and `runner-result.json`.

The three passing cases satisfy the existing held-out, loss, field-error,
independent-residual, conservation, and source-identity checks. Mushy Driven
VMS completed all 3,050 steps and passed the held-out loss, physics/boundary
loss, and residual checks, but pressure normalized RMSE was 0.11633, exceeding
the unchanged 0.1 per-field limit. This is the measured reason its combined
`heldout_and_conservation_gates` check failed; the other fields and residuals
do not override that pressure failure.

Liquid Dynamo VMS failed before a final audit. After progress reached primary
epoch 862, its worker log shows CUDA out of memory during backpropagation in an
L-BFGS closure, followed by another CUDA out-of-memory error while restoring
weights. At about 218 s the worker wrote an `ERROR` result, but its process did
not exit before the deadline. The runner waits for process exit and ultimately
recorded TIMEOUT at 3,659.95 s and qualification `NOT_RUN`; the reason the
process remained alive is not established. No completed training metadata was
published, and the intermediate loss is not a passing result. The retained
attempt and diagnostics are under
`.cache/test-suite/20261026T133138.135583Z/cases/pinn.fe.liquid.dynamo.vms/attempt-0002/`.

The report for that targeted selection recorded 10 cases: eight PASS
(five sources and three PINNs), one FAIL, and one TIMEOUT. Its 119 unselected
results were retained, and full coverage remained false. The exact replay
command was:

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --profile full --resume ".cache/test-suite/20261026T133138.135583Z" --case "pinn.preset.iron" --case "pinn.preset.regolith-composite" --case "pinn.fe.freezing" --case "pinn.fe.liquid.dynamo.vms" --case "pinn.fe.mushy.driven.vms"
```

These replay results did not enable additional desktop PINN modes or qualify the
wider coupled family. The failed pressure case, other unsuccessful PINNs, and
complete profiles remain open; later targeted Liquid Dynamo and water precision
runs passed their existing gates.

## Pressure and CUDA follow-up (2026-10-27)

### Liquid Dynamo targeted rerun

The later user-run full-profile selection passed the Liquid Dynamo case and
its FE prerequisite. The PINN completed 3,050 steps in 640.703 s on CUDA with
seed 42 and the unchanged qualification gates: held-out MSE 0.00304978,
maximum field normalized RMSE 0.097568 (pressure), maximum independently checked
residual RMS 0.035691 (magnetic consistency), physics loss 0.00021645, and
boundary loss 5.30e-37. This is a pass for that source, not qualification of
all evolving dynamos. The earlier OOM remains unexplained; telemetry did not
record one during this rerun. Original case evidence remains under
`.cache/test-suite/20261026T133138.135583Z/cases/pinn.fe.liquid.dynamo.vms/attempt-0003/`.
The later review fixes can invalidate this attempt's fingerprint on resume;
they do not change the historical result. The enlarged water case is separate.

The later fixed 1,048,576-history water-precision case also passed, with
relative standard error 3.5828% against the unchanged 5% gate. Both the water
case and its prerequisite passed; the targeted selection does not qualify the
complete feature or full profile. Evidence:
`.cache/test-suite/20261026T133138.135583Z/cases/dose.water-precision/attempt-0002/result.json`.

The research-workflow review fixes were verified with eight focused suite cases
(94 tests), including real desktop, publication, temporal checkpoint, and
supervisor coverage. This targeted selection does not qualify a complete profile.
After the reporting compatibility adjustment, the 31 publication and supervisor
tests passed again. Evidence: `.cache/review-fixes-20261027/report.json` and
`.cache/review-fixes-final-20261027/report.json`.

### Timestep, operator, Huber and memory investigation

A new 12-nominal-step source passed the existing FE source checks. On the
matched conductor mesh at accepted t=0.020 s, node-weighted pressure RMS
differences were 0.0047032 Pa (original versus six steps), 0.0073358 Pa
(original versus 12), and 0.0026523 Pa (six versus 12). The last comparison
has velocity RMS difference 2.5696e-9 m/s and maximum liquid-fraction
difference 3.6123e-9. These are differences among three discrete calculations,
not a convergence proof or a direct comparison to the all-time PINN gate.
The new report records each common accepted time, source-specific normalized
residuals, nodal/volume-weighted errors and quantiles without interpolation.
At 0.020 s, held-out pressure absolute error had median 0.0024242 Pa, 90th
percentile 0.0073822 Pa, and maximum 0.0182440 Pa. Volume-weighted held-out
statistics describe only sampled nodes and are not a whole-volume estimate.
Evidence: `.cache/pinn-pressure-followup/diagnostics-refined12-isolated/`.

The independent worker/PyTorch comparison at 0.01667–0.020 s used matching
fields, mesh, masks and gauge. The five analytic directional Jacobian actions
and four raw momentum, continuity, induction and charge residuals matched the
declared comparison tolerances. Raw enthalpy residuals differed by up to
3.5654e-4 (worker maximum 2.0260e-4), so the strict raw comparison **failed**;
the discrepancy after the existing qualification normalization was 1.9028e-12,
well below the 0.1 residual limit. A counterfactual with equal current and
previous temperature reduced the raw difference to 7.39e-12, consistent with
transient enthalpy cancellation but not proof of its cause. This counterfactual
is not an accepted solve. No qualification gate was changed. Evidence:
`.cache/pinn-pressure-followup/operator-probe-final/report.json` and
`.cache/pinn-pressure-followup/operator-probe-v14-counterfactual/report.json`.
On the distinct accepted 0–0.01667 s interval, the same four residual and five
Jacobian comparisons matched; raw heat again failed (maximum difference
1.1214e-4, normalized maximum 4.49e-13). Evidence:
`.cache/pinn-pressure-followup/operator-probe-second-interval/report.json`.
New operator probes record the worker-output SHA-256 for verified reuse. Older
reports without this digest remain readable, but their worker output cannot be
reused; run a fresh probe to create verifiable evidence.
A fresh source-bound worker and its verified reuse produced matching probe hashes
and both retained the documented raw enthalpy comparison failure. Evidence:
`.cache/review-probe-digest-20261027-fresh/report.json` and
`.cache/review-probe-digest-20261027-reuse/report.json`.

The source-bound Huber pilot used 0.003 Pa and the training-only pressure scale;
its quadratic region matches MSE. Against a matched control, training pressure
normalized RMSE was 0.0857890 versus 0.0850952, so Huber was not selected for
a refined-source screen. After that selection was frozen, held-out pressure
errors were 0.1174506 versus 0.1158900; both fail the unchanged 0.1 gate. Both
completed 200 Adam and 10 L-BFGS steps, with unchanged input and implementation
hashes. Evidence: `.cache/pinn-pressure-followup/huber-selection-20261027.json`
and the `huber-control-20261027` / `huber-candidate-20261027` reports.

Opt-in telemetry records CUDA allocation, reservation, process peaks, device
free memory, stage/closure counts, checkpoints, rollback and errors. A saved
Liquid epoch-850 checkpoint completed 13 L-BFGS steps and 26 closures in 16.36 s
without OOM. Allocated memory after each step was 21,912,064 bytes and peak
allocation was 203,548,672 bytes. This does not identify the earlier intermittent
failure or prove a full run will pass. Telemetry begins after model preparation,
so preparation failures are not logged. It does not empty the CUDA cache.
Evidence: `.cache/pinn-memory-liquid-probe-20261027/report.json` and its JSONL log.

Five focused cases passed through the suite runner (44 individual tests),
covering operator comparisons, catalog, CPU/CUDA telemetry, pressure pilots,
and pressure diagnostics. This partial selection does not qualify either
complete profile. Evidence:
`.cache/pinn-pressure-followup/investigation-suite-20261027/report.json`.

Enable telemetry for the targeted Liquid rerun in PowerShell:

```powershell
$env:MAGNETIC_TORUS_PINN_MEMORY_LOG = (Join-Path (Get-Location) '.cache\pinn-memory-liquid.jsonl')
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --profile full --resume ".cache/test-suite/20261026T133138.135583Z" --case "pinn.fe.liquid.dynamo.vms"
Remove-Item Env:MAGNETIC_TORUS_PINN_MEMORY_LOG
```

The log is opt-in, append-only and PID-tagged. The runner reuses prerequisites
only when their fingerprints and artifacts remain valid. The Liquid and water
reruns above have since passed as separate targeted selections.

Exploratory signed and absolute pressure maps, per-snapshot statistics, radial
sample counts, and the accepted-field Lorentz-force reconstruction are saved in
`.cache/pinn-pressure-followup/diagnostics-v5/`. At the original accepted
0.020 s snapshot, 112 of 679 held-out pressure samples account for 55.56% of
their total squared error. The conductor liquid fraction is nearly uniform
(0.60000150–0.60000157), so this snapshot does not establish a phase-front
cause. Outer-radius samples are numerous on this mesh; their error share alone
does not establish a wall layer. The source and prediction hashes were unchanged.

A separate driven FE source with six nominal steps and split seed 123 passed
the accepted-source energy and independent weak-operator checks. Its seed field
is inactive in driven mode. At the common accepted 0.020 s snapshot, with mesh
coordinates and pressure anchor matched, its pressure differs from the original
by 0.0047032 Pa RMS across conductor nodes; velocity differs by 1.6506e-8 m/s
RMS and liquid fraction by at most 4.20e-9. No temporal interpolation was used.
This single refinement shows sensitivity, not mesh/timestep convergence or an
independent physical experiment. The refined-source aggregate difference was
seen before pilot selection, so the later screen is not fully blinded.
Evidence: `.cache/pinn-pressure-followup/refined/` and
`.cache/pinn-pressure-followup/diagnostics-v5/report.json`.

Three source-bound warm-start pilots each completed 200 Adam and 10 L-BFGS
steps. The control, temporal-pressure sampler, and pressure-only Fourier-16
head produced training pressure normalized RMSE 0.0850952, 0.0839624, and
0.0847815, respectively. The temporal sampler was selected from these
training-only diagnostics, before evaluating any candidate on the refined
source. It samples adjacent-time pressure variation only from matched training
nodes, with weights capped at 3:1. Research training uses training-only weak
scales; each final audit restores the original unweighted source oracle.
On the already examined original source, all three pilots still failed the
pressure field gate: held-out normalized RMSE was 0.115890, 0.114909, and
0.116011, respectively. The Fourier candidate is a zero-initialized additive
pressure-only sinusoidal head; its outcome does not rule out richer models.
The comparison and hashes are in
`.cache/pinn-pressure-followup/selection.json` and the three `pilot-*` reports.

The refined-source screen trained a fresh source-bound core for 210 steps,
then trained the frozen temporal candidate for another 210 steps using only
that source's checkpoint. The latter completed in 55.0 s but **failed** the
unchanged gates: held-out MSE 0.0165178, pressure normalized RMSE 0.125310,
electric-potential normalized RMSE 0.288336, and maximum residual RMS 0.05229
(charge). Physics loss 0.000394955 and boundary loss 1.33e-37 passed. Source
and implementation hashes were unchanged during both stages. These are short
research fits; no candidate was promoted and no desktop mode was enabled.
Evidence: `.cache/pinn-pressure-followup/refined-pretrain/report.json` and
`.cache/pinn-pressure-followup/refined-temporal-screen/report.json`.

For the saved Liquid source and epoch-850 checkpoint, the full FE graph,
bounded neural recomputation, CPU-saved tensors, and history-size-10 alternatives
matched the original full assembled objective and gradients in short probes.
Three-repeat median times per optimizer step were 1.142, 1.381, 2.502, and
1.628 s respectively; the fastest full-graph path remains the normal default.
Peak GPU allocations were approximately 192–194 MiB, with about 7.2–7.4 GB
free before the probes. Three further 13-step full-graph probes, each starting
with the checkpoint's 20 history entries, completed all 39 steps without OOM;
median time was 14.55 s per 13 steps. These bounded results do **not** prove
that the earlier intermittent CUDA OOM is fixed. The original exception is now
preserved if rollback also fails. Evidence:
`.cache/pinn-fe-vjp-profile-20261027-b/report.json` and
`.cache/pinn-fe-vjp-profile-20261027-c/report.json`.

Five focused cases passed through the suite runner, covering the catalog,
runner lifecycle, CPU/CUDA FE gradient equivalence, pressure pilots, and
pressure diagnostics. After adding private source staging, the pressure-pilot
case passed again (15 tests); a real-source CPU equivalence probe and a
one-step Liquid GPU probe also passed. Evidence is under
`.cache/pinn-pressure-followup/suite-check/`, `suite-check-final/`, and
`.cache/pinn-fe-vjp-profile-private-check/`. These targeted runs do not
constitute feature or full-suite qualification.

The suite runner now reports a valid current-attempt worker failure after a
five-second exit grace and owned-process cleanup, rather than waiting until a
case timeout. PASS still requires a successful process exit. Cancellation and
deadlines take precedence, and the worker error remains available separately.
The pressure candidates remain experimental, so the targeted long rerun is
Liquid only:

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --profile full --resume ".cache/test-suite/20261026T133138.135583Z" --case "pinn.fe.liquid.dynamo.vms"
```

## Bounded proposal experiments: 2026-10-26

Nine real pilots on saved accepted sources tried front/residual sampling, cached
causal weighting, time FiLM, a temporal potential branch, and inverse
label-gradient balancing. All completed; all still failed whole-model
qualification under the original gates. Native front + causal reduced enthalpy
normalized RMSE from 0.8051 to 0.7511, still above 0.1. Lunar FiLM + causal
reduced pressure error from 0.6072 to 0.4622 but worsened conservation; Iron
label-gradient balancing worsened pressure/potential error.

The temporal Lunar potential branch reduced phi normalized RMSE from 0.64046 to
0.00000584837 without changing reference labels or the FE gauge. The other
fields were frozen; continuity RMS 0.8604, charge RMS 1.5266 and physics loss
0.38385 still failed. This component improvement does not qualify Lunar or
enable PINN radiation. No disabled desktop training mode was re-enabled.

See [PINN_EXPERIMENTS.md](PINN_EXPERIMENTS.md) for protocol differences,
training-only adaptation, source/checkpoint hashes, timings and the private
research command. Evidence and comparison images are under
`.cache/pinn-proposal-research/`. These short single-seed pilots do not replace
canonical full-profile replay, the enlarged water run, or the existing release
gates. The original completed full report and accepted sources were preserved.

## Release gates still open

- Full coupled manufactured-solution convergence and published MHD Taylor–Green comparison for the adapted A–phi mixed VMS closure.
- Exterior-domain and mesh refinement; robust large-mesh linear solver scaling; global MHD/subgrid energy budget qualification and closure sensitivity.
- Qualified temperature-dependent material/composite datasets and applicable ranges.
- Independent PINN qualification across the selected numerical benchmark family, rather than one scenario's successful fit.
- Stopping/range references, Geant4 physics-list cross-checks, field-map refinement, and normalized representative heavy-ion transport benchmark comparisons.
- At most 5% Monte Carlo statistical uncertainty in selected nonzero dose regions. Zero-hit or insufficiently sampled organs cannot be labelled successful.
- Reproducible approximately 30 FPS performance on a specified medium-detail scene, plus responsive controls under the user's intended workload.

Water and male/female phantom runtime smoke tests exercise the installed transport engine; their completion and statistics must be evaluated separately from the unit tests. No transport smoke result establishes shielding effectiveness.

## Recorded Geant4 transport checks

Geant4 11.4.3 runs as a compiled sequential worker in Ubuntu WSL. Its toolkit,
development libraries, and physics datasets are recorded in
`env/geant4-linux-64.explicit.txt` as package URLs with integrity digests.
`scripts/setup_geant4.sh` replays that lock and downloads the official CERN
ICRP110 dataset. Set `TORUS_TEST_GEANT4=1` to run the three optional real-transport
checks in `tests/test_torus_radiation.py`; the ordinary suite runs ten radiation
physics/provenance tests without launching Geant4.

The following runs used 100 MeV protons, seed 42, a uniform incident beam, and
one particle/m² incident fluence. The first four used explicitly synthetic
zero-field fixtures, a 1.6 m major radius, 0.2 m reservoir radius, 1 m spherical
air-filled habitat, and a 100 µm production cut. Counts are per field ensemble;
identical primary samples and per-event random streams were used for field on
and field off. Maximum doses are recorded to check execution and normalization.

| Target / physics | Primaries | Scored mass | Maximum observed Gy | Relative standard error at that region | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| Water / Shielding + EM4 | 1024 | 64.000000 kg | 1.41951e-15 | 100% | 19.91 s |
| ICRP110 male / Shielding + EM4 | 64 | 73.145486 kg | 3.77194e-12 | 100% | 32.49 s |
| ICRP110 female / Shielding + EM4 | 64 | 60.028763 kg | 2.38395e-12 | 100% | 38.53 s |
| Water / QGSP_BIC_HP + EM4 | 256 | 64.000000 kg | 5.71329e-16 | 86.4% | 18.31 s |

All zero-field pairs produced exactly identical absorbed-dose arrays. The male
and female phantoms scored 137 and 136 tissue regions respectively. The official
slice files' label 141 is assigned the skin material, following the
[upstream Geant4 ICRP110 implementation](https://github.com/Geant4/geant4/blob/master/examples/advanced/ICRP110_HumanPhantoms/src/ICRP110PhantomConstruction.cc).
Internal body air is transported but excluded from organ-dose reporting.

A further integration check loaded the accepted `.cache/fe_coupled` conductor
and vacuum mesh directly: 1961 nodes, 11485 tetrahedra, physical time 0.1 s, and
maximum element field 0.632591 T. Eight Boris paths and 64 Geant4 primaries per
ensemble completed using the exact element curl of the accepted total P1 vector
potential. This exercised actual FE node ordering, tetrahedral interpolation,
field extents, and transport without an out-of-map fallback. No dose was observed
in that small water sample; it does not establish zero physical dose. Runtime
was 21.39 s including trajectory tracing. Artifacts are in
`.cache/radiation-smoke`, `.cache/radiation-qgsp-smoke`, and
`.cache/radiation-fe-smoke`.

Dose uses event-summed deposited energy divided by scoring mass, including all
secondary tracks. Standard errors use primary histories as independent samples;
correlated deposits from a primary and its descendants remain in the same sample.
Both means and standard errors are multiplied by incident fluence × projected
source area. Secondary-only contributions also retain event-level errors.
Sparse voxel artifacts contain observed deposition only. Zero-hit voxels and
organs are statistically unresolved. These samples miss the 5% precision gate,
and the QGSP run is an execution cross-check rather than a qualified comparison
with Shielding. Stopping/range validation, representative heavy-ion spectra,
production-cut and field-map convergence, and independent phantom dose
benchmarks remain open.
