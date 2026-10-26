# Validation record

Reference workstation: Windows, Python 3.12 `torch-gpu`, Radeon 780M rendering, RTX 4060 Laptop CUDA training, 16 GB host RAM; Ubuntu WSL DOLFINx 0.10.0. These are integration/reference checks, not a qualified shielding design.

Reproduce with `python -B scripts/run_checks.py`, `python -B scripts/desktop_smoke.py`, and `python -B -m magnetic_torus.physics --benchmark-report .cache/physics_validation.json --include-fenics`.

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
current script alone is insufficient. Legacy baseline JSON remains readable but
is marked as lacking verified graphics-environment identity. The current harness
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
