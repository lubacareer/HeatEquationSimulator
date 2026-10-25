# Validation record

Reference workstation: Windows, Python 3.12 `torch-gpu`, Radeon 780M rendering, RTX 4060 Laptop CUDA training, 16 GB host RAM; Ubuntu WSL DOLFINx 0.10.0. These are integration/reference checks, not a qualified shielding design.

Reproduce with `python -B scripts/run_checks.py`, `python -B scripts/desktop_smoke.py`, and `python -B -m magnetic_torus.physics --benchmark-report .cache/physics_validation.json --include-fenics`.

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

Reproduce paired timing using the same Python environment and graphical session:

```powershell
python -B scripts/benchmark_desktop.py --output .cache/playback-before.json
python -B scripts/benchmark_desktop.py --output .cache/playback-after.json --baseline .cache/playback-before.json
```

Run the first command before changes and the second after them. To compare the
historical pre-refactor checkout, copy the benchmark script onto that checkout;
the harness supports both workspace interfaces. It rejects mismatched fixture
sizes/counts and a slowdown greater than 10%.

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
