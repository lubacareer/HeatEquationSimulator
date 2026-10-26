# Magnetic Torus Simulator

A visualization-first Python desktop for phase-change thermal storage, resistive magnetohydrodynamics (MHD), physics-informed neural networks (PINNs), and charged-particle transport. Built on HeatEquationSimulator, it combines an interactive torus scene with linked scientific plots and animation. The original application remains available with `--legacy`.

**Status: experimental research implementation.** Every selected module has executable code. Integration checks are separate from scientific qualification: MHD Taylor–Green, full coupled manufactured-solution convergence, exterior-domain refinement, subgrid total-energy validation, and reference radiation transport qualification remain acceptance gates. A successful run or PINN fit does not establish habitat protection or a self-sustaining dynamo.

## Quick start: Windows desktop

Use Python 3.12 and run commands from the repository root. If your `torch-gpu` environment is already configured, activate it and launch:

```powershell
conda activate torch-gpu
python interface3.py
```

Select **Thermal benchmark**, set **Simulation duration [s]** in **Run setup** at the top of the sidebar, then select **Run simulation** for the quickest computed animation. Geometry and thermal properties remain editable in the same window; press **Recalculate scenario** without restarting. After running, right-click the 3D view to inspect a point. Accepted results remain visible while a job runs and survive failures and cancellation. **Iron** and **Lunar habitat** select the full FEniCS worker; their engineering/material values are illustrative.

The thermal energy chart defaults to **Change (ΔE)**; select **Stored total** for absolute values. Stored energy is recoverable heat relative to the configured discharge temperature; it includes heat present before charging. Probe temperatures use the computed temperature range, with an annotation when the melting point is outside that range.

For a new desktop environment:

```powershell
git clone https://github.com/lubacareer/HeatEquationSimulator.git
cd HeatEquationSimulator
conda create -n torch-gpu python=3.12
conda activate torch-gpu
python -m pip install -r requirements-desktop.txt
python interface3.py
```

The desktop and native thermal benchmark work without PyTorch or WSL. PINN training and the full test suite require PyTorch; install the CPU or CUDA build appropriate for your machine using the [official PyTorch installer](https://pytorch.org/get-started/locally/). Select the same environment as your Python interpreter in your IDE.

| Capability | Additional environment |
| --- | --- |
| Desktop, plots, native thermal benchmark, imposed-coil trajectory diagnostics | Windows desktop dependencies |
| Thermal/coupled PINN training | PyTorch in the Windows environment; CUDA optional |
| Coupled numerical MHD and FE field snapshots | FEniCSx/Gmsh/PETSc in WSL Ubuntu |
| Quantitative water or anatomical dose | Accepted FE field plus the Geant4 WSL worker and datasets |

Check dependencies, reopen saved data, or launch the original interface:

```powershell
python -m magnetic_torus.diagnostics
python interface3.py --load C:\path\to\result.h5
python interface3.py --legacy
```

The tested Windows Python 3.12 package versions are recorded in [requirements-desktop.lock.txt](requirements-desktop.lock.txt), including the CUDA 12.8 PyTorch build. This is a reference package snapshot; [requirements-desktop.txt](requirements-desktop.txt) installs the desktop stack while leaving your PyTorch build under your control. Legacy dependencies and instructions are retained in [docs/LEGACY_README.md](docs/LEGACY_README.md).

## Explore, animate, compare

- Rotate/zoom the PyVista scene or choose a camera preset. Toggle cutaway, phase contour, velocity arrows, integrated flow tracers, magnetic field lines, and a movable cross-section. Hide **Plots** for a larger scene. Opacity and visual detail do not alter the computed solution.
- Right-click the reservoir to choose a probe. The T(E) marker and history update together: temperature rises for E<0, stays at the melting temperature for 0≤E≤LH, and rises again for E>LH. Zero enthalpy marks solid material at its melting point. The chart displays kJ/kg and K; internal quantities use SI.
- FEniCS results also show their labelled regularized computational T(E) curve, and their probe markers follow that curve. The sharp reference plateau remains visible; conserved solver enthalpy is not converted for display.
- Play, step, scrub, loop, or click history plots. The main clock displays accepted thermal/MHD snapshots. Particle replay has its own flight-time clock. Tracers interpolate accepted velocities for display; they are not additional solver states.
- Thermal charts offer probe/cross-section temperature, liquid fraction, and thermal/pumping power. Dynamo charts show magnetic energy and toroidal/poloidal components. Shielding shows paired first-contact outcomes and field-on/off absorbed dose with standard errors. PINN shows training, held-out metrics, and overlays.
- A completed PINN fit can fail its accuracy gates. The PINN view names failed checks, shows actual errors and trained epochs, and provides **Adjust training** to reveal the existing epoch/batch settings. Accepted numerical results remain available; failed PINNs remain labelled and cannot supply radiation-analysis fields.
- Select two saved runs for linked cameras, common color limits, and aligned timelines. **Run difference** is available for matching point geometries; time interpolation is labelled as a display operation. Radiation always references an accepted frozen field snapshot.
- Expand individual plots with their arrow. Advanced parameters and diagnostics remain collapsible. Export PNG scenes, SVG charts, annotated MP4, JSON scenarios, and SI HDF5 results on request. HDF5 embeds accepted exterior-field and weak-operator artifacts for portability.

## Architecture and result history

Qt/PyVista/Matplotlib runs natively on Windows. Each solve, training job, and analysis runs in a separate process; one heavy job is allowed per desktop. The current conda environment can load conflicting Intel OpenMP runtimes when SciPy and PyTorch share a process, so workers and test modules deliberately remain isolated. Do not bypass this with `KMP_DUPLICATE_LIB_OK`.

Immutable `ScenarioConfig`, `RunResult`, `PINNResult`, and `DoseResult` contracts record configuration hashes, source runs, accepted snapshots, seeds, versions, and validation status. Editable inputs are separate from accepted results. Geometry previews update before the first solve; later geometry edits appear as a labelled wireframe over accepted fields until recalculation. Run history defaults to `%LOCALAPPDATA%\MagneticTorus`; override with `MAGNETIC_TORUS_DATA` or `--data-dir`.

New results and analyses are staged privately and published as complete bundles. Cancellation before publication also discards copied observation-only scenarios. Imported field artifacts must be embedded and checksum verified; metadata paths cannot authorize local file reads. Existing results without embedded field artifacts remain viewable, but their coupled analysis requires recalculation or a self-contained export.

### FEniCSx / Gmsh / PETSc in WSL Ubuntu

From the Windows clone directory, with WSL2 and an Ubuntu distribution installed:

```powershell
wsl -d Ubuntu -- bash scripts/setup_wsl_physics.sh
```

WSL inherits the current clone directory. Alternatively, open Ubuntu, change to the same clone under `/mnt/`, and run `bash scripts/setup_wsl_physics.sh`. Setup requires administrative access and installs the Ubuntu FE packages. The tested environment is **Ubuntu 26.04 with DOLFINx 0.10**; other package versions require verification. The script records installed versions, and [env/physics-ubuntu.lock.txt](env/physics-ubuntu.lock.txt) preserves the tested package snapshot; the apt installer does not replay that snapshot automatically. Set `TORUS_WSL_DISTRO` if your distribution is named differently from `Ubuntu`. See [DOLFINx installation guidance](https://github.com/FEniCS/dolfinx/blob/main/README.md).

The full worker uses conservative regularized enthalpy; buoyancy, pumping and Lorentz feedback; no-slip/enthalpy-porosity damping; Joule/viscous/Darcy heating; phase-dependent electrical transport; global vector potential A and conductor electric potential; truncated vacuum exterior; explicit gauges; and fully iterated implicit timesteps with rejection. Only accepted snapshots are published. Imposed currents and circulation are independent; dynamo mode removes its seed and reports actual growth/decay. Recoverable energy is volume weighted relative to the discharge temperature, with efficiency and pumping/imposed-field diagnostics accounted separately.

The mixed residual-VMS closure includes momentum and induction terms plus residual-based eddy transport using [Sondak et al.](https://arxiv.org/html/1412.1055). The A–phi/tetrahedral/exterior adaptation remains experimental; it does not reproduce the published B–psi Taylor–Green qualification. Thermal turbulent Prandtl number 0.85 is an assumption. The native finite-volume thermal benchmark charges at its boundary; the full FE source is volumetric, as recorded in metadata. Iron heat capacities use J/(kg K). Presets are illustrative constant-property models, not qualified temperature-dependent material datasets.

Imposed, induced, and mutual magnetic stored-energy contributions and integrated pumping work are separate diagnostics. The imposed coil is an ideal prescribed current; a coil circuit, resistive supply losses, and power electronics are not modeled. These stored-field diagnostics do not establish net electrical profitability. The lunar preset starts with molten iron; the thermal benchmark starts below melting to show the moving phase front.

### PINNs

The desktop enables training for single-phase native thermal sources and stationary coupled sources. Phase-changing thermal and time-varying coupled MHD training are disabled, with an explanation beside **Train PINN**, because those modes remain unqualified. This does not disable their numerical simulations, plots, or radiation analysis using accepted numerical fields. Existing PINNs remain inspectable; only passed, source-matched models can provide radiation fields. The research backend and full diagnostic cases remain available for investigation and still enforce their original gates.

Per-scenario PyTorch models use nondimensional variables and sparse numerical samples. Full coupled models enforce the same exported differentiable FE weak operator, including closure, conductor/exterior interfaces, gauges, and boundary conditions. Native models enforce the corresponding discrete conservative enthalpy balance and supply only the thermal fields they train. Batches are bounded; checkpoints, cancellation, held-out field errors, conservation defects, and histories are recorded.

Training method version 5 adds a fixed native physics warmup and, when training-only field/equation diagnostics require it, bounded correction stages. Coupled corrections use parameterization version 3 with independent thermofluid/elliptic branches and a frozen base network; the base parameterization remains version 2. The correction budget adds at most `min(2000, 2 × requested steps)` joint Adam steps and `min(50, requested steps / 20 rounded down)` joint L-BFGS steps, retaining data, physics, and boundary losses. Checkpoints record the architecture, requested and actual steps, stages, and source identity. Held-out labels do not select or fit the corrections, and qualification limits are unchanged. This is not qualification of the disabled research modes.

Training can fail its validation gates. Failed models remain inspectable and labelled failed. **PINN-sourced** radiation requires a passed model with the originating scenario and full global A prediction. Numerical FE fields can supply radiation analyses without training a PINN. Agreement with a numerical source does not independently qualify that source's physics.

Nine bounded research pilots tried front/residual sampling, causal weighting, time FiLM, temporal potential correction and label-gradient balancing. A Lunar potential correction reduced that field's normalized error from 0.6405 to 0.00000585, but every whole-model pilot still failed its original accuracy/conservation gates. The disabled desktop modes remain disabled. The separate research command, measured comparisons and limitations are documented in [docs/PINN_EXPERIMENTS.md](docs/PINN_EXPERIMENTS.md).

### Geant4 and ICRP110 targets

```powershell
wsl -d Ubuntu -- bash scripts/setup_geant4.sh
```

Run this from the same Windows clone directory, or run `bash scripts/setup_geant4.sh` from the clone inside Ubuntu. Setup needs administrative access and downloads the toolkit and datasets. It installs Geant4 11.4.3, builds the C++ worker, downloads official physics/ICRP110 data, and replays the integrity-pinned Linux package URLs in [env/geant4-linux-64.explicit.txt](env/geant4-linux-64.explicit.txt). Bootstrap, source, and phantom downloads are checked against [pinned SHA-256 digests](env/geant4-downloads.sha256), including cached archives; extraction uses fresh unprivileged staging. It uses prebuilt conda-forge toolkit binaries by default; `TORUS_GEANT4_PREBUILT=0` selects a source build. Runtime wrapper: `/opt/magnetic-torus/bin/torus-dose-run`; override with `TORUS_GEANT4_WORKER`.

Relativistic Boris trajectories support protons, alpha particles, and heavier ions. Native thermal runs provide an explicitly labelled imposed-coil Biot–Savart diagnostic only. Dose requires the full accepted FE exterior map. Physical transport evaluates cellwise curl(A), including selected PINN A; smoothed nodal B is for display.

Geant4 uses [Shielding](https://geant4.web.cern.ch/documentation/dev/plg_html/PhysicsListGuide/reference_PL/Shielding.html) with EM option 4 and supports a QGSP_BIC_HP cross-check. Geometry includes reservoir, vessel, coils, habitat wall, optional ground, water, and [ICRP110 male/female anatomical targets](https://geant4.web.cern.ch/docs/advanced_examples_doc/example_ICRP110Phantom). Anatomical targets require a large enough habitat; the lunar preset fits. Absorbed dose is Gy per specified incident fluence, with event-wise standard errors, secondary contributions, and explicit precision status. Zero-hit organs do not meet the ≤5% uncertainty target. Effective dose in Sv is outside this release; unnormalized dose rates are not reported.

Spectrum JSON uses `schema_version: 1`, `energy_unit: "MeV/nucleon"`, `angular_distribution: "beam"` or `"isotropic"`, `provenance`, and `bins` with integer `z`, integer `a`, positive `energy_mev_nucleon`, and positive integrated-bin `weight`. Fluence is a separate scenario parameter. Field-on/off transport reuses identical incident primaries.

With ground enabled, beam primaries sample only the exposed part of the incident disk. Isotropic primaries start above the ground and travel downward; their normalization uses the joint sky-and-boundary acceptance relative to unobstructed 4π scalar fluence. Ground shadowing is included, and primary albedo is not imposed. Source metadata records the sampled area and fluence convention.

## Reproduce checks

Run the representative modern-workspace feature suite from the repository root:

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py
```

The default `features` profile tests one fixed scenario per major feature, including real desktop interactions/exports, thermal/MHD solvers, CPU/CUDA PINNs, particles, water/anatomical dose, and playback performance. It retains short regressions and the existing scientific accuracy limits. Both profiles now run one fixed **1,048,576-history** water ensemble and require nonzero dose with measured relative standard error ≤5%; neither old attempts nor paired field-on/off history counts are pooled. The command allows 55 minutes of work plus five minutes for cleanup/reporting, with a one-hour supervisor backstop. Feature PINNs retain 10-minute case limits and water retains 30 minutes. An unfinished case or failed accuracy gate cannot pass. Complete timing of the revised feature profile remains unmeasured. Normal application history stays separate.

The exhaustive multi-hour matrix remains available explicitly:

```powershell
conda run --no-capture-output -n torch-gpu python -B scripts/run_all_tests.py --profile full
```

A representative pass does not qualify all presets, option combinations, female anatomy, or dynamic/phase-changing PINNs. Existing full-suite scientific failures remain documented. A resumed report keeps its recorded profile; resuming an old `full` report still runs the exhaustive profile.

The completed full run at revision `47c87d7` (`.cache/test-suite/20261026T133138.135583Z/`) tested 129 cases in **4 h 30 min 28 s**: **120 passed, seven failed, and two timed out**. Its desktop benchmark passed at **48.13 FPS**; the last 524,288-history water ensemble measured **5.08554%** relative standard error and correctly failed the 5% gate. The remaining eight unsuccessful cases were PINNs. That report has not been rewritten as a pass. Their full-profile deadlines are now one hour each, with source scenarios and accuracy limits unchanged; canonical reruns of those eight PINNs and the larger water ensemble remain pending. See [the targeted resume command](docs/TEST_MATRIX.md#rerun-the-nine-outstanding-canonical-cases).

Live progress and per-case evidence are saved under `.cache/test-suite/<run>/`; open `report.html` or share `report.json` with its case logs. Reports distinguish representative feature coverage from exhaustive matrix coverage. Budget exhaustion returns exit code 124 and retains evidence for resume. See [the coverage and rerun workflow](docs/TEST_MATRIX.md) for selected scenarios, limits, and commands.

For shorter development checks and individual benchmark tools:

```powershell
python -B scripts/run_checks.py
python -B scripts/desktop_smoke.py
python -B scripts/benchmark_desktop.py --output .cache/playback.json
python -B -m magnetic_torus.physics --benchmark-report .cache/physics_validation.json --include-fenics
python -B -m magnetic_torus.pinn_validation --native --device cpu --epochs 1000
```

Run these in the desktop Python environment with PyTorch installed. The FE benchmark additionally needs the WSL worker; desktop smoke needs a graphical session. The test runner isolates modules to avoid OpenMP conflicts. Desktop smoke exercises real Qt/VTK, repeated edits/runs, probes, comparisons, cancellation, failure/stale isolation, separate particle replay, an explicitly synthetic PINN display fixture, and decoding every exported MP4 frame. Actual PINN training and qualification run in separate cases: selected representatives in `features`, and the larger matrix in `full`. Reports record scene size, measured FPS, completed checks, and outstanding qualification gates separately. The PINN validation CLI returns exit code **3** for a completed fit that fails validation.

See [docs/VALIDATION.md](docs/VALIDATION.md) for actual benchmark values, the measured 31.8 FPS reference scene, real water/anatomical transport runs, failed coupled PINN fits, and open release gates. The Ubuntu FE package lock and integrity-pinned Geant4 environment are in `env/`.

For component ownership, worker data flow, and before/after playback measurements, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Generated run histories, worker artifacts, datasets, checkpoints, and `.cache/` benchmark reports are kept out of Git. The paths in the validation record describe local evidence; rerun the commands above to generate reports on your machine. The recorded dynamic coupled PINN fits fail their accuracy gates, the small dose samples miss the 5% precision target, and the remaining MHD qualification benchmarks are still open.

The original author and documentation are retained in [docs/LEGACY_README.md](docs/LEGACY_README.md).
