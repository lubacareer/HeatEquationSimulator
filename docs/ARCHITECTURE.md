# Code ownership and data flow

`python interface3.py` starts the modern desktop. `--legacy` retains the original
Tk application. The original scientific import surfaces (`physics`, `radiation`,
and `pinn`) remain available; their implementations have separate owners.

## Desktop

The main window composes widgets, connects signals, and coordinates jobs. It does
not own chart construction, scenario parsing, playback clocks, or encoders.

| Component | Responsibility |
| --- | --- |
| `ScenarioEditor` | Editable fields, presets, parsing, validation, and debounced draft notifications |
| `WorkspaceSession` | Accepted results and their source associations, comparison fields, physical time, probes, and selection revisions |
| `ChartPanel` | Plot construction when selections change; lightweight artist updates during playback |
| `PlaybackController` | Separate thermal/MHD and particle-flight clocks, scrubbing, stepping, and looping |
| `ExportController` | Requested plot/data/image exports and incremental MP4 encoding |
| `SceneWidget` | Geometry, interpolation, accepted-field rendering, and snapshot-bound analysis layers |

`WorkspaceSession`, `DisplaySelection`, and `SceneOptions` have no Qt/VTK
dependency. A session resolves the selected fields once; charts and the scene
consume the same selection. Renderer changes go through its public methods,
rather than edits to actors or caches from the main window.

The scene coalesces render requests with a single-shot Qt timer. Exporters call
`flush_render()` before reading pixels. Surface and section interpolation
operators are reused while geometry is unchanged. Difference bounds and flow
availability are cached for the selected immutable fields and invalidated when
their source changes. Color scales remain stable across physical time.

## Scientific implementations

`numerics` owns constitutive thermal laws and native transport, coil fields, the
FE process protocol, differentiable weak residuals, and benchmark reporting.
`transport` owns source sampling, immutable frozen fields, relativistic
trajectories, and the Geant4 protocol. Shared WSL configuration is limited to
path conversion and distribution selection; FE stdout and Geant4 progress-file
protocols retain their separate cancellation/error handling.

`learning` separates field/operator contracts, training preparation and losses,
optimization and prediction, and held-out validation/conservation checks.
Splitting these stages does not change equations, sampling order, precision,
loss weights, checkpoints, or qualification thresholds.

Scientific modules import no desktop components. Torch and SciPy imports remain
lazy where required. Native numerical jobs and PyTorch jobs run in separate
processes because this Windows environment contains conflicting OpenMP
runtimes. Do not bypass that isolation with `KMP_DUPLICATE_LIB_OK`.

## Jobs and publication

Editors produce configuration snapshots; accepted results never change when a
draft changes. The controller launches one isolated worker at a time. The worker
prepares source identity, executes an action, serializes its result, then commits
one `JobPublication`. Its shared lifecycle handles cancellation and failures.
Only successful publications enter run history.

The session revision prevents late completions from replacing a newly selected
run. PINNs, trajectories, and dose retain source-run/configuration/snapshot
checks. Field artifact bytes are captured explicitly; metadata path strings do
not authorize arbitrary filesystem reads. Result schemas and checkpoint formats
are unchanged by this refactor.

## Verification and performance

Run `python -B scripts/run_checks.py` for isolated regression modules. Set
`TORUS_TEST_DESKTOP=1` for the real Qt/VTK tests, then run the desktop smoke
workflow for repeated runs, analysis layers, cancellation, comparisons, and MP4
encoding/decoding. Optional FE/Geant4 tests require their installed workers.

Measure full playback, including Qt event processing, in a graphical session:

```powershell
python -B scripts/benchmark_desktop.py --output .cache/playback-before.json
python -B scripts/benchmark_desktop.py --output .cache/playback-after.json --baseline .cache/playback-before.json
```

Use the same machine, scene, window size, and background workload for both runs.
Defaults are the resolution-8 thermal reference (2048 points), a 1282×780
window, five repetitions of 60 frames, Medium detail, opacity 100%, and optional
overlays off. A slowdown greater than 10% returns exit code 3. Generated results
stay outside Git; exports remain user initiated.
