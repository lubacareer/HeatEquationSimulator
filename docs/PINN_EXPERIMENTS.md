# Bounded PINN experiments: 2026-10-26

The proposed conditioning changes were tried against saved accepted numerical
sources. **All nine completed pilots still fail the original whole-model
qualification.** A temporal electric-potential correction fixes one Lunar field
error, but does not fix the other fields or conservation. No disabled desktop
training mode was re-enabled. The normal trainer and qualification limits remain
unchanged by this experiment.

## What the proposal correctly identified, and what differs in this code

- Native thermal already predicts specific enthalpy directly and derives
  temperature and liquid fraction with the sharp enthalpy law. It does not
  invert a temperature prediction through the phase plateau. Its remaining
  failure is spatial/temporal enthalpy and local balance accuracy. A small global
  energy defect can coexist with large local errors because positive and
  negative cell errors can cancel in the integral.
- Temperature, enthalpy and liquid-fraction errors have different normalizations.
  Their ratio does not establish error amplification. A bounded dimensional
  slope of T(H) does not guarantee a well-conditioned normalized PINN loss.
- Pressure already has a hard finite-element point anchor. Subtracting its
  spatial mean would change that reference gauge. The weak electric-potential
  mass gauge also does not make the saved reference mean exactly zero. The
  current equation includes `-dA/dt - grad(phi) + u cross B`; dropping `-dA/dt`
  changes the coupled equations. Absence of an explicit pressure/potential time
  derivative is not evidence that these fields are stationary.
- Residual-adaptive sampling and causal training are plausible research
  directions, not guaranteed fixes. The relevant primary sources are the
  [RAD/RAR-D study](https://arxiv.org/abs/2207.10289) and the
  [causal PINN study](https://arxiv.org/abs/2203.07404). The implementations below
  are limited adaptations to this discrete conservative/FE worker, not exact
  reproductions of those papers. No universal speedup was demonstrated.

A read-only audit of the original checkpoints used training labels to estimate
the part of each field error explained by a per-snapshot constant offset:

| Original checkpoint / field | Fraction of training squared error explained | Normalized error remaining after diagnostic offset |
| --- | ---: | ---: |
| Iron pressure | 5.42% | 0.2006 |
| Iron potential | 11.35% | 0.2273 |
| Lunar pressure | 40.09% | 0.4451 |
| Lunar potential | 99.999999996% | 0.00000401 |

These diagnostic offsets were not applied to accepted labels or used to change
the final validation gauge. Evidence is in
`.cache/pinn-proposal-gauge-audit/report.json`. They motivated a trainable temporal
potential branch for Lunar, rather than a blanket mean-zero projection.

## Research methods and isolation

`scripts/pinn_research.py` explicitly selects a saved accepted source and writes
a new attributed source copy, checkpoint, predictions and report into an empty
private directory. It does not modify the normal application history or attach
old weights to another source implicitly. Warm starts require an explicit
source-matched checkpoint, supported architecture and matching training-only
normalization. The normal workspace does not select any of these strategies.

The experiment supports:

- **Front/residual sampling:** a 50% uniform / 50% adaptive mixture using
  predicted mushy fraction and absolute conservative residual, a separate seeded
  RNG, bounded proposals, and importance weights preserving the uniform-node
  objective in expectation. Reference liquid-fraction labels do not select
  collocation points. The thermal residual helper preserves the default
  trainer's equations, random draws and gradients.
- **Causal weighting:** detached, cached training-loss prefixes weight accepted
  intervals with `exp(-epsilon * prefix)`, with a fixed positive floor of 0.05.
  An initial pass evaluates all accepted intervals, then sampled intervals update
  their cache. This stochastic cached-prefix variant incurs extra FE evaluations
  and is not a full temporal-batch implementation of the paper.
- **Time FiLM:** a separate temporal network modulates spatial features in a
  fresh coupled network. Zero-initialized modulation/output preserves the fresh
  particular solution; the existing outer masks retain initial/boundary
  constraints. No tensor decomposition or guaranteed speedup is claimed.
- **Temporal potential branch:** a frozen clone of an explicitly selected
  legacy coupled checkpoint plus zero-initialized phi-output biases on accepted
  time knots, interpolated linearly. All other field outputs and hard outer
  constraints are preserved. The branch learns through the original joint loss;
  it does not subtract a label mean. Values between knots are interpolation,
  not demonstrated temporal generalization.
- **Inverse label-gradient balancing:** a simple detached inverse-gradient-norm
  weighting, clamped to relative weights 0.1 through 10, refreshed every 50
  steps. It balances training-label tasks only; physics/boundary coefficients
  remain one. This is not a full GradNorm or NTK algorithm.
- **Temporal pressure sampling:** the `temporal-pressure` strategy weights
  pressure-label sampling using differences at nodes present in the training
  split at both accepted times. Relative weights are capped at 3:1. It does not
  use importance correction, so the sampled training objective changes; the
  final source oracle and all qualification gates remain unchanged.
- **Pressure Fourier 16:** the `pressure-fourier` architecture adds a
  zero-initialized, pressure-only sinusoidal head to an explicitly selected,
  source-bound core or correction checkpoint. It preserves every output at
  creation; subsequent training still uses the joint data, physics, and
  boundary losses.

The `method5` and `pressure-fourier` research architectures require an explicit
checkpoint bound to the selected source. They run in an owned child with a hard
`--seconds` deadline and separate worker/supervisor reports. The pressure
follow-up's selection, reference comparison, and unsuccessful independent screen
are recorded in [VALIDATION.md](VALIDATION.md#pressure-and-cuda-follow-up-2026-10-27).

`scripts/pinn_refine_source.py refine --source SOURCE.h5 --output NEW_DIRECTORY`
creates the six-step, seed-123 driven FE follow-up and checks it in a separate
process. `scripts/pinn_pressure_diagnostics.py --source SOURCE.h5 --pinn PINN.h5
--refined REFINED.h5 --output NEW_DIRECTORY` writes exploratory maps and
statistics; omit `--refined` to inspect only the original fit. Both commands
require a new empty output directory and copy inputs before extracting artifacts.

An auxiliary strong pressure-Poisson loss was not added: its boundary conditions
and turbulence terms would have to be derived and checked against the exported
discrete weak operator first. The trials retain that original coupled operator
rather than assuming a second differential residual is equivalent.

Comparisons use the same conditional-time free-DOF training-label policy and
separate seeded time/label RNGs. Coupled research fitting derives weak residual
amplitude scales from training labels only. The original FE residual oracle and
its original normalization are restored before the final raw objective and
qualification. Held-out labels are observed for reporting, never used to select
sampling, branches, schedules, objectives or optimized weights. Neural forwarding
remains bounded, in float64, with cancellation and the eight-closure refinement
safeguard. Native pilots retain fixed data warmup/physics ramp phases.

The final limits are still held-out/physics/boundary loss at most 0.01,
normalized field RMSE and residual RMS at most 0.1, and native relative energy
defect at most 0.05, together with source-identity and required-field checks.
Completing training cannot bypass any gate.

## Measured results

All pilots used seed 42 and the accepted sources from the completed full report
`.cache/test-suite/20261026T133138.135583Z/`. Native used the phase-changing CPU
source; Iron and Lunar used their preset sources. Native fresh fits used
800 Adam + 20 joint unweighted L-BFGS steps. Fresh coupled fits used 250 Adam +
10 L-BFGS steps. These bounded comparisons do not reproduce the longer canonical
budgets. The temporal branch used the explicitly selected old Lunar epoch-900
checkpoint and ten added joint L-BFGS steps.

| Pilot | Training seconds | Enthalpy normalized RMSE | Other selected errors | Whole-model outcome |
| --- | ---: | ---: | --- | --- |
| Native uniform | 78.89 | 0.8051 | T 0.1632; liquid fraction 0.3586; enthalpy residual 2.9520 | FAIL |
| Native front/residual | 94.75 | 0.7842 | T 0.1631; liquid fraction 0.3499; enthalpy residual 2.9333 | FAIL |
| Native front + causal | 112.84 | 0.7511 | T 0.1627; liquid fraction 0.3336; enthalpy residual 2.6801 | FAIL |
| Lunar temporal potential branch | 55.08 | 0.3291 | phi **0.00000585**; pressure 0.5664; velocity 0.2429 | FAIL |
| Lunar uniform | 93.37 | 0.3319 | pressure 0.6072; phi 0.6405 | FAIL |
| Lunar causal | 104.89 | 0.3312 | pressure 0.5857; phi 0.6405 | FAIL |
| Lunar FiLM + causal | 102.34 | 0.3268 | pressure 0.4622; phi 0.6405 | FAIL |
| Iron uniform | 63.62 | 0.001008 | pressure 0.3808; phi 0.4213 | FAIL |
| Iron inverse label-gradient | 66.70 | 0.003843 | pressure 0.5450; phi 0.6792 | FAIL |

Native front + causal reduces enthalpy error by approximately 6.7% relative to
this uniform research baseline, but remains far above the field and residual
limits. FiLM improves Lunar pressure error compared with the uniform research
baseline, while its continuity/charge residuals worsen relative to that baseline;
it does not qualify the model. Simple
inverse label-gradient balancing worsens Iron pressure and potential.

The Lunar temporal branch reduces potential normalized RMSE from approximately
0.64046 to **0.00000584837** without changing the FE reference gauge. This is a
real component-level improvement. A nearly spatially constant correction does
not repair electric gradients/current or coupled conservation. The candidate
still has continuity RMS **0.8604**, charge RMS **1.5266**, physics loss
**0.38385**, and failed thermal/flow/pressure errors. It remains failed and
cannot supply PINN radiation fields. Other fields remain frozen during that
experiment.

Per-pilot strict JSON reports, private copied results and checkpoints are
under `.cache/pinn-proposal-research/<pilot>/`. `comparison.json` records all nine
reports, and `comparison.png` / `comparison.svg` visualize the comparisons. Source
and selected original-checkpoint SHA-256 hashes were unchanged. All pilots
except the first native-uniform pilot also record implementation hashes before
and after execution. Completion/cancellation reporting and provenance checks
were strengthened between pilots; the declared optimization budgets/equations
were unchanged. These are exploratory single-seed comparisons, not a claim of
statistical reproducibility or qualification of the full scenario matrix.

Focused verification passed **126 tests with no skips** across the six new
research regression modules, existing conditioning/refinement/batching/
correction tests, the 16-test PINN module (including CUDA), and import boundaries.
Modules ran in separate Python processes. Checks include default residual and
gradient equivalence, held-out-independent optimized weights, bounded caches,
known-architecture/source restoration, identical loaded/hash checkpoint bytes,
actual step counts and interrupted-result rejection. Focused Ruff and whitespace
checks passed. The CLI now fingerprints the entire Python package; the earlier
pilots' implementation manifests cover the files listed in their reports.

## Run another bounded experiment

From the repository root, select an existing accepted result explicitly. For
example, in PowerShell:

```powershell
$source = (Get-ChildItem ".cache/test-suite/20261026T133138.135583Z/cases/pinn.thermal.cpu/attempt-0001/store/runs/*/result.h5" | Select-Object -First 1).FullName
conda run --no-capture-output -n torch-gpu python -B scripts/pinn_research.py --source "$source" --output ".cache/pinn-experiment-new" --strategy front-causal --adam-steps 800 --lbfgs-steps 20 --seconds 180 --device cpu
```

Inspect the selected path before running; a different accepted source is also
supported. Use a new empty output directory each time. For a coupled fresh fit,
select a coupled FE result and use `--strategy causal --architecture film`.
For the temporal branch, select the exact originating source and add
`--architecture temporal-phi --checkpoint <its-checkpoint.pt>`; do not use a
checkpoint from a different source. `--balance-labels` enables the independent
label-gradient experiment.

`--seconds` is a cooperative training deadline, including preparation. Final
diagnostics/checkpoint writing take additional time. The separate representative
suite retains its existing one-hour supervisor backstop. The research command
returns 0 only for a completed qualified candidate, 3 for completed scientific
failure, 124 for training timeout, 130 for interruption, and 1 for execution or
provenance errors. Incomplete training cannot pass even if its final diagnostics
happen to meet some limits.

The full canonical cases and measured water precision rerun remain pending.
The water ensemble is already fixed at 1,048,576 histories in both suite profiles;
this experiment did not rerun it or claim it can never fail the measured 5% gate.
Use the existing [test commands and resume workflow](TEST_MATRIX.md). The
one-hour default and disabled desktop scope remain unchanged.
