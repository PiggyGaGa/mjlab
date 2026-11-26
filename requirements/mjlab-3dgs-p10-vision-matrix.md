# P10 initial GPU vision-pipeline calibration

Prepared and audited: 2026-11-25. Status: all four pipeline checks passed.

## Purpose and reuse

After completing fixed-asset collision comparisons and CPU preprocessing-cache
validation, first calibrate a small visual-training workload on the user's RTX 3060.
Reuse `tests/run_g1_3dgs_vision_smoke.py` unchanged rather than copying PPO or
modifying the generic trainer, sensor, play/demo, P7 protocol, or P8 assets.

The fixed input is the existing apartment manifest and its original collision.
No automatic collision is substituted mid-run. Use seed 42, initial base-pose
randomization disabled, 12 rollout steps and one PPO update per case. The runner
also validates finite observations/losses, camera updates and checkpoint roundtrip.
This is a tiny pipeline check, not new policy training to convergence or a
visual-control task-benefit experiment.

| Case | Worlds | Capture resolution | Rollout environment transitions |
|---|---:|---|---:|
| 1w-64x48 | 1 | 64x48 | 12 |
| 1w-128x96 | 1 | 128x96 | 12 |
| 4w-64x48 | 4 | 64x48 | 48 |
| 4w-128x96 | 4 | 128x96 | 48 |

The primary measured loops request four small PPO updates and 120 rollout
transitions. Checkpoint validation additionally performs one resumed rollout and
update per case, bringing these two phases to eight updates and 240 transitions;
initialization, inference checks and reset checks also run.
Each case is a separate process and executes serially. Failures stop the script,
while each completed/failed runner report and its log remain available for audit.

## Desktop command

```bash
cd /home/liana/SSD/Ubuntu/robot/mjlab
bash artifacts/p10-vision-matrix/run-desktop.sh
```

Outputs: `artifacts/p10-vision-matrix/<case>/report.json`, snapshots, checkpoints
and `run.log`. The script does not modify drivers or synchronize/install packages;
it uses the current environment. A missing CUDA device stops before the matrix.
The newly uploaded MuJoCo wheel is not silently installed by this experiment.

## Measurement limits

- The existing smoke runner captures **RGB and depth** even for its depth policy;
  these timings are not a depth-only production workload.
- Rollout timing includes validation, tensor synchronization and snapshot writing.
  Do not interpret it as pure renderer timing or a stable training SPS benchmark.
- Torch peak allocation excludes some allocations owned by MuJoCo/Warp/OpenGL;
  it is not complete device VRAM usage.
- Kernel compilation/cache state affects initialization, especially the first case.
- Disabled initial pose randomization is not a frozen robot throughout rollout.
- This does not cover Viser transport, no-3DGS controls, rendering-only timing,
  large world/splat budgets or statistically repeated measurements.

The formal P10 benchmark/matrix checkboxes stay open. Actual results must be
checked before selecting a larger workload or claiming throughput benefits.

## Audited desktop results

The user completed all four cases on 2026-11-25, 20:19–20:21 Asia/Shanghai.
Logs identify an RTX 3060 (12 GiB), Warp 1.18.0, MuJoCo 3.16.0+3dgs and
rsl-rl-lib 5.5.1. The package version alone does not identify the installed wheel
digest. No wheel installation or driver changes were performed during this audit.

| Case | Status | Init (s) | Rollout (s) | Update (s) | Rollout env steps/s | Torch peak (MiB) |
|---|---|---:|---:|---:|---:|---:|
| 1w-64x48 | passed | 32.273 | 0.355 | 0.519 | 33.84 | 34.51 |
| 1w-128x96 | passed | 2.731 | 0.487 | 0.381 | 24.63 | 37.61 |
| 4w-64x48 | passed | 3.291 | 0.617 | 0.383 | 77.74 | 37.86 |
| 4w-128x96 | passed | 2.713 | 1.411 | 0.427 | 34.02 | 56.78 |

All reports have finite primary/resumed losses, positive CNN parameter changes,
changing depth frames, and positive action changes when camera input is zeroed.
Checkpoint actor/critic/optimizer roundtrip and resumed updates passed in all cases.
Both four-world cases passed the other-world state/depth partial-reset check.
Camera sensitivity establishes structural use of input, not a learned task benefit.

Audit recomputed each checkpoint SHA256 against its report and matched the current
manifest SHA256 in all reports:
`f2fc9518db66ff61fb5f38583c4abe07778ea64a63f05f20eb250e9f7c9c3d76`.
Report SHA256 values, in table order:

```text
aa96b146573d3d097f60d9e927805479606e42e81deb21d06c039668c8669e07
0f8052c858d2bdd56dbcdb7fa678509b764a24671c09897da6c681dfae79c6fb
5390ad74a868e1b4a4af4b327355ccbd931c47b1bcd995d96eb094548e327507
da7f49374ad57fa19e6f3db4065ca39c68ae8fb56d3411534a14c5aa569ef969
```

The first log includes kernel compilation, explaining why its initialization is
not comparable with the later cached runs. Logs also warn that capsule/cylinder
CCD pairs support at most one contact; this is not a failed smoke assertion.
No traceback was found in the four logs.

The higher resolution increased measured rollout time in both world counts, while
actor-only inference stayed around 0.34–0.37 ms per batch. This supports separating
rendering/environment stepping from policy inference in the next benchmark; it
does not isolate the renderer as the cause. Next, use warmed, repeated measurements
without snapshot I/O and record physics/render/policy/update costs separately.
The formal performance matrix and full repository test acceptance remain open.

## Repeated throughput experiment (four desktop cases audited)

`scripts/benchmarks/measure_3dgs_throughput.py` reuses the upstream
`measure_throughput.py` physics and full-environment timing helpers unchanged.
The script adds stationary `sim.sense()` batches, fixed-observation actor batches,
and PPO rollout/update batches without snapshots, per-step assertions or disk I/O
inside the timed loops. CUDA synchronization surrounds each batch. No runtime,
play/demo or shared trainer modifications are required.

Run on the desktop:

```bash
bash artifacts/p10-vision-matrix/run-throughput.sh
```

Each invocation creates a fresh `throughput-XXXXXXXX` directory so previous reports
are preserved. The four configurations match the smoke matrix. Each process
discards one full repetition, then measures three repetitions of 32-step batches
with 16 warmup steps. Raw samples and medians, source/manifest hashes, package
versions, PPO settings and GPU name are saved; exceptions produce failed reports
and stop the shell script. The script does not synchronize packages or change
drivers. The completed desktop run is audited below; local checks alone cannot
establish performance.

These are **independent workload measurements**, not an additive decomposition:
raw physics advances simulation without action managers; full-env uses zero
actions and includes sensing/reset/observations; stationary sensing includes BVH,
rendering and camera conversion at a fixed state; actor inference reuses fixed
observations. PPO rollout includes action/critic inference, stepping, storage and
returns. Policy parameters evolve between repetitions; this is a short synthetic
training workload, not a frozen-policy or convergence comparison. The first
discarded update also warms optimizer allocation. State resets and warmup are
outside measured batches; same seed does not imply identical stochastic workload
across resolutions. Torch memory is not total VRAM. No Viser/no-3DGS control or
pure render-kernel timing is supplied yet, so formal P10 remains open.

Local validation on 2026-11-25: `make check` passed (Ruff, ty, pyright),
`make docs` succeeded, CLI `--help` and shell syntax checks passed. Three focused
existing task/runner/default-off configuration regressions passed (12 deselected).
Checks used the current environment with dependency synchronization disabled.
No GPU execution or numerical performance claim is implied by these checks.

The first desktop throughput attempt failed at an explicit reset: the preceding
inference-mode rollout replaced contact-history buffers with inference tensors,
which cannot be mutated outside inference mode. The benchmark now uses `no_grad`
for rollout and fixed-observation inference; PPO updates retain autograd. Shared
environment/sensor code is unchanged. A CPU regression exercises two rollout/reset
cycles with persistent rolled history and confirms that backpropagation remains
available after rollout. The subsequent desktop rerun passed all four cases;
failed timing reports are not accepted as performance evidence.

Fix validation: `tests/test_3dgs_throughput.py` passed on CPU (one test),
`make check` passed with zero type errors/warnings, and `git diff --check` passed.
New reports explicitly record `rollout_and_actor_context: torch.no_grad`.

### Audited repeated measurements

Accepted directory: `artifacts/p10-vision-matrix/throughput-DA7sGc22/`.
The previous `throughput-Q3PbfhFY/` contains only the failed first case and is
excluded. The accepted four reports all record `passed` and `torch.no_grad`,
32 steps, 16 warmup steps and three retained samples numbered 1/2/3. Logs have no
traceback. GPU: RTX 3060; package versions: MuJoCo 3.16.0+3dgs, Warp 1.18.0,
Torch 2.14.0. Versions do not establish the installed wheel's SHA256.

Audit recomputed every median, checked positive finite timings and finite losses,
and verified the manifest and PLY hashes against the current source files. The
manifest hash is the same as the smoke run; the PLY hash is
`f82a890d3fc9565c62fde7b95618c4d6229156dbd3f33b4dd0235c785d7ba5c8`.

All entries below are medians. Physics SPS uses environment-equivalent units
(physics substeps are divided by decimation), multiplied by world count. Full-env
SPS also sums world transitions. Sense/actor milliseconds are per **batch** call
across all worlds, calculated as batch seconds / 32; update seconds are per PPO
update. These units must not be mixed with per-world latency.

| Case | Physics env-equivalent SPS | Full-env SPS | Sense ms/batch | Actor ms/batch | PPO rollout s | PPO update s | Torch peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1w-64x48 | 527.68 | 55.30 | 12.390 | 0.366 | 0.656 | 0.126 | 27.46 |
| 1w-128x96 | 534.03 | 33.35 | 23.765 | 0.368 | 1.098 | 0.128 | 37.98 |
| 4w-64x48 | 1892.13 | 95.79 | 28.587 | 0.368 | 1.362 | 0.128 | 38.31 |
| 4w-128x96 | 1893.01 | 34.18 | 83.164 | 0.366 | 3.540 | 0.133 | 79.99 |

Full-env SPS ranges across the three samples are respectively 52.48–55.46,
33.30–33.51, 95.74–95.85 and 33.25–34.22. These are within-process repeats of
short batches, not independent trials or confidence intervals.

Accepted report SHA256 values in table order:

```text
c7663b355e85ebfa871516380f291ec83e0cb2fd6f9ee9515b094bc9218b43f6
fbe5f6029cb80f7111d3c3089d5583412b3be78c629050e0026adef9a131f893
271507f54164c1487e683f7db1931c02b8c25c866b5143bbc06c137c28d3f557
2db0885edaaf7db77e18fd6d3ac906a2c6d46acb3f622e8372cca7af38b36aa0
```

Fixed-observation actor latency is much smaller than stationary sensing latency.
Increasing resolution increases sensing time while the raw physics baseline
remains similar at fixed world count. This points to the sensing pipeline as a
priority for further profiling; it does not isolate pure Gaussian rendering,
nor establish a causal percentage of end-to-end runtime. Higher resolution also
removes most of the aggregate full-env throughput gain from four worlds in this
small workload. Use 64x48 as the initial low-cost reference, not as a proven
task-quality optimum.

This repeated apartment submatrix is complete. The full P10 remains open: next
prepare matched runs with the camera retained and Gaussian assets disabled,
preserving the same collision and settings, to estimate incremental Gaussian
cost. Pure renderer attribution, other splat/world budgets, Viser and full test
acceptance remain separate work.

### Matched Gaussian-on / Gaussian-off experiment (eight cases audited)

The same benchmark now accepts `--without-gaussians` (default false). It first
applies the manifest normally, records source and collision/dependency SHA256
values, then clears only `cfg.scene.gaussian_splats` for the control. Camera
configuration, collision loading, pose/seed, task and PPO settings are retained.
Reports distinguish input assets from effective loaded assets and explicitly
record Gaussian enablement and RGB/depth capture. No shared runtime changes.

```bash
bash artifacts/p10-vision-matrix/run-camera-comparison.sh
```

The launcher creates a fresh `camera-comparison-XXXXXXXX` directory and runs eight
separate processes serially: four existing configurations, each with and without
Gaussians. Pair order alternates across configurations. Each process discards its
first complete repetition and retains three 32-step repeats after warmup. Both
arms are rerun rather than comparing new controls only against older measurements.
CUDA synchronization, `no_grad` rollout and normal PPO-update gradients are reused.
No dependency synchronization or driver changes occur. Failed cases stop execution
and retain their reports/logs; original assets and prior results are preserved.

This control retains geometry-based camera rendering; it is not a no-camera or
physics-only control. Removing Gaussians changes the observations, so stochastic
actions, losses, contacts and resets can diverge despite equal initial seeds.
Stationary sensing at the reset state is the primary incremental-cost comparison;
PPO/end-to-end changes are workload-specific and do not establish task benefit or
an exact Gaussian-render-kernel cost. Pair order alternation across different
configurations is not an independent repeated crossover trial. Full P10 remains
open until actual results and remaining benchmark coverage are audited.

Local validation: both CPU benchmark regressions passed, including construction
of the two effective configurations before GPU allocation. The control test
checks retained collision path/hash, camera capture and PPO settings, and verifies
that only the control omits effective Gaussian assets. `make check` passed with
zero type errors/warnings; `make docs` succeeded with one warning. Shell syntax
and `git diff --check` passed. These do not substitute for the eight desktop runs.

#### Desktop evidence

Accepted directory:
`artifacts/p10-vision-matrix/camera-comparison-5zGTMuoE/`.
All eight reports passed. Audit verified three retained samples numbered 1/2/3,
recomputed all medians, checked positive finite timings/finite losses, and found
no traceback in the eight logs. Pairwise equality checks passed for GPU, versions,
manifest/source/collision hashes, input assets, camera capture, PPO configuration
and rollout context. Effective Gaussian enablement differs as requested. Current
source files match all hashes. The collision SHA256 in every report is
`5f873d4fdab71c8e4e377eeb5258d77549f985315e1aa54c6754ae13094441a1`.

GPU and package versions match the previous repeated experiment. Each value below
is a median of three within-process batches, not an independent-trial estimate.
Sense latency covers all worlds in one batch call, not one world. Increment is
the difference of the two medians; no samplewise pairing is assumed.

| Case | Sense on ms/batch | Sense off ms/batch | Increment ms/batch | Full-env SPS on | Full-env SPS off |
|---|---:|---:|---:|---:|---:|
| 1w-64x48 | 12.296 | 0.026 | 12.271 | 55.39 | 131.95 |
| 1w-128x96 | 23.544 | 0.034 | 23.511 | 33.25 | 133.13 |
| 4w-64x48 | 29.617 | 0.038 | 29.579 | 92.12 | 518.79 |
| 4w-128x96 | 89.162 | 0.069 | 89.094 | 32.03 | 503.55 |

Across three samples, sense-on ranges are 12.295–12.364, 23.438–23.866,
28.557–29.656 and 87.675–91.058 ms/batch respectively. Actor medians across
the eight cases range from 0.360 to 0.460 ms/batch, while PPO update medians
range from 0.127 to 0.147 s/update. Raw physics env-equivalent SPS remains of
similar magnitude within pairs (518/530, 525/528, 1883/1862, 1801/1844).

Torch peak allocation is identical within each pair (27.46/37.98/38.31/79.99
MiB by configuration). This does **not** mean Gaussian GPU memory costs nothing:
the metric omits memory owned by Warp and other renderer components.

Report SHA256 values in table order, on then off for each row:

```text
3664fcf8beb3695bdd9b36232447aa3982151d5917d953b76e90a2d8b814e2a1
878c96d5573346a8507b8ed5c4343cb0475340ab021fd1e390d48cba8e7f961e
5bb6a267bfef1b08c890bdd20e0b122a68ed490186254dd271d942f08e7f77d4
da28a6706d91bd1d4a2ed74386d389861a8f0da89027c419c19c9a78f48ab360
da695a5744e8c93d76294b2f0a3b2e332658775d33b2e7ad2c829c5532711eb9
2c8a571acbf599c0c123d2aaccab1342016c09a5955a66f2c15c0a78ec9e05c5
b06f907a0988d2076e25bc27284396d8a7b4fa3e5137f4e68803810a10c6193f
00d2cad53107f821860f8aed3f976d53a8ebf4c9d1f2e25eb5e3e50e004dd613
```

This subexperiment establishes a large measured incremental cost of enabling
Gaussian assets in the sensing pipeline for this apartment workload. It does not
separate Gaussian BVH/refit/render/conversion kernel costs, or imply identical
policy trajectories in the two arms. No training-quality comparison was made.
The on-arm numbers are of similar magnitude to the prior repeated experiment,
with visible run-to-run variation, especially at four worlds and 128x96.

The matched-control subexperiment is complete. Next investigate Gaussian-count
budgets at fixed 64x48 resolution, documenting the effect on visual coverage as
well as timing. Broader budgets, Viser, pure-kernel attribution and complete
repository test acceptance remain outside this completed subexperiment; P10
overall remains in progress.

### Gaussian-count budget experiment (eight desktop cases audited)

The benchmark accepts `--max-splats N`, mutually exclusive with
`--without-gaussians`. The default remains uncapped. It uses the existing
`GaussianSplatAssetCfg.max_splats` option via dataclass replacement, with no new
sampling algorithm: after existing filtering, the loader stably ranks descending
opacity and retains up to N points per asset. This can leave spatial holes; it is
not spatially uniform sampling, geometry simplification or a quality guarantee.
No source PLY, collision or shared runtime is modified.

```bash
bash artifacts/p10-vision-matrix/run-splat-budgets.sh
```

The launcher creates a fresh `splat-budgets-XXXXXXXX` directory. Eight processes
cover 1/4 worlds, fixed 64x48 RGB/depth, and caps 50,000/100,000/200,000/full.
The current full apartment asset contains 374,410 Gaussians. Each process retains
three 32-step repeats after discarding a full first repetition, reusing the
previous warmup protocol. Budget order reverses for four worlds; this is not
an independent repeated crossover trial. Full references are rerun in this batch.

Effective import settings and the actual normalized splat count shared across
worlds are recorded alongside original asset/source/collision provenance. The
count is not multiplied by world count and is not GPU memory consumption.
`--save-capture` explicitly enables an untimed seed-42 default-reset capture before
the warmup/discarded repetition. `<case>.capture/reset.npz` contains raw RGB,
depth and qpos for all worlds; PNG RGB/depth previews use a fixed 5 m depth cutoff.
The report records positive-finite depth fractions by world. Capture I/O is never
inside the measured loops. Without this flag the benchmark saves no captures.

Compare each capped capture with the full reference at equal qpos before reporting
pixel differences. Positive-finite depth is an availability measure, not semantic
coverage or correctness. Geometry can still supply depth where Gaussian content
is absent. A single reset view per world cannot establish whole-scene coverage or
task quality. Images and timing must be assessed together before choosing a cap;
no automatic budget recommendation or change to task defaults is made yet.

CPU configuration regression covers capped/default/disabled variants, verifying
retained collision, camera, PPO and original asset provenance. GPU execution and
visual comparison were subsequently audited; P10 remains in progress.

Local checks: the two CPU benchmark regressions and the existing loader
filter/transform/cap test passed (three tests total). `make check`, shell syntax
and `git diff --check` passed. `make docs` succeeded with one warning because the
Python intersphinx inventory could not be fetched through the restricted proxy.
GPU capture output and budget timings were subsequently validated in the desktop
audit described below.

Accepted desktop run: `splat-budgets-qg3ZlRyj`, eight cases passed. Counts,
source/collision hashes, medians and equal-qpos comparisons were checked. RGB/depth
comparison figures show severe visual loss at 50k/100k; 200k resembles the full
reset views more closely, with remaining depth holes/errors. At 200k, stationary
sense is 4.605/11.045 ms per batch for 1/4 worlds, versus full 14.076/30.614 ms.
The candidate remains opt-in, pending broader views; no task defaults changed.
Detailed timing, pixel metrics and report hashes are recorded in
[budget audit](validation/p10-splat-budgets-2026-11-25.md).

### Multi-view candidate validation (16 desktop captures audited)

The benchmark adds `--capture-only`, `--spawn-point` and finite
`--view-yaw-degrees` (default 0). Named poses use the existing manifest spawn
loader; yaw is a fixed reset-event offset relative to the spawn orientation.
No new motion controller or pose-write implementation is added. Static capture
mode saves seed-42 RGB/depth/qpos and the requested view, then closes the environment
before constructing a PPO runner. Its passed status means capture completed, not
that visual similarity, pose safety or training passed. No throughput medians
are generated in this mode. Existing timing mode remains available.

```bash
bash artifacts/p10-vision-matrix/run-multiview.sh
```

Sixteen serial processes compare 200,000 with full at two existing apartment
spawns (`default`, `p7_free`) and yaw offsets 0/90/180/270 degrees. Each uses one
world and 64x48 RGB/depth; pair order alternates. `p7_free` is reused solely as
an existing camera location, not as a reopened P7 policy/map acceptance protocol.
Outputs go to a fresh `multiview-XXXXXXXX` directory, preserving old experiments.
The launcher does not install dependencies, change drivers or modify source assets.

Audit must match qpos, view requests, camera/collision/source hashes within each
pair before comparing images. Missing/empty depth is reported rather than treated
as proof of similarity. Examine each view and the worst view, not just pooled
averages. Full renders are a reference, not geometry ground truth. Eight static
views in one apartment cannot certify global coverage, collision-free poses,
motion consistency or training quality. No default budget changes are authorized
by this preparation; P10 remains in progress.

Local validation: three CPU regressions passed, including named-spawn/yaw
configuration, unchanged collision provenance, and static-mode close without
wrapper/PPO construction. `make check` passed (Ruff, ty, pyright), shell syntax and
diff checks passed. Documentation built successfully with one warning for an
unreachable online intersphinx inventory. Desktop output was subsequently audited
in `multiview-3LNvW1Qt`: all 16 captures passed, with equal qpos and provenance
within each pair. 200k has visible wall speckling/depth holes, with worst-view
reference-valid depth loss 4.40% and common-valid >0.1 m error fraction 7.78%.
Static protocol completion is not visual equivalence; full remains the default.
See [multi-view audit](validation/p10-multiview-2026-11-25.md) for per-view metrics,
artifact hashes and the prepared 250k/300k/full refinement experiment. The latter
uses existing capture functionality and passed a mocked 24-command launcher
check. Its 24 desktop captures have now been audited: 300k is closer to full than
250k across the eight sampled views, but this is not performance or task-quality
acceptance. Detailed metrics, comparison artifacts and updated priorities are in
[refinement evidence](validation/p10-multiview-refinement-2026-11-25.md).

The user subsequently reported fixing the sensing bottleneck in MuJoCo source.
Earlier timing results remain historical baselines, not verified performance of
the fixed build. Further work should first fingerprint/rebenchmark full rendering
and check output consistency; budget caps remain optional visual tradeoffs.

Current installed MuJoCo native files have now been verified against the supplied
wheel. Installed MuJoCo-Warp remains at 8ffab7a despite newer neighboring source.
The user selected testing this installed environment first. The fingerprinted four-case
full benchmark plus eight matched-view comparisons completed in `current-full-zztT0pGh`:
all 12 reports passed, eight RGB/depth pairs are exact, and camera timings are of
similar magnitude to the historical baseline. Installed Warp remains at 8ffab7a;
the newer neighboring Warp optimization was not evaluated.
See [build verification/retest protocol](validation/p10-current-full-2026-11-25.md).
This does not infer training-camera improvements from native-viewer code changes.

The independent native-viewer retest completed using the existing native policy
benchmark unchanged: three repetitions, stable pre/post CPU build/input snapshots,
six ready/zero-overflow results. Median Actual RT is 1.02978/1.02834 for reference/
automatic variants. This is native workload throughput, not camera acceleration; see
[native retest protocol](validation/p10-native-retest-2026-11-26.md).
