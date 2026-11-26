# P10 installed-build verification and full-render retest

Date: 2026-11-25. Build verification and desktop performance/output retest
audited. The user explicitly selected testing the current installed
environment before updating MuJoCo-Warp.

## Installed environment verified on CPU

The loaded MuJoCo module is under the project's Python 3.13 `.venv`, package
version `3.16.0+3dgs`, native `mj_versionString()` 3.16.0. All 11 installed native
files match the previously downloaded and SHA256-verified wheel byte-for-byte:
`33ef7f10e5dde0697987509f893e92703801c457cd73e5c8e2886099bdd0bc6b`.
`/proc/self/maps` confirms the actual mapped library is the project's
`mujoco/libmujoco.so.3.16.0`, with SHA256
`87d2aff959f9813c5e5b66c823a80728c1418722588918fb7903a7e71c32b05a`.

Installed MuJoCo-Warp is 3.16.0 with direct-url Git revision
`8ffab7a5810bf82ef6156025b6765e5122532cde`. This differs from the neighboring
source checkout at `b4f0688a3254544a4202d42cccf7ba47ce77962e`, whose latest commit
specializes Gaussian shading for render-context outputs. The MuJoCo checkout is
clean at `4248ddf4aab5f4ee643712fbd52f149287e5105e`; its recent performance work is
under `simulate/gaussian` (native viewer). Neither checkout HEAD is assumed to
identify the installed wheel's source revision: no embedded revision was found
in the recorded package metadata.

Native viewer GLSL and MJWarp training-camera rendering are distinct paths. This
retest evaluates the current installed training-camera workload, not a claim that
the neighboring MuJoCo-Warp optimization was installed or that a native-viewer
fix must improve this camera. No packages or drivers were changed.

Installed file identity is saved to
`artifacts/p10-vision-matrix/current-installed-build.json`:

- MuJoCo runtime file tree SHA256:
  `d87ce2bc7067b970d9c1d3d442a1dfbb5f0fc118d00ced598de488db9391a566`.
- MuJoCo-Warp runtime file tree SHA256:
  `f635c513b1c0da16928ea4b50f88fa4f9b70e243d4069f0f69f716703e11bfe1`.

The tree identity covers installed `.py`, `.glsl` and `.so*` files by relative name
and content, ignoring bytecode caches. It is not a hash of every package resource,
JIT cache, GPU driver or generated device binary. Absolute locations are recorded
separately. The benchmark now records these identities, direct-url metadata,
mapped MuJoCo library hashes and its own/upstream timing-helper source hashes
before initialization/timed loops. Generic runtime/task/viewer behavior is unchanged.

## Desktop command

```bash
bash artifacts/p10-vision-matrix/run-current-full.sh
```

The launcher uses the existing environment with dependency synchronization disabled
and writes a fresh `current-full-XXXXXXXX` directory. It does not update Warp,
install a wheel, clear caches or alter source assets. It first checks that all
eight archived reference reports/captures exist.

1. Four timed full-apartment configurations: 1/4 worlds × 64x48/128x96 RGB/depth.
   The primary timing protocol remains 32 steps, 16 warmup steps, one discarded
   complete repetition and three retained repetitions. No splat cap is applied.
   Raw initial frames/qpos are saved outside measured loops for future comparison.
2. Eight separate full-only static captures at default/p7_free × four yaw offsets,
   one world, 64x48, no PPO updates.
3. Each static report is immediately compared on CPU against its corresponding
   full report in `multiview-refinement-iOIrPPHa`. Eight `*.comparison.json` files
   contain equality/error metrics, input hashes and available build identities.

Snapshots use the same fixed seed and reset configuration. The comparison tool
requires passed reports, identical camera/manifest/source/collision/effective
asset/count inputs, equal array shapes and exact finite qpos before computing
differences. Invalid depth is counted separately; empty common-valid sets produce
null error metrics. Unsigned RGB is converted before subtraction. A mismatch
writes `not_comparable` and stops the launcher. Nonzero image errors are recorded
as `compared`, not silently labeled quality passed; there is no invented tolerance.
Historical references lack build fingerprints, so even identical frames do not
establish a controlled before/after performance comparison.

128x96 captures have no prior matched-qpos NPZ reference in this protocol; they
are archived as new baselines, not declared unchanged. Torch memory remains an
incomplete measure of renderer VRAM. Timing phases are independent workloads,
not additive pure-kernel attribution. Four small PPO runs are pipeline benchmarks,
not converged training or evidence of task benefit. Further P10 coverage remains open.

## Local verification

Seven CPU regressions passed across the capture audit and existing benchmark
tests, covering same-version native-file changes, ignored bytecode caches,
unsigned RGB differences, missing depth, rejected qpos/collision mismatch,
rollout/reset/autograd boundaries and static-mode omission of PPO construction.
The comparison CLI also compared a real archived full capture with itself and
reported exact RGB/depth equality. A mocked launcher dry run checked four timed,
eight static and eight comparison commands, with no caps or Gaussian disabling.

`make check`, shell syntax and `git diff --check` passed. `make docs` succeeded
with one warning for an unreachable online intersphinx inventory. These CPU/local
checks do not constitute desktop GPU timing or new rendered-output acceptance.

## Audited desktop run

Accepted directory: `artifacts/p10-vision-matrix/current-full-zztT0pGh/`.
Four timed reports and eight static capture reports passed. All 12 actual build
identities exactly match the verified installed snapshot above, including mapped
library and package tree hashes; installed Warp remains at 8ffab7a. The benchmark
and timing-helper source hashes match the current scripts. Full 374,410 Gaussians
were loaded in every case, with no cap or disabled Gaussian branch.

Audit checked current manifest/source/collision hashes, capture dimensions and
finite poses, three retained samples numbered 1/2/3, finite losses, positive finite
timings, and all medians by recomputation. No traceback appeared in the 12 logs.
`audit-manifest.json` retains each report and capture SHA256 for this run.
Its SHA256 is
`9ef31abeddd9ede4f12adda4374791dd80afb110de1a65e820b0c2549df557f9`.

| Configuration | Sense ms/batch | Three-sample range ms | Full-env SPS | PPO rollout s | PPO update s |
|---|---:|---:|---:|---:|---:|
| 1w-64x48 | 12.512 | 12.442–12.528 | 54.76 | 0.667 | 0.128 |
| 1w-128x96 | 24.401 | 23.999–25.127 | 32.78 | 1.139 | 0.129 |
| 4w-64x48 | 28.544 | 28.525–28.570 | 95.57 | 1.368 | 0.134 |
| 4w-128x96 | 83.448 | 83.399–83.464 | 34.11 | 3.496 | 0.134 |

Sense latency is batch seconds / 32, covering all worlds. Full-env SPS sums world
transitions. These remain independent short workload timings; phases are not an
additive renderer-kernel decomposition. Historical `throughput-DA7sGc22` sense
medians in the same configuration order were 12.390/23.765/28.587/83.164 ms/batch.
The current values are similar in magnitude and do not demonstrate a large camera
speedup. Historical runs lack installed-binary fingerprints, so no controlled
before/after speedup or confidence claim is made from this comparison.

All eight static `*.comparison.json` results were independently recomputed from
the raw NPZs and report inputs. Exact qpos, RGB and raw-depth equality held in all
eight views; RGB/depth MAE and lost-reference-valid-depth fractions are zero.
High-resolution captures are retained as new baselines; no archived matched-pose
128x96 equivalence is claimed. Static frame equality is not a policy-quality or
temporal-consistency test.

This closes the selected current-environment four-case/eight-view subexperiment.
It does not evaluate the newer Warp shading revision visible in the neighboring
checkout, nor measure native-viewer rendering performance. Future work should
separate those paths: evaluate an explicitly updated Warp build with fingerprints
and the same protocol if selected, or benchmark the native viewer separately.
No package updates or GPU experiments were launched during this CPU audit.
Full P10 acceptance remains open.
