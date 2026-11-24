# P8 collision representation and fixed-budget validation

Date: 2026-11-21

This record covers the internal representation comparison, the executable
fixed-budget semantic protocol, the SplatTransform external reproduction, and CPU
and CUDA policy/native-viewer evaluation. It does not claim completion of
SceneSplat inference or real-label semantic quality.

## Artanim representation comparison

Input:

- `src/mjlab/asset_zoo/3dgs/ArtanimMotionCaptureStudio/scene_transform_20p.ply`
- SHA-256: `0451b2e220fe01ce603f8b61e32167fc4241d26582ea444c945d23018d361ba0`
- bounds: `[-2, -2, -0.5]` to `[2, 2, 3]`
- 23,933 bounded Gaussians; voxel size 0.1 m; TSDF truncation 0.3 m

Command:

```bash
uv run scripts/tools/compare_3dgs_collision_representations.py \
  src/mjlab/asset_zoo/3dgs/ArtanimMotionCaptureStudio/scene_transform_20p.ply \
  /tmp/p8-representations-final \
  --bounds -2 -2 -0.5 2 2 3 \
  --voxel-size 0.1 --tsdf-truncation 0.3 \
  --simplify-cell-size 0.2 --max-evaluation-points 10000
```

Results:

| representation | size | mean / p95 / max source distance (m) | topology |
|---|---:|---:|---|
| primitives | 43 geoms | 0.05939 / 0.13669 / 0.18088 | n/a |
| exact occupancy | 5,653 vertices, 11,752 faces | 0.04372 / 0.09185 / 0.10365 | 245 non-manifold edges |
| TSDF | 18,490 vertices, 37,000 faces | 0.04533 / 0.09330 / 0.11570 | watertight, consistent winding |
| coarse TSDF | 4,878 vertices, 9,720 faces | 0.13560 / 0.19488 / 0.22402 | watertight, consistent winding |

The 42×42 lowest-run heightfield was fully observed but reported 205 ambiguous
multilevel columns. The mixed output compiled with one heightfield and 24 residual
primitives. MuJoCo treats a user mesh as a convex collision shape, so the concave OBJ
exports are geometry/topology baselines rather than valid whole-room single geoms.

The same 3×3 probes used 1,000 warmup and 10,000 measured steps:

| representation | physics SPS | mean contacts | mean constraints | mean solver iterations |
|---|---:|---:|---:|---:|
| 43 primitives | 23,144.93 | 13.7613 | 55.0452 | 3.3550 |
| heightfield | 20,772.11 | 19.7733 | 79.0932 | 1.4867 |
| heightfield + 24 residual primitives | 23,691.50 | 13.7684 | 55.0736 | 3.3587 |

These are single CPU runs and are not evidence of a statistically stable speedup.

## Fixed-budget semantic diagnostic

The synthetic scene contains two support levels and one distractor component. Manual
integer labels are diagnostic ground truth. Both variants are capped at exactly two
primitives and evaluated with the same surface/free-space points and 2×2 falling
probes.

```bash
uv run scripts/tools/compare_3dgs_collision_semantics.py \
  /tmp/p8-semantic-synthetic/scene.ply \
  /tmp/p8-semantic-synthetic/labels.npy \
  /tmp/p8-semantic-synthetic/priorities.json \
  /tmp/p8-semantic-synthetic/results \
  --max-primitives 2 \
  --evaluation-points /tmp/p8-semantic-synthetic/evaluation.npz \
  --voxel-size 0.1 --opacity-threshold 0.5 \
  --lower-quantile 0 --upper-quantile 1 \
  --min-points-per-voxel 1 --min-component-voxels 1 \
  --min-box-volume 0 --warmup-steps 100 --measured-steps 2000 \
  --probe-grid 2 --probe-spacing 0.3 --probe-z 0.8
```

| variant | output geoms | false occupied | surface mean / p95 / max (m) | CPU SPS |
|---|---:|---:|---:|---:|
| geometry only | 2 | 1/1 | 0.04741 / 0.05000 / 0.07500 | 94,605.47 |
| semantic priority | 2 | 0/1 | 0.12601 / 0.65000 / 0.65192 | 96,690.26 |

The semantic variant preserves the inter-level free probe while discarding a
low-priority surface, making the fixed-budget quality tradeoff explicit. Contacts,
constraints, and solver iterations were equal in this small probe. This validates the
comparison protocol and manual-label upper-bound path, not SceneSplat label quality.

## External reproduction status

- SceneSplat commit: `07e2c946203467e7dfc402832fad2b8b1e49b810`.
  Official preprocessing produced finite, aligned arrays of 659,010 rows for
  coord/color/opacity/scale/quat. The official language checkpoint is gated at
  `GaussianWorld/SceneSplat_checkpoint` and requires the user to accept terms and
  authenticate. Desktop CUDA availability for the mjlab measurements does not
  supply that checkpoint or provision SceneSplat's separate
  CUDA/spconv/flash-attention/custom-pointops environment.
- SplatTransform commit: `5d5be2b1733b807a2a30a213cc677ce82a4c41ce`
  (package 3.12.0). The official Node 22.23.5 archive was verified with SHA-256
  `cdeb8670a0c81f51b4dc4ac61919f9b1ad917b304bf8221eb939570663d240f8`.
  The official npm package was installed after approval. Although `nvidia-smi`
  cannot communicate with a driver, WebGPU enumerated llvmpipe and completed both
  `faces` and default `smooth` collision exports for the same Artanim crop, 0.1 m
  voxel size, and 0.1 opacity threshold. The smooth run took 50.76 s with about
  492 MB peak CPU and 13.6 MB WebGPU memory. Its GLB SHA-256 is
  `5dd76e3537f4426d376013c63cdffefbc50ffd05e3aa6f3e62fd7d96b329690a` and contains
  2,227 vertices, 4,406 faces, 12 components, no boundary/non-manifold edges, and a
  watertight consistently wound surface. Distance from 10,000 uniformly sampled crop
  centers was 0.10244/0.21737/0.64472 m mean/p95/max. The `faces` output has
  SHA-256 `d9402472fdbf158af98f65184a55cc36118d33b9d22a2e34f78ca0ad7b41d207`,
  2,764 vertices, 5,222 faces, 550 boundary edges, and 26 non-manifold edges.
  Software-adapter time is not a real-GPU benchmark. The concave GLB also cannot be
  loaded as one MuJoCo mesh geom without changing collision to its global convex hull;
  it was therefore converted to explicit multi-convex bundles before contact tests.

Component-wise hull conversion produced 12 MuJoCo mesh geoms. Total hull volume was
1.442× the watertight source volume, and source-vertex-to-hull distance was
0.07637/0.25702/0.38158 m mean/p95/max. Deterministic V-HACD with 64 hulls,
resolution 200,000, and at most 64 vertices per hull reduced these to 1.172× and
0.02874/0.08851/0.17274 m. Both representations left 21 manifest-spawn free probes
(seven unique XY positions at z=0.2/0.6/1.0) outside every hull; this sparse set does
not prove global free space. With the same 3×3, 1,000-warmup, 10,000-measured probe:

| SplatTransform MuJoCo conversion | geoms | physics SPS | mean contacts | mean constraints | mean solver iterations |
|---|---:|---:|---:|---:|---:|
| component hulls | 12 | 27,455.70 | 9.0000 | 36.0000 | 1.7890 |
| V-HACD | 64 | 20,688.66 | 11.0624 | 44.2496 | 3.7173 |

File-backed benchmarking is opt-in and accepts only mesh dependencies resolving
inside the MJCF bundle; every dependency is hashed. V-HACD asynchronous processing
was disabled, and two runs produced identical MJCF and sampled first/last OBJ hashes.

## G1 tracking policy joint evaluation

The manifest in
`requirements/validation/p8-g1-policy-variants-2026-11-21.json` binds the flat-plane
control and four Artanim collision representations to their already measured
geometry evidence. The local evaluator used the cached standard demo checkpoint and
motion, fixed motion-start sampling, seeds 42/43/44, 10 untimed warmup steps, and at
most 500 measured control steps per run. Push events were disabled; seeded startup
randomization and actor observation corruption remained enabled. Auto-reset was
disabled so a terminal step was measured before any reset.

```bash
WARP_CACHE_PATH=/tmp/mjlab-warp-cache \
UV_CACHE_DIR=/tmp/mjlab-uv-cache UV_NO_SYNC=1 \
MPLCONFIGDIR=/tmp/mjlab-matplotlib \
uv run scripts/tools/benchmark_g1_collision_policy.py \
  requirements/validation/p8-g1-policy-variants-2026-11-21.json \
  --checkpoint /tmp/mjlab_cache/demo_ckpt.pt \
  --motion /tmp/mjlab_cache/lafan1_dance1_subject1_demo_motion.npz \
  --output /tmp/p8-g1-collision-policy-final.json \
  --device cpu --seeds 42 43 44 --warmup-steps 10 --max-steps 500
```

The report SHA-256 is
`5f4910332dc3017e6a5a82910018576f065b49e1c966846d99db4d450a0999ae`.
The checkpoint and motion hashes are respectively
`f1bc404f5165b833a3933ac98ff223450392b78df3eb76b0c36cd1360ff22319` and
`f08d15d4b5bb605e17b6928ccdb44ae6ce7bf2038111e8b145f12a176cd096d4`.

| representation | success | MPKPE (m) | R-MPKPE (m) | EE pos. error (m) | contacts | constraints | solver iterations | control SPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat-plane control | 1.000 | 0.1172 ± 0.0412 | 0.0341 ± 0.0010 | 0.0519 ± 0.0024 | 14.415 ± 0.797 | 57.664 ± 3.191 | 2.285 ± 0.191 | 27.82 ± 0.86 |
| 43 primitives | 0.333 | 0.2206 ± 0.0544 | 0.0985 ± 0.0040 | 0.1070 ± 0.0048 | 10.785 ± 2.446 | 43.195 ± 9.759 | 2.291 ± 0.062 | 26.95 ± 0.63 |
| heightfield + 24 residual | 0.333 | 0.2203 ± 0.0527 | 0.0991 ± 0.0036 | 0.1074 ± 0.0045 | 10.883 ± 2.742 | 43.588 ± 10.941 | 2.301 ± 0.049 | 22.98 ± 0.08 |
| SplatTransform component hulls | 0.333 | 0.1546 ± 0.0420 | 0.0779 ± 0.0017 | 0.0878 ± 0.0017 | 8.919 ± 0.453 | 35.704 ± 1.801 | 3.100 ± 0.186 | 21.83 ± 0.24 |
| SplatTransform V-HACD 64 | 0.333 | 0.1485 ± 0.0378 | 0.0789 ± 0.0012 | 0.0892 ± 0.0011 | 10.952 ± 1.099 | 43.837 ± 4.387 | 2.903 ± 0.191 | 22.24 ± 0.03 |

All four proxy variants completed seed 42, but seeds 43/44 terminated after 401--428
steps, mostly on `ee_body_pos`; some SplatTransform runs also triggered
`anchor_pos`. None of the proxy runs set a Warp overflow flag. The plane control
completed all seeds, but seed 44 set `LS_ITERATIONS` (bit 1024) on 172 steps. This is
a line-search iteration-limit flag, not 1,024 overflow events and not contact-buffer
overflow. The evaluator now records bitwise flags and nonzero-step counts explicitly.

The four proxies replace the task plane with Artanim geometry whose support height
and local surface differ from the flat-ground tracking reference. This is an intended
task-level test and exposes alignment sensitivity; it does not isolate representation
quality from scene/task coordinate mismatch. The geometry fields also retain their
original definitions: primitive distance is measured to retained Gaussian centres,
whereas SplatTransform conversion distance is source-mesh vertex to convex hull.
Consequently the table is a joint diagnostic, not a claim that every geometry number
is an interchangeable ground-truth error.

The CPU throughput column covers policy inference plus `env.step` and excludes
rendering. An additive native-viewer companion accepts the same manifest and records
actual realtime with native 3DGS submission. Its finite timing pass disables task
terminations so representation-dependent reset cost is excluded; policy success
continues to come from the joint evaluator above.

## CUDA policy and native 3DGS evidence

The desktop run used the same checkpoint, motion, manifest, seeds, warmup, horizon,
and input hashes as the CPU report. The CUDA policy report is
`/tmp/p8-g1-collision-policy-cuda.json`, SHA-256
`3222e0e12f0b79976ac75886d90cc5c41adbd0b513d0c103da341ccf7babb79e`.

| representation | success | R-MPKPE (m) | control SPS | CPU-relative SPS |
|---|---:|---:|---:|---:|
| flat-plane control | 1.000 | 0.0336 | 80.20 | 2.88x |
| 43 primitives | 0.333 | 0.1012 | 79.63 | 2.96x |
| heightfield + 24 residual | 0.333 | 0.1012 | 79.46 | 3.46x |
| SplatTransform component hulls | 0.000 | 0.0802 | 76.10 | 3.49x |
| SplatTransform V-HACD 64 | 0.000 | 0.0801 | 77.55 | 3.49x |

The plane and internal representations retained their CPU success rates. Both
SplatTransform variants changed from 1/3 CPU success to 0/3 CUDA success; their
three CUDA runs terminated after 405--471 steps. Seeded observation corruption and
the task protocol were fixed, but CPU and CUDA simulation/inference are not asserted
to be trajectory-identical. The success difference is therefore reported rather
than hidden or interpreted as a representation ranking. `LS_ITERATIONS` occurred in
one plane run, one component-hull run, and two V-HACD runs; it remains a solver flag,
not contact-buffer overflow.

The native report is `/tmp/p8-g1-native-3dgs.json`, SHA-256
`a2cfaade76c6aa8486d4005b0f6b01b965aed1b157456fe2c653063958a0cf1a`.
It identifies device `cuda:0`, MuJoCo `3.16.0+3dgs`, and Torch `2.14.0`; all five
variants reached `gaussian_status=ready`, completed 500 measured submissions, and
reported zero overflow bitmasks.

| representation | actual realtime | viewer submissions/s | setup + ready (s) |
|---|---:|---:|---:|
| flat-plane control | 0.498 | 24.90 | 1.476 |
| 43 primitives | 0.547 | 27.35 | 1.180 |
| heightfield + 24 residual | 0.533 | 26.67 | 1.188 |
| SplatTransform component hulls | 0.571 | 28.54 | 1.187 |
| SplatTransform V-HACD 64 | 0.528 | 26.41 | 1.150 |

Viewer submissions are not monitor refresh FPS. `actual_realtime` includes policy,
MuJoCo Warp, CPU viewer synchronization, and native 3DGS submission. These results
close the requested CUDA policy/native-visual measurement, but not the fixed-budget
semantic acceptance item because no actual SceneSplat labels were available.

The separately generated GPU-path SplatTransform files are
`/tmp/p8-splat-transform-gpu/artanim-smooth.voxel.json` and
`artanim-smooth.collision.glb`, with SHA-256
`9c7448e58c46fc262506d847fe8d54b03ba2622f326cc7a49b017234154f1f42` and
`5dd76e3537f4426d376013c63cdffefbc50ffd05e3aa6f3e62fd7d96b329690a`.
The GLB is byte-identical to the llvmpipe result and independently loads as one
watertight geometry with 2,227 vertices and 4,406 faces. The voxel JSON records the
generator and grid, but not adapter identity or elapsed time. A rerun preserved the
console at `/tmp/p8-splat-transform-gpu/run.log`, SHA-256
`bba65145779f32d403d5689c7843f182584f0ae9cb50627778447e2ac5ecc4f2`.
With `--gpu 0 --gpu-backend vulkan`, voxelization took 0.036 s and the complete
command took 0.714 s, with 335.2 MB peak CPU memory and 13.6 MB peak GPU memory. The
corresponding llvmpipe smooth run took about 49.86 s for voxelization and 50.76 s
overall, so the observed ratios are about 1,385x and 71x respectively. These are
single runs and the phases differ in how much CPU/file work they contain; they are
not statistical speedup claims.

The four warnings in the log reduce unusually large dynamic uniform/storage buffer
limits to 16 and are followed by successful extraction and writes; no output failure
occurred. The log says that GPU adapters were detected but does not print adapter 0's
name. The desktop `nvidia-smi` record identifies one NVIDIA GeForce RTX 3060 with
driver 580.82.07 and 12,288 MiB memory. The record is
`/tmp/p8-splat-transform-gpu/nvidia-smi.log`, SHA-256
`c29ec277b2f59b29563cf762cf98c7b350b75bd8331dc3336d2206b0a8d23cd5`.
The SplatTransform adapter record explicitly lists `[0] NVIDIA GeForce RTX 3060`,
`[1] NVIDIA GeForce RTX 3060`, and `[2] llvmpipe (LLVM 15.0.7, 256 bits)`. It is
`/tmp/p8-splat-transform-gpu/adapters.log`, SHA-256
`5dfbb5e3329028c94664d6583b7b1211c38c892552b84b42e422b21d4f59a30d`.
Because the timing command selected `--gpu 0`, the 0.036 s voxelization and 0.714 s
total are now directly bound to the RTX 3060 rather than inferred from host state.

The following commands reproduce the archived device evidence:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
/tmp/node-v22.23.5-linux-x64/bin/node \
  /tmp/p8-splat-runtime/node_modules/@playcanvas/splat-transform/bin/cli.mjs \
  --list-gpus --gpu-backend vulkan \
  2>&1 | tee /tmp/p8-splat-transform-gpu/adapters.log
```

The CUDA policy/native-viewer and Vulkan GPU timing reports above are complete. The
captured timing command was:

```bash
mkdir -p /tmp/p8-splat-transform-gpu
/tmp/node-v22.23.5-linux-x64/bin/node \
  /tmp/p8-splat-runtime/node_modules/@playcanvas/splat-transform/bin/cli.mjs \
  --gpu 0 --gpu-backend vulkan --memory --overwrite \
  src/mjlab/asset_zoo/3dgs/ArtanimMotionCaptureStudio/scene_transform_20p.ply \
  --filter-box=-2,-2,-0.5,2,2,3 \
  /tmp/p8-splat-transform-gpu/artanim-smooth.voxel.json \
  --voxel-size 0.1 --voxel-opacity 0.1 --collision-mesh smooth \
  2>&1 | tee /tmp/p8-splat-transform-gpu/run.log
```

Shared `src/mjlab/scripts/play.py` and `src/mjlab/scripts/demo.py` have zero diff.
