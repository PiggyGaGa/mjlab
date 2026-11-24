![Project banner](https://raw.githubusercontent.com/mujocolab/mjlab/main/docs/source/_static/mjlab-banner.jpg)

# mjlab

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjlab/ci.yml?branch=main)](https://github.com/mujocolab/mjlab/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation](https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg)](https://mujocolab.github.io/mjlab/)
[![License](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)
[![MuJoCo Warp](https://img.shields.io/badge/MuJoCo_Warp-3.16.0-blue)](https://github.com/google-deepmind/mujoco_warp/releases/tag/v3.16.0)
[![Nightly Benchmarks](https://img.shields.io/badge/Nightly-Benchmarks-blue)](https://mujocolab.github.io/mjlab/nightly/)
[![PyPI](https://img.shields.io/pypi/v/mjlab)](https://pypi.org/project/mjlab/)
[![PyPI downloads](https://img.shields.io/pypi/dm/mjlab?color=blue)](https://pypistats.org/packages/mjlab)

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), a GPU-accelerated version of [MuJoCo](https://github.com/google-deepmind/mujoco).
The framework provides composable building blocks for environment design,
with minimal dependencies and direct access to native MuJoCo data structures.

## Getting Started

mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Try it now:**

Run the demo (no installation needed):

```bash
uvx --from mjlab --refresh demo
```

Or try in [Google Colab](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb) (no local setup required).

**Install from source:**

```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv run demo
```

For alternative installation methods (PyPI, Docker), see the [Installation Guide](https://mujocolab.github.io/mjlab/main/source/installation.html).

## 3D Gaussian Splatting

mjlab can place trained robots in a static 3D Gaussian Splatting (3DGS) scene.
Viser renders the splats natively alongside the robot, while MuJoCo continues to
use separate collision geometry.

The goal is broader than photorealistic playback. A reconstructed real environment
can become a robot-learning scene whose visual representation, collision proxy,
sensor observations, training configuration, and policy evaluation share one
coordinate system. The native MuJoCo Warp path also avoids introducing a second
Gaussian renderer and its CUDA dependency stack.

This creates a foundation for several robotics directions:

- vision-based locomotion, navigation, and manipulation in reconstructed places;
- automatic conversion of visual reconstructions into contact-stable collision
  proxies for robot training;
- measuring and reducing visual-to-physical mismatch in simulation;
- generating realistic RGB/depth observations for policy robustness and
  sim-to-sim or sim-to-real evaluation;
- task-aware scene simplification that balances contact quality, visual fidelity,
  training throughput, and memory use.

Optional repeated-load preprocessing caches are enabled per asset with
``GaussianSplatAssetCfg(path="scene.ply", cache_dir="artifacts/splat-cache")``.
They are off by default, invalidate on source content or import/transform changes,
and rebuild corrupt entries with atomic writes. Measure CPU loading separately with
``uv run scripts/tools/benchmark_gaussian_splat_loading.py scene.ply report.json
--cache-dir artifacts/splat-cache``. This does not cache GPU BVHs or increase
rendering FPS by itself.

The current milestone provides native visualization and sensor integration plus a
collision-only MJCF input for externally prepared primitives, convex meshes, and
heightfields. A deterministic center-only baseline can also voxelize a Graphdeco
PLY and merge occupied cells into collision boxes:

```bash
uv run scripts/tools/generate_3dgs_collision.py \
  /path/to/splats.ply /path/to/collision.xml \
  --report /path/to/collision_report.json --floor-z 0 \
  --primitive-mode auto
```

The tool writes a hashed report and a compressed mapping from each retained box
or fitted primitive to the original PLY vertex IDs. ``auto`` uses local covariance
and surface-fit error to select oriented boxes, spheres, cylinders, or capsules;
failed fits fall back to the original bounded axis-aligned boxes. The default
``--primitive-mode boxes`` preserves the original baseline. Round primitives must
also reduce normalized surface error by 30% relative to an oriented-box fit; tune
this conservative gate with ``--min-round-error-improvement``. For stepped or bent
surfaces, ``--max-planar-merge-residual 0.04`` prevents an approximately planar
merged candidate from spanning more than 4 cm away from its best-fit plane. This
scale-dependent guard is opt-in so existing outputs remain unchanged. The tool assumes the
input is already right-handed, Z-up, and measured in metres; it does not infer
semantic structure or guarantee contact-quality geometry, so inspect the result
before training. When camera rays, manual annotations, or another trusted source
provide known free space, pass an ``(N, 3)`` world-coordinate NumPy array with
``--known-free-points /path/to/free_points.npy``. Merging and automatic primitive
fitting will not cover those samples; conflicting occupied/free voxels are rejected.
The PLY alone cannot supply this negative evidence. In auto mode, combine it with
``--planar-slab-thickness 0.1`` to place a planar patch's fitted surface on the
known-free face and extend the requested thickness only toward the inferred solid
side. Ambiguous or missing free-side evidence retains the symmetric legacy fit.
For a covariance-aware comparison, ``--use-gaussian-covariance`` adds each
Graphdeco splat covariance to the candidate's center covariance before PCA;
``--max-gaussian-scale`` can remove extreme scale outliers first. Both are opt-in.
For fragmented horizontal patches, add ``--merge-horizontal-surfaces`` in auto
mode. This aligns nearby existing box-top levels and retiles the exact union of
their voxel-candidate footprints into finite slabs. It preserves holes in that
union, distinct height groups, semantic boundaries and explicit known-free
constraints. It is not global floor recognition: uncertain patches and ramps stay
unchanged, and existing unseen holes cannot be inferred from source points alone.
``--horizontal-level-tolerance`` (0.04 m), ``--horizontal-surface-residual``
(0.06 m), ``--horizontal-slab-thickness`` (0.1 m), and
``--horizontal-max-tilt-degrees`` (5 degrees) control this opt-in experiment.
An integer ``(N,)`` ``--semantic-labels`` NumPy sidecar can provide a replaceable
per-vertex frontend: voxel labels use deterministic majority voting, merging does
not cross another label's occupied cells, and the configured unknown label remains
as a geometric residual group. In auto mode, ``--max-primitives`` imposes a
deterministic geom budget; ``--semantic-priorities`` and ``--point-importance``
rank candidates within that budget while retaining exact source-ID mappings.
``scripts/tools/decode_3dgs_semantic_features.py`` converts aligned SceneSplat-style
language features and user-supplied class embeddings into this integer sidecar using
blocked cosine similarity. Similarity and top-two margin thresholds retain uncertain
rows as ``unknown``; the report hashes every input and records confidence statistics.
It is a decoder, not a bundled SceneSplat checkpoint or a claim of label accuracy.
For automatic use with uncertain labels,
``uv run scripts/tools/generate_3dgs_collision_guarded.py scene.ply labels.npy output/``
always generates a geometry baseline and falls back to it when unknown labels
exceed 50%. Otherwise it accepts semantic grouping only if source mapping coverage
does not decrease and source-distance mean/p95/max do not increase by more than 5%.
The thresholds are configurable engineering guards, not calibrated confidence or
a safety guarantee. Unknown categories never imply free space; the selected proxy
and rejection reasons are recorded in ``output/guard_report.json``.
An independent ``--evaluation-points`` NPZ can report reference-surface distance
and false occupied free-space probes without affecting generation.
``--evaluate-source-surface`` adds an input-coverage diagnostic. Compare CPU
collision-only cost with identical probes using
``uv run scripts/tools/benchmark_collision_mjcf.py ... --output report.json``;
the benchmark reports physics SPS, contacts, constraints, and solver iterations,
and does not measure 3DGS rendering. It rejects file assets by default; the explicit
``--allow-file-assets`` mode accepts only bundle-local mesh files and hashes every
dependency.
For an explicitly matched-budget geometry/semantic experiment, run
``scripts/tools/compare_3dgs_collision_semantics.py`` with one label per original
PLY vertex, a priority JSON file, and independent evaluation probes. It generates
both MJCFs and records generation time, geometric/free-space error, physics SPS,
contacts, constraints, and solver iterations under identical probes. Its report
leaves visual throughput unset unless that separate CUDA-visible experiment is
actually run.

``scripts/tools/compare_3dgs_collision_representations.py`` provides an offline
comparison of voxel primitives, exact occupancy surfaces, TSDF marching-tetrahedra
surfaces, coarse manifold remeshing, and lowest-surface heightfield plus residual
primitives. It writes topology diagnostics and deterministic OBJ/MJCF artifacts.
The OBJ files are reconstruction baselines: MuJoCo collision for a single concave
mesh uses its convex hull, so a whole room OBJ must not be loaded as one collision
geom without convex decomposition.
``scripts/tools/convert_collision_mesh_to_mjcf.py`` converts an external triangle
mesh either to one convex hull per connected component or to deterministic V-HACD
pieces, exports a bundle-local MJCF, and reports volume inflation, source-to-hull
distance, compile counts, and optional independent free-space probes.
For a collision replacement experiment with extra common obstacles, first use
``scripts/tools/prepare_collision_comparison.py reference.xml automatic.xml
scene.ply output/ --common-geoms obstacle_a obstacle_b``. This freezes independent
primitive copies with identical resolved common obstacles; it does not change the
source scene. Optional ``--automatic-report`` binds the automatic MJCF and visual
PLY to their generation hashes. See
[the P10 protocol](requirements/mjlab-3dgs-p10-benchmarks.md) for the apartment setup.

``scripts/tools/benchmark_g1_collision_policy.py`` consumes a JSON variant manifest
and evaluates every proxy with the same local G1 checkpoint, motion, seeds, and
finite horizon. Its report keeps geometry evidence beside tracking error,
termination reason, contacts, constraints, solver flags, and policy-plus-physics
throughput. It disables auto-reset so the terminal step is measurable and does not
modify ``play`` or ``demo``. On a CUDA-visible desktop, the companion
``scripts/tools/benchmark_native_3dgs_collision_policy.py`` runs the same manifest
through the native 3DGS viewer and records actual realtime and viewer submission
rate; submissions are not display refresh FPS.
Add ``--debug-color-by-type`` in auto mode to render boxes as translucent green,
spheres as magenta, cylinders as orange, and capsules as blue;
this changes only MJCF appearance. Run the pretrained G1 tracking demo with aligned
visual and collision assets:

```bash
uv run demo \
  --use-3dgs True \
  --scene-manifest /path/to/scene/manifest.json
```

The versioned manifest binds relative visual and collision paths, coordinate
conventions, one shared world transform, and spawn-point metadata. The loader checks
that its dependency list exactly matches every file referenced by the collision
MJCF. The spawn named ``default`` is applied to robot resets; choose another with
``--scene-spawn-point NAME``. For ad-hoc experiments, the independent
``--gaussian-splat-file`` and
``--collision-mjcf-file`` options remain available; they cannot be combined with
``--scene-manifest``.

The repository also includes a reproducible room1 smoke scene with a flat
collision floor and six fixed pseudo-random pillars:

```bash
uv run tests/run_g1_3dgs_collision_demo.py
```

With the custom MuJoCo 3DGS wheel installed, test the native desktop viewer:

```bash
uv run --no-sync tests/run_g1_3dgs_native_demo.py --steps 200
uv run --no-sync tests/run_g1_3dgs_native_demo.py --interactive
```

The smoke test waits for the Gaussian renderer, runs the pretrained policy for
the requested steps, checks finite joint positions, and closes the window.
Add `--collision-visible` to display the collision proxy. On Linux x86_64 with
Python 3.12 or 3.13, uv downloads the custom wheel from the MuJoCo fork's
[v1.0 Release](https://github.com/PiggyGaGa/mujoco/releases/tag/v1.0).
Normal locked installation needs no adjacent MuJoCo checkout. These wheels require glibc
2.35 or newer; other platforms and Python versions use the official MuJoCo package
and do not provide this custom native 3DGS viewer.

When a release wheel is replaced at the same URL, update it without maintaining
the repository's wheel hash:

```sh
uv run --no-sync scripts/tools/update_mujoco_wheel.py
uv run --no-sync tests/run_g1_3dgs_native_demo.py --interactive --collision-visible
```

The updater installs the current interpreter's wheel directly, bypassing lockfile
hash validation. Continue using `--no-sync` afterwards: ordinary `uv sync` and
`uv run` still enforce the lockfile, whose URL wheel hashes are mandatory in uv.

To check native viewer controls, collision visibility, window cleanup, and
runtime performance with a velocity checkpoint:

```bash
uv run tests/run_g1_3dgs_acceptance.py --checkpoint /path/to/model.pt
```

The script loads the room1 manifest and opens four native windows in sequence
with 3DGS off/on/off/on. Each window warms up for 10 policy steps, then measures
10 seconds by default. It checks callback-driven pause, single-step, reset and
resume, camera parameters, robot state synchronization, CPU contact replay with
collision proxies hidden/shown, and window/temporary PLY cleanup.

The August 2026 velocity run with 98 actor inputs requires the explicit
`--legacy-phase-observations` adapter. For that checkpoint, run:

```bash
uv run tests/run_g1_3dgs_acceptance.py \
  --checkpoint /path/to/model_10000.pt \
  --legacy-phase-observations \
  --seconds 10 \
  --output /tmp/p3-acceptance.json
```

The JSON report defaults to `/tmp/p3-acceptance.json` and records Actual RT,
viewer setup time, submission frequency, device-wide GPU memory, and check
results. Submission frequency is not display FPS, and CPU contact replay does
not establish full MJWarp policy trajectory equivalence. Physical keyboard/mouse
operation and visual occlusion still need manual verification.

For an interactive comparison using the same legacy checkpoint:

```bash
uv run tests/run_g1_3dgs_acceptance.py \
  --checkpoint /path/to/model_10000.pt \
  --legacy-phase-observations --interactive --viewer native --collision-visible
```

Replace `--viewer native` with `--viewer viser` to compare the browser frontend.
Interactive mode skips the four-window benchmark; closing it records the session
ending, not a passed manual acceptance check. See the
[interface contract and validation limits](requirements/mjlab-3dgs-native-interface.md).

To play a locally trained G1 velocity policy in the same manifest scene:

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1-3DGS \
  --scene-manifest src/mjlab/asset_zoo/3dgs/room1/manifest.json \
  --scene-spawn-point default \
  --checkpoint-file logs/rsl_rl/g1_velocity/<run>/model_<iteration>.pt \
  --num-envs 1 --viewer viser --collision-visible True
```

During Viser playback, use ``Visualization > Collision > Enabled`` to show or
hide the collision proxy without restarting the simulation.

See the [3D Gaussian Splatting guide](https://mujocolab.github.io/mjlab/main/source/sensors/gaussian_splats.html)
for scene configuration, training, playback, multiple robots, recording, and
performance notes.

## Training Examples

For the G1 3DGS depth-training pipeline smoke check:

```bash
uv run tests/run_g1_3dgs_vision_smoke.py --num-envs 1
uv run tests/run_g1_3dgs_vision_smoke.py --num-envs 2 \
  --scene-manifest src/mjlab/asset_zoo/3dgs/room1/manifest.json \
  --output /tmp/p6-vision-smoke
```

To inspect the two camera-visible apartment pillars against the 3DGS wall, keep
the robot at the manifest's ``default`` spawn, which has the same transform as
``p7_both_open``:

```bash
uv run --no-sync tests/run_g1_3dgs_vision_smoke.py \
  --scene-manifest src/mjlab/asset_zoo/3dgs/apartment/manifest.json \
  --fixed-base-pose \
  --validity-mask \
  --depth-metrics \
  --width 640 \
  --height 480 \
  --output /tmp/p6-apartment-pillars
```

Compare ``reset-env0-rgb.png``, ``reset-env0-raw-depth.png``,
``reset-env0-policy-depth.png``, and ``reset-env0-validity.png`` under
``/tmp/p6-apartment-pillars``. The fixed-pose option disables reset-time
``x``/``y``/yaw offsets so the pillar view remains reproducible.

Add `--depth-metrics` to record raw missing, near, far, and in-range fractions
for every world and rollout step in `rollout_depth.json`. The report summarizes
these fractions and done counts; sampling includes auto-reset frames and adds
CPU overhead, so use a separate run for performance measurements. See the
[four-world rollout and multi-asset isolation results](requirements/validation/p6-rollout-isolation-2026-10-31.md).

For a two-channel depth + validity comparison, add `--validity-mask`:

```bash
uv run tests/run_g1_3dgs_vision_smoke.py --num-envs 2 --iterations 4 \
  --scene-manifest src/mjlab/asset_zoo/3dgs/room1/manifest.json \
  --output /tmp/p6-depth-baseline
uv run tests/run_g1_3dgs_vision_smoke.py --validity-mask --num-envs 2 --iterations 4 \
  --scene-manifest src/mjlab/asset_zoo/3dgs/room1/manifest.json \
  --output /tmp/p6-depth-validity
```

The baseline task stays `Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth`; the additional
task is `Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity`. Channel order is
depth, then raw finite-positive validity. Missing depth and valid depth beyond
5m become `(1, 0)` and `(1, 1)` respectively. This mask indicates renderer
availability, not geometric accuracy or free space. Task experiment names are
separate, and one-/two-channel checkpoints cannot be loaded interchangeably.
See the [validity evaluation](requirements/validation/p6-validity-2026-10-31.md)
for evidence and limits; short PPO checks do not establish task-level gains.

This runs short CNN/PPO updates, checks changing depth frames and partial reset
isolation, saves/reloads a visual checkpoint, and resumes an update. Outputs
include diagnostic depth PNG/NPY files and a JSON report; no external logging
service is used. It validates the pipeline, not locomotion convergence or visual
benefit. The method combines PPO/GAE with spatial-softmax visual features,
reusing the repository's YAM Vision implementation. See the
[paper sources and method](requirements/mjlab-3dgs-p6-method.md),
[BibTeX](requirements/mjlab-3dgs-p6-references.bib), and
[runtime evidence and limitations](requirements/validation/p6-runtime-2026-10-30.md).

To diagnose camera depth at a fixed pose with combined, robot-only, and
Gaussian-only rendering:

```bash
uv run tests/run_g1_depth_diagnostics.py --output /tmp/p6-depth-diagnostics
uv run tests/run_g1_depth_diagnostics.py --pose-sweep --output /tmp/p6-depth-poses
```

The G1 depth task maps missing zero-depth pixels to the far cutoff, so they are
not presented as nearby obstacles. Missing and out-of-range pixels share the
same normalized value; this does not establish free space. See the
[depth quality report](requirements/validation/p6-depth-quality-2026-10-30.md).
`--pose-sweep` additionally compares seven controlled root/arm configurations
and reports self-occlusion and missing-depth fractions. These are kinematic
configurations, not a simulated walking trajectory; see the
[pose-sweep results](requirements/validation/p6-pose-sweep-2026-10-31.md).

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**Multi-GPU Training:** Scale to multiple GPUs using `--gpu-ids`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 2. Motion Imitation

Train a humanoid to mimic reference motions. See the [motion imitation guide](https://mujocolab.github.io/mjlab/main/source/training/motion_imitation.html) for preprocessing setup.

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name your-org/motions/motion-name --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 3. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions
```

When running motion-tracking tasks, add `--registry-name your-org/motions/motion-name` to the command.


## Documentation

Full documentation is available at **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**.

## Development

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format and lint
make docs          # Build docs locally
```

For development setup: `uvx pre-commit install`

## Citation

mjlab is used in published research and open-source robotics projects. See the [Research](https://mujocolab.github.io/mjlab/main/source/research.html) page for publications and projects, or share your own in [Show and Tell](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell).

If you use mjlab in your research, please consider citing:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.
