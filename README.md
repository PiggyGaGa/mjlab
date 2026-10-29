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

The current milestone provides native visualization and sensor integration plus a
collision-only MJCF input for externally prepared primitives, convex meshes, and
heightfields. Automatic collision-proxy generation from 3DGS remains planned. Run
the pretrained G1 tracking demo with aligned visual and collision assets:

```bash
uv run demo \
  --use-3dgs True \
  --scene-manifest /path/to/scene/manifest.json
```

The versioned manifest binds relative visual and collision paths, SHA-256
fingerprints, coordinate conventions, one shared world transform, and spawn-point
metadata. The loader checks that its dependency list exactly matches every file
referenced by the collision MJCF. The spawn named ``default`` is applied to robot
resets; choose another with ``--scene-spawn-point NAME``. For ad-hoc experiments,
the independent ``--gaussian-splat-file`` and
``--collision-mjcf-file`` options remain available; they cannot be combined with
``--scene-manifest``.

The repository also includes a reproducible room1 smoke scene with a flat
collision floor and six fixed pseudo-random pillars:

```bash
uv run tests/run_g1_3dgs_collision_demo.py
```

With the custom MuJoCo 3DGS wheel installed, test the native desktop viewer:

```bash
uv run tests/run_g1_3dgs_native_demo.py --steps 200
uv run tests/run_g1_3dgs_native_demo.py --interactive
```

The smoke test waits for the Gaussian renderer, runs the pretrained policy for
the requested steps, checks finite joint positions, and closes the window.
Add `--collision-visible` to display the collision proxy. On Linux x86_64 with
Python 3.12 or 3.13, uv downloads the custom wheel from the MuJoCo fork's
[v1.0 Release](https://github.com/PiggyGaGa/mujoco/releases/tag/v1.0).
No adjacent MuJoCo checkout or `--no-sync` is needed. These wheels require glibc
2.35 or newer; other platforms and Python versions use the official MuJoCo package
and do not provide this custom native 3DGS viewer.

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
