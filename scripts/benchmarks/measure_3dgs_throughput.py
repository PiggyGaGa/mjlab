"""Repeated 3DGS pipeline measurements; run on the desktop GPU explicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from gaussian_build_identity import installed_build_identity
from measure_throughput import measure_env_sps, measure_physics_sps
from PIL import Image
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scene import apply_scene_manifest
from mjlab.sensor import CameraSensor, CameraSensorCfg
from mjlab.tasks.g1_3dgs import G1_3DGS_DEPTH_SENSOR
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

TASK = "Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth"


def save_reset_capture(env: ManagerBasedRlEnv, output: Path) -> dict:
  """Save the fixed reset view outside every measured batch."""
  env.reset(seed=42)
  sensor = env.scene[G1_3DGS_DEPTH_SENSOR]
  assert isinstance(sensor, CameraSensor)
  rgb, depth = sensor.data.rgb, sensor.data.depth
  assert rgb is not None and depth is not None
  rgb_array = rgb.detach().cpu().numpy()
  depth_array = depth.detach().cpu().numpy()[..., 0]
  valid = np.isfinite(depth_array) & (depth_array > 0)
  output.mkdir(parents=True, exist_ok=False)
  np.savez(
    output / "reset.npz",
    rgb=rgb_array,
    depth=depth_array,
    qpos=env.sim.data.qpos.detach().cpu().numpy(),
  )
  for index, frame in enumerate(rgb_array):
    Image.fromarray(frame).save(output / f"env{index}-rgb.png")
    display = np.where(valid[index], depth_array[index], 5.0)
    Image.fromarray((np.clip(display / 5.0, 0, 1) * 255).astype(np.uint8)).save(
      output / f"env{index}-depth.png"
    )
  return {
    "directory": str(output),
    "view": "seed-42 configured reset; one viewpoint per world",
    "positive_finite_depth_fraction": valid.mean(axis=(1, 2)).tolist(),
    "display_depth_cutoff_m": 5.0,
  }


def elapsed(operation, device: str) -> float:
  torch.cuda.synchronize(device)
  start = time.perf_counter()
  operation()
  torch.cuda.synchronize(device)
  return time.perf_counter() - start


@torch.no_grad()
def collect_rollout(
  runner: MjlabOnPolicyRunner,
  wrapped: RslRlVecEnvWrapper,
  obs: TensorDict,
  steps: int,
) -> TensorDict:
  # Environment history buffers survive into resets outside this context.
  # no_grad avoids autograd graphs without creating immutable inference tensors.
  for _ in range(steps):
    actions = runner.alg.act(obs)
    obs, rewards, dones, extras = wrapped.step(actions)
    runner.alg.process_env_step(obs, rewards, dones, extras)
  runner.alg.compute_returns(obs)
  return obs


def run(args: argparse.Namespace, report: dict) -> None:
  configure_torch_backends()
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = 42
  manifest = apply_scene_manifest(
    cfg.scene, args.scene_manifest, spawn_point=args.spawn_point
  )
  camera = next(
    sensor for sensor in cfg.scene.sensors or () if sensor.name == G1_3DGS_DEPTH_SENSOR
  )
  assert isinstance(camera, CameraSensorCfg)
  camera.width, camera.height = args.width, args.height
  camera.data_types = ("rgb", "depth")
  cfg.events["reset_base"].params["pose_range"].update(
    {axis: (0.0, 0.0) for axis in ("x", "y", "yaw")}
  )
  yaw = math.radians(args.view_yaw_degrees)
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (yaw, yaw)
  report["requested_view"] = {
    "spawn_point": args.spawn_point,
    "spawn_position": cfg.scene.spawn_position,
    "spawn_quaternion_wxyz": cfg.scene.spawn_quaternion_wxyz,
    "yaw_offset_degrees": args.view_yaw_degrees,
  }
  agent = load_rl_cfg(TASK)
  agent.seed = 42
  agent.num_steps_per_env = args.steps
  agent.upload_model = False
  report["agent_cfg"] = asdict(agent)
  report["manifest_sha256"] = hashlib.sha256(
    args.scene_manifest.read_bytes()
  ).hexdigest()
  assert cfg.scene.gaussian_splats is not None
  report["scene_assets"] = asdict(cfg.scene.gaussian_splats)
  report["source_sha256"] = {
    asset.path: hashlib.sha256(Path(asset.path).read_bytes()).hexdigest()
    for asset in cfg.scene.gaussian_splats.assets
  }
  report["collision_sha256"] = {
    str(asset.path): hashlib.sha256(asset.path.read_bytes()).hexdigest()
    for asset in (manifest.collision, *manifest.collision.dependencies)
  }
  report["input_scene_assets"] = report["scene_assets"]
  if args.max_splats is not None:
    cfg.scene.gaussian_splats = replace(
      cfg.scene.gaussian_splats,
      assets=tuple(
        replace(asset, max_splats=args.max_splats)
        for asset in cfg.scene.gaussian_splats.assets
      ),
    )
    report["scene_assets"] = asdict(cfg.scene.gaussian_splats)
  if args.without_gaussians:
    cfg.scene.gaussian_splats = None
    report["scene_assets"] = None
  report["gaussians_enabled"] = cfg.scene.gaussian_splats is not None
  report["camera_capture"] = {
    "width": camera.width,
    "height": camera.height,
    "data_types": list(camera.data_types),
  }
  started = time.perf_counter()
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    torch.cuda.synchronize(args.device)
    report["initialization_seconds"] = time.perf_counter() - started
    context = env.scene.sensor_context
    splats = context.splat_data if context is not None else None
    report["loaded_splats_shared_across_worlds"] = (
      len(splats.position) if splats is not None else 0
    )
    if args.save_capture or args.capture_only:
      report["reset_capture"] = save_reset_capture(
        env, args.output.with_suffix(".capture")
      )
    if args.capture_only:
      report["mode"] = "static_capture_only_no_ppo"
      return
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=args.device)
    actor = runner.alg.get_policy()
    samples = []
    report["samples"] = samples
    # Discard an entire repetition, including the first PPO update.
    for repeat in range(args.repeats + 1):
      env.reset(seed=42)
      action = torch.zeros(args.num_envs, wrapped.num_actions, device=args.device)
      for _ in range(args.warmup_steps):
        env.step(action)
      torch.cuda.synchronize(args.device)
      torch.cuda.reset_peak_memory_stats(args.device)
      sample: dict = {"repeat": repeat}
      sample["physics_env_equivalent_sps"] = measure_physics_sps(env, args.steps)
      env.reset(seed=42)
      for _ in range(args.warmup_steps):
        env.step(action)
      sample["zero_action_full_env_sps"] = measure_env_sps(env, args.steps)
      env.reset(seed=42)
      for _ in range(args.warmup_steps):
        env.sim.sense()

      def sense_batch() -> None:
        for _ in range(args.steps):
          env.sim.sense()

      sample["stationary_sense_seconds"] = elapsed(sense_batch, args.device)
      obs = wrapped.get_observations()
      runner.alg.eval_mode()

      def policy_batch(observations=obs) -> None:
        with torch.no_grad():
          for _ in range(args.steps):
            actor(observations)

      policy_batch()
      sample["fixed_observation_actor_seconds"] = elapsed(policy_batch, args.device)
      runner.alg.train_mode()

      def rollout() -> None:
        nonlocal obs
        obs = collect_rollout(runner, wrapped, obs, args.steps)

      sample["ppo_rollout_seconds"] = elapsed(rollout, args.device)
      losses = {}

      def update(update_losses=losses) -> None:
        update_losses.update(runner.alg.update())

      sample["ppo_update_seconds"] = elapsed(update, args.device)
      # Validate outside measured loops; no image/checkpoint/disk I/O in them.
      assert all(math.isfinite(float(value)) for value in losses.values())
      for key in obs.keys():
        value = obs[key]
        assert isinstance(value, torch.Tensor)
        assert torch.isfinite(value).all()
      sample["losses"] = losses
      sample["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated(
        args.device
      )
      if repeat:
        samples.append(sample)
    report["median"] = {
      key: statistics.median(sample[key] for sample in samples)
      for key in samples[0]
      if key not in ("repeat", "losses")
    }
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--scene-manifest", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--width", type=int, default=64)
  parser.add_argument("--height", type=int, default=48)
  parser.add_argument("--steps", type=int, default=32)
  parser.add_argument("--warmup-steps", type=int, default=16)
  parser.add_argument("--repeats", type=int, default=3)
  visual = parser.add_mutually_exclusive_group()
  visual.add_argument(
    "--without-gaussians",
    action="store_true",
    help="Keep manifest collision and RGB/depth camera, omit Gaussian assets",
  )
  visual.add_argument(
    "--max-splats",
    type=int,
    help="Per-asset opacity-ranked cap, using the existing Gaussian loader",
  )
  parser.add_argument("--save-capture", action="store_true")
  parser.add_argument("--capture-only", action="store_true")
  parser.add_argument("--spawn-point", help="Named existing manifest spawn")
  parser.add_argument("--view-yaw-degrees", type=float, default=0.0)
  args = parser.parse_args()
  if args.max_splats is not None and args.max_splats < 1:
    parser.error("max-splats must be positive")
  if not math.isfinite(args.view_yaw_degrees):
    parser.error("view-yaw-degrees must be finite")
  if min(args.num_envs, args.width, args.height, args.warmup_steps, args.repeats) < 1:
    parser.error("Counts and dimensions must be positive")
  if args.steps < 4 or args.steps * args.num_envs % 4:
    parser.error("Steps must be >= 4 and steps * num-envs divisible by four")
  if not args.device.startswith("cuda"):
    parser.error("This benchmark requires a CUDA device")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  if args.output.exists():
    parser.error("Output already exists; choose a new path")
  report = {
    "scope": (
      "Static capture only; no throughput or policy evaluation"
      if args.capture_only
      else "Independent pipeline timings, not additive or task benefit"
    ),
    "rollout_and_actor_context": "torch.no_grad",
    "arguments": {
      k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    },
    "versions": {name: version(name) for name in ("mujoco", "warp-lang", "torch")},
  }
  try:
    report["build_identity"] = installed_build_identity()
    report["benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["throughput_helper_sha256"] = hashlib.sha256(
      Path(__file__).with_name("measure_throughput.py").read_bytes()
    ).hexdigest()
    report["gpu"] = torch.cuda.get_device_name(args.device)
    with torch.cuda.device(args.device):
      run(args, report)
    report["status"] = "passed"
  except Exception as exc:
    report.update(status="failed", error=repr(exc))
    raise
  finally:
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
  main()
