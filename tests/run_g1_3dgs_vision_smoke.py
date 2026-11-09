"""Validate G1 vision observations, CNN/PPO updates, and checkpoint roundtrip.

uv run tests/run_g1_3dgs_vision_smoke.py --num-envs 1 --fixed-base-pose
uv run tests/run_g1_3dgs_vision_smoke.py --num-envs 2 --scene-manifest PATH
This is a pipeline check, not a trained locomotion or visual-benefit result.
"""

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from PIL import Image
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scene import apply_scene_manifest
from mjlab.sensor import CameraSensor, CameraSensorCfg
from mjlab.tasks.g1_3dgs import G1_3DGS_DEPTH_SENSOR
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

_TASK = "Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth"
_VALIDITY_TASK = _TASK + "-Validity"
_HISTORY_TASK = _VALIDITY_TASK + "-History"
_ROBUST_TASK = _HISTORY_TASK + "-Robust"
_ASYMMETRIC_TASK = _ROBUST_TASK + "-Asymmetric"


def camera_tensor(obs: TensorDict) -> torch.Tensor:
  depth = obs["camera"]
  assert isinstance(depth, torch.Tensor)
  return depth


def latest_camera_frame(camera: torch.Tensor) -> torch.Tensor:
  if camera.ndim == 5:
    return camera[:, -1]
  return camera


def check_depth(
  depth: torch.Tensor,
  num_envs: int,
  height: int,
  width: int,
  channels: int = 1,
  history_length: int = 0,
) -> None:
  expected = (
    (num_envs, history_length, channels, height, width)
    if history_length > 0
    else (num_envs, channels, height, width)
  )
  assert depth.shape == expected
  assert depth.dtype == torch.float32
  assert torch.isfinite(depth).all()
  assert ((depth >= 0) & (depth <= 1)).all()
  latest = latest_camera_frame(depth)
  if channels == 2:
    assert ((latest[:, 1] == 0) | (latest[:, 1] == 1)).all()


def raw_depth_tensor(env: ManagerBasedRlEnv) -> torch.Tensor:
  sensor = env.scene[G1_3DGS_DEPTH_SENSOR]
  assert isinstance(sensor, CameraSensor)
  raw = sensor.data.depth
  assert raw is not None
  return raw


def policy_camera_cfg(cfg: ManagerBasedRlEnvCfg) -> CameraSensorCfg:
  scene = cfg.scene
  sensor = next(
    sensor for sensor in scene.sensors or () if sensor.name == G1_3DGS_DEPTH_SENSOR
  )
  assert isinstance(sensor, CameraSensorCfg)
  return sensor


def check_raw_depth_signal(env: ManagerBasedRlEnv) -> None:
  raw = raw_depth_tensor(env)
  assert torch.isfinite(raw).all()
  flattened = raw.flatten(1)
  assert (flattened.amax(1) > flattened.amin(1)).all(), (
    "Raw renderer depth is spatially constant"
  )


def save_snapshot(
  env: ManagerBasedRlEnv,
  depth: torch.Tensor,
  output: Path,
  label: str,
  cutoff_distance: float = 5.0,
) -> None:
  """Save aligned RGB, raw depth, and policy-ready depth for comparison."""
  full_arrays = depth.detach().cpu().numpy()
  np.save(output / f"{label}-policy-depth.npy", full_arrays)
  arrays = latest_camera_frame(depth).detach().cpu().numpy()
  sensor = env.scene[G1_3DGS_DEPTH_SENSOR]
  assert isinstance(sensor, CameraSensor)
  raw_depth = sensor.data.depth
  rgb = sensor.data.rgb
  assert raw_depth is not None
  assert rgb is not None
  raw_depth_arrays = raw_depth.detach().cpu().numpy()[..., 0]
  rgb_arrays = rgb.detach().cpu().numpy()
  np.save(output / f"{label}-raw-depth.npy", raw_depth_arrays)
  np.save(output / f"{label}-rgb.npy", rgb_arrays)
  for index, policy_frame in enumerate(arrays[:, 0]):
    raw_frame = raw_depth_arrays[index]
    display_depth = np.where(
      np.isfinite(raw_frame) & (raw_frame > 0), raw_frame, cutoff_distance
    )
    display_depth = np.clip(display_depth / cutoff_distance, 0.0, 1.0)
    Image.fromarray(rgb_arrays[index]).save(output / f"{label}-env{index}-rgb.png")
    Image.fromarray((display_depth * 255).astype(np.uint8)).save(
      output / f"{label}-env{index}-raw-depth.png"
    )
    Image.fromarray((policy_frame * 255).astype(np.uint8)).save(
      output / f"{label}-env{index}-policy-depth.png"
    )
    if arrays.shape[1] == 2:
      Image.fromarray((arrays[index, 1] * 255).astype(np.uint8)).save(
        output / f"{label}-env{index}-validity.png"
      )


def sample_raw_depth(env: ManagerBasedRlEnv, phase: str, dones: torch.Tensor) -> dict:
  raw = raw_depth_tensor(env)
  values = raw.detach().cpu().numpy().reshape(env.num_envs, -1)
  assert np.isfinite(values).all()
  return {
    "phase": phase,
    "post_step_done": dones.cpu().tolist(),
    "missing_fraction": (values == 0).mean(axis=1).tolist(),
    "positive_near_fraction": ((values > 0) & (values <= 0.05)).mean(axis=1).tolist(),
    "beyond_cutoff_fraction": (values >= 5).mean(axis=1).tolist(),
    "in_range_fraction": ((values > 0.05) & (values < 5)).mean(axis=1).tolist(),
  }


def run(args: argparse.Namespace, report: dict) -> None:
  configure_torch_backends()
  if args.asymmetric_critic:
    task = _ASYMMETRIC_TASK
  elif args.robust_history:
    task = _ROBUST_TASK
  elif args.temporal_history:
    task = _HISTORY_TASK
  elif args.validity_mask:
    task = _VALIDITY_TASK
  else:
    task = _TASK
  channels = 1 if task == _TASK else 2
  cfg = load_env_cfg(task)
  history_length = cfg.observations["camera"].history_length or 0
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  if args.fixed_base_pose:
    reset_base = cfg.events["reset_base"]
    pose_range = reset_base.params["pose_range"]
    assert isinstance(pose_range, dict)
    pose_range.update({axis: (0.0, 0.0) for axis in ("x", "y", "yaw")})
  camera_cfg = policy_camera_cfg(cfg)
  camera_cfg.width = args.width
  camera_cfg.height = args.height
  camera_cfg.data_types = ("rgb", "depth")
  report["camera_capture"] = {
    "width": args.width,
    "height": args.height,
    "data_types": list(camera_cfg.data_types),
    "depth_visualization_cutoff_m": 5.0,
  }
  if args.scene_manifest is not None:
    apply_scene_manifest(cfg.scene, args.scene_manifest)
    report["manifest_sha256"] = hashlib.sha256(
      args.scene_manifest.read_bytes()
    ).hexdigest()
  agent = load_rl_cfg(task)
  agent.seed = args.seed
  agent.upload_model = False
  agent.logger = "tensorboard"
  agent.num_steps_per_env = args.steps
  # Keep the real PPO settings; a smoke run changes only rollout/iteration counts.
  report["agent_cfg"] = asdict(agent)
  report["environment_seed"] = args.seed
  assert cfg.scene.gaussian_splats is not None
  report["scene_assets"] = asdict(cfg.scene.gaussian_splats)
  if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
  started = time.perf_counter()
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    report["initialization_seconds"] = time.perf_counter() - started
    report["cuda_graph_enabled"] = env.sim.use_cuda_graph
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    obs = wrapped.get_observations()
    check_raw_depth_signal(env)
    check_depth(
      camera_tensor(obs),
      args.num_envs,
      args.height,
      args.width,
      channels,
      history_length,
    )
    save_snapshot(env, camera_tensor(obs), args.output, "reset")
    report["observation_shapes"] = {k: list(v.shape) for k, v in obs.items()}
    if args.num_envs > 1:
      before_qpos = env.sim.data.qpos[1:].clone()
      before_depth = obs["camera"][1:].clone()
      env.reset(env_ids=torch.tensor([0], device=env.device))
      after = wrapped.get_observations()
      torch.testing.assert_close(env.sim.data.qpos[1:], before_qpos, rtol=0, atol=0)
      torch.testing.assert_close(after["camera"][1:], before_depth, rtol=0, atol=0)
      report["partial_reset_other_world_state_and_depth"] = "passed"
      obs = after
    previous_raw = raw_depth_tensor(env).clone()
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=args.device)
    actor = runner.alg.get_policy()
    report["actor_parameter_count"] = sum(p.numel() for p in actor.parameters())
    runner.alg.eval_mode()
    with torch.inference_mode():
      for _ in range(10):
        actor(obs)
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      inference_started = time.perf_counter()
      for _ in range(100):
        actor(obs)
      if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    report["actor_inference"] = {
      "batches": 100,
      "batch_size": args.num_envs,
      "total_seconds": inference_seconds,
      "milliseconds_per_batch": inference_seconds * 10.0,
      "environment_observations_per_second": 100 * args.num_envs / inference_seconds,
    }
    cnn_before = {
      name: p.detach().clone()
      for name, p in actor.named_parameters()
      if name.startswith("cnns.")
    }
    assert cnn_before, "Actor has no CNN parameters"
    loss_history = []
    deltas = []
    raw_deltas = []
    depth_samples = []
    previous = obs["camera"].clone()
    rollout_seconds = 0.0
    update_seconds = 0.0
    started = time.perf_counter()
    for iteration in range(args.iterations):
      runner.alg.train_mode()
      rollout_started = time.perf_counter()
      with torch.inference_mode():
        for step in range(args.steps):
          actions = runner.alg.act(obs)
          assert actions.shape == (args.num_envs, wrapped.num_actions)
          assert torch.isfinite(actions).all()
          obs, rewards, dones, extras = wrapped.step(actions)
          if args.depth_metrics:
            depth_samples.append(sample_raw_depth(env, "training", dones))
          assert torch.isfinite(rewards).all()
          check_depth(
            camera_tensor(obs),
            args.num_envs,
            args.height,
            args.width,
            channels,
            history_length,
          )
          deltas.append(float((obs["camera"] - previous).abs().max()))
          raw = raw_depth_tensor(env)
          raw_deltas.append(float((raw - previous_raw).abs().max()))
          previous = obs["camera"].clone()
          previous_raw = raw.clone()
          runner.alg.process_env_step(obs, rewards, dones, extras)
          if step in (0, args.steps - 1):
            save_snapshot(
              env,
              camera_tensor(obs),
              args.output,
              f"iter{iteration}-step{step}",
            )
        runner.alg.compute_returns(obs)
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      rollout_seconds += time.perf_counter() - rollout_started
      update_started = time.perf_counter()
      losses = runner.alg.update()
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      update_seconds += time.perf_counter() - update_started
      assert all(math.isfinite(float(value)) for value in losses.values())
      loss_history.append(losses)
      runner.current_learning_iteration = iteration + 1
    report["rollout_and_update_seconds"] = time.perf_counter() - started
    report["rollout_seconds"] = rollout_seconds
    report["update_seconds"] = update_seconds
    report["rollout_environment_steps_per_second"] = (
      args.num_envs * args.steps * args.iterations / rollout_seconds
    )
    report["loss_history"] = loss_history
    report["depth_frame_max_abs_deltas"] = deltas
    report["raw_depth_frame_max_abs_deltas"] = raw_deltas
    assert any(delta > 1e-5 for delta in deltas), "Depth appears frozen"
    assert any(delta > 1e-5 for delta in raw_deltas), (
      "Raw renderer depth appears frozen"
    )
    report["actor_cnn_parameter_max_abs_change"] = max(
      float((p.detach() - cnn_before[name]).abs().max())
      for name, p in actor.named_parameters()
      if name in cnn_before
    )
    assert report["actor_cnn_parameter_max_abs_change"] > 0
    for model in (actor, runner.alg.critic):
      assert all(torch.isfinite(p).all() for p in model.parameters())
      assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
      )
    runner.alg.eval_mode()
    with torch.no_grad():
      expected = actor(obs).clone()
      altered = obs.clone()
      altered["camera"] = torch.zeros_like(camera_tensor(obs))
      sensitivity = float((actor(altered) - expected).abs().max())
    report["camera_zeroing_action_max_abs_delta"] = sensitivity
    assert sensitivity > 0, "Actor ignores camera tensor structurally"
    if channels == 2:
      with torch.no_grad():
        altered = obs.clone()
        camera = camera_tensor(obs).clone()
        if camera.ndim == 5:
          camera[:, :, 1] = 1 - camera[:, :, 1]
        else:
          camera[:, 1] = 1 - camera[:, 1]
        altered["camera"] = camera
        mask_sensitivity = float((actor(altered) - expected).abs().max())
      report["validity_flipping_action_max_abs_delta"] = mask_sensitivity
      assert mask_sensitivity > 0, "Actor ignores validity channel structurally"
    checkpoint = args.output / "vision_smoke.pt"
    runner.save(
      str(checkpoint),
      infos={
        "task": task,
        "camera_encoding": report["camera_encoding"],
        "camera_channels": channels,
      },
    )
    restored = MjlabOnPolicyRunner(wrapped, asdict(agent), device=args.device)
    loaded_info = restored.load(str(checkpoint), map_location=args.device)
    assert loaded_info["camera_encoding"] == report["camera_encoding"]
    assert loaded_info["camera_channels"] == channels
    restored.alg.eval_mode()
    with torch.no_grad():
      torch.testing.assert_close(
        restored.alg.get_policy()(obs), expected, rtol=0, atol=0
      )
      torch.testing.assert_close(
        restored.alg.critic(obs), runner.alg.critic(obs), rtol=0, atol=0
      )
    assert restored.alg.optimizer.state, "Optimizer state was not restored"
    assert restored.current_learning_iteration == args.iterations
    report["checkpoint_actor_critic_optimizer_roundtrip"] = "passed"
    restored.alg.train_mode()
    with torch.inference_mode():
      for _ in range(args.steps):
        actions = restored.alg.act(obs)
        assert torch.isfinite(actions).all()
        obs, rewards, dones, extras = wrapped.step(actions)
        if args.depth_metrics:
          depth_samples.append(sample_raw_depth(env, "resumed", dones))
        check_depth(
          camera_tensor(obs),
          args.num_envs,
          args.height,
          args.width,
          channels,
          history_length,
        )
        assert torch.isfinite(rewards).all()
        restored.alg.process_env_step(obs, rewards, dones, extras)
      restored.alg.compute_returns(obs)
    resumed_losses = restored.alg.update()
    assert all(math.isfinite(float(value)) for value in resumed_losses.values())
    report["resumed_update_losses"] = resumed_losses
    report["checkpoint_resume_update"] = "passed"
    report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if args.depth_metrics:
      metrics_path = args.output / "rollout_depth.json"
      metrics_path.write_text(json.dumps(depth_samples, indent=2) + "\n")
      report["rollout_depth_metrics"] = {
        "samples_per_world": len(depth_samples),
        "post_step_done_count": np.asarray(
          [row["post_step_done"] for row in depth_samples]
        )
        .sum(axis=0)
        .tolist(),
        "fraction_summary_by_world": {
          key: {
            "min": np.asarray([row[key] for row in depth_samples]).min(axis=0).tolist(),
            "mean": np.asarray([row[key] for row in depth_samples])
            .mean(axis=0)
            .tolist(),
            "max": np.asarray([row[key] for row in depth_samples]).max(axis=0).tolist(),
          }
          for key in (
            "missing_fraction",
            "positive_near_fraction",
            "beyond_cutoff_fraction",
            "in_range_fraction",
          )
        },
        "scope": "Early PPO policy; post-step images include auto-reset frames; no self-occlusion attribution",
        "timing_note": "CPU depth sampling adds diagnostic overhead to rollout timing",
      }
    if torch.cuda.is_available():
      torch.cuda.synchronize()
      report["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--steps", type=int, default=24)
  parser.add_argument("--iterations", type=int, default=2)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--width", type=int, default=320)
  parser.add_argument("--height", type=int, default=240)
  parser.add_argument("--fixed-base-pose", action="store_true")
  parser.add_argument("--depth-metrics", action="store_true")
  modes = parser.add_mutually_exclusive_group()
  modes.add_argument("--validity-mask", action="store_true")
  modes.add_argument("--temporal-history", action="store_true")
  modes.add_argument("--robust-history", action="store_true")
  modes.add_argument("--asymmetric-critic", action="store_true")
  parser.add_argument("--scene-manifest", type=Path)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path, default=Path("/tmp/p6-vision-smoke"))
  args = parser.parse_args()
  if min(args.num_envs, args.iterations, args.width, args.height) < 1:
    parser.error("Environments, iterations, width, and height must be positive")
  if args.steps < 4:
    parser.error("At least four steps are required")
  args.output.mkdir(parents=True, exist_ok=True)
  report = {
    "date": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "task": (
      _ASYMMETRIC_TASK
      if args.asymmetric_critic
      else _ROBUST_TASK
      if args.robust_history
      else _HISTORY_TASK
      if args.temporal_history
      else _VALIDITY_TASK
      if args.validity_mask
      else _TASK
    ),
    "camera_encoding": (
      "strided-depth+finite-positive-validity-robust-v1"
      if args.robust_history or args.asymmetric_critic
      else "temporal-depth+finite-positive-validity-v1"
      if args.temporal_history
      else "depth+finite-positive-validity-v1"
      if args.validity_mask
      else "depth-zero-as-far-v1"
    ),
    "mujoco": version("mujoco"),
    "rsl_rl": version("rsl-rl-lib"),
    "arguments": {
      key: str(value) if isinstance(value, Path) else value
      for key, value in vars(args).items()
    },
    "scope": "Pipeline smoke only; not visual task performance or convergence",
  }
  try:
    run(args, report)
    report["status"] = "passed"
  except Exception as exc:
    report["status"] = "failed"
    report["error"] = repr(exc)
    raise
  finally:
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
  main()
