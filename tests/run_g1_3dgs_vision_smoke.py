"""Validate G1 depth observations, CNN/PPO updates, and checkpoint roundtrip.

uv run tests/run_g1_3dgs_vision_smoke.py --num-envs 1
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

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scene import apply_scene_manifest
from mjlab.sensor import CameraSensor
from mjlab.tasks.g1_3dgs import G1_3DGS_DEPTH_SENSOR
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

_TASK = "Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth"
_VALIDITY_TASK = _TASK + "-Validity"


def camera_tensor(obs: TensorDict) -> torch.Tensor:
  depth = obs["camera"]
  assert isinstance(depth, torch.Tensor)
  return depth


def check_depth(depth: torch.Tensor, num_envs: int, channels: int = 1) -> None:
  assert depth.shape == (num_envs, channels, 48, 64)
  assert depth.dtype == torch.float32
  assert torch.isfinite(depth).all()
  assert ((depth >= 0) & (depth <= 1)).all()
  image = depth[:, :1].flatten(1)
  assert (image.amax(1) > image.amin(1)).all()
  if channels == 2:
    assert ((depth[:, 1] == 0) | (depth[:, 1] == 1)).all()


def save_depth(depth: torch.Tensor, output: Path, label: str) -> None:
  arrays = depth.detach().cpu().numpy()
  np.save(output / f"{label}.npy", arrays)
  for index, frame in enumerate(arrays[:, 0]):
    # Fixed 0..5m normalization; white is far, black is near.
    Image.fromarray((frame * 255).astype(np.uint8)).save(
      output / f"{label}-env{index}.png"
    )
    if arrays.shape[1] == 2:
      Image.fromarray((arrays[index, 1] * 255).astype(np.uint8)).save(
        output / f"{label}-validity-env{index}.png"
      )


def sample_raw_depth(env: ManagerBasedRlEnv, phase: str, dones: torch.Tensor) -> dict:
  sensor = env.scene[G1_3DGS_DEPTH_SENSOR]
  assert isinstance(sensor, CameraSensor)
  raw = sensor.data.depth
  assert raw is not None
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
  task = _VALIDITY_TASK if args.validity_mask else _TASK
  channels = 2 if args.validity_mask else 1
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
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
    check_depth(camera_tensor(obs), args.num_envs, channels)
    save_depth(camera_tensor(obs), args.output, "reset")
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
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=args.device)
    actor = runner.alg.get_policy()
    report["actor_parameter_count"] = sum(p.numel() for p in actor.parameters())
    cnn_before = {
      name: p.detach().clone()
      for name, p in actor.named_parameters()
      if name.startswith("cnns.")
    }
    assert cnn_before, "Actor has no CNN parameters"
    loss_history = []
    deltas = []
    depth_samples = []
    previous = obs["camera"].clone()
    started = time.perf_counter()
    for iteration in range(args.iterations):
      runner.alg.train_mode()
      with torch.inference_mode():
        for step in range(args.steps):
          actions = runner.alg.act(obs)
          assert actions.shape == (args.num_envs, wrapped.num_actions)
          assert torch.isfinite(actions).all()
          obs, rewards, dones, extras = wrapped.step(actions)
          if args.depth_metrics:
            depth_samples.append(sample_raw_depth(env, "training", dones))
          assert torch.isfinite(rewards).all()
          check_depth(camera_tensor(obs), args.num_envs, channels)
          deltas.append(float((obs["camera"] - previous).abs().max()))
          previous = obs["camera"].clone()
          runner.alg.process_env_step(obs, rewards, dones, extras)
          if step in (0, args.steps - 1):
            save_depth(camera_tensor(obs), args.output, f"iter{iteration}-step{step}")
        runner.alg.compute_returns(obs)
      losses = runner.alg.update()
      assert all(math.isfinite(float(value)) for value in losses.values())
      loss_history.append(losses)
      runner.current_learning_iteration = iteration + 1
    report["rollout_and_update_seconds"] = time.perf_counter() - started
    report["loss_history"] = loss_history
    report["depth_frame_max_abs_deltas"] = deltas
    assert any(delta > 1e-5 for delta in deltas), "Depth appears frozen"
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
    if args.validity_mask:
      with torch.no_grad():
        altered = obs.clone()
        camera = camera_tensor(obs).clone()
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
        check_depth(camera_tensor(obs), args.num_envs, channels)
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
  parser.add_argument("--depth-metrics", action="store_true")
  parser.add_argument("--validity-mask", action="store_true")
  parser.add_argument("--scene-manifest", type=Path)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path, default=Path("/tmp/p6-vision-smoke"))
  args = parser.parse_args()
  if min(args.num_envs, args.iterations) < 1 or args.steps < 4:
    parser.error("Positive environments/iterations and at least four steps required")
  args.output.mkdir(parents=True, exist_ok=True)
  report = {
    "date": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "task": _VALIDITY_TASK if args.validity_mask else _TASK,
    "camera_encoding": "depth+finite-positive-validity-v1"
    if args.validity_mask
    else "depth-zero-as-far-v1",
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
