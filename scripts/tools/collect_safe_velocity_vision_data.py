"""Collect temporal 3DGS observations with collision-truth safety labels.

Example:
  uv run --no-sync python scripts/tools/collect_safe_velocity_vision_data.py \
    --checkpoint logs/rsl_rl/g1_3dgs_safe_velocity_blind/RUN/model_99.pt \
    --output-dir datasets/safe_velocity_vision/both_open_seed0 \
    --scenario both_open --num-envs 16 --steps 1000 --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.velocity.config.g1.safe_velocity import (
  SAFE_VELOCITY_VISION_DATA_TASK_ID,
  unitree_g1_safe_velocity_vision_data_3dgs_env_cfg,
)
from mjlab.tasks.velocity.config.g1.safe_velocity_baseline import (
  SafeVelocityPrivilegedSafetyLabels,
  SafeVelocitySafetyTeacherCfg,
)
from mjlab.tasks.velocity.config.g1.safe_velocity_scenarios import (
  SAFE_VELOCITY_SCENARIO_CATALOG,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.torch import configure_torch_backends


def _checkpoint_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for block in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _prepare_output_dir(path: Path) -> None:
  path.mkdir(parents=True, exist_ok=True)
  if any(path.iterdir()):
    raise FileExistsError(f"Output directory must be empty: {path}")


def _fix_scenario_command(cfg: Any, command: tuple[float, float, float]) -> None:
  twist = cfg.commands["twist"]
  if not isinstance(twist, UniformVelocityCommandCfg):
    raise TypeError("Vision data collection requires UniformVelocityCommandCfg")
  vx, vy, wz = command
  twist.ranges.lin_vel_x = (vx, vx)
  twist.ranges.lin_vel_y = (vy, vy)
  twist.ranges.ang_vel_z = (wz, wz)
  twist.rel_forward_envs = 0.0


def _enable_collection_camera_corruption(cfg: Any) -> None:
  """Augment camera data while keeping frozen locomotion observations deterministic."""
  camera_term = cfg.observations["camera"].terms["depth_validity"]
  camera_term.params["enable_corruption"] = True
  command_term = cfg.observations["actor"].terms["command"]
  command_term.params["debug_print"] = False


def _configure_collection_dynamics(cfg: Any, enable_randomization: bool) -> None:
  """Keep the first offline dataset on nominal frozen-actor dynamics."""
  if enable_randomization:
    return
  for event_name in ("foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(event_name, None)


def _configure_rollout_command(cfg: Any, rollout_command: str) -> None:
  """Select behavior command without changing collision-truth supervision."""
  if rollout_command == "teacher":
    return
  if rollout_command != "nominal":
    raise ValueError(f"Unsupported rollout command: {rollout_command}")
  for group_name in ("actor", "critic"):
    command_term = cfg.observations[group_name].terms["command"]
    command_term.func = envs_mdp.generated_commands
    command_term.params = {"command_name": "twist"}


def _termination_flags(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return active termination flags in manager registration order."""
  return torch.stack(
    tuple(env.termination_manager.get_term(name) for name in _termination_names(env)),
    dim=1,
  )


def _termination_names(env: ManagerBasedRlEnv) -> tuple[str, ...]:
  names = tuple(env.termination_manager.active_terms)
  if not names:
    raise RuntimeError("Vision data collection requires termination terms")
  return names


def _sample_batch(
  env: ManagerBasedRlEnv,
  observations: Any,
  reset_mask: torch.Tensor,
  rollout_command: str,
) -> dict[str, torch.Tensor]:
  label_term = env.observation_manager.get_term_cfg("safety_teacher", "labels").func
  if not isinstance(label_term, SafeVelocityPrivilegedSafetyLabels):
    raise TypeError("Vision data task did not build the safety-teacher labels")
  if label_term.last_output is None:
    raise RuntimeError("Safety-teacher output is unavailable")
  nominal = env.command_manager.get_command("twist")
  if not isinstance(nominal, torch.Tensor):
    raise RuntimeError("Nominal velocity command is unavailable")
  robot = env.scene["robot"]
  if not isinstance(robot, Entity):
    raise TypeError("Vision data collection requires the robot entity")
  measured = torch.cat(
    (
      robot.data.root_link_lin_vel_b[:, :2],
      robot.data.root_link_ang_vel_b[:, 2:3],
    ),
    dim=1,
  )
  safe_command = label_term.last_output.command
  executed_command = nominal if rollout_command == "nominal" else safe_command
  return {
    "env_index": torch.arange(env.num_envs, dtype=torch.int32),
    "camera": observations["camera"]
    .detach()
    .to(device="cpu", dtype=torch.float16)
    .clone(),
    "teacher_labels": observations["safety_teacher"].detach().cpu().clone(),
    "nominal_command": nominal.detach().cpu().clone(),
    "safe_command": safe_command.detach().cpu().clone(),
    "executed_command": executed_command.detach().cpu().clone(),
    "measured_velocity": measured.detach().cpu().clone(),
    "base_height": robot.data.root_link_pos_w[:, 2].detach().cpu().clone(),
    "episode_step": env.episode_length_buf.detach().cpu().clone(),
    "reset": reset_mask.detach().to(device="cpu", dtype=torch.bool).clone(),
    "termination": _termination_flags(env).detach().cpu().clone(),
  }


def _flush_chunk(
  output_dir: Path,
  chunk_index: int,
  buffer: dict[str, list[torch.Tensor]],
) -> tuple[Path, int]:
  chunk = {name: torch.cat(values, dim=0) for name, values in buffer.items()}
  sample_count = int(chunk["camera"].shape[0])
  final_path = output_dir / f"chunk_{chunk_index:06d}.pt"
  temporary_path = final_path.with_suffix(".pt.tmp")
  torch.save(chunk, temporary_path)
  temporary_path.replace(final_path)
  buffer.clear()
  return final_path, sample_count


def collect(args: argparse.Namespace) -> dict[str, Any]:
  configure_torch_backends()
  checkpoint = args.checkpoint.expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
  output_dir = args.output_dir.expanduser().resolve()
  _prepare_output_dir(output_dir)

  scenario = SAFE_VELOCITY_SCENARIO_CATALOG.scenario(args.scenario)
  # The frozen locomotion actor was accepted under deterministic play observations.
  # Keep that interface stable, then independently restore camera augmentation for
  # the vision student dataset.
  env_cfg = unitree_g1_safe_velocity_vision_data_3dgs_env_cfg(
    play=True,
    scenario_id=scenario.id,
    isolate_teacher_rays=not args.legacy_shared_render_context,
  )
  _enable_collection_camera_corruption(env_cfg)
  _configure_collection_dynamics(env_cfg, args.enable_dynamics_randomization)
  _configure_rollout_command(env_cfg, args.rollout_command)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  _fix_scenario_command(env_cfg, scenario.nominal_command)
  agent_cfg = load_rl_cfg(SAFE_VELOCITY_VISION_DATA_TASK_ID)
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  chunks: list[dict[str, Any]] = []
  total_samples = 0
  alpha_sum = 0.0
  stop_samples = 0
  completed_steps = 0
  termination_counts: dict[str, int] = {}
  buffer: dict[str, list[torch.Tensor]] = defaultdict(list)
  buffered_samples = 0
  try:
    if not args.legacy_shared_render_context:
      context = env.scene.sensor_context
      if context is None:
        raise RuntimeError("Visual collection requires a camera render context")
      geom_ids = context.render_context.enabled_geom_ids.numpy()
      actual_groups = set(env.sim.mj_model.geom_group[geom_ids].tolist())
      if not actual_groups <= {1, 2}:
        raise RuntimeError(
          f"Camera BVH contains non-visual geom groups: {sorted(actual_groups)}"
        )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    observations = wrapped.get_observations()
    reset_mask = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
    termination_names = _termination_names(env)
    termination_counts = {name: 0 for name in termination_names}

    with torch.inference_mode():
      for _ in range(args.steps):
        batch = _sample_batch(env, observations, reset_mask, args.rollout_command)
        completed_steps += 1
        for name, value in batch.items():
          buffer[name].append(value)
        batch_size = int(batch["camera"].shape[0])
        buffered_samples += batch_size
        total_samples += batch_size
        alpha_sum += float(batch["teacher_labels"][:, 2].sum().item())
        stop_samples += int(batch["teacher_labels"][:, 3].sum().item())
        for index, name in enumerate(termination_names):
          termination_counts[name] += int(batch["termination"][:, index].sum().item())

        if buffered_samples >= args.chunk_size:
          path, sample_count = _flush_chunk(output_dir, len(chunks), buffer)
          chunks.append({"file": path.name, "samples": sample_count})
          buffered_samples = 0

        if (
          args.stop_after_stop_samples is not None
          and stop_samples >= args.stop_after_stop_samples
        ):
          break

        actions = policy(observations)
        observations, _, dones, _ = wrapped.step(actions)
        reset_mask = dones.bool()

    if buffered_samples:
      path, sample_count = _flush_chunk(output_dir, len(chunks), buffer)
      chunks.append({"file": path.name, "samples": sample_count})
  finally:
    env.close()

  metadata: dict[str, Any] = {
    "schema_version": 2,
    "visual_input": {
      "sources": ["3dgs", "visible_mesh"]
      + (["collision_only"] if args.legacy_shared_render_context else []),
      "camera_geom_groups": (
        [1, 2, 3] if args.legacy_shared_render_context else [1, 2]
      ),
      "teacher_geom_groups": [1, 3],
      "teacher_query": (
        "shared_render_bvh"
        if args.legacy_shared_render_context
        else "direct_collision_rays"
      ),
      "collision_only_in_camera": args.legacy_shared_render_context,
    },
    "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "task_id": SAFE_VELOCITY_VISION_DATA_TASK_ID,
    "scenario": asdict(scenario),
    "seed": args.seed,
    "device": device,
    "num_envs": args.num_envs,
    "steps": completed_steps,
    "requested_steps": args.steps,
    "stop_sample_target": args.stop_after_stop_samples,
    "rollout_command": args.rollout_command,
    "dynamics_randomization": args.enable_dynamics_randomization,
    "samples": total_samples,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": _checkpoint_sha256(checkpoint),
    "camera_shape": [7, 2, 48, 64],
    "camera_dtype": "float16",
    "teacher_label_order": [
      "normalized_clearance",
      "validity",
      "alpha",
      "stop",
    ],
    "teacher_thresholds": asdict(SafeVelocitySafetyTeacherCfg()),
    "termination_order": list(termination_counts),
    "termination_counts": termination_counts,
    "mean_alpha": alpha_sum / total_samples,
    "stop_fraction": stop_samples / total_samples,
    "chunks": chunks,
  }
  manifest_path = output_dir / "manifest.json"
  manifest_path.write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  return metadata


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--scenario",
    choices=tuple(item.id for item in SAFE_VELOCITY_SCENARIO_CATALOG.scenarios),
    default="both_open",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--chunk-size", type=int, default=1024)
  parser.add_argument(
    "--stop-after-stop-samples",
    type=int,
    default=None,
    help="Stop before the next action once this many positive stop labels exist.",
  )
  parser.add_argument("--device", type=str, default=None)
  parser.add_argument(
    "--legacy-shared-render-context",
    action="store_true",
    help="Reproduce old collision-assisted images; hidden collision enters depth.",
  )
  parser.add_argument(
    "--rollout-command",
    choices=("teacher", "nominal"),
    default="teacher",
    help="Command executed by the frozen locomotion actor; labels always use teacher.",
  )
  parser.add_argument(
    "--enable-dynamics-randomization",
    action="store_true",
    help="Enable startup friction, encoder-bias, and base-COM randomization.",
  )
  args = parser.parse_args()
  if args.num_envs < 1:
    parser.error("--num-envs must be positive")
  if args.steps < 1:
    parser.error("--steps must be positive")
  if args.chunk_size < 1:
    parser.error("--chunk-size must be positive")
  if args.stop_after_stop_samples is not None and args.stop_after_stop_samples < 1:
    parser.error("--stop-after-stop-samples must be positive")
  return args


def main() -> None:
  metadata = collect(_parse_args())
  print(
    f"Collected {metadata['samples']} samples in {len(metadata['chunks'])} chunks; "
    f"mean_alpha={metadata['mean_alpha']:.4f}, "
    f"stop_fraction={metadata['stop_fraction']:.4f}, "
    f"terminations={metadata['termination_counts']}, "
    f"steps={metadata['steps']}/{metadata['requested_steps']}"
  )


if __name__ == "__main__":
  main()
