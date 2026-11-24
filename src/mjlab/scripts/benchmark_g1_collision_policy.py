"""Jointly benchmark collision geometry, G1 tracking, contacts, and throughput."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from xml.etree import ElementTree as ET

import mujoco_warp as mjwarp
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.metrics import (
  compute_ee_orientation_error,
  compute_ee_position_error,
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class CollisionVariant:
  """One collision representation and its precomputed geometry evidence."""

  name: str
  collision_mjcf: Path | None
  keep_plane: bool
  geometry: dict[str, object]


@dataclass(frozen=True)
class BenchmarkSettings:
  """Settings held constant across collision variants."""

  task_id: str
  checkpoint: Path
  motion: Path
  device: str
  seeds: tuple[int, ...]
  warmup_steps: int
  max_steps: int
  nconmax: int
  njmax: int
  observation_corruption: bool


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _resolve_path(value: str, *, relative_to: Path) -> Path:
  path = Path(value).expanduser()
  if not path.is_absolute():
    path = relative_to / path
  return path.resolve()


def load_variant_manifest(path: Path) -> list[CollisionVariant]:
  """Load a versioned variant manifest, resolving paths relative to the manifest."""
  manifest_path = path.expanduser().resolve()
  payload = json.loads(manifest_path.read_text())
  if not isinstance(payload, dict) or payload.get("schema_version") != 1:
    raise ValueError("Variant manifest must be an object with schema_version 1.")
  raw_variants = payload.get("variants")
  if not isinstance(raw_variants, list) or not raw_variants:
    raise ValueError("Variant manifest must contain a non-empty variants list.")

  variants: list[CollisionVariant] = []
  seen_names: set[str] = set()
  for index, raw in enumerate(raw_variants):
    if not isinstance(raw, dict):
      raise ValueError(f"Variant {index} must be an object.")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
      raise ValueError(f"Variant {index} must have a non-empty string name.")
    if name in seen_names:
      raise ValueError(f"Duplicate variant name: {name}")
    seen_names.add(name)

    raw_mjcf = raw.get("collision_mjcf")
    if raw_mjcf is not None and not isinstance(raw_mjcf, str):
      raise ValueError(f"Variant '{name}' collision_mjcf must be a string or null.")
    collision_mjcf = (
      _resolve_path(raw_mjcf, relative_to=manifest_path.parent)
      if raw_mjcf is not None
      else None
    )
    if collision_mjcf is not None and not collision_mjcf.is_file():
      raise FileNotFoundError(
        f"Variant '{name}' collision MJCF not found: {collision_mjcf}"
      )
    keep_plane = raw.get("keep_plane", False)
    if not isinstance(keep_plane, bool):
      raise ValueError(f"Variant '{name}' keep_plane must be a boolean.")
    if collision_mjcf is None and keep_plane:
      raise ValueError(f"Variant '{name}' cannot keep a plane without an MJCF.")
    geometry = raw.get("geometry", {})
    if not isinstance(geometry, dict):
      raise ValueError(f"Variant '{name}' geometry must be an object.")
    variants.append(
      CollisionVariant(
        name=name,
        collision_mjcf=collision_mjcf,
        keep_plane=keep_plane,
        geometry=geometry,
      )
    )
  return variants


def collision_bundle_hashes(path: Path) -> dict[str, object]:
  """Hash an MJCF and every local file asset referenced by it."""
  bundle_root = path.resolve().parent
  root = ET.parse(path).getroot()
  if root.find(".//include") is not None:
    raise ValueError("Collision benchmark manifests do not support MJCF includes.")
  compiler = root.find("compiler")
  asset_dir = Path(compiler.get("assetdir", "")) if compiler is not None else Path()
  mesh_dir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
  texture_dir = Path(compiler.get("texturedir", "")) if compiler is not None else Path()
  dependencies: list[dict[str, str]] = []
  for element in root.iter():
    file_name = element.get("file")
    if file_name is None:
      continue
    directory = asset_dir
    if element.tag == "mesh" and mesh_dir != Path():
      directory = mesh_dir
    elif element.tag == "texture" and texture_dir != Path():
      directory = texture_dir
    dependency = (bundle_root / directory / file_name).resolve()
    try:
      relative_path = dependency.relative_to(bundle_root)
    except ValueError as exc:
      raise ValueError(f"MJCF asset must stay inside its bundle: {file_name}") from exc
    if not dependency.is_file():
      raise FileNotFoundError(f"MJCF dependency not found: {dependency}")
    dependencies.append(
      {"path": relative_path.as_posix(), "sha256": _sha256(dependency)}
    )
  dependencies.sort(key=lambda item: item["path"])
  return {
    "path": str(path.resolve()),
    "sha256": _sha256(path),
    "dependencies": dependencies,
  }


def _sync_device(device: str) -> None:
  if device.startswith("cuda") and torch.cuda.is_available():
    torch.cuda.synchronize(torch.device(device))


def _sample_stats(values: list[float]) -> dict[str, float | int | None]:
  if not values:
    return {"mean": None, "max": None, "samples": 0}
  return {
    "mean": sum(values) / len(values),
    "max": max(values),
    "samples": len(values),
  }


def _overflow_stats(values: list[float]) -> dict[str, object]:
  bitmasks = [int(value) for value in values]
  bitmask_or = 0
  for bitmask in bitmasks:
    bitmask_or |= bitmask
  flags = [flag.name for flag in mjwarp.OverflowType if bitmask_or & int(flag.value)]
  return {
    "any": bitmask_or != 0,
    "bitmask_or": bitmask_or,
    "flags": flags,
    "nonzero_steps": sum(bitmask != 0 for bitmask in bitmasks),
    "samples": len(bitmasks),
  }


def _configure_env(
  variant: CollisionVariant,
  settings: BenchmarkSettings,
  *,
  seed: int,
):
  env_cfg = load_env_cfg(settings.task_id, play=False)
  motion_cfg = env_cfg.commands.get("motion")
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise ValueError(f"Task '{settings.task_id}' is not a tracking task.")
  motion_cfg.motion_file = str(settings.motion)
  motion_cfg.sampling_mode = "start"
  motion_cfg.debug_vis = False
  env_cfg.seed = seed
  env_cfg.auto_reset = False
  env_cfg.scene.num_envs = 1
  env_cfg.observations["actor"].enable_corruption = settings.observation_corruption
  env_cfg.events.pop("push_robot", None)
  env_cfg.sim.nconmax = max(env_cfg.sim.nconmax or 0, settings.nconmax)
  env_cfg.sim.njmax = max(env_cfg.sim.njmax or 0, settings.njmax)

  if variant.collision_mjcf is not None:
    terrain_cfg = env_cfg.scene.terrain
    if terrain_cfg is None:
      env_cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="mjcf",
        mjcf_path=str(variant.collision_mjcf),
        mjcf_keep_plane=variant.keep_plane,
      )
    else:
      env_cfg.scene.terrain = replace(
        terrain_cfg,
        terrain_type="mjcf",
        terrain_generator=None,
        mjcf_path=str(variant.collision_mjcf),
        mjcf_keep_plane=variant.keep_plane,
      )
  return env_cfg


def _run_one(
  variant: CollisionVariant,
  settings: BenchmarkSettings,
  *,
  seed: int,
) -> dict[str, object]:
  env_cfg = _configure_env(variant, settings, seed=seed)
  agent_cfg = load_rl_cfg(settings.task_id)
  init_start = time.perf_counter()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=settings.device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  environment_init_seconds = time.perf_counter() - init_start

  try:
    runner_cls = load_runner_cls(settings.task_id) or MjlabOnPolicyRunner
    load_start = time.perf_counter()
    runner = runner_cls(env, asdict(agent_cfg), device=settings.device)
    runner.load(
      str(settings.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=settings.device,
    )
    policy = runner.get_inference_policy(device=settings.device)
    policy_load_seconds = time.perf_counter() - load_start

    command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
    ee_body_names = tuple(env_cfg.terminations["ee_body_pos"].params["body_names"])
    metric_sums = {
      "mpkpe": 0.0,
      "r_mpkpe": 0.0,
      "joint_vel_error": 0.0,
      "ee_pos_error": 0.0,
      "ee_ori_error": 0.0,
    }
    active_steps = 0
    contacts: list[float] = []
    constraints: list[float] = []
    solver_iterations: list[float] = []
    overflow: list[float] = []
    terminated = False
    truncated = False
    termination_terms: list[str] = []
    elapsed_step_seconds = 0.0
    executed_steps = 0
    obs = env.get_observations()

    for _ in range(settings.warmup_steps):
      with torch.no_grad():
        actions = policy(obs)
      obs, _, warmup_dones, _ = env.step(actions)
      if bool(warmup_dones[0].item()):
        obs, _ = env.reset()
    obs, _ = env.reset()

    for _ in range(settings.max_steps):
      ref = SimpleNamespace(
        num_envs=command.num_envs,
        device=command.device,
        cfg=command.cfg,
        body_pos_w=command.body_pos_w.clone(),
        body_pos_relative_w=command.body_pos_relative_w.clone(),
        body_quat_relative_w=command.body_quat_relative_w.clone(),
        joint_vel=command.joint_vel.clone(),
      )
      _sync_device(settings.device)
      step_start = time.perf_counter()
      with torch.no_grad():
        actions = policy(obs)
      obs, _, dones, _ = env.step(actions)
      _sync_device(settings.device)
      elapsed_step_seconds += time.perf_counter() - step_start
      executed_steps += 1

      done = bool(dones[0].item())
      ref.robot_body_pos_w = command.robot_body_pos_w
      ref.robot_body_quat_w = command.robot_body_quat_w
      ref.robot_joint_vel = command.robot_joint_vel
      ref_command = cast(MotionCommand, ref)
      metric_sums["mpkpe"] += float(compute_mpkpe(ref_command)[0].item())
      metric_sums["r_mpkpe"] += float(
        compute_root_relative_mpkpe(ref_command)[0].item()
      )
      metric_sums["joint_vel_error"] += float(
        compute_joint_velocity_error(ref_command)[0].item()
      )
      metric_sums["ee_pos_error"] += float(
        compute_ee_position_error(ref_command, ee_body_names)[0].item()
      )
      metric_sums["ee_ori_error"] += float(
        compute_ee_orientation_error(ref_command, ee_body_names)[0].item()
      )
      active_steps += 1

      sim_data = env.unwrapped.sim.data
      contacts.append(float(sim_data.nacon[0].item()))
      constraints.append(float(sim_data.nefc[0].item()))
      solver_iterations.append(float(sim_data.solver_niter[0].sum().item()))
      overflow.append(float(sim_data.overflow[0].item()))

      if done:
        terminated = bool(env.unwrapped.termination_manager.terminated[0].item())
        truncated = bool(env.unwrapped.termination_manager.time_outs[0].item())
        termination_terms = [
          name
          for name in env.unwrapped.termination_manager.active_terms
          if bool(env.unwrapped.termination_manager.get_term(name)[0].item())
        ]
        break

    denominator = max(active_steps, 1)
    tracking = {name: value / denominator for name, value in metric_sums.items()}
    control_sps = executed_steps / elapsed_step_seconds
    physics_sps = executed_steps * env_cfg.decimation / elapsed_step_seconds
    return {
      "seed": seed,
      "environment_init_seconds": environment_init_seconds,
      "policy_load_seconds": policy_load_seconds,
      "executed_control_steps": executed_steps,
      "active_metric_steps": active_steps,
      "completion": {
        "terminated": terminated,
        "truncated": truncated,
        "success": truncated and not terminated,
        "censored_at_max_steps": not terminated and not truncated,
        "triggered_terms": termination_terms,
      },
      "tracking": tracking,
      "contacts_total": _sample_stats(contacts),
      "constraints": _sample_stats(constraints),
      "solver_iterations": _sample_stats(solver_iterations),
      "overflow": _overflow_stats(overflow),
      "throughput": {
        "timed_policy_and_env_step_seconds": elapsed_step_seconds,
        "control_steps_per_second": control_sps,
        "physics_steps_per_second": physics_sps,
      },
      "model": {
        "ngeom": int(env.unwrapped.sim.mj_model.ngeom),
        "nconmax": env_cfg.sim.nconmax,
        "njmax": env_cfg.sim.njmax,
        "timestep": env_cfg.sim.mujoco.timestep,
        "decimation": env_cfg.decimation,
      },
    }
  finally:
    env.close()


def _mean_std(values: list[float]) -> dict[str, float]:
  mean = sum(values) / len(values)
  variance = sum((value - mean) ** 2 for value in values) / len(values)
  return {"mean": mean, "std": math.sqrt(variance)}


def summarize_runs(runs: list[dict[str, object]]) -> dict[str, object]:
  """Summarize the comparable scalar metrics across deterministic seeds."""
  metric_paths = {
    "success": ("completion", "success"),
    "mpkpe": ("tracking", "mpkpe"),
    "r_mpkpe": ("tracking", "r_mpkpe"),
    "joint_vel_error": ("tracking", "joint_vel_error"),
    "ee_pos_error": ("tracking", "ee_pos_error"),
    "ee_ori_error": ("tracking", "ee_ori_error"),
    "contacts_mean": ("contacts_total", "mean"),
    "constraints_mean": ("constraints", "mean"),
    "solver_iterations_mean": ("solver_iterations", "mean"),
    "overflow_run_rate": ("overflow", "any"),
    "overflow_nonzero_steps": ("overflow", "nonzero_steps"),
    "control_steps_per_second": ("throughput", "control_steps_per_second"),
    "physics_steps_per_second": ("throughput", "physics_steps_per_second"),
  }
  summary: dict[str, object] = {"run_count": len(runs)}
  for output_name, path in metric_paths.items():
    values: list[float] = []
    for run in runs:
      value: Any = run
      for key in path:
        value = value[key]
      if value is not None:
        values.append(float(cast(float | int | bool, value)))
    summary[output_name] = _mean_std(values) if values else None
  return summary


def benchmark_variants(
  variants: list[CollisionVariant], settings: BenchmarkSettings
) -> dict[str, object]:
  """Run all variants under the same local policy evaluation protocol."""
  results: list[dict[str, object]] = []
  for variant in variants:
    print(f"[INFO] Benchmarking collision variant: {variant.name}")
    runs = [_run_one(variant, settings, seed=seed) for seed in settings.seeds]
    collision = (
      collision_bundle_hashes(variant.collision_mjcf)
      if variant.collision_mjcf is not None
      else None
    )
    results.append(
      {
        "name": variant.name,
        "collision": collision,
        "keep_plane": variant.keep_plane,
        "geometry": variant.geometry,
        "runs": runs,
        "aggregate": summarize_runs(runs),
      }
    )
  return {
    "schema_version": 1,
    "protocol": {
      "task_id": settings.task_id,
      "device": settings.device,
      "seeds": list(settings.seeds),
      "warmup_steps": settings.warmup_steps,
      "max_steps": settings.max_steps,
      "nconmax": settings.nconmax,
      "njmax": settings.njmax,
      "observation_corruption": settings.observation_corruption,
      "push_event_enabled": False,
      "motion_sampling_mode": "start",
      "num_envs": 1,
      "auto_reset": False,
      "throughput_scope": "policy inference plus env.step; rendering excluded",
      "contact_scope": "all active MuJoCo contacts, including robot self-contact",
    },
    "inputs": {
      "checkpoint": {
        "path": str(settings.checkpoint),
        "sha256": _sha256(settings.checkpoint),
      },
      "motion": {"path": str(settings.motion), "sha256": _sha256(settings.motion)},
    },
    "results": results,
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("variant_manifest", type=Path)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--motion", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--task-id", default="Mjlab-Tracking-Flat-Unitree-G1")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--seeds", type=int, nargs="+", default=[42])
  parser.add_argument("--warmup-steps", type=int, default=10)
  parser.add_argument("--max-steps", type=int, default=500)
  parser.add_argument("--nconmax", type=int, default=256)
  parser.add_argument("--njmax", type=int, default=1024)
  parser.add_argument("--disable-observation-corruption", action="store_true")
  return parser.parse_args()


def main() -> None:
  import mjlab.tasks  # noqa: F401

  args = _parse_args()
  if args.max_steps <= 0:
    raise ValueError("--max-steps must be positive.")
  if args.warmup_steps < 0:
    raise ValueError("--warmup-steps must be non-negative.")
  if args.nconmax <= 0 or args.njmax <= 0:
    raise ValueError("--nconmax and --njmax must be positive.")
  if not args.seeds:
    raise ValueError("--seeds must contain at least one seed.")
  checkpoint = args.checkpoint.expanduser().resolve()
  motion = args.motion.expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
  if not motion.is_file():
    raise FileNotFoundError(f"Motion not found: {motion}")

  configure_torch_backends()
  variants = load_variant_manifest(args.variant_manifest)
  settings = BenchmarkSettings(
    task_id=args.task_id,
    checkpoint=checkpoint,
    motion=motion,
    device=args.device,
    seeds=tuple(args.seeds),
    warmup_steps=args.warmup_steps,
    max_steps=args.max_steps,
    nconmax=args.nconmax,
    njmax=args.njmax,
    observation_corruption=not args.disable_observation_corruption,
  )
  report = benchmark_variants(variants, settings)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"[INFO] Wrote joint collision-policy report: {args.output}")


if __name__ == "__main__":
  main()
