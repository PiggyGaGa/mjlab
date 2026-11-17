"""Evaluate safe-velocity deterministic-adapter scenario outcomes.

uv run tests/run_g1_3dgs_safe_velocity_adapter.py \
  --checkpoint logs/rsl_rl/g1_velocity/RUN/model.pt \
  --controller search \
  --scenario free left_open right_open both_open dead_end false_alarm \
  --seed 0 1 2 --require-pass

Each run records a structured trace. A multi-scenario or multi-seed invocation also
emits an acceptance matrix against the catalog's expected outcomes.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scene import load_scene_manifest
from mjlab.sensor import ContactSensor
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.velocity.config.g1.safe_velocity import (
  SAFE_VELOCITY_ADAPTER_TASK_ID,
  SAFE_VELOCITY_COLLISION_SENSOR_PREFIX,
  SAFE_VELOCITY_DEPTH_STOP_TASK_ID,
  unitree_g1_safe_velocity_adapter_3dgs_env_cfg,
  unitree_g1_safe_velocity_depth_stop_3dgs_env_cfg,
)
from mjlab.tasks.velocity.config.g1.safe_velocity_baseline import (
  SafeVelocityAdapterMode,
  SafeVelocityTrajectoryEvaluator,
  SafeVelocityTrajectoryEvaluatorCfg,
  SafeVelocityTrajectorySample,
  adapted_velocity_command,
)
from mjlab.tasks.velocity.config.g1.safe_velocity_scenarios import (
  SAFE_VELOCITY_SCENARIO_CATALOG,
)
from mjlab.utils.torch import configure_torch_backends


def matching_scene_spawn_points(
  scene_manifest: Path,
  prefix: str,
) -> tuple[str, ...]:
  """Return manifest spawn names matching a non-empty prefix in manifest order."""
  if not prefix:
    raise ValueError("Scene spawn prefix must not be empty")
  manifest = load_scene_manifest(scene_manifest.expanduser().resolve())
  matches = tuple(
    spawn.name for spawn in manifest.spawn_points if spawn.name.startswith(prefix)
  )
  if not matches:
    raise ValueError(f"No scene spawn points match prefix: {prefix}")
  return matches


def acceptance_matrix(reports: list[dict]) -> dict:
  """Summarize safety and trajectory ranges across acceptance reports."""
  failures = [
    {
      "scenario": item["scenario"]["id"],
      "scene_spawn_point": item["scene_spawn_point"],
      "seed": item["seed"],
      "outcome": item["summary"]["outcome"],
      "expected_outcome": item["summary"]["expected_outcome"],
    }
    for item in reports
    if not item["summary"]["accepted"]
  ]
  trajectories = [item["summary"]["trajectory"] for item in reports]
  clearances = [item["min_sensor_clearance_m"] for item in trajectories]
  path_lengths = [item["path_length_m"] for item in trajectories]
  durations = [item["duration_s"] for item in trajectories]
  return {
    "total": len(reports),
    "accepted": len(reports) - len(failures),
    "all_accepted": not failures,
    "collision_count": sum(item["collision"] for item in trajectories),
    "fall_count": sum(item["fall"] for item in trajectories),
    "termination_count": sum(item["summary"]["terminated"] for item in reports),
    "base_contact_steps": sum(
      item["summary"].get("base_contact_steps", 0) for item in reports
    ),
    "min_clearance_m": min(clearances),
    "max_clearance_m": max(clearances),
    "min_path_length_m": min(path_lengths),
    "max_path_length_m": max(path_lengths),
    "min_duration_s": min(durations),
    "max_duration_s": max(durations),
    "failures": failures,
  }


def _collision_metrics(
  env: ManagerBasedRlEnv,
  contact_events: list[dict] | None = None,
) -> tuple[bool, float, float]:
  collision = False
  max_force_n = 0.0
  impulse_ns = 0.0
  for name, sensor in env.scene.sensors.items():
    if not name.startswith(SAFE_VELOCITY_COLLISION_SENSOR_PREFIX) or not isinstance(
      sensor, ContactSensor
    ):
      continue
    data = sensor.data
    found_count = int(data.found[0].sum().item()) if data.found is not None else 0
    sensor_contact = found_count > 0
    sensor_force_n = 0.0
    history = data.force_history
    if history is not None:
      force_magnitudes = torch.linalg.vector_norm(history[0], dim=-1)
      sensor_force_n = float(force_magnitudes.max().item())
      impulse_ns += float(force_magnitudes.sum().item()) * env.physics_dt
    elif data.force is not None:
      force = data.force
      force_magnitudes = torch.linalg.vector_norm(force[0], dim=-1)
      sensor_force_n = float(force_magnitudes.max().item())
      impulse_ns += float(force_magnitudes.sum().item()) * env.step_dt
    sensor_contact |= sensor_force_n > 0.0
    collision |= sensor_contact
    max_force_n = max(max_force_n, sensor_force_n)
    if sensor_contact and contact_events is not None:
      secondary = sensor.cfg.secondary
      contact_events.append(
        {
          "sensor": name,
          "obstacle_geom": secondary.pattern if secondary is not None else None,
          "current_found": found_count,
          "peak_force_n": sensor_force_n,
          "history_available": history is not None,
        }
      )
  return collision, max_force_n, impulse_ns


def _row(env: ManagerBasedRlEnv, step: int, done: bool) -> dict:
  term_cfg = env.observation_manager.get_term_cfg("actor", "command")
  term = term_cfg.func
  if not isinstance(term, adapted_velocity_command) or term.last_output is None:
    raise RuntimeError("Safe-velocity adapter diagnostics are unavailable")
  output = term.last_output
  robot = env.scene["robot"]
  mode = SafeVelocityAdapterMode(int(output.mode[0].item()))
  if term.adapter.cfg.stop_only:
    front_clearance_m = float(term.adapter.decision_clearance_m[0].item())
    speed_scale = float(term.adapter.speed_scale[0].item())
  else:
    front_clearance_m = float(output.sector_clearance_m[0, 1].item())
    speed_scale = 1.0
  nominal = env.command_manager.get_command("twist")
  assert isinstance(nominal, torch.Tensor)
  contact_events: list[dict] = []
  collision, collision_force_n, collision_impulse_ns = _collision_metrics(
    env, contact_events
  )
  fell_over = (
    bool(env.termination_manager.get_term("fell_over")[0].item())
    if "fell_over" in env.termination_manager.active_terms
    else False
  )
  return {
    "step": step,
    "sim_time_s": step * env.step_dt,
    "mode": mode.name.lower(),
    "nominal_command": nominal[0].detach().cpu().tolist(),
    "adapted_command": output.command[0].detach().cpu().tolist(),
    "sector_clearance_m": output.sector_clearance_m[0].detach().cpu().tolist(),
    "sector_valid_ratio": output.sector_valid_ratio[0].detach().cpu().tolist(),
    "front_clearance_m": front_clearance_m,
    "speed_scale": speed_scale,
    "search_elapsed_s": float(output.search_elapsed_s[0].item()),
    "intervention": bool(output.intervention[0].item()),
    "search_timed_out": bool(output.search_timed_out[0].item()),
    "collision": collision,
    "base_contacts": contact_events,
    "collision_force_n": collision_force_n,
    "collision_impulse_ns": collision_impulse_ns,
    "fell_over": fell_over,
    "episode_timeout": bool(env.termination_manager.time_outs[0].item()),
    "base_position_w": robot.data.root_link_pos_w[0].detach().cpu().tolist(),
    "base_heading_w": float(robot.data.heading_w[0].item()),
    "base_linear_velocity_b": (
      robot.data.root_link_lin_vel_b[0].detach().cpu().tolist()
    ),
    "done": done,
    "termination_reasons": [
      name
      for name in env.termination_manager.active_terms
      if env.termination_manager.get_term(name)[0].item()
    ]
    if done
    else [],
  }


def _trajectory_sample(row: dict) -> SafeVelocityTrajectorySample:
  return SafeVelocityTrajectorySample(
    time_s=row["sim_time_s"],
    mode=SafeVelocityAdapterMode[row["mode"].upper()],
    position_xy=(
      float(row["base_position_w"][0]),
      float(row["base_position_w"][1]),
    ),
    planar_speed_mps=(
      float(row["base_linear_velocity_b"][0]) ** 2
      + float(row["base_linear_velocity_b"][1]) ** 2
    )
    ** 0.5,
    front_clearance_m=row["front_clearance_m"],
    collision=row["collision"],
    collision_force_n=row["collision_force_n"],
    collision_impulse_ns=row["collision_impulse_ns"],
    fall=row["fell_over"],
    episode_timeout=row["episode_timeout"],
    search_timed_out=row["search_timed_out"],
  )


def stationary_tail_duration(
  rows: list[dict], cfg: SafeVelocityTrajectoryEvaluatorCfg
) -> float:
  """Audit the final contiguous low-speed interval from recorded samples."""
  end_time = rows[-1]["sim_time_s"]
  start_time = end_time
  modes = {mode.name.lower() for mode in cfg.blocked_modes}
  for row in reversed(rows):
    velocity = row["base_linear_velocity_b"]
    if (
      row["mode"] not in modes
      or math.hypot(velocity[0], velocity[1]) > cfg.stationary_speed_mps
    ):
      break
    start_time = row["sim_time_s"]
  return end_time - start_time


def run(args: argparse.Namespace) -> dict:
  configure_torch_backends()
  checkpoint = args.checkpoint.expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

  catalog = SAFE_VELOCITY_SCENARIO_CATALOG
  scenario = catalog.scenario(args.scenario)
  scene_manifest = (
    args.scene_manifest.expanduser().resolve()
    if args.scene_manifest is not None
    else None
  )
  nominal_command = (
    tuple(args.nominal_command)
    if args.nominal_command is not None
    else scenario.nominal_command
  )
  if args.controller == "depth_stop":
    task_id = SAFE_VELOCITY_DEPTH_STOP_TASK_ID
    env_cfg = unitree_g1_safe_velocity_depth_stop_3dgs_env_cfg(
      play=True,
      scenario_id=scenario.id,
      scene_manifest=scene_manifest,
      scene_spawn_point=args.scene_spawn_point,
      nominal_command=nominal_command,
    )
    expected_outcome = args.expected_outcome or (
      "safe_blocked_stop" if scenario.blocking_geoms else "continuous_tracking"
    )
    blocked_modes = (
      SafeVelocityAdapterMode.BRAKE,
      SafeVelocityAdapterMode.STOP,
      SafeVelocityAdapterMode.BLOCKED,
    )
  else:
    task_id = SAFE_VELOCITY_ADAPTER_TASK_ID
    env_cfg = unitree_g1_safe_velocity_adapter_3dgs_env_cfg(
      play=True,
      scenario_id=scenario.id,
    )
    expected_outcome = scenario.expected_outcome
    blocked_modes = (SafeVelocityAdapterMode.BLOCKED,)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.seed
  env_cfg.auto_reset = False
  env_cfg.observations["actor"].terms["command"].params["debug_print"] = False
  agent_cfg = load_rl_cfg(task_id)
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    context = env.scene.sensor_context
    if context is None or not context.has_cameras:
      raise RuntimeError("Depth acceptance requires a camera render context")
    geom_ids = context.render_context.enabled_geom_ids.numpy()
    camera_groups = sorted(set(env.sim.mj_model.geom_group[geom_ids].tolist()))
    if args.controller == "depth_stop" and (
      context.has_raycasts or not set(camera_groups) <= {1, 2}
    ):
      raise RuntimeError("Depth-stop acceptance requires visible-only camera inputs")
    steps = (
      args.steps
      if args.steps is not None
      else round(scenario.evaluation_time_s / env.step_dt)
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

    env.reset(seed=args.seed)
    nominal = torch.tensor(
      nominal_command,
      dtype=torch.float32,
      device=env.device,
    )
    command = env.command_manager.get_command("twist")
    assert isinstance(command, torch.Tensor)
    command[:] = nominal
    command_term = env.command_manager.get_term("twist")
    assert isinstance(command_term, CommandTerm)
    command_term.time_left.fill_((steps + 1) * env.step_dt)

    term_cfg = env.observation_manager.get_term_cfg("actor", "command")
    term = term_cfg.func
    if not isinstance(term, adapted_velocity_command):
      raise TypeError("Safe-velocity task did not build adapted_velocity_command")
    env.observation_manager.reset()
    observations = wrapped.get_observations()
    rows = [_row(env, step=0, done=False)]
    evaluator = SafeVelocityTrajectoryEvaluator(
      SafeVelocityTrajectoryEvaluatorCfg(
        allowed_region_xy=(
          tuple(args.allowed_region)
          if args.allowed_region is not None
          else scenario.allowed_region_xy
        ),
        blocked_modes=blocked_modes,
      )
    )
    evaluator.update(_trajectory_sample(rows[0]))
    accepted_early = False

    with torch.inference_mode():
      for step in range(1, steps + 1):
        actions = policy(observations)
        observations, _, dones, _ = wrapped.step(actions)
        done = bool(dones[0].item())
        rows.append(_row(env, step=step, done=done))
        evaluator.update(_trajectory_sample(rows[-1]))
        if done:
          break
        if expected_outcome != "continuous_tracking":
          partial_outcome = evaluator.summarize().outcome
          if partial_outcome == expected_outcome:
            accepted_early = True
            break

    mode_counts = Counter(row["mode"] for row in rows)
    trajectory_summary = asdict(evaluator.summarize())
    outcome = trajectory_summary["outcome"]
    accepted = outcome == expected_outcome
    stationary_hold_s = stationary_tail_duration(rows, evaluator.cfg)
    if args.controller == "depth_stop" and rows[-1]["done"]:
      accepted = False
    if expected_outcome == "safe_blocked_stop":
      accepted &= stationary_hold_s >= evaluator.cfg.blocked_hold_s
    return {
      "date": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
      "scope": "Safe-velocity deterministic-adapter scenario acceptance",
      "controller": args.controller,
      "scenario": asdict(scenario),
      "scene_manifest": str(scene_manifest) if scene_manifest is not None else None,
      "scene_spawn_point": args.scene_spawn_point,
      "nominal_command": nominal_command,
      "checkpoint": str(checkpoint),
      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      "device": device,
      "seed": args.seed,
      "evaluation": {
        "stationary_stop_rule": "continuous_hold",
        "required_stationary_hold_s": evaluator.cfg.blocked_hold_s,
        "stationary_speed_mps": evaluator.cfg.stationary_speed_mps,
      },
      "perception": {
        "sources": ["3dgs", "visible_mesh"],
        "camera_bvh_geom_groups": camera_groups,
        "has_shared_raycast_sensors": context.has_raycasts,
        "collision_signal": "pelvis_contact_sensor",
        "contact_rule": "current_found_positive_or_positive_force_history",
      },
      "requested_steps": steps,
      "recorded_steps": len(rows) - 1,
      "step_dt": env.step_dt,
      "summary": {
        "outcome": trajectory_summary.pop("outcome"),
        "expected_outcome": expected_outcome,
        "accepted": accepted,
        "accepted_early": accepted_early,
        "mode_counts": dict(mode_counts),
        "intervention_steps": sum(row["intervention"] for row in rows),
        "resumed": any(row["mode"] == "resume" for row in rows),
        "blocked": any(row["mode"] == "blocked" for row in rows),
        "terminated": rows[-1]["done"],
        "termination_reasons": rows[-1]["termination_reasons"],
        "base_contact_steps": sum(row["collision"] for row in rows),
        "contact_obstacles": sorted(
          {
            event["obstacle_geom"]
            for row in rows
            for event in row["base_contacts"]
            if event["obstacle_geom"] is not None
          }
        ),
        "final_stationary_hold_s": stationary_hold_s,
        "trajectory": trajectory_summary,
      },
      "trace": rows,
    }
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
    "--controller",
    choices=("search", "depth_stop"),
    default="search",
    help="Select the original search adapter or the depth-only stop layer.",
  )
  parser.add_argument(
    "--scenario",
    nargs="+",
    choices=(
      "free",
      "left_open",
      "right_open",
      "both_open",
      "dead_end",
      "false_alarm",
    ),
    default=("left_open",),
  )
  parser.add_argument(
    "--scene-manifest",
    type=Path,
    default=None,
    help="Build a custom map before depth-stop contact sensors are created.",
  )
  parser.add_argument("--scene-spawn-point", default=None)
  parser.add_argument(
    "--scene-spawn-prefix",
    default=None,
    help="Evaluate every manifest spawn whose name starts with this prefix.",
  )
  parser.add_argument(
    "--nominal-command",
    type=float,
    nargs=3,
    metavar=("VX", "VY", "WZ"),
    default=None,
  )
  parser.add_argument(
    "--allowed-region",
    type=float,
    nargs=4,
    metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
    default=None,
    help="Evaluator bounds for a custom map; defaults to the selected scenario.",
  )
  parser.add_argument(
    "--expected-outcome",
    choices=("continuous_tracking", "safe_blocked_stop"),
    default=None,
    help="Required for custom maps so acceptance semantics are explicit.",
  )
  parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="Override the catalog evaluation window.",
  )
  parser.add_argument("--seed", type=int, nargs="+", default=(0,))
  parser.add_argument("--device", default=None)
  parser.add_argument(
    "--require-pass",
    action="store_true",
    help="Exit nonzero when any scenario outcome differs from the catalog.",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("/tmp/safe-velocity-adapter-diagnostics.json"),
  )
  args = parser.parse_args()
  if args.steps is not None and args.steps < 1:
    parser.error("--steps must be positive")
  if args.scene_spawn_point is not None and args.scene_manifest is None:
    parser.error("--scene-spawn-point requires --scene-manifest")
  if args.scene_spawn_prefix is not None and args.scene_manifest is None:
    parser.error("--scene-spawn-prefix requires --scene-manifest")
  if args.scene_spawn_point is not None and args.scene_spawn_prefix is not None:
    parser.error("--scene-spawn-point and --scene-spawn-prefix are mutually exclusive")
  if args.scene_manifest is not None:
    if args.controller != "depth_stop":
      parser.error("--scene-manifest currently requires --controller depth_stop")
    if not args.scene_manifest.expanduser().is_file():
      parser.error(f"Scene manifest not found: {args.scene_manifest}")
    if args.expected_outcome is None:
      parser.error("--expected-outcome is required with --scene-manifest")
    if len(args.scenario) != 1:
      parser.error("Custom map evaluation accepts exactly one --scenario protocol")
  if args.allowed_region is not None:
    x_min, x_max, y_min, y_max = args.allowed_region
    if x_min >= x_max or y_min >= y_max:
      parser.error("--allowed-region bounds must be ordered")

  if args.scene_spawn_prefix is not None:
    try:
      scene_spawn_points: tuple[str | None, ...] = matching_scene_spawn_points(
        args.scene_manifest,
        args.scene_spawn_prefix,
      )
    except ValueError as error:
      parser.error(str(error))
  else:
    scene_spawn_points = (args.scene_spawn_point,)

  reports = []
  for scenario_id in args.scenario:
    for scene_spawn_point in scene_spawn_points:
      for seed in args.seed:
        run_args = argparse.Namespace(**vars(args))
        run_args.scenario = scenario_id
        run_args.scene_spawn_point = scene_spawn_point
        run_args.seed = seed
        result = run(run_args)
        reports.append(result)
        summary = result["summary"]
        print(
          f"[safe_velocity_acceptance] scenario={scenario_id} "
          f"spawn={scene_spawn_point or 'scenario_default'} seed={seed} "
          f"outcome={summary['outcome']} accepted={summary['accepted']} "
          f"contacts={summary['base_contact_steps']} "
          f"terminations={summary['termination_reasons']}",
          flush=True,
        )

  if len(reports) == 1:
    report = reports[0]
  else:
    report = {
      "scope": "Safe-velocity deterministic-adapter acceptance matrix",
      "checkpoint": reports[0]["checkpoint"],
      "checkpoint_sha256": reports[0]["checkpoint_sha256"],
      "device": reports[0]["device"],
      "scene_manifest": reports[0]["scene_manifest"],
      "scene_spawn_prefix": args.scene_spawn_prefix,
      "summary": acceptance_matrix(reports),
      "reports": reports,
    }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
  print(json.dumps(report["summary"], indent=2))
  print(f"Wrote {args.output}")
  if (
    args.require_pass
    and reports
    and not all(item["summary"]["accepted"] for item in reports)
  ):
    raise SystemExit(1)


if __name__ == "__main__":
  main()
