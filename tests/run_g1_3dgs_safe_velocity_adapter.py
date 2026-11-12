"""Evaluate safe-velocity deterministic-adapter scenario outcomes.

uv run tests/run_g1_3dgs_safe_velocity_adapter.py \
  --checkpoint logs/rsl_rl/g1_velocity/RUN/model.pt \
  --scenario free left_open right_open both_open dead_end false_alarm \
  --seed 0 1 2 --require-pass

Each run records a structured trace. A multi-scenario or multi-seed invocation also
emits an acceptance matrix against the catalog's expected outcomes.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.velocity.config.g1.safe_velocity import (
  SAFE_VELOCITY_ADAPTER_TASK_ID,
  SAFE_VELOCITY_COLLISION_SENSOR_PREFIX,
  unitree_g1_safe_velocity_adapter_3dgs_env_cfg,
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


def _collision_metrics(env: ManagerBasedRlEnv) -> tuple[bool, float, float]:
  collision = False
  max_force_n = 0.0
  impulse_ns = 0.0
  for name, sensor in env.scene.sensors.items():
    if not name.startswith(SAFE_VELOCITY_COLLISION_SENSOR_PREFIX) or not isinstance(
      sensor, ContactSensor
    ):
      continue
    history = sensor.data.force_history
    if history is not None:
      force_magnitudes = torch.linalg.vector_norm(history[0], dim=-1)
      collision |= bool(torch.any(force_magnitudes > 0).item())
      max_force_n = max(max_force_n, float(force_magnitudes.max().item()))
      impulse_ns += float(force_magnitudes.sum().item()) * env.physics_dt
      continue
    force = sensor.data.force
    if force is not None:
      force_magnitudes = torch.linalg.vector_norm(force[0], dim=-1)
      collision |= bool(torch.any(force_magnitudes > 0).item())
      max_force_n = max(max_force_n, float(force_magnitudes.max().item()))
      impulse_ns += float(force_magnitudes.sum().item()) * env.step_dt
    found = sensor.data.found
    if found is not None:
      collision |= bool(torch.any(found[0] > 0).item())
  return collision, max_force_n, impulse_ns


def _row(env: ManagerBasedRlEnv, step: int, done: bool) -> dict:
  term_cfg = env.observation_manager.get_term_cfg("actor", "command")
  term = term_cfg.func
  if not isinstance(term, adapted_velocity_command) or term.last_output is None:
    raise RuntimeError("Safe-velocity adapter diagnostics are unavailable")
  output = term.last_output
  robot = env.scene["robot"]
  mode = SafeVelocityAdapterMode(int(output.mode[0].item()))
  nominal = env.command_manager.get_command("twist")
  assert isinstance(nominal, torch.Tensor)
  collision, collision_force_n, collision_impulse_ns = _collision_metrics(env)
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
    "search_elapsed_s": float(output.search_elapsed_s[0].item()),
    "intervention": bool(output.intervention[0].item()),
    "search_timed_out": bool(output.search_timed_out[0].item()),
    "collision": collision,
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
    front_clearance_m=row["sector_clearance_m"][1],
    collision=row["collision"],
    collision_force_n=row["collision_force_n"],
    collision_impulse_ns=row["collision_impulse_ns"],
    fall=row["fell_over"],
    episode_timeout=row["episode_timeout"],
    search_timed_out=row["search_timed_out"],
  )


def run(args: argparse.Namespace) -> dict:
  configure_torch_backends()
  checkpoint = args.checkpoint.expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

  catalog = SAFE_VELOCITY_SCENARIO_CATALOG
  scenario = catalog.scenario(args.scenario)
  env_cfg = unitree_g1_safe_velocity_adapter_3dgs_env_cfg(
    play=True,
    scenario_id=scenario.id,
  )
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.seed
  env_cfg.auto_reset = False
  agent_cfg = load_rl_cfg(SAFE_VELOCITY_ADAPTER_TASK_ID)
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
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
      scenario.nominal_command,
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
      SafeVelocityTrajectoryEvaluatorCfg(allowed_region_xy=scenario.allowed_region_xy)
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
        if scenario.expected_outcome != "continuous_tracking":
          partial_outcome = evaluator.summarize().outcome
          if partial_outcome == scenario.expected_outcome:
            accepted_early = True
            break

    mode_counts = Counter(row["mode"] for row in rows)
    trajectory_summary = asdict(evaluator.summarize())
    outcome = trajectory_summary["outcome"]
    accepted = outcome == scenario.expected_outcome
    return {
      "date": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
      "scope": "Safe-velocity deterministic-adapter scenario acceptance",
      "scenario": asdict(scenario),
      "checkpoint": str(checkpoint),
      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      "device": device,
      "seed": args.seed,
      "requested_steps": steps,
      "recorded_steps": len(rows) - 1,
      "step_dt": env.step_dt,
      "summary": {
        "outcome": trajectory_summary.pop("outcome"),
        "expected_outcome": scenario.expected_outcome,
        "accepted": accepted,
        "accepted_early": accepted_early,
        "mode_counts": dict(mode_counts),
        "intervention_steps": sum(row["intervention"] for row in rows),
        "resumed": any(row["mode"] == "resume" for row in rows),
        "blocked": any(row["mode"] == "blocked" for row in rows),
        "terminated": rows[-1]["done"],
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

  reports = []
  for scenario_id in args.scenario:
    for seed in args.seed:
      run_args = argparse.Namespace(**vars(args))
      run_args.scenario = scenario_id
      run_args.seed = seed
      reports.append(run(run_args))

  if len(reports) == 1:
    report = reports[0]
  else:
    failures = [
      {
        "scenario": item["scenario"]["id"],
        "seed": item["seed"],
        "outcome": item["summary"]["outcome"],
        "expected_outcome": item["summary"]["expected_outcome"],
      }
      for item in reports
      if not item["summary"]["accepted"]
    ]
    report = {
      "scope": "Safe-velocity deterministic-adapter acceptance matrix",
      "checkpoint": reports[0]["checkpoint"],
      "checkpoint_sha256": reports[0]["checkpoint_sha256"],
      "device": reports[0]["device"],
      "summary": {
        "total": len(reports),
        "accepted": len(reports) - len(failures),
        "all_accepted": not failures,
        "failures": failures,
      },
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
