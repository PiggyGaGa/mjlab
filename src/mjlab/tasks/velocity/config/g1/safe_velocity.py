"""P7 safe perceptive velocity tracking configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import apply_scene_manifest, load_scene_manifest
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from .env_cfgs import (
  unitree_g1_flat_3dgs_depth_validity_history_robust_env_cfg,
)
from .rl_cfg import (
  unitree_g1_3dgs_depth_validity_history_robust_asymmetric_ppo_runner_cfg,
  unitree_g1_ppo_runner_cfg,
)
from .safe_velocity_baseline import P7DepthAdapterCfg, adapted_velocity_command

P7_TASK_ID = "Mjlab-Velocity-Safe-Unitree-G1-3DGS"
P7_ADAPTER_TASK_ID = "Mjlab-Velocity-Safe-Adapter-Unitree-G1-3DGS"
P7_SCENARIO_CATALOG_PATH = Path(__file__).with_name("p7_scenarios.json")


@dataclass(frozen=True)
class P7CommandProtocol:
  """Frozen initial command ranges for P7."""

  resampling_time_s: tuple[float, float]
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  ang_vel_z: tuple[float, float]


@dataclass(frozen=True)
class P7SafetyThresholds:
  """Initial values to calibrate during P7-0 and P7-3."""

  collision_risk_distance_m: float
  stop_clearance_m: float
  search_time_budget_s: float
  search_angle_budget_deg: float


@dataclass(frozen=True)
class P7Scenario:
  """One deterministic local-obstacle scenario."""

  id: str
  split: Literal["train", "validation", "test"]
  spawn_point: str
  nominal_command: tuple[float, float, float]
  blocking_geoms: tuple[str, ...]
  safe_exit_sectors_deg: tuple[tuple[float, float], ...]
  allowed_region_xy: tuple[float, float, float, float]
  expected_behavior: str
  min_clearance_m: float
  search_time_budget_s: float
  search_angle_budget_deg: float
  review_status: Literal["pending", "approved"]


@dataclass(frozen=True)
class P7ScenarioCatalog:
  """Versioned P7 scenario protocol resolved to the apartment manifest."""

  source_path: Path
  scene_manifest: Path
  command: P7CommandProtocol
  thresholds: P7SafetyThresholds
  scenarios: tuple[P7Scenario, ...]

  def scenario(self, scenario_id: str) -> P7Scenario:
    """Return a scenario by ID with a useful error for CLI/config callers."""
    match = next((item for item in self.scenarios if item.id == scenario_id), None)
    if match is not None:
      return match
    available = ", ".join(item.id for item in self.scenarios)
    raise ValueError(f"Unknown P7 scenario '{scenario_id}'; available: {available}")


def _float_tuple(value: Any, length: int, label: str) -> tuple[float, ...]:
  if not isinstance(value, list) or len(value) != length:
    raise ValueError(f"{label} must contain exactly {length} numbers")
  try:
    result = tuple(float(item) for item in value)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{label} must contain only numbers") from exc
  return result


def _range(value: Any, label: str) -> tuple[float, float]:
  result = cast(tuple[float, float], _float_tuple(value, 2, label))
  if result[0] > result[1]:
    raise ValueError(f"{label} must be ordered")
  return result


def _positive_float(value: Any, label: str) -> float:
  try:
    result = float(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{label} must be a number") from exc
  if result <= 0:
    raise ValueError(f"{label} must be positive")
  return result


def load_p7_scenario_catalog(
  path: str | Path = P7_SCENARIO_CATALOG_PATH,
) -> P7ScenarioCatalog:
  """Load and validate the task-local P7 scenario catalog."""
  source_path = Path(path).resolve()
  raw = json.loads(source_path.read_text())
  if not isinstance(raw, dict) or raw.get("schema_version") != 1:
    raise ValueError("P7 scenario catalog must use schema_version 1")

  manifest_value = raw.get("scene_manifest")
  if not isinstance(manifest_value, str) or not manifest_value:
    raise ValueError("scene_manifest must be a non-empty relative path")
  scene_manifest = (source_path.parent / manifest_value).resolve()
  manifest = load_scene_manifest(scene_manifest)
  spawn_names = {spawn.name for spawn in manifest.spawn_points}

  command_raw = raw.get("command")
  if not isinstance(command_raw, dict):
    raise ValueError("command must be an object")
  command = P7CommandProtocol(
    resampling_time_s=_range(
      command_raw.get("resampling_time_s"), "command.resampling_time_s"
    ),
    lin_vel_x=_range(command_raw.get("lin_vel_x"), "command.lin_vel_x"),
    lin_vel_y=_range(command_raw.get("lin_vel_y"), "command.lin_vel_y"),
    ang_vel_z=_range(command_raw.get("ang_vel_z"), "command.ang_vel_z"),
  )

  thresholds_raw = raw.get("thresholds")
  if not isinstance(thresholds_raw, dict):
    raise ValueError("thresholds must be an object")
  thresholds = P7SafetyThresholds(
    collision_risk_distance_m=_positive_float(
      thresholds_raw.get("collision_risk_distance_m"),
      "thresholds.collision_risk_distance_m",
    ),
    stop_clearance_m=_positive_float(
      thresholds_raw.get("stop_clearance_m"), "thresholds.stop_clearance_m"
    ),
    search_time_budget_s=_positive_float(
      thresholds_raw.get("search_time_budget_s"),
      "thresholds.search_time_budget_s",
    ),
    search_angle_budget_deg=_positive_float(
      thresholds_raw.get("search_angle_budget_deg"),
      "thresholds.search_angle_budget_deg",
    ),
  )

  scenarios_raw = raw.get("scenarios")
  if not isinstance(scenarios_raw, list) or not scenarios_raw:
    raise ValueError("scenarios must be a non-empty list")
  scenarios: list[P7Scenario] = []
  ids: set[str] = set()
  for index, item in enumerate(scenarios_raw):
    label = f"scenarios[{index}]"
    if not isinstance(item, dict):
      raise ValueError(f"{label} must be an object")
    scenario_id = item.get("id")
    if not isinstance(scenario_id, str) or not scenario_id:
      raise ValueError(f"{label}.id must be a non-empty string")
    if scenario_id in ids:
      raise ValueError(f"Duplicate P7 scenario id: {scenario_id}")
    ids.add(scenario_id)

    split = item.get("split")
    if split not in ("train", "validation", "test"):
      raise ValueError(f"{label}.split must be train, validation, or test")
    spawn_point = item.get("spawn_point")
    if spawn_point not in spawn_names:
      raise ValueError(f"{label}.spawn_point is not in the scene manifest")
    review_status = item.get("review_status")
    if review_status not in ("pending", "approved"):
      raise ValueError(f"{label}.review_status must be pending or approved")

    sectors_raw = item.get("safe_exit_sectors_deg")
    if not isinstance(sectors_raw, list):
      raise ValueError(f"{label}.safe_exit_sectors_deg must be a list")
    sectors = tuple(
      _range(sector, f"{label}.safe_exit_sectors_deg") for sector in sectors_raw
    )
    nominal_command = cast(
      tuple[float, float, float],
      _float_tuple(item.get("nominal_command"), 3, f"{label}.nominal_command"),
    )
    command_ranges = (command.lin_vel_x, command.lin_vel_y, command.ang_vel_z)
    if any(
      not low <= value <= high
      for value, (low, high) in zip(nominal_command, command_ranges, strict=True)
    ):
      raise ValueError(f"{label}.nominal_command is outside command ranges")

    blocking_geoms = item.get("blocking_geoms")
    if not isinstance(blocking_geoms, list) or not all(
      isinstance(name, str) and name for name in blocking_geoms
    ):
      raise ValueError(f"{label}.blocking_geoms must contain geom names")
    expected_behavior = item.get("expected_behavior")
    if not isinstance(expected_behavior, str) or not expected_behavior:
      raise ValueError(f"{label}.expected_behavior must be a non-empty string")
    allowed_region = cast(
      tuple[float, float, float, float],
      _float_tuple(item.get("allowed_region_xy"), 4, f"{label}.allowed_region_xy"),
    )
    if allowed_region[0] >= allowed_region[1] or allowed_region[2] >= allowed_region[3]:
      raise ValueError(f"{label}.allowed_region_xy bounds must be ordered")
    scenarios.append(
      P7Scenario(
        id=scenario_id,
        split=split,
        spawn_point=cast(str, spawn_point),
        nominal_command=nominal_command,
        blocking_geoms=tuple(blocking_geoms),
        safe_exit_sectors_deg=sectors,
        allowed_region_xy=allowed_region,
        expected_behavior=expected_behavior,
        min_clearance_m=_positive_float(
          item.get("min_clearance_m"), f"{label}.min_clearance_m"
        ),
        search_time_budget_s=_positive_float(
          item.get("search_time_budget_s"), f"{label}.search_time_budget_s"
        ),
        search_angle_budget_deg=_positive_float(
          item.get("search_angle_budget_deg"),
          f"{label}.search_angle_budget_deg",
        ),
        review_status=cast(Literal["pending", "approved"], review_status),
      )
    )

  required = {"free", "left_open", "right_open", "both_open", "dead_end", "false_alarm"}
  missing = required - ids
  if missing:
    raise ValueError(f"P7 scenario catalog is missing: {', '.join(sorted(missing))}")
  return P7ScenarioCatalog(
    source_path=source_path,
    scene_manifest=scene_manifest,
    command=command,
    thresholds=thresholds,
    scenarios=tuple(scenarios),
  )


def unitree_g1_safe_velocity_3dgs_env_cfg(
  play: bool = False,
  scenario_id: str = "both_open",
) -> ManagerBasedRlEnvCfg:
  """Create the isolated P7 task skeleton for one deterministic scenario."""
  cfg = unitree_g1_flat_3dgs_depth_validity_history_robust_env_cfg(play=play)
  catalog = load_p7_scenario_catalog()
  scenario = catalog.scenario(scenario_id)
  apply_scene_manifest(
    cfg.scene,
    catalog.scene_manifest,
    spawn_point=scenario.spawn_point,
  )

  # Keep replicated apartments isolated when more than one world is requested.
  cfg.scene.env_spacing = 12.0
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 256)
  cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch, 256)

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_heading_envs = 0.0
  twist.rel_world_envs = 0.0
  twist.rel_standing_envs = 0.0
  twist.rel_forward_envs = 0.8
  twist.resampling_time_range = catalog.command.resampling_time_s
  twist.ranges.lin_vel_x = catalog.command.lin_vel_x
  twist.ranges.lin_vel_y = catalog.command.lin_vel_y
  twist.ranges.ang_vel_z = catalog.command.ang_vel_z

  # P7-0 uses deterministic scenario poses. Randomized spawn curricula are added
  # only after the catalog has passed visual/collision review.
  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.02, 0.02),
    "yaw": (0.0, 0.0),
  }
  cfg.events.pop("push_robot", None)

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  return cfg


def unitree_g1_safe_velocity_3dgs_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reuse the P6 actor-camera/asymmetric-critic runner for P7."""
  cfg = unitree_g1_3dgs_depth_validity_history_robust_asymmetric_ppo_runner_cfg()
  cfg.experiment_name = "g1_3dgs_safe_velocity"
  return cfg


def unitree_g1_safe_velocity_adapter_3dgs_env_cfg(
  play: bool = False,
  scenario_id: str = "both_open",
) -> ManagerBasedRlEnvCfg:
  """Create the evaluation task for a blind policy behind the depth adapter."""
  cfg = unitree_g1_safe_velocity_3dgs_env_cfg(
    play=play,
    scenario_id=scenario_id,
  )
  catalog = load_p7_scenario_catalog()
  cfg.observations["actor"].terms["command"] = ObservationTermCfg(
    func=adapted_velocity_command,
    params={
      "command_name": "twist",
      "sensor_name": "g1_3dgs_depth",
      "adapter_cfg": P7DepthAdapterCfg(
        collision_risk_distance_m=(catalog.thresholds.collision_risk_distance_m),
        stop_clearance_m=catalog.thresholds.stop_clearance_m,
        search_time_budget_s=catalog.thresholds.search_time_budget_s,
      ),
    },
  )
  # The camera remains in the scene for the adapter, but the blind policy and its
  # MLP runner do not consume an image observation group.
  cfg.observations.pop("camera")
  return cfg


def unitree_g1_safe_velocity_adapter_3dgs_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Keep the adapter baseline actor compatible with the standard G1 MLP."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_3dgs_safe_velocity_adapter"
  return cfg


__all__ = [
  "P7_SCENARIO_CATALOG_PATH",
  "P7_ADAPTER_TASK_ID",
  "P7_TASK_ID",
  "P7Scenario",
  "P7ScenarioCatalog",
  "load_p7_scenario_catalog",
  "unitree_g1_safe_velocity_3dgs_env_cfg",
  "unitree_g1_safe_velocity_3dgs_ppo_runner_cfg",
  "unitree_g1_safe_velocity_adapter_3dgs_env_cfg",
  "unitree_g1_safe_velocity_adapter_3dgs_ppo_runner_cfg",
]
