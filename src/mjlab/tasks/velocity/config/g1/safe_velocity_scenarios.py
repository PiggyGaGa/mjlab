"""Typed scenario protocol for G1 safe perceptive velocity tracking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SafeVelocityCommandProtocol:
  """Frozen initial command ranges for safe velocity tracking."""

  resampling_time_s: tuple[float, float]
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  ang_vel_z: tuple[float, float]


@dataclass(frozen=True)
class SafeVelocityThresholds:
  """Initial collision, stopping, and search thresholds."""

  collision_risk_distance_m: float
  stop_clearance_m: float
  search_trigger_distance_m: float
  search_time_budget_s: float
  search_angle_budget_deg: float


@dataclass(frozen=True)
class SafeVelocityScenario:
  """One deterministic local-obstacle scenario."""

  id: str
  split: Literal["train", "validation", "test"]
  spawn_point: str
  nominal_command: tuple[float, float, float]
  blocking_geoms: tuple[str, ...]
  safe_exit_sectors_deg: tuple[tuple[float, float], ...]
  allowed_region_xy: tuple[float, float, float, float]
  expected_behavior: str
  expected_outcome: Literal[
    "continuous_tracking", "avoidance_resume", "safe_blocked_stop"
  ]
  evaluation_time_s: float
  min_clearance_m: float = 0.6
  search_time_budget_s: float = 18.0
  search_angle_budget_deg: float = 180.0
  review_status: Literal["pending", "approved"] = "approved"


@dataclass(frozen=True)
class SafeVelocityScenarioCatalog:
  """Versioned safe-velocity scenarios bound to one scene manifest."""

  scene_manifest: Path
  command: SafeVelocityCommandProtocol
  thresholds: SafeVelocityThresholds
  scenarios: tuple[SafeVelocityScenario, ...]

  def scenario(self, scenario_id: str) -> SafeVelocityScenario:
    """Return a scenario by ID with a useful error for config callers."""
    match = next((item for item in self.scenarios if item.id == scenario_id), None)
    if match is not None:
      return match
    available = ", ".join(item.id for item in self.scenarios)
    raise ValueError(f"Unknown safe-velocity scenario '{scenario_id}'; {available}")


_APARTMENT_MANIFEST = (
  Path(__file__).parents[4] / "asset_zoo/3dgs/apartment/manifest.json"
).resolve()
_ALLOWED_REGION = (-2.2, 2.5, -5.4, 2.9)

SAFE_VELOCITY_SCENARIO_CATALOG = SafeVelocityScenarioCatalog(
  scene_manifest=_APARTMENT_MANIFEST,
  command=SafeVelocityCommandProtocol(
    resampling_time_s=(8.0, 12.0),
    lin_vel_x=(0.0, 0.8),
    lin_vel_y=(0.0, 0.0),
    ang_vel_z=(-0.6, 0.6),
  ),
  thresholds=SafeVelocityThresholds(
    collision_risk_distance_m=1.2,
    stop_clearance_m=0.6,
    search_trigger_distance_m=1.0,
    search_time_budget_s=18.0,
    search_angle_budget_deg=180.0,
  ),
  scenarios=(
    SafeVelocityScenario(
      id="free",
      split="train",
      spawn_point="p7_free",
      nominal_command=(0.5, 0.0, 0.0),
      blocking_geoms=(),
      safe_exit_sectors_deg=((-20.0, 20.0),),
      allowed_region_xy=_ALLOWED_REGION,
      expected_behavior="track",
      expected_outcome="continuous_tracking",
      evaluation_time_s=10.0,
    ),
    SafeVelocityScenario(
      id="left_open",
      split="train",
      spawn_point="p7_left_open",
      nominal_command=(0.5, 0.0, 0.0),
      blocking_geoms=("pillar_01",),
      safe_exit_sectors_deg=((30.0, 80.0),),
      allowed_region_xy=_ALLOWED_REGION,
      expected_behavior="brake_search_left_resume",
      expected_outcome="avoidance_resume",
      evaluation_time_s=22.0,
    ),
    SafeVelocityScenario(
      id="right_open",
      split="validation",
      spawn_point="p7_right_open",
      nominal_command=(0.5, 0.0, 0.0),
      blocking_geoms=("pillar_00",),
      safe_exit_sectors_deg=((-80.0, -30.0),),
      allowed_region_xy=_ALLOWED_REGION,
      expected_behavior="brake_search_right_resume",
      expected_outcome="avoidance_resume",
      evaluation_time_s=22.0,
    ),
    SafeVelocityScenario(
      id="both_open",
      split="train",
      spawn_point="p7_both_open",
      nominal_command=(0.5, 0.0, 0.0),
      blocking_geoms=("pillar_00", "pillar_01"),
      safe_exit_sectors_deg=((-80.0, -30.0), (30.0, 80.0)),
      allowed_region_xy=_ALLOWED_REGION,
      expected_behavior="brake_search_either_resume",
      expected_outcome="avoidance_resume",
      evaluation_time_s=22.0,
    ),
    SafeVelocityScenario(
      id="dead_end",
      split="test",
      spawn_point="p7_dead_end",
      nominal_command=(0.5, 0.0, 0.0),
      blocking_geoms=("surface_0002", "surface_0003"),
      safe_exit_sectors_deg=(),
      allowed_region_xy=_ALLOWED_REGION,
      expected_behavior="brake_stop_blocked",
      expected_outcome="safe_blocked_stop",
      evaluation_time_s=22.0,
    ),
    SafeVelocityScenario(
      id="false_alarm",
      split="test",
      spawn_point="p7_false_alarm",
      nominal_command=(0.5, 0.0, 0.0),
      blocking_geoms=(),
      safe_exit_sectors_deg=((-20.0, 20.0),),
      allowed_region_xy=_ALLOWED_REGION,
      expected_behavior="track_despite_visual_structure",
      expected_outcome="continuous_tracking",
      evaluation_time_s=10.0,
    ),
  ),
)


__all__ = [
  "SAFE_VELOCITY_SCENARIO_CATALOG",
  "SafeVelocityCommandProtocol",
  "SafeVelocityScenario",
  "SafeVelocityScenarioCatalog",
  "SafeVelocityThresholds",
]
