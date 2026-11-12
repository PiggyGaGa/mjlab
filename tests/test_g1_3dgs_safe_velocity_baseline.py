"""Deterministic tests for safe-velocity adapter and outcome semantics."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

from mjlab.tasks.velocity.config.g1.safe_velocity_baseline import (
  SafeVelocityAdapterMode,
  SafeVelocityDepthAdapter,
  SafeVelocityDepthAdapterCfg,
  SafeVelocityOutcome,
  SafeVelocityTrajectoryEvaluator,
  SafeVelocityTrajectoryEvaluatorCfg,
  SafeVelocityTrajectorySample,
  classify_safe_velocity_outcome,
  local_geometry_depth_validity_from_ray_distances,
  local_geometry_stop_command,
  safe_velocity_base_obstacle_collision_cost,
)


def _observation(
  left_m: float,
  front_m: float,
  right_m: float,
  *,
  batch: int = 1,
  valid: bool = True,
) -> torch.Tensor:
  observation = torch.ones((batch, 2, 12, 12))
  for index, clearance in enumerate((left_m, front_m, right_m)):
    observation[:, 0, :, index * 4 : (index + 1) * 4] = clearance / 5.0
  observation[:, 1] = float(valid)
  return observation


def _adapter(**kwargs) -> SafeVelocityDepthAdapter:
  return SafeVelocityDepthAdapter(
    num_envs=1,
    device="cpu",
    cfg=SafeVelocityDepthAdapterCfg(**kwargs),
  )


def test_adapter_tracks_and_brakes_proportionally() -> None:
  adapter = _adapter()
  nominal = torch.tensor([[0.6, 0.0, 0.0]])

  clear = adapter.step(_observation(2.0, 2.0, 2.0), nominal, dt=0.1)
  assert clear.mode.item() == SafeVelocityAdapterMode.TRACK
  torch.testing.assert_close(clear.command, nominal)
  assert not clear.intervention.item()

  braking = adapter.step(_observation(2.0, 1.1, 2.0), nominal, dt=0.1)
  assert braking.mode.item() == SafeVelocityAdapterMode.BRAKE
  assert braking.command[0, 0].item() == pytest.approx(0.5)
  assert braking.intervention.item()


def test_adapter_starts_search_before_reaching_minimum_clearance() -> None:
  adapter = _adapter()
  nominal = torch.tensor([[0.5, 0.0, 0.0]])

  stopped = adapter.step(_observation(2.0, 0.95, 0.4), nominal, dt=0.1)

  assert stopped.mode.item() == SafeVelocityAdapterMode.STOP
  assert torch.count_nonzero(stopped.command) == 0


@pytest.mark.parametrize(
  "left_m,right_m,expected_sign",
  [(2.0, 0.4, 1.0), (0.4, 2.0, -1.0)],
)
def test_adapter_stops_then_searches_safer_side(
  left_m: float, right_m: float, expected_sign: float
) -> None:
  adapter = _adapter()
  nominal = torch.tensor([[0.5, 0.0, 0.0]])

  stopped = adapter.step(_observation(left_m, 0.4, right_m), nominal, dt=0.1)
  assert stopped.mode.item() == SafeVelocityAdapterMode.STOP
  assert torch.count_nonzero(stopped.command) == 0

  searching = adapter.step(_observation(left_m, 0.4, right_m), nominal, dt=0.1)
  assert searching.mode.item() == SafeVelocityAdapterMode.SEARCH
  assert searching.command[0, 2].item() == pytest.approx(expected_sign * 0.5)
  assert searching.command[0, :2].abs().sum().item() == 0.0
  assert searching.search_elapsed_s.item() == pytest.approx(0.1)


def test_adapter_keeps_search_direction_and_resumes_on_clear_front() -> None:
  adapter = _adapter()
  nominal = torch.tensor([[0.5, 0.0, 0.0]])
  adapter.step(_observation(2.0, 0.4, 0.4), nominal, dt=0.1)
  first_search = adapter.step(_observation(0.4, 0.4, 2.0), nominal, dt=0.1)
  assert first_search.command[0, 2] > 0

  resumed = adapter.step(_observation(0.4, 2.0, 2.0), nominal, dt=0.1)
  assert resumed.mode.item() == SafeVelocityAdapterMode.RESUME
  assert resumed.command[0, 0].item() == pytest.approx(0.3)
  assert resumed.command[0, 2].item() == 0.0


def test_adapter_blocks_after_search_budget() -> None:
  adapter = _adapter(search_time_budget_s=0.2)
  nominal = torch.tensor([[0.5, 0.0, 0.0]])
  blocked = _observation(0.4, 0.4, 0.4)

  adapter.step(blocked, nominal, dt=0.1)
  adapter.step(blocked, nominal, dt=0.1)
  result = adapter.step(blocked, nominal, dt=0.1)

  assert result.mode.item() == SafeVelocityAdapterMode.BLOCKED
  assert result.search_timed_out.item()
  assert result.search_elapsed_s.item() == pytest.approx(0.2)
  assert torch.count_nonzero(result.command) == 0


def test_adapter_treats_invalid_depth_conservatively() -> None:
  adapter = _adapter()
  nominal = torch.tensor([[0.5, 0.0, 0.0]])

  result = adapter.step(_observation(5.0, 5.0, 5.0, valid=False), nominal, dt=0.1)

  assert result.mode.item() == SafeVelocityAdapterMode.STOP
  assert torch.count_nonzero(result.command) == 0
  assert result.sector_valid_ratio.max().item() == 0.0


def test_disabled_adapter_is_exact_passthrough() -> None:
  adapter = _adapter(enabled=False)
  nominal = torch.tensor([[0.5, -0.1, 0.2]])

  result = adapter.step(_observation(0.2, 0.2, 0.2), nominal, dt=0.1)

  torch.testing.assert_close(result.command, nominal)
  assert result.mode.item() == SafeVelocityAdapterMode.TRACK
  assert not result.intervention.item()
  assert adapter.intervention_count.item() == 0


def test_local_geometry_treats_raycast_miss_as_known_clear() -> None:
  distances = torch.tensor(
    [[-1.0, 0.5, float("nan"), float("inf")]],
    dtype=torch.float32,
  )

  observation = local_geometry_depth_validity_from_ray_distances(
    distances,
    cutoff_distance_m=5.0,
  )

  torch.testing.assert_close(
    observation[:, 0, 0],
    torch.tensor([[1.0, 0.1, 1.0, 1.0]]),
  )
  torch.testing.assert_close(
    observation[:, 1, 0],
    torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
  )


def test_raycast_misses_allow_adapter_tracking_and_search_resume() -> None:
  adapter = _adapter()
  nominal = torch.tensor([[0.5, 0.0, 0.0]])
  clear_rays = local_geometry_depth_validity_from_ray_distances(
    torch.full((1, 9), -1.0),
    cutoff_distance_m=5.0,
  )
  blocked_rays = local_geometry_depth_validity_from_ray_distances(
    torch.tensor([[2.0, 2.0, 2.0, 0.4, 0.4, 0.4, -1.0, -1.0, -1.0]]),
    cutoff_distance_m=5.0,
  )

  tracking = adapter.step(clear_rays, nominal, dt=0.1)
  assert tracking.mode.item() == SafeVelocityAdapterMode.TRACK
  torch.testing.assert_close(tracking.command, nominal)

  stopped = adapter.step(blocked_rays, nominal, dt=0.1)
  assert stopped.mode.item() == SafeVelocityAdapterMode.STOP
  searching = adapter.step(blocked_rays, nominal, dt=0.1)
  assert searching.mode.item() == SafeVelocityAdapterMode.SEARCH
  assert searching.command[0, 2] < 0

  resumed = adapter.step(clear_rays, nominal, dt=0.1)
  assert resumed.mode.item() == SafeVelocityAdapterMode.RESUME
  assert resumed.command[0, 0].item() == pytest.approx(0.3)


def test_local_geometry_stop_command_ignores_nominal_speed_at_obstacle() -> None:
  cfg = SafeVelocityDepthAdapterCfg()
  nominal = torch.tensor([[4.0, -2.0, 3.0], [0.5, 0.0, 0.2]])
  distances = torch.tensor(
    [
      [-1.0, -1.0, -1.0, 0.8, 0.7, 0.9, -1.0, -1.0, -1.0],
      [0.4, 0.5, 0.6, -1.0, -1.0, -1.0, 0.4, 0.5, 0.6],
    ]
  )
  observation = local_geometry_depth_validity_from_ray_distances(
    distances,
    cutoff_distance_m=cfg.cutoff_distance_m,
  )

  command, stop = local_geometry_stop_command(observation, nominal, cfg)

  assert stop.tolist() == [True, False]
  torch.testing.assert_close(command[0], torch.zeros(3))
  torch.testing.assert_close(command[1], nominal[1])


def test_local_geometry_stop_command_is_conservative_for_invalid_front() -> None:
  cfg = SafeVelocityDepthAdapterCfg()
  nominal = torch.tensor([[0.8, 0.0, -0.6]])
  distances = torch.full((1, 9), -1.0)
  distances[0, 4] = torch.nan
  observation = local_geometry_depth_validity_from_ray_distances(
    distances,
    cutoff_distance_m=cfg.cutoff_distance_m,
  )

  command, stop = local_geometry_stop_command(observation, nominal, cfg)

  assert stop.item()
  torch.testing.assert_close(command, torch.zeros_like(nominal))


def test_base_obstacle_collision_cost_aggregates_policy_substeps() -> None:
  clear = torch.zeros((2, 1, 4, 3))
  contact = clear.clone()
  contact[0, 0, 1, 0] = 11.0
  contact[0, 0, 3, 0] = 12.0
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    scene={
      "obstacle_0": SimpleNamespace(data=SimpleNamespace(force_history=contact)),
      "obstacle_1": SimpleNamespace(data=SimpleNamespace(force_history=clear)),
    },
  )

  cost = safe_velocity_base_obstacle_collision_cost(
    cast("ManagerBasedRlEnv", env),
    sensor_names=("obstacle_0", "obstacle_1"),
    force_threshold=10.0,
  )

  torch.testing.assert_close(cost, torch.tensor([2.0, 0.0]))


def test_adapter_supports_batched_history_and_partial_reset() -> None:
  adapter = SafeVelocityDepthAdapter(num_envs=2, device="cpu")
  nominal = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
  frame = torch.cat(
    (
      _observation(2.0, 2.0, 2.0),
      _observation(2.0, 0.4, 0.4),
    )
  )
  history = frame[:, None].expand(-1, 3, -1, -1, -1).clone()

  result = adapter.step(history, nominal, dt=0.1)

  assert result.mode.tolist() == [
    SafeVelocityAdapterMode.TRACK,
    SafeVelocityAdapterMode.STOP,
  ]
  assert adapter.intervention_count.tolist() == [0, 1]
  adapter.reset(torch.tensor([1]))
  assert adapter.mode.tolist() == [
    SafeVelocityAdapterMode.TRACK,
    SafeVelocityAdapterMode.TRACK,
  ]
  assert adapter.intervention_count.tolist() == [0, 0]


@pytest.mark.parametrize(
  "flag,expected",
  [
    ("continuous_tracking", SafeVelocityOutcome.CONTINUOUS_TRACKING),
    ("avoidance_resume", SafeVelocityOutcome.AVOIDANCE_RESUME),
    ("safe_blocked_stop", SafeVelocityOutcome.SAFE_BLOCKED_STOP),
    ("collision", SafeVelocityOutcome.COLLISION),
    ("fall", SafeVelocityOutcome.FALL),
    ("out_of_bounds", SafeVelocityOutcome.OUT_OF_BOUNDS),
    ("search_timeout", SafeVelocityOutcome.SEARCH_TIMEOUT),
    ("stagnation", SafeVelocityOutcome.STAGNATION),
    ("episode_timeout", SafeVelocityOutcome.EPISODE_TIMEOUT),
  ],
)
def test_all_primary_outcomes_are_reachable(
  flag: str, expected: SafeVelocityOutcome
) -> None:
  assert classify_safe_velocity_outcome(**{flag: True}) == expected


def test_outcome_precedence_is_safety_first() -> None:
  assert (
    classify_safe_velocity_outcome(
      collision=True,
      avoidance_resume=True,
      continuous_tracking=True,
      episode_timeout=True,
    )
    == SafeVelocityOutcome.COLLISION
  )


def test_outcome_requires_a_terminal_condition() -> None:
  with pytest.raises(ValueError, match="At least one"):
    classify_safe_velocity_outcome()


def _trajectory_outcome(
  samples: list[SafeVelocityTrajectorySample],
) -> SafeVelocityOutcome:
  evaluator = SafeVelocityTrajectoryEvaluator(
    SafeVelocityTrajectoryEvaluatorCfg(allowed_region_xy=(-1.0, 1.0, -1.0, 1.0))
  )
  for sample in samples:
    evaluator.update(sample)
  return evaluator.summarize().outcome


def _sample(
  time_s: float,
  mode: SafeVelocityAdapterMode,
  x: float = 0.0,
  **kwargs,
) -> SafeVelocityTrajectorySample:
  return SafeVelocityTrajectorySample(
    time_s=time_s,
    mode=mode,
    position_xy=(x, 0.0),
    planar_speed_mps=kwargs.pop("planar_speed_mps", 0.2),
    front_clearance_m=kwargs.pop("front_clearance_m", 1.0),
    **kwargs,
  )


@pytest.mark.parametrize(
  "samples,expected",
  [
    (
      [
        _sample(0.0, SafeVelocityAdapterMode.TRACK),
        _sample(1.0, SafeVelocityAdapterMode.TRACK, 0.2),
      ],
      SafeVelocityOutcome.CONTINUOUS_TRACKING,
    ),
    (
      [
        _sample(0.0, SafeVelocityAdapterMode.BRAKE),
        _sample(1.0, SafeVelocityAdapterMode.RESUME),
        _sample(2.0, SafeVelocityAdapterMode.TRACK, 0.2),
      ],
      SafeVelocityOutcome.AVOIDANCE_RESUME,
    ),
    (
      [
        _sample(0.0, SafeVelocityAdapterMode.SEARCH),
        _sample(
          1.0,
          SafeVelocityAdapterMode.BLOCKED,
          planar_speed_mps=0.0,
          search_timed_out=True,
        ),
        _sample(2.1, SafeVelocityAdapterMode.BLOCKED, planar_speed_mps=0.0),
      ],
      SafeVelocityOutcome.SAFE_BLOCKED_STOP,
    ),
    (
      [_sample(0.0, SafeVelocityAdapterMode.TRACK, collision=True)],
      SafeVelocityOutcome.COLLISION,
    ),
    (
      [_sample(0.0, SafeVelocityAdapterMode.TRACK, fall=True)],
      SafeVelocityOutcome.FALL,
    ),
    (
      [_sample(0.0, SafeVelocityAdapterMode.TRACK, x=1.1)],
      SafeVelocityOutcome.OUT_OF_BOUNDS,
    ),
    (
      [
        _sample(0.0, SafeVelocityAdapterMode.SEARCH),
        _sample(
          1.0,
          SafeVelocityAdapterMode.BLOCKED,
          planar_speed_mps=0.0,
          search_timed_out=True,
        ),
      ],
      SafeVelocityOutcome.SEARCH_TIMEOUT,
    ),
    (
      [
        _sample(0.0, SafeVelocityAdapterMode.TRACK),
        _sample(2.1, SafeVelocityAdapterMode.TRACK, 0.01),
      ],
      SafeVelocityOutcome.STAGNATION,
    ),
    (
      [
        _sample(0.0, SafeVelocityAdapterMode.BRAKE),
        _sample(1.0, SafeVelocityAdapterMode.BRAKE, episode_timeout=True),
      ],
      SafeVelocityOutcome.EPISODE_TIMEOUT,
    ),
  ],
)
def test_trajectory_evaluator_triggers_all_primary_outcomes(
  samples: list[SafeVelocityTrajectorySample], expected: SafeVelocityOutcome
) -> None:
  assert _trajectory_outcome(samples) == expected


def test_trajectory_evaluator_accumulates_safety_metrics() -> None:
  evaluator = SafeVelocityTrajectoryEvaluator(
    SafeVelocityTrajectoryEvaluatorCfg(allowed_region_xy=(-1.0, 1.0, -1.0, 1.0))
  )
  evaluator.update(_sample(0.0, SafeVelocityAdapterMode.TRACK, front_clearance_m=1.2))
  evaluator.update(
    _sample(
      1.0,
      SafeVelocityAdapterMode.TRACK,
      x=0.3,
      front_clearance_m=0.7,
      collision_force_n=12.0,
      collision_impulse_ns=0.4,
    )
  )

  summary = evaluator.summarize()

  assert summary.outcome == SafeVelocityOutcome.COLLISION
  assert summary.path_length_m == pytest.approx(0.3)
  assert summary.min_sensor_clearance_m == pytest.approx(0.7)
  assert summary.max_collision_force_n == pytest.approx(12.0)
  assert summary.collision_impulse_ns == pytest.approx(0.4)
