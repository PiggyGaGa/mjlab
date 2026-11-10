"""Deterministic tests for the P7 depth adapter and outcome semantics."""

import pytest
import torch

from mjlab.tasks.velocity.config.g1.safe_velocity_baseline import (
  P7AdapterMode,
  P7DepthAdapterCfg,
  P7DeterministicDepthAdapter,
  P7Outcome,
  classify_p7_outcome,
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


def _adapter(**kwargs) -> P7DeterministicDepthAdapter:
  return P7DeterministicDepthAdapter(
    num_envs=1,
    device="cpu",
    cfg=P7DepthAdapterCfg(**kwargs),
  )


def test_adapter_tracks_and_brakes_proportionally() -> None:
  adapter = _adapter()
  nominal = torch.tensor([[0.6, 0.0, 0.0]])

  clear = adapter.step(_observation(2.0, 2.0, 2.0), nominal, dt=0.1)
  assert clear.mode.item() == P7AdapterMode.TRACK
  torch.testing.assert_close(clear.command, nominal)
  assert not clear.intervention.item()

  braking = adapter.step(_observation(2.0, 0.9, 2.0), nominal, dt=0.1)
  assert braking.mode.item() == P7AdapterMode.BRAKE
  assert braking.command[0, 0].item() == pytest.approx(0.3)
  assert braking.intervention.item()


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
  assert stopped.mode.item() == P7AdapterMode.STOP
  assert torch.count_nonzero(stopped.command) == 0

  searching = adapter.step(_observation(left_m, 0.4, right_m), nominal, dt=0.1)
  assert searching.mode.item() == P7AdapterMode.SEARCH
  assert searching.command[0, 2].item() == pytest.approx(expected_sign * 0.5)
  assert searching.command[0, :2].abs().sum().item() == 0.0


def test_adapter_keeps_search_direction_and_resumes_on_clear_front() -> None:
  adapter = _adapter()
  nominal = torch.tensor([[0.5, 0.0, 0.0]])
  adapter.step(_observation(2.0, 0.4, 0.4), nominal, dt=0.1)
  first_search = adapter.step(_observation(0.4, 0.4, 2.0), nominal, dt=0.1)
  assert first_search.command[0, 2] > 0

  resumed = adapter.step(_observation(0.4, 2.0, 2.0), nominal, dt=0.1)
  assert resumed.mode.item() == P7AdapterMode.RESUME
  assert resumed.command[0, 0].item() == pytest.approx(0.3)
  assert resumed.command[0, 2].item() == 0.0


def test_adapter_blocks_after_search_budget() -> None:
  adapter = _adapter(search_time_budget_s=0.2)
  nominal = torch.tensor([[0.5, 0.0, 0.0]])
  blocked = _observation(0.4, 0.4, 0.4)

  adapter.step(blocked, nominal, dt=0.1)
  adapter.step(blocked, nominal, dt=0.1)
  result = adapter.step(blocked, nominal, dt=0.1)

  assert result.mode.item() == P7AdapterMode.BLOCKED
  assert result.search_timed_out.item()
  assert torch.count_nonzero(result.command) == 0


def test_adapter_treats_invalid_depth_conservatively() -> None:
  adapter = _adapter()
  nominal = torch.tensor([[0.5, 0.0, 0.0]])

  result = adapter.step(_observation(5.0, 5.0, 5.0, valid=False), nominal, dt=0.1)

  assert result.mode.item() == P7AdapterMode.STOP
  assert torch.count_nonzero(result.command) == 0
  assert result.sector_valid_ratio.max().item() == 0.0


def test_disabled_adapter_is_exact_passthrough() -> None:
  adapter = _adapter(enabled=False)
  nominal = torch.tensor([[0.5, -0.1, 0.2]])

  result = adapter.step(_observation(0.2, 0.2, 0.2), nominal, dt=0.1)

  torch.testing.assert_close(result.command, nominal)
  assert result.mode.item() == P7AdapterMode.TRACK
  assert not result.intervention.item()
  assert adapter.intervention_count.item() == 0


def test_adapter_supports_batched_history_and_partial_reset() -> None:
  adapter = P7DeterministicDepthAdapter(num_envs=2, device="cpu")
  nominal = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
  frame = torch.cat(
    (
      _observation(2.0, 2.0, 2.0),
      _observation(2.0, 0.4, 0.4),
    )
  )
  history = frame[:, None].expand(-1, 3, -1, -1, -1).clone()

  result = adapter.step(history, nominal, dt=0.1)

  assert result.mode.tolist() == [P7AdapterMode.TRACK, P7AdapterMode.STOP]
  assert adapter.intervention_count.tolist() == [0, 1]
  adapter.reset(torch.tensor([1]))
  assert adapter.mode.tolist() == [P7AdapterMode.TRACK, P7AdapterMode.TRACK]
  assert adapter.intervention_count.tolist() == [0, 0]


@pytest.mark.parametrize(
  "flag,expected",
  [
    ("continuous_tracking", P7Outcome.CONTINUOUS_TRACKING),
    ("avoidance_resume", P7Outcome.AVOIDANCE_RESUME),
    ("safe_blocked_stop", P7Outcome.SAFE_BLOCKED_STOP),
    ("collision", P7Outcome.COLLISION),
    ("fall", P7Outcome.FALL),
    ("out_of_bounds", P7Outcome.OUT_OF_BOUNDS),
    ("search_timeout", P7Outcome.SEARCH_TIMEOUT),
    ("stagnation", P7Outcome.STAGNATION),
    ("episode_timeout", P7Outcome.EPISODE_TIMEOUT),
  ],
)
def test_all_primary_outcomes_are_reachable(flag: str, expected: P7Outcome) -> None:
  assert classify_p7_outcome(**{flag: True}) == expected


def test_outcome_precedence_is_safety_first() -> None:
  assert (
    classify_p7_outcome(
      collision=True,
      avoidance_resume=True,
      continuous_tracking=True,
      episode_timeout=True,
    )
    == P7Outcome.COLLISION
  )


def test_outcome_requires_a_terminal_condition() -> None:
  with pytest.raises(ValueError, match="At least one"):
    classify_p7_outcome()
