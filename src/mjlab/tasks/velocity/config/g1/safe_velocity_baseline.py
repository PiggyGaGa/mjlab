"""Deterministic depth baseline and outcome semantics for P7."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TYPE_CHECKING

import torch

from mjlab.sensor import CameraSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg


class P7AdapterMode(IntEnum):
  """Observable phases of the deterministic depth adapter."""

  TRACK = 0
  BRAKE = 1
  STOP = 2
  SEARCH = 3
  RESUME = 4
  BLOCKED = 5


class P7Outcome(str, Enum):
  """Mutually exclusive primary episode outcomes required by P7."""

  CONTINUOUS_TRACKING = "continuous_tracking"
  AVOIDANCE_RESUME = "avoidance_resume"
  SAFE_BLOCKED_STOP = "safe_blocked_stop"
  COLLISION = "collision"
  FALL = "fall"
  OUT_OF_BOUNDS = "out_of_bounds"
  SEARCH_TIMEOUT = "search_timeout"
  STAGNATION = "stagnation"
  EPISODE_TIMEOUT = "episode_timeout"


@dataclass(frozen=True)
class P7DepthAdapterCfg:
  """Deployable thresholds for the task-local deterministic baseline."""

  cutoff_distance_m: float = 5.0
  collision_risk_distance_m: float = 1.2
  stop_clearance_m: float = 0.6
  minimum_valid_ratio: float = 0.1
  clearance_quantile: float = 0.1
  search_yaw_rate: float = 0.5
  search_time_budget_s: float = 6.0
  resume_speed: float = 0.3
  vertical_roi: tuple[float, float] = (0.2, 0.85)
  enabled: bool = True

  def __post_init__(self) -> None:
    if self.cutoff_distance_m <= 0:
      raise ValueError("cutoff_distance_m must be positive")
    if not 0 < self.stop_clearance_m < self.collision_risk_distance_m:
      raise ValueError("clearance thresholds must satisfy 0 < stop < risk")
    if not 0 <= self.minimum_valid_ratio <= 1:
      raise ValueError("minimum_valid_ratio must be in [0, 1]")
    if not 0 <= self.clearance_quantile <= 1:
      raise ValueError("clearance_quantile must be in [0, 1]")
    if self.search_yaw_rate <= 0 or self.search_time_budget_s <= 0:
      raise ValueError("search rate and time budget must be positive")
    if self.resume_speed < 0:
      raise ValueError("resume_speed must be non-negative")
    if not 0 <= self.vertical_roi[0] < self.vertical_roi[1] <= 1:
      raise ValueError("vertical_roi must satisfy 0 <= min < max <= 1")


@dataclass(frozen=True)
class P7DepthAdapterOutput:
  """One batch of adapter decisions and diagnostics."""

  command: torch.Tensor
  mode: torch.Tensor
  sector_clearance_m: torch.Tensor
  sector_valid_ratio: torch.Tensor
  intervention: torch.Tensor
  search_timed_out: torch.Tensor


class P7DeterministicDepthAdapter:
  """Brake, rotate toward free depth, and resume without localization.

  Input depth is the normalized P6 depth+validity observation. Both a single
  frame ``(B, 2, H, W)`` and temporal history ``(B, T, 2, H, W)`` are accepted;
  the baseline deliberately consumes only the latest frame.
  """

  def __init__(
    self,
    num_envs: int,
    device: str | torch.device,
    cfg: P7DepthAdapterCfg | None = None,
  ) -> None:
    if num_envs < 1:
      raise ValueError("num_envs must be positive")
    self.cfg = cfg or P7DepthAdapterCfg()
    self.mode = torch.full(
      (num_envs,), int(P7AdapterMode.TRACK), dtype=torch.long, device=device
    )
    self.search_elapsed_s = torch.zeros(num_envs, device=device)
    self.search_direction = torch.ones(num_envs, device=device)
    self.intervention_count = torch.zeros(num_envs, dtype=torch.long, device=device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset all or selected adapter states."""
    ids = slice(None) if env_ids is None else env_ids
    self.mode[ids] = int(P7AdapterMode.TRACK)
    self.search_elapsed_s[ids] = 0.0
    self.search_direction[ids] = 1.0
    self.intervention_count[ids] = 0

  def step(
    self,
    depth_validity: torch.Tensor,
    nominal_command: torch.Tensor,
    dt: float,
  ) -> P7DepthAdapterOutput:
    """Return a safe command and structured intervention diagnostics."""
    if dt <= 0:
      raise ValueError("dt must be positive")
    frame = self._latest_frame(depth_validity)
    if nominal_command.shape != (len(self.mode), 3):
      raise ValueError(
        f"nominal_command must have shape {(len(self.mode), 3)}, got "
        f"{tuple(nominal_command.shape)}"
      )
    if nominal_command.device != self.mode.device:
      raise ValueError("nominal_command and adapter state must use the same device")

    clearance, valid_ratio = self._sector_statistics(frame)
    command = nominal_command.clone()
    timed_out = torch.zeros_like(self.mode, dtype=torch.bool)
    if not self.cfg.enabled:
      intervention = torch.zeros_like(self.mode, dtype=torch.bool)
      return P7DepthAdapterOutput(
        command, self.mode.clone(), clearance, valid_ratio, intervention, timed_out
      )

    front_clearance = clearance[:, 1]
    front_valid = valid_ratio[:, 1] >= self.cfg.minimum_valid_ratio
    front_safe = front_valid & (front_clearance >= self.cfg.collision_risk_distance_m)
    must_stop = (~front_valid) | (front_clearance <= self.cfg.stop_clearance_m)
    risk = (~front_safe) & (~must_stop)

    current = self.mode.clone()
    active_motion = (
      (current == int(P7AdapterMode.TRACK))
      | (current == int(P7AdapterMode.BRAKE))
      | (current == int(P7AdapterMode.RESUME))
    )
    track = active_motion & front_safe
    self.mode[track] = int(P7AdapterMode.TRACK)

    braking = active_motion & risk
    scale = (
      (front_clearance - self.cfg.stop_clearance_m)
      / (self.cfg.collision_risk_distance_m - self.cfg.stop_clearance_m)
    ).clamp(0.0, 1.0)
    command[braking, 0] = command[braking, 0].clamp(min=0.0) * scale[braking]
    command[braking, 1:] = 0.0
    self.mode[braking] = int(P7AdapterMode.BRAKE)

    stopping = active_motion & must_stop
    command[stopping] = 0.0
    self.mode[stopping] = int(P7AdapterMode.STOP)
    self.search_elapsed_s[stopping] = 0.0
    prefer_left = clearance[:, 0] >= clearance[:, 2]
    self.search_direction[stopping] = torch.where(prefer_left[stopping], 1.0, -1.0)

    start_search = current == int(P7AdapterMode.STOP)
    searching = current == int(P7AdapterMode.SEARCH)
    search_active = start_search | (searching & ~front_safe)
    self.search_elapsed_s[search_active] += dt
    command[search_active] = 0.0
    command[search_active, 2] = (
      self.search_direction[search_active] * self.cfg.search_yaw_rate
    )
    self.mode[search_active] = int(P7AdapterMode.SEARCH)

    timed_out = search_active & (self.search_elapsed_s >= self.cfg.search_time_budget_s)
    command[timed_out] = 0.0
    self.mode[timed_out] = int(P7AdapterMode.BLOCKED)

    resume = (searching | (current == int(P7AdapterMode.BLOCKED))) & front_safe
    command[resume] = 0.0
    command[resume, 0] = nominal_command[resume, 0].clamp(
      min=0.0, max=self.cfg.resume_speed
    )
    self.mode[resume] = int(P7AdapterMode.RESUME)
    self.search_elapsed_s[resume] = 0.0

    blocked = (current == int(P7AdapterMode.BLOCKED)) & ~front_safe
    command[blocked] = 0.0

    intervention = ~torch.isclose(command, nominal_command).all(dim=-1)
    self.intervention_count += intervention.long()
    return P7DepthAdapterOutput(
      command=command,
      mode=self.mode.clone(),
      sector_clearance_m=clearance,
      sector_valid_ratio=valid_ratio,
      intervention=intervention,
      search_timed_out=timed_out,
    )

  def _latest_frame(self, observation: torch.Tensor) -> torch.Tensor:
    if observation.ndim == 5:
      observation = observation[:, -1]
    expected = (len(self.mode), 2)
    if observation.ndim != 4 or observation.shape[:2] != expected:
      raise ValueError("depth_validity must have shape (B, 2, H, W) or (B, T, 2, H, W)")
    if observation.device != self.mode.device:
      raise ValueError("observation and adapter state must use the same device")
    return observation

  def _sector_statistics(
    self, observation: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    depth = observation[:, 0] * self.cfg.cutoff_distance_m
    validity = observation[:, 1] > 0.5
    height, width = depth.shape[-2:]
    row_start = int(height * self.cfg.vertical_roi[0])
    row_end = max(row_start + 1, int(height * self.cfg.vertical_roi[1]))
    column_edges = (0, width // 3, 2 * width // 3, width)
    clearances: list[torch.Tensor] = []
    valid_ratios: list[torch.Tensor] = []
    for start, end in zip(column_edges[:-1], column_edges[1:], strict=True):
      sector_depth = depth[:, row_start:row_end, start:end]
      sector_valid = validity[:, row_start:row_end, start:end]
      masked = torch.where(
        sector_valid,
        sector_depth,
        torch.full_like(sector_depth, torch.nan),
      )
      sector_clearance = torch.nanquantile(
        masked.flatten(1), self.cfg.clearance_quantile, dim=1
      ).nan_to_num(0.0)
      clearances.append(sector_clearance)
      valid_ratios.append(sector_valid.float().mean(dim=(-2, -1)))
    return torch.stack(clearances, dim=1), torch.stack(valid_ratios, dim=1)


class adapted_velocity_command:
  """Observation term that exposes an adapter command to a blind policy.

  The nominal command remains owned by ``CommandManager`` for evaluation. The
  actor receives only this safe body-frame twist; raw depth is consumed internally
  and is not added to the actor observation group.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv) -> None:
    adapter_cfg = cfg.params.get("adapter_cfg")
    if not isinstance(adapter_cfg, P7DepthAdapterCfg):
      raise TypeError("adapter_cfg must be P7DepthAdapterCfg")
    self.adapter = P7DeterministicDepthAdapter(
      num_envs=env.num_envs,
      device=env.device,
      cfg=adapter_cfg,
    )
    self.last_output: P7DepthAdapterOutput | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.adapter.reset(env_ids)
    self.last_output = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    adapter_cfg: P7DepthAdapterCfg,
  ) -> torch.Tensor:
    del adapter_cfg
    sensor = env.scene[sensor_name]
    if not isinstance(sensor, CameraSensor):
      raise TypeError(
        f"adapted_velocity_command requires CameraSensor, got {type(sensor).__name__}"
      )
    raw = sensor.data.depth
    if raw is None:
      raise ValueError(f"Camera '{sensor_name}' has no depth data")
    valid = torch.isfinite(raw) & (raw > 0)
    cutoff = self.adapter.cfg.cutoff_distance_m
    depth = torch.where(valid, raw, cutoff).clamp_(0.0, cutoff) / cutoff
    observation = torch.cat(
      (
        depth.permute(0, 3, 1, 2),
        valid.permute(0, 3, 1, 2).to(depth.dtype),
      ),
      dim=1,
    )
    nominal = env.command_manager.get_command(command_name)
    if nominal is None:
      raise ValueError(f"Velocity command '{command_name}' is not configured")
    self.last_output = self.adapter.step(observation, nominal, env.step_dt)
    return self.last_output.command


def classify_p7_outcome(
  *,
  collision: bool = False,
  fall: bool = False,
  out_of_bounds: bool = False,
  search_timeout: bool = False,
  stagnation: bool = False,
  avoidance_resume: bool = False,
  safe_blocked_stop: bool = False,
  continuous_tracking: bool = False,
  episode_timeout: bool = False,
) -> P7Outcome:
  """Choose exactly one primary outcome using a fixed safety-first precedence."""
  ordered = (
    (collision, P7Outcome.COLLISION),
    (fall, P7Outcome.FALL),
    (out_of_bounds, P7Outcome.OUT_OF_BOUNDS),
    (search_timeout, P7Outcome.SEARCH_TIMEOUT),
    (stagnation, P7Outcome.STAGNATION),
    (avoidance_resume, P7Outcome.AVOIDANCE_RESUME),
    (safe_blocked_stop, P7Outcome.SAFE_BLOCKED_STOP),
    (continuous_tracking, P7Outcome.CONTINUOUS_TRACKING),
    (episode_timeout, P7Outcome.EPISODE_TIMEOUT),
  )
  match = next((outcome for active, outcome in ordered if active), None)
  if match is None:
    raise ValueError("At least one P7 outcome condition must be true")
  return match


__all__ = [
  "P7AdapterMode",
  "P7DepthAdapterCfg",
  "P7DepthAdapterOutput",
  "P7DeterministicDepthAdapter",
  "P7Outcome",
  "adapted_velocity_command",
  "classify_p7_outcome",
]
