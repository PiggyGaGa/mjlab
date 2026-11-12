"""Deterministic depth baseline and outcome semantics for safe velocity."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import CameraSensor, RayCastSensor
from mjlab.tasks.velocity.mdp import self_collision_cost

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg


class SafeVelocityAdapterMode(IntEnum):
  """Observable phases of the deterministic depth adapter."""

  TRACK = 0
  BRAKE = 1
  STOP = 2
  SEARCH = 3
  RESUME = 4
  BLOCKED = 5


class SafeVelocityOutcome(str, Enum):
  """Mutually exclusive primary safe-velocity episode outcomes."""

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
class SafeVelocityTrajectoryEvaluatorCfg:
  """Thresholds for task-local, privileged rollout evaluation."""

  allowed_region_xy: tuple[float, float, float, float]
  collision_force_threshold_n: float = 10.0
  stagnation_window_s: float = 2.0
  stagnation_distance_m: float = 0.05
  resume_distance_m: float = 0.1
  blocked_hold_s: float = 1.0
  stationary_speed_mps: float = 0.1

  def __post_init__(self) -> None:
    x_min, x_max, y_min, y_max = self.allowed_region_xy
    if x_min >= x_max or y_min >= y_max:
      raise ValueError("allowed_region_xy bounds must be ordered")
    positive = (
      self.collision_force_threshold_n,
      self.stagnation_window_s,
      self.stagnation_distance_m,
      self.resume_distance_m,
      self.blocked_hold_s,
      self.stationary_speed_mps,
    )
    if any(value <= 0 for value in positive):
      raise ValueError("trajectory evaluator thresholds must be positive")


@dataclass(frozen=True)
class SafeVelocityTrajectorySample:
  """Privileged state and adapter diagnostics for one rollout instant."""

  time_s: float
  mode: SafeVelocityAdapterMode
  position_xy: tuple[float, float]
  planar_speed_mps: float
  front_clearance_m: float
  collision: bool = False
  collision_force_n: float = 0.0
  collision_impulse_ns: float = 0.0
  fall: bool = False
  episode_timeout: bool = False
  search_timed_out: bool = False


@dataclass(frozen=True)
class SafeVelocityTrajectorySummary:
  """One mutually exclusive outcome plus reusable safety metrics."""

  outcome: SafeVelocityOutcome
  duration_s: float
  path_length_m: float
  min_sensor_clearance_m: float
  max_collision_force_n: float
  collision_impulse_ns: float
  collision: bool
  fall: bool
  out_of_bounds: bool
  search_timed_out: bool
  stagnation: bool
  avoidance_resumed: bool
  safe_blocked_stop: bool
  intervention: bool


class SafeVelocityTrajectoryEvaluator:
  """Reduce a scripted or simulated trajectory to one safety-first outcome.

  World position and collision signals are deliberately evaluator-only. They are
  not observations and never enter the deployable adapter or policy.
  """

  def __init__(self, cfg: SafeVelocityTrajectoryEvaluatorCfg) -> None:
    self.cfg = cfg
    self._samples: list[SafeVelocityTrajectorySample] = []
    self._active_motion: deque[SafeVelocityTrajectorySample] = deque()
    self._collision = False
    self._fall = False
    self._out_of_bounds = False
    self._search_timed_out = False
    self._stagnation = False
    self._intervention = False
    self._resume_position: tuple[float, float] | None = None
    self._avoidance_resumed = False
    self._blocked_hold_s = 0.0
    self._path_length_m = 0.0
    self._min_clearance_m = math.inf
    self._max_collision_force_n = 0.0
    self._collision_impulse_ns = 0.0

  def update(self, sample: SafeVelocityTrajectorySample) -> None:
    """Consume one chronological sample and update trajectory events."""
    if sample.time_s < 0:
      raise ValueError("sample time_s must be non-negative")
    if sample.planar_speed_mps < 0:
      raise ValueError("sample planar_speed_mps must be non-negative")
    if sample.front_clearance_m < 0:
      raise ValueError("sample front_clearance_m must be non-negative")
    if sample.collision_force_n < 0 or sample.collision_impulse_ns < 0:
      raise ValueError("collision diagnostics must be non-negative")
    if self._samples and sample.time_s <= self._samples[-1].time_s:
      raise ValueError("trajectory sample times must be strictly increasing")

    previous = self._samples[-1] if self._samples else None
    dt = sample.time_s - previous.time_s if previous is not None else 0.0
    if previous is not None:
      self._path_length_m += math.dist(previous.position_xy, sample.position_xy)
    self._samples.append(sample)

    self._min_clearance_m = min(self._min_clearance_m, sample.front_clearance_m)
    self._max_collision_force_n = max(
      self._max_collision_force_n, sample.collision_force_n
    )
    self._collision_impulse_ns += sample.collision_impulse_ns
    self._collision |= sample.collision or (
      sample.collision_force_n >= self.cfg.collision_force_threshold_n
    )
    self._fall |= sample.fall
    self._search_timed_out |= sample.search_timed_out
    self._out_of_bounds |= not self._inside_allowed_region(sample.position_xy)
    self._intervention |= sample.mode != SafeVelocityAdapterMode.TRACK

    if sample.mode == SafeVelocityAdapterMode.RESUME and self._resume_position is None:
      self._resume_position = sample.position_xy
    if self._resume_position is not None:
      self._avoidance_resumed |= (
        math.dist(self._resume_position, sample.position_xy)
        >= self.cfg.resume_distance_m
      )

    if (
      sample.mode == SafeVelocityAdapterMode.BLOCKED
      and previous is not None
      and previous.mode == SafeVelocityAdapterMode.BLOCKED
      and sample.planar_speed_mps <= self.cfg.stationary_speed_mps
      and previous.planar_speed_mps <= self.cfg.stationary_speed_mps
    ):
      self._blocked_hold_s += dt
    elif sample.mode != SafeVelocityAdapterMode.BLOCKED:
      self._blocked_hold_s = 0.0

    if sample.mode in (SafeVelocityAdapterMode.TRACK, SafeVelocityAdapterMode.RESUME):
      self._active_motion.append(sample)
      cutoff = sample.time_s - self.cfg.stagnation_window_s
      while len(self._active_motion) > 1 and self._active_motion[1].time_s <= cutoff:
        self._active_motion.popleft()
      oldest = self._active_motion[0]
      if (
        sample.time_s - oldest.time_s >= self.cfg.stagnation_window_s
        and math.dist(oldest.position_xy, sample.position_xy)
        <= self.cfg.stagnation_distance_m
      ):
        self._stagnation = True
    else:
      self._active_motion.clear()

  def summarize(self) -> SafeVelocityTrajectorySummary:
    """Finalize the current evaluation window without mutating its state."""
    if not self._samples:
      raise ValueError("Cannot summarize an empty safe-velocity trajectory")
    last = self._samples[-1]
    safe_blocked = self._blocked_hold_s >= self.cfg.blocked_hold_s
    continuous = not self._intervention
    outcome = classify_safe_velocity_outcome(
      collision=self._collision,
      fall=self._fall,
      out_of_bounds=self._out_of_bounds,
      search_timeout=self._search_timed_out and not safe_blocked,
      stagnation=self._stagnation,
      avoidance_resume=self._avoidance_resumed,
      safe_blocked_stop=safe_blocked,
      continuous_tracking=continuous,
      episode_timeout=(
        last.episode_timeout
        or not (
          self._search_timed_out
          or self._stagnation
          or self._avoidance_resumed
          or safe_blocked
          or continuous
        )
      ),
    )
    return SafeVelocityTrajectorySummary(
      outcome=outcome,
      duration_s=last.time_s - self._samples[0].time_s,
      path_length_m=self._path_length_m,
      min_sensor_clearance_m=self._min_clearance_m,
      max_collision_force_n=self._max_collision_force_n,
      collision_impulse_ns=self._collision_impulse_ns,
      collision=self._collision,
      fall=self._fall,
      out_of_bounds=self._out_of_bounds,
      search_timed_out=self._search_timed_out,
      stagnation=self._stagnation,
      avoidance_resumed=self._avoidance_resumed,
      safe_blocked_stop=safe_blocked,
      intervention=self._intervention,
    )

  def _inside_allowed_region(self, position_xy: tuple[float, float]) -> bool:
    x_min, x_max, y_min, y_max = self.cfg.allowed_region_xy
    return x_min <= position_xy[0] <= x_max and y_min <= position_xy[1] <= y_max


@dataclass(frozen=True)
class SafeVelocityDepthAdapterCfg:
  """Deployable thresholds for the task-local deterministic baseline."""

  cutoff_distance_m: float = 5.0
  collision_risk_distance_m: float = 1.2
  stop_clearance_m: float = 0.6
  search_trigger_distance_m: float = 1.0
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
    if not (
      0
      < self.stop_clearance_m
      < self.search_trigger_distance_m
      < self.collision_risk_distance_m
    ):
      raise ValueError(
        "clearance thresholds must satisfy 0 < stop < search trigger < risk"
      )
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
class SafeVelocityDepthAdapterOutput:
  """One batch of adapter decisions and diagnostics."""

  command: torch.Tensor
  mode: torch.Tensor
  sector_clearance_m: torch.Tensor
  sector_valid_ratio: torch.Tensor
  intervention: torch.Tensor
  search_timed_out: torch.Tensor
  search_elapsed_s: torch.Tensor


class SafeVelocityDepthAdapter:
  """Brake, rotate toward free depth, and resume without localization.

  Input depth is the normalized P6 depth+validity observation. Both a single
  frame ``(B, 2, H, W)`` and temporal history ``(B, T, 2, H, W)`` are accepted;
  the baseline deliberately consumes only the latest frame.
  """

  def __init__(
    self,
    num_envs: int,
    device: str | torch.device,
    cfg: SafeVelocityDepthAdapterCfg | None = None,
  ) -> None:
    if num_envs < 1:
      raise ValueError("num_envs must be positive")
    self.cfg = cfg or SafeVelocityDepthAdapterCfg()
    self.mode = torch.full(
      (num_envs,), int(SafeVelocityAdapterMode.TRACK), dtype=torch.long, device=device
    )
    self.search_elapsed_s = torch.zeros(num_envs, device=device)
    self.search_direction = torch.ones(num_envs, device=device)
    self.intervention_count = torch.zeros(num_envs, dtype=torch.long, device=device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset all or selected adapter states."""
    ids = slice(None) if env_ids is None else env_ids
    self.mode[ids] = int(SafeVelocityAdapterMode.TRACK)
    self.search_elapsed_s[ids] = 0.0
    self.search_direction[ids] = 1.0
    self.intervention_count[ids] = 0

  def step(
    self,
    depth_validity: torch.Tensor,
    nominal_command: torch.Tensor,
    dt: float,
  ) -> SafeVelocityDepthAdapterOutput:
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
      return SafeVelocityDepthAdapterOutput(
        command=command,
        mode=self.mode.clone(),
        sector_clearance_m=clearance,
        sector_valid_ratio=valid_ratio,
        intervention=intervention,
        search_timed_out=timed_out,
        search_elapsed_s=self.search_elapsed_s.clone(),
      )

    front_clearance = clearance[:, 1]
    front_valid = valid_ratio[:, 1] >= self.cfg.minimum_valid_ratio
    front_safe = front_valid & (front_clearance >= self.cfg.collision_risk_distance_m)
    must_stop = (~front_valid) | (front_clearance <= self.cfg.search_trigger_distance_m)
    risk = (~front_safe) & (~must_stop)

    current = self.mode.clone()
    active_motion = (
      (current == int(SafeVelocityAdapterMode.TRACK))
      | (current == int(SafeVelocityAdapterMode.BRAKE))
      | (current == int(SafeVelocityAdapterMode.RESUME))
    )
    track = active_motion & front_safe
    self.mode[track] = int(SafeVelocityAdapterMode.TRACK)

    braking = active_motion & risk
    scale = (
      (front_clearance - self.cfg.stop_clearance_m)
      / (self.cfg.collision_risk_distance_m - self.cfg.stop_clearance_m)
    ).clamp(0.0, 1.0)
    command[braking, 0] = command[braking, 0].clamp(min=0.0) * scale[braking]
    command[braking, 1:] = 0.0
    self.mode[braking] = int(SafeVelocityAdapterMode.BRAKE)

    stopping = active_motion & must_stop
    command[stopping] = 0.0
    self.mode[stopping] = int(SafeVelocityAdapterMode.STOP)
    self.search_elapsed_s[stopping] = 0.0
    prefer_left = clearance[:, 0] >= clearance[:, 2]
    self.search_direction[stopping] = torch.where(prefer_left[stopping], 1.0, -1.0)

    start_search = current == int(SafeVelocityAdapterMode.STOP)
    searching = current == int(SafeVelocityAdapterMode.SEARCH)
    search_active = start_search | (searching & ~front_safe)
    self.search_elapsed_s[search_active] += dt
    command[search_active] = 0.0
    command[search_active, 2] = (
      self.search_direction[search_active] * self.cfg.search_yaw_rate
    )
    self.mode[search_active] = int(SafeVelocityAdapterMode.SEARCH)

    timed_out = search_active & (self.search_elapsed_s >= self.cfg.search_time_budget_s)
    command[timed_out] = 0.0
    self.mode[timed_out] = int(SafeVelocityAdapterMode.BLOCKED)

    resume = (
      searching | (current == int(SafeVelocityAdapterMode.BLOCKED))
    ) & front_safe
    command[resume] = 0.0
    command[resume, 0] = nominal_command[resume, 0].clamp(
      min=0.0, max=self.cfg.resume_speed
    )
    self.mode[resume] = int(SafeVelocityAdapterMode.RESUME)
    self.search_elapsed_s[resume] = 0.0

    blocked = (current == int(SafeVelocityAdapterMode.BLOCKED)) & ~front_safe
    command[blocked] = 0.0

    intervention = ~torch.isclose(command, nominal_command).all(dim=-1)
    self.intervention_count += intervention.long()
    return SafeVelocityDepthAdapterOutput(
      command=command,
      mode=self.mode.clone(),
      sector_clearance_m=clearance,
      sector_valid_ratio=valid_ratio,
      intervention=intervention,
      search_timed_out=timed_out,
      search_elapsed_s=self.search_elapsed_s.clone(),
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


def _camera_depth_validity(
  sensor: CameraSensor,
  cutoff_distance_m: float,
  sensor_name: str,
) -> torch.Tensor:
  """Convert native camera depth to normalized depth and validity channels."""
  raw = sensor.data.depth
  if raw is None:
    raise ValueError(f"Camera '{sensor_name}' has no depth data")
  valid = torch.isfinite(raw) & (raw > 0)
  depth = torch.where(valid, raw, cutoff_distance_m)
  depth = depth.clamp(0.0, cutoff_distance_m) / cutoff_distance_m
  return torch.cat(
    (
      depth.permute(0, 3, 1, 2),
      valid.permute(0, 3, 1, 2).to(depth.dtype),
    ),
    dim=1,
  )


def safe_velocity_twist_tracking_reward(
  actual_linear_velocity_b: torch.Tensor,
  actual_angular_velocity_b: torch.Tensor,
  safe_command: torch.Tensor,
  linear_std: float,
  angular_std: float,
) -> torch.Tensor:
  """Reward tracking a safe body-frame twist with the standard velocity shape."""
  if linear_std <= 0 or angular_std <= 0:
    raise ValueError("safe-velocity tracking standard deviations must be positive")
  linear_error = torch.sum(
    torch.square(safe_command[:, :2] - actual_linear_velocity_b[:, :2]), dim=1
  ) + torch.square(actual_linear_velocity_b[:, 2])
  angular_error = torch.square(
    safe_command[:, 2] - actual_angular_velocity_b[:, 2]
  ) + torch.sum(torch.square(actual_angular_velocity_b[:, :2]), dim=1)
  return torch.exp(-linear_error / linear_std**2) + torch.exp(
    -angular_error / angular_std**2
  )


class SafeVelocityTrackingReward:
  """Track the depth adapter's safe twist instead of an unsafe nominal command."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    adapter_cfg = cfg.params.get("adapter_cfg")
    if not isinstance(adapter_cfg, SafeVelocityDepthAdapterCfg):
      raise TypeError("adapter_cfg must be SafeVelocityDepthAdapterCfg")
    self.adapter = SafeVelocityDepthAdapter(
      num_envs=env.num_envs,
      device=env.device,
      cfg=adapter_cfg,
    )
    self.last_output: SafeVelocityDepthAdapterOutput | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.adapter.reset(env_ids)
    self.last_output = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    adapter_cfg: SafeVelocityDepthAdapterCfg,
    linear_std: float,
    angular_std: float,
  ) -> torch.Tensor:
    del adapter_cfg
    sensor = env.scene[sensor_name]
    if not isinstance(sensor, CameraSensor):
      raise TypeError(
        f"SafeVelocityTrackingReward requires CameraSensor, got {type(sensor).__name__}"
      )
    observation = _camera_depth_validity(
      sensor,
      cutoff_distance_m=self.adapter.cfg.cutoff_distance_m,
      sensor_name=sensor_name,
    )
    nominal = env.command_manager.get_command(command_name)
    if nominal is None:
      raise ValueError(f"Velocity command '{command_name}' is not configured")
    self.last_output = self.adapter.step(observation, nominal, env.step_dt)
    robot: Entity = env.scene["robot"]
    return safe_velocity_twist_tracking_reward(
      robot.data.root_link_lin_vel_b,
      robot.data.root_link_ang_vel_b,
      self.last_output.command,
      linear_std,
      angular_std,
    )


def safe_velocity_base_obstacle_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Count policy substeps where the G1 base contacts any scene obstacle."""
  cost = torch.zeros(env.num_envs, device=env.device)
  for sensor_name in sensor_names:
    cost += self_collision_cost(
      env,
      sensor_name=sensor_name,
      force_threshold=force_threshold,
    )
  return cost


def _local_geometry_adapter_input(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
  cutoff_distance_m: float,
) -> torch.Tensor:
  """Build depth/validity channels from privileged collision-ray distances."""
  if not sensor_names:
    raise ValueError("sensor_names must contain at least one local-geometry ray")
  distances = []
  for sensor_name in sensor_names:
    sensor = env.scene[sensor_name]
    if not isinstance(sensor, RayCastSensor):
      raise TypeError(
        f"Local geometry requires RayCastSensor, got {type(sensor).__name__}"
      )
    distances.append(sensor.data.distances)
  raw = torch.cat(distances, dim=-1)
  return local_geometry_depth_validity_from_ray_distances(
    raw,
    cutoff_distance_m=cutoff_distance_m,
  )


def local_geometry_depth_validity_from_ray_distances(
  distances: torch.Tensor,
  cutoff_distance_m: float,
) -> torch.Tensor:
  """Convert collision-ray results to known-clear depth and validity channels.

  ``RayCastSensor`` reports ``-1`` when no geometry is hit before its maximum
  distance. Unlike a missing camera-depth sample, that is a successful query and
  means the ray is known to be clear up to ``cutoff_distance_m``. Only non-finite
  sensor values are treated as invalid.
  """
  if distances.ndim != 2:
    raise ValueError("distances must have shape (B, N)")
  if cutoff_distance_m <= 0:
    raise ValueError("cutoff_distance_m must be positive")
  valid = torch.isfinite(distances)
  hit = valid & (distances >= 0.0)
  depth = torch.where(hit, distances, cutoff_distance_m)
  depth = depth.clamp(0.0, cutoff_distance_m) / cutoff_distance_m
  return torch.stack((depth, valid.to(dtype=depth.dtype)), dim=1).unsqueeze(2)


def safe_velocity_local_geometry(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
  cutoff_distance_m: float,
) -> torch.Tensor:
  """Expose normalized collision-ray distance and validity to a privileged actor."""
  observation = _local_geometry_adapter_input(
    env,
    sensor_names=sensor_names,
    cutoff_distance_m=cutoff_distance_m,
  )
  return observation.flatten(start_dim=1)


def local_geometry_stop_command(
  depth_validity: torch.Tensor,
  nominal_command: torch.Tensor,
  cfg: SafeVelocityDepthAdapterCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Stop on a front obstacle without searching, turning, or resuming around it."""
  if depth_validity.ndim == 5:
    depth_validity = depth_validity[:, -1]
  batch = nominal_command.shape[0]
  if depth_validity.ndim != 4 or depth_validity.shape[:2] != (batch, 2):
    raise ValueError("depth_validity must have shape (B, 2, H, W)")
  if nominal_command.shape != (batch, 3):
    raise ValueError("nominal_command must have shape (B, 3)")

  depth_m = depth_validity[:, 0] * cfg.cutoff_distance_m
  valid = depth_validity[:, 1] > 0.5
  width = depth_m.shape[-1]
  front_start = width // 3
  front_end = 2 * width // 3
  front_depth = depth_m[..., front_start:front_end]
  front_valid = valid[..., front_start:front_end]
  front_known = front_valid.flatten(1).all(dim=1)
  front_clearance = (
    torch.where(
      front_valid,
      front_depth,
      torch.full_like(front_depth, torch.inf),
    )
    .flatten(1)
    .amin(dim=1)
  )
  stop = (~front_known) | (front_clearance <= cfg.collision_risk_distance_m)
  command = nominal_command.clone()
  command[stop] = 0.0
  return command, stop


class SafeVelocityLocalGeometryTrackingReward:
  """Track a stateless stop-on-obstacle command from privileged geometry."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    adapter_cfg = cfg.params.get("adapter_cfg")
    if not isinstance(adapter_cfg, SafeVelocityDepthAdapterCfg):
      raise TypeError("adapter_cfg must be SafeVelocityDepthAdapterCfg")
    self.cfg = adapter_cfg
    self.last_command: torch.Tensor | None = None
    self.last_stop: torch.Tensor | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids
    self.last_command = None
    self.last_stop = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_names: tuple[str, ...],
    adapter_cfg: SafeVelocityDepthAdapterCfg,
    linear_std: float,
    angular_std: float,
  ) -> torch.Tensor:
    del adapter_cfg
    nominal = env.command_manager.get_command(command_name)
    if nominal is None:
      raise ValueError(f"Velocity command '{command_name}' is not configured")
    observation = _local_geometry_adapter_input(
      env,
      sensor_names=sensor_names,
      cutoff_distance_m=self.cfg.cutoff_distance_m,
    )
    self.last_command, self.last_stop = local_geometry_stop_command(
      observation,
      nominal,
      self.cfg,
    )
    robot: Entity = env.scene["robot"]
    return safe_velocity_twist_tracking_reward(
      robot.data.root_link_lin_vel_b,
      robot.data.root_link_ang_vel_b,
      self.last_command,
      linear_std,
      angular_std,
    )


class adapted_velocity_command:
  """Observation term that exposes an adapter command to a blind policy.

  The nominal command remains owned by ``CommandManager`` for evaluation. The
  actor receives only this safe body-frame twist; raw depth is consumed internally
  and is not added to the actor observation group.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv) -> None:
    adapter_cfg = cfg.params.get("adapter_cfg")
    if not isinstance(adapter_cfg, SafeVelocityDepthAdapterCfg):
      raise TypeError("adapter_cfg must be SafeVelocityDepthAdapterCfg")
    self.adapter = SafeVelocityDepthAdapter(
      num_envs=env.num_envs,
      device=env.device,
      cfg=adapter_cfg,
    )
    self.last_output: SafeVelocityDepthAdapterOutput | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.adapter.reset(env_ids)
    self.last_output = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    adapter_cfg: SafeVelocityDepthAdapterCfg,
  ) -> torch.Tensor:
    del adapter_cfg
    sensor = env.scene[sensor_name]
    if not isinstance(sensor, CameraSensor):
      raise TypeError(
        f"adapted_velocity_command requires CameraSensor, got {type(sensor).__name__}"
      )
    observation = _camera_depth_validity(
      sensor,
      cutoff_distance_m=self.adapter.cfg.cutoff_distance_m,
      sensor_name=sensor_name,
    )
    nominal = env.command_manager.get_command(command_name)
    if nominal is None:
      raise ValueError(f"Velocity command '{command_name}' is not configured")
    self.last_output = self.adapter.step(observation, nominal, env.step_dt)
    return self.last_output.command


def classify_safe_velocity_outcome(
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
) -> SafeVelocityOutcome:
  """Choose exactly one primary outcome using a fixed safety-first precedence."""
  ordered = (
    (collision, SafeVelocityOutcome.COLLISION),
    (fall, SafeVelocityOutcome.FALL),
    (out_of_bounds, SafeVelocityOutcome.OUT_OF_BOUNDS),
    (search_timeout, SafeVelocityOutcome.SEARCH_TIMEOUT),
    (stagnation, SafeVelocityOutcome.STAGNATION),
    (avoidance_resume, SafeVelocityOutcome.AVOIDANCE_RESUME),
    (safe_blocked_stop, SafeVelocityOutcome.SAFE_BLOCKED_STOP),
    (continuous_tracking, SafeVelocityOutcome.CONTINUOUS_TRACKING),
    (episode_timeout, SafeVelocityOutcome.EPISODE_TIMEOUT),
  )
  match = next((outcome for active, outcome in ordered if active), None)
  if match is None:
    raise ValueError("At least one safe-velocity outcome condition must be true")
  return match


__all__ = [
  "SafeVelocityAdapterMode",
  "SafeVelocityDepthAdapterCfg",
  "SafeVelocityDepthAdapterOutput",
  "SafeVelocityDepthAdapter",
  "SafeVelocityOutcome",
  "SafeVelocityLocalGeometryTrackingReward",
  "SafeVelocityTrackingReward",
  "SafeVelocityTrajectoryEvaluator",
  "SafeVelocityTrajectoryEvaluatorCfg",
  "SafeVelocityTrajectorySample",
  "SafeVelocityTrajectorySummary",
  "adapted_velocity_command",
  "classify_safe_velocity_outcome",
  "local_geometry_stop_command",
  "local_geometry_depth_validity_from_ray_distances",
  "safe_velocity_base_obstacle_collision_cost",
  "safe_velocity_local_geometry",
  "safe_velocity_twist_tracking_reward",
]
