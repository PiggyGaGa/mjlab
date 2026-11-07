"""Project-specific robust vision observation for G1 3DGS tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from mjlab.sensor import CameraSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import ObservationTermCfg


class camera_depth_validity_robust:
  """Aligned depth-validity corruption with multi-rate sample-and-hold.

  This term is intentionally task-local. Generic observation delay and history stay
  in :class:`ObservationManager`; this class only covers image-specific corruption
  that must be coherent across the depth and validity channels.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self._env = env
    self._cached: torch.Tensor | None = None
    self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._last_step = torch.full(
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids: Any = slice(None) if env_ids is None else env_ids
    self._initialized[ids] = False
    self._last_step[ids] = -1

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    cutoff_distance: float,
    min_depth: float = 0.01,
    update_period: int = 2,
    enable_corruption: bool = True,
    range_noise_std: float = 0.01,
    blur_kernel_size: int = 3,
    structured_dropout_prob: float = 0.1,
    frame_dropout_prob: float = 0.02,
    dropout_size_range: tuple[float, float] = (0.1, 0.3),
  ) -> torch.Tensor:
    if update_period < 1:
      raise ValueError("update_period must be at least one")
    current = self._read_camera(env, sensor_name, cutoff_distance, min_depth)
    if enable_corruption:
      current = self._corrupt(
        current,
        range_noise_std=range_noise_std,
        blur_kernel_size=blur_kernel_size,
        structured_dropout_prob=structured_dropout_prob,
        frame_dropout_prob=frame_dropout_prob,
        dropout_size_range=dropout_size_range,
      )

    step = torch.full_like(self._last_step, env.common_step_counter)
    scheduled = env.episode_length_buf.remainder(update_period) == 0
    refresh = ~self._initialized | (scheduled & (self._last_step != step))
    if self._cached is None:
      self._cached = current.clone()
    mask = refresh.reshape(-1, 1, 1, 1)
    self._cached = torch.where(mask, current, self._cached)
    self._initialized |= refresh
    self._last_step = torch.where(refresh, step, self._last_step)
    return self._cached

  @staticmethod
  def _read_camera(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    cutoff_distance: float,
    min_depth: float,
  ) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    if not isinstance(sensor, CameraSensor):
      raise TypeError(
        f"camera_depth_validity_robust requires CameraSensor, got "
        f"{type(sensor).__name__}"
      )
    raw = sensor.data.depth
    if raw is None:
      raise ValueError(f"Camera '{sensor_name}' has no depth data")
    valid = torch.isfinite(raw) & (raw > 0)
    depth = torch.where(valid, raw, cutoff_distance)
    depth = torch.clamp(depth, min=min_depth, max=cutoff_distance)
    depth = depth / cutoff_distance
    return torch.cat(
      (depth.permute(0, 3, 1, 2), valid.permute(0, 3, 1, 2).to(depth.dtype)),
      dim=1,
    )

  @staticmethod
  def _corrupt(
    observation: torch.Tensor,
    range_noise_std: float,
    blur_kernel_size: int,
    structured_dropout_prob: float,
    frame_dropout_prob: float,
    dropout_size_range: tuple[float, float],
  ) -> torch.Tensor:
    if range_noise_std < 0:
      raise ValueError("range_noise_std must be non-negative")
    if blur_kernel_size < 1 or blur_kernel_size % 2 == 0:
      raise ValueError("blur_kernel_size must be a positive odd integer")
    if not 0 <= structured_dropout_prob <= 1:
      raise ValueError("structured_dropout_prob must be in [0, 1]")
    if not 0 <= frame_dropout_prob <= 1:
      raise ValueError("frame_dropout_prob must be in [0, 1]")
    size_min, size_max = dropout_size_range
    if not 0 < size_min <= size_max <= 1:
      raise ValueError("dropout_size_range must satisfy 0 < min <= max <= 1")

    depth = observation[:, :1]
    validity = observation[:, 1:2]
    if blur_kernel_size > 1:
      padding = blur_kernel_size // 2
      depth = F.pad(depth, (padding, padding, padding, padding), mode="replicate")
      depth = F.avg_pool2d(depth, kernel_size=blur_kernel_size, stride=1)
    if range_noise_std > 0:
      depth = depth + torch.randn_like(depth) * range_noise_std * validity

    batch, _, height, width = observation.shape
    dtype = observation.dtype
    device = observation.device
    center_y = torch.rand((batch, 1, 1), device=device)
    center_x = torch.rand((batch, 1, 1), device=device)
    size = size_min + torch.rand((batch, 2), device=device) * (size_max - size_min)
    grid_y = torch.linspace(0, 1, height, device=device, dtype=dtype).reshape(
      1, height, 1
    )
    grid_x = torch.linspace(0, 1, width, device=device, dtype=dtype).reshape(
      1, 1, width
    )
    rectangle = (torch.abs(grid_y - center_y) <= size[:, :1, None] / 2) & (
      torch.abs(grid_x - center_x) <= size[:, 1:, None] / 2
    )
    use_rectangle = torch.rand((batch, 1, 1), device=device) < structured_dropout_prob
    drop_frame = torch.rand((batch, 1, 1), device=device) < frame_dropout_prob
    missing = (rectangle & use_rectangle) | drop_frame
    missing = missing.unsqueeze(1)
    depth = torch.where(missing, torch.ones_like(depth), depth).clamp_(0.0, 1.0)
    validity = torch.where(missing, torch.zeros_like(validity), validity)
    return torch.cat((depth, validity), dim=1)
