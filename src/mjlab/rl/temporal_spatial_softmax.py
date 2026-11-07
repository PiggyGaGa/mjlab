"""Spatial-softmax CNN adapter for image observations with frame history."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from tensordict import TensorDict

from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel


class _MergeTimeChannels(nn.Module):
  """Merge ``(T, C)`` before forwarding a frame history to a 2D encoder."""

  def __init__(
    self,
    encoder: nn.Module,
    history_length: int,
    temporal_frames: int,
    temporal_stride: int,
  ) -> None:
    super().__init__()
    self.encoder = cast(Any, encoder)
    self.history_length = history_length
    self.temporal_frames = temporal_frames
    self.temporal_stride = temporal_stride

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # RSL-RL's ONNX wrapper supplies its flattened 2D metadata as an already
    # channel-stacked tensor. Runtime training supplies the structured history.
    if x.ndim == 4:
      return self.encoder(x)
    if x.ndim != 5:
      raise ValueError("Expected (B, T, C, H, W) or stacked (B, C, H, W)")
    batch, history, channels, height, width = x.shape
    if history != self.history_length:
      raise ValueError(
        f"Expected {self.history_length} history frames, received {history}"
      )
    selected = x[:, :: self.temporal_stride]
    if selected.shape[1] != self.temporal_frames:
      raise ValueError(
        f"Expected {self.temporal_frames} selected frames, received {selected.shape[1]}"
      )
    return self.encoder(
      selected.reshape(batch, self.temporal_frames * channels, height, width)
    )

  @property
  def output_dim(self) -> Any:
    """Expose the wrapped encoder's flattened output dimension."""
    return self.encoder.output_dim

  @property
  def output_channels(self) -> Any:
    """Expose the wrapped encoder's output-channel marker."""
    return self.encoder.output_channels


class TemporalSpatialSoftmaxCNNModel(SpatialSoftmaxCNNModel):
  """Spatial-softmax model that stacks image history along the channel axis.

  RSL-RL's CNN model accepts ``(B, C, H, W)`` observations. Mjlab's observation
  history keeps image structure as ``(B, T, C, H, W)`` when
  ``flatten_history_dim=False``. This adapter reuses both implementations and only
  merges ``T`` and ``C`` at the encoder boundary.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    cnn_cfg: dict[str, dict] | dict[str, Any],
    cnns: nn.ModuleDict | None = None,
    hidden_dims: tuple[int] | list[int] = [256, 256, 256],  # noqa: B006
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict[str, Any] | None = None,
  ) -> None:
    self._temporal_obs_groups: set[str] = set()
    self._temporal_history_lengths: dict[str, int] = {}
    cnn_cfg = dict(cnn_cfg)
    temporal_frames = cnn_cfg.pop("temporal_frames", 0)
    temporal_stride = cnn_cfg.pop("temporal_stride", 1)
    if not isinstance(temporal_frames, int) or not isinstance(temporal_stride, int):
      raise TypeError("temporal_frames and temporal_stride must be integers")
    self._temporal_frames = temporal_frames
    self._temporal_stride = temporal_stride
    if self._temporal_frames < 0:
      raise ValueError("temporal_frames must be non-negative")
    if self._temporal_stride < 1:
      raise ValueError("temporal_stride must be at least one")
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      cnn_cfg=cnn_cfg,
      cnns=cnns,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    for obs_group in self._temporal_obs_groups:
      if not isinstance(self.cnns[obs_group], _MergeTimeChannels):
        history_length = self._temporal_history_lengths[obs_group]
        temporal_frames = self._temporal_frames or history_length
        expected_history = 1 + (temporal_frames - 1) * self._temporal_stride
        if history_length != expected_history:
          raise ValueError(
            f"Observation group '{obs_group}' has history length {history_length}; "
            f"expected {expected_history} for {temporal_frames} frames with "
            f"stride {self._temporal_stride}"
          )
        self.cnns[obs_group] = _MergeTimeChannels(
          self.cnns[obs_group],
          history_length=history_length,
          temporal_frames=temporal_frames,
          temporal_stride=self._temporal_stride,
        )

  def _get_obs_dim(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
  ) -> tuple[list[str], int]:
    active_groups = obs_groups[obs_set]
    adapted = obs.copy()
    for obs_group in active_groups:
      value = cast(torch.Tensor, obs[obs_group])
      if value.ndim != 5:
        continue
      batch, history, channels, height, width = value.shape
      temporal_frames = self._temporal_frames or history
      expected_history = 1 + (temporal_frames - 1) * self._temporal_stride
      if history != expected_history:
        raise ValueError(
          f"Observation group '{obs_group}' has history length {history}; "
          f"expected {expected_history}"
        )
      adapted[obs_group] = value.new_empty(
        batch, temporal_frames * channels, height, width
      )
      self._temporal_obs_groups.add(obs_group)
      self._temporal_history_lengths[obs_group] = history
    return super()._get_obs_dim(adapted, obs_groups, obs_set)
