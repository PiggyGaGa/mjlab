"""Offline visual student components for the G1 safe-velocity task."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset

from mjlab.rl.spatial_softmax import SpatialSoftmax

_REQUIRED_CHUNK_KEYS = ("camera", "teacher_labels", "nominal_command")
SAFE_VELOCITY_STUDENT_MODEL_VERSION = 2


class SafeVelocityVisionDataset(Dataset[dict[str, Tensor]]):
  """Load schema-v2 safe-velocity chunks while enforcing catalog splits."""

  def __init__(
    self,
    directories: list[Path],
    *,
    allowed_splits: frozenset[str] | None = None,
  ) -> None:
    if not directories:
      raise ValueError("At least one dataset directory is required")

    cameras: list[Tensor] = []
    labels: list[Tensor] = []
    commands: list[Tensor] = []
    self.manifests: list[dict[str, Any]] = []
    self.sources: list[tuple[str, int, int]] = []
    total_samples = 0

    for directory in directories:
      manifest_path = directory / "manifest.json"
      if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      if manifest.get("schema_version") != 2:
        raise ValueError(f"Unsupported schema in {manifest_path}")

      scenario = manifest.get("scenario", {})
      split = scenario.get("split")
      if allowed_splits is not None and split not in allowed_splits:
        expected = ", ".join(sorted(allowed_splits))
        raise ValueError(
          f"Scenario {scenario.get('id')!r} uses split {split!r}; expected {expected}"
        )

      directory_samples = 0
      for chunk_info in manifest.get("chunks", []):
        chunk_path = directory / chunk_info["file"]
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=True)
        missing = [key for key in _REQUIRED_CHUNK_KEYS if key not in chunk]
        if missing:
          raise ValueError(f"Missing {missing} in {chunk_path}")

        camera = chunk["camera"]
        teacher_labels = chunk["teacher_labels"]
        nominal_command = chunk["nominal_command"]
        samples = camera.shape[0]
        if camera.ndim != 5 or camera.shape[2] != 2:
          raise ValueError(
            f"Invalid camera shape {tuple(camera.shape)} in {chunk_path}"
          )
        if teacher_labels.shape != (samples, 4):
          raise ValueError(
            f"Invalid teacher label shape {tuple(teacher_labels.shape)} in {chunk_path}"
          )
        if nominal_command.shape != (samples, 3):
          raise ValueError(
            f"Invalid nominal command shape {tuple(nominal_command.shape)} in {chunk_path}"
          )
        if samples != chunk_info["samples"]:
          raise ValueError(f"Chunk sample count mismatch in {chunk_path}")

        cameras.append(camera)
        labels.append(teacher_labels)
        commands.append(nominal_command)
        directory_samples += samples

      if directory_samples != manifest.get("samples"):
        raise ValueError(f"Manifest sample count mismatch in {manifest_path}")
      self.manifests.append(manifest)
      scenario_id = str(scenario.get("id"))
      self.sources.append(
        (scenario_id, total_samples, total_samples + directory_samples)
      )
      total_samples += directory_samples

    self.camera = torch.cat(cameras)
    self.teacher_labels = torch.cat(labels).to(torch.float32)
    self.nominal_command = torch.cat(commands).to(torch.float32)

  def __len__(self) -> int:
    return self.camera.shape[0]

  def __getitem__(self, index: int) -> dict[str, Tensor]:
    return {
      "camera": self.camera[index].to(torch.float32),
      "teacher_labels": self.teacher_labels[index],
      "nominal_command": self.nominal_command[index],
    }

  @property
  def stop_positive_count(self) -> int:
    """Return the number of positive binary stop labels."""
    return int(self.teacher_labels[:, 3].sum().item())


class SafeVelocityVisionStudent(nn.Module):
  """Encode temporal depth/validity and predict four teacher quantities."""

  def __init__(
    self,
    history_length: int = 7,
    image_shape: tuple[int, int] = (48, 64),
    stop_alpha_threshold: float = 0.1,
    stop_temperature: float = 0.02,
  ) -> None:
    super().__init__()
    if history_length < 1:
      raise ValueError("history_length must be positive")
    if stop_temperature <= 0:
      raise ValueError("stop_temperature must be positive")
    self.history_length = history_length
    self.image_shape = image_shape
    self.stop_alpha_threshold = stop_alpha_threshold
    self.stop_temperature = stop_temperature
    input_channels = history_length * 2
    self.encoder = nn.Sequential(
      nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
      nn.GroupNorm(4, 32),
      nn.SiLU(),
      nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
      nn.GroupNorm(8, 64),
      nn.SiLU(),
      nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
      nn.GroupNorm(8, 96),
      nn.SiLU(),
      nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
      nn.GroupNorm(8, 128),
      nn.SiLU(),
    )
    with torch.no_grad():
      feature_shape = self.encoder(
        torch.zeros(1, input_channels, image_shape[0], image_shape[1])
      ).shape
    feature_channels = feature_shape[1]
    self.spatial_softmax = SpatialSoftmax(feature_shape[2], feature_shape[3])
    self.head = nn.Sequential(
      nn.Linear(feature_channels * 3 + 3, 128),
      nn.SiLU(),
      nn.Linear(128, 3),
    )

  def forward(self, camera: Tensor, nominal_command: Tensor) -> Tensor:
    if camera.ndim != 5 or camera.shape[1:3] != (self.history_length, 2):
      raise ValueError(
        "camera must have shape "
        f"(B, {self.history_length}, 2, H, W), got {tuple(camera.shape)}"
      )
    if nominal_command.shape != (camera.shape[0], 3):
      raise ValueError("nominal_command must have shape (B, 3)")
    if camera.shape[-2:] != self.image_shape:
      raise ValueError(f"camera image shape must be {self.image_shape}")
    batch, history, channels, height, width = camera.shape
    features = self.encoder(camera.reshape(batch, history * channels, height, width))
    spatial = self.spatial_softmax(features)
    magnitude = features.mean(dim=(-2, -1))
    core_logits = self.head(torch.cat((spatial, magnitude, nominal_command), dim=-1))
    alpha = torch.sigmoid(core_logits[:, 2])
    stop_logit = (self.stop_alpha_threshold - alpha) / self.stop_temperature
    return torch.cat((core_logits, stop_logit.unsqueeze(-1)), dim=-1)


def decode_safe_velocity_student(logits: Tensor) -> Tensor:
  """Map the four student logits to normalized teacher-label ranges."""
  if logits.ndim != 2 or logits.shape[1] != 4:
    raise ValueError("Student logits must have shape (B, 4)")
  return torch.sigmoid(logits)


@dataclass(frozen=True)
class SafeVelocityStudentLossCfg:
  """Weights for conservative visual safety supervision."""

  clearance_weight: float = 0.5
  validity_weight: float = 0.1
  alpha_weight: float = 2.0
  stop_weight: float = 1.0
  unsafe_alpha_weight: float = 4.0
  stop_positive_weight: float = 1.0


def safe_velocity_student_loss(
  logits: Tensor,
  teacher_labels: Tensor,
  cfg: SafeVelocityStudentLossCfg,
) -> dict[str, Tensor]:
  """Compute a loss that penalizes unsafe alpha overestimation more strongly."""
  if logits.shape != teacher_labels.shape or logits.ndim != 2 or logits.shape[1] != 4:
    raise ValueError("logits and teacher_labels must both have shape (B, 4)")
  predictions = decode_safe_velocity_student(logits)

  clearance = F.smooth_l1_loss(predictions[:, 0], teacher_labels[:, 0])
  validity = F.binary_cross_entropy_with_logits(logits[:, 1], teacher_labels[:, 1])
  alpha_pointwise = F.smooth_l1_loss(
    predictions[:, 2], teacher_labels[:, 2], reduction="none"
  )
  alpha_weights = torch.where(
    predictions[:, 2] > teacher_labels[:, 2],
    cfg.unsafe_alpha_weight,
    1.0,
  )
  alpha = (alpha_pointwise * alpha_weights).mean()
  positive_weight = logits.new_tensor(cfg.stop_positive_weight)
  stop = F.binary_cross_entropy_with_logits(
    logits[:, 3],
    teacher_labels[:, 3],
    pos_weight=positive_weight,
  )
  total = (
    cfg.clearance_weight * clearance
    + cfg.validity_weight * validity
    + cfg.alpha_weight * alpha
    + cfg.stop_weight * stop
  )
  return {
    "loss": total,
    "clearance_loss": clearance,
    "validity_loss": validity,
    "alpha_loss": alpha,
    "stop_loss": stop,
  }


__all__ = [
  "SAFE_VELOCITY_STUDENT_MODEL_VERSION",
  "SafeVelocityStudentLossCfg",
  "SafeVelocityVisionDataset",
  "SafeVelocityVisionStudent",
  "decode_safe_velocity_student",
  "safe_velocity_student_loss",
]
