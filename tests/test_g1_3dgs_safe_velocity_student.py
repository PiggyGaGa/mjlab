from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mjlab.tasks.velocity.config.g1.safe_velocity_student import (
  SafeVelocityStudentLossCfg,
  SafeVelocityVisionDataset,
  SafeVelocityVisionStudent,
  decode_safe_velocity_student,
  safe_velocity_student_loss,
)


def _write_dataset(directory: Path, *, split: str) -> None:
  directory.mkdir()
  chunk = {
    "camera": torch.rand(3, 7, 2, 48, 64, dtype=torch.float16),
    "teacher_labels": torch.tensor(
      [[0.1, 1.0, 0.0, 1.0], [0.5, 1.0, 0.5, 0.0], [1.0, 1.0, 1.0, 0.0]]
    ),
    "nominal_command": torch.tensor([[0.5, 0.0, 0.0]]).repeat(3, 1),
  }
  torch.save(chunk, directory / "chunk_000000.pt")
  manifest = {
    "schema_version": 2,
    "samples": 3,
    "chunks": [{"file": "chunk_000000.pt", "samples": 3}],
    "scenario": {"id": "fixture", "split": split},
  }
  (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_student_model_outputs_normalized_teacher_shape() -> None:
  model = SafeVelocityVisionStudent()
  logits = model(torch.rand(2, 7, 2, 48, 64), torch.rand(2, 3))
  predictions = decode_safe_velocity_student(logits)

  assert logits.shape == (2, 4)
  assert torch.all((predictions >= 0.0) & (predictions <= 1.0))
  assert torch.equal(predictions[:, 3] >= 0.5, predictions[:, 2] <= 0.1)


def test_student_loss_penalizes_unsafe_alpha_overestimate() -> None:
  cfg = SafeVelocityStudentLossCfg(
    clearance_weight=0.0,
    validity_weight=0.0,
    stop_weight=0.0,
    unsafe_alpha_weight=4.0,
  )
  teacher = torch.tensor([[0.5, 1.0, 0.5, 0.0]])
  unsafe = torch.tensor([[0.0, 0.0, torch.logit(torch.tensor(0.7)), 0.0]])
  conservative = torch.tensor([[0.0, 0.0, torch.logit(torch.tensor(0.3)), 0.0]])

  unsafe_loss = safe_velocity_student_loss(unsafe, teacher, cfg)["loss"]
  conservative_loss = safe_velocity_student_loss(conservative, teacher, cfg)["loss"]

  assert unsafe_loss > conservative_loss * 3.9


def test_vision_dataset_loads_chunks_and_enforces_split(tmp_path: Path) -> None:
  dataset_path = tmp_path / "train"
  _write_dataset(dataset_path, split="train")

  dataset = SafeVelocityVisionDataset(
    [dataset_path], allowed_splits=frozenset({"train"})
  )

  assert len(dataset) == 3
  assert dataset.stop_positive_count == 1
  assert dataset.sources == [("fixture", 0, 3)]
  assert dataset[0]["camera"].dtype == torch.float32
  with pytest.raises(ValueError, match="expected validation"):
    SafeVelocityVisionDataset([dataset_path], allowed_splits=frozenset({"validation"}))
