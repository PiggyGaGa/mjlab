#!/usr/bin/env python3
"""Train the visual safe-velocity student from schema-v2 offline chunks."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from mjlab.tasks.velocity.config.g1.safe_velocity_student import (
  SAFE_VELOCITY_STUDENT_MODEL_VERSION,
  SafeVelocityStudentLossCfg,
  SafeVelocityVisionDataset,
  SafeVelocityVisionStudent,
  decode_safe_velocity_student,
  safe_velocity_student_loss,
)


def _seed_everything(seed: int) -> None:
  random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _loader(
  dataset: Dataset[dict[str, Tensor]],
  *,
  batch_size: int,
  shuffle: bool,
  device: torch.device,
) -> DataLoader[dict[str, Tensor]]:
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=0,
    pin_memory=device.type == "cuda",
  )


def _run_epoch(
  model: SafeVelocityVisionStudent,
  loader: DataLoader[dict[str, Tensor]],
  loss_cfg: SafeVelocityStudentLossCfg,
  device: torch.device,
  optimizer: AdamW | None,
) -> dict[str, float]:
  training = optimizer is not None
  model.train(training)
  totals = {
    "loss": 0.0,
    "clearance_loss": 0.0,
    "validity_loss": 0.0,
    "alpha_loss": 0.0,
    "stop_loss": 0.0,
  }
  sample_count = 0
  alpha_absolute_error = 0.0
  unsafe_alpha_count = 0
  false_stop_count = 0
  negative_stop_count = 0
  missed_stop_count = 0
  positive_stop_count = 0

  context = torch.enable_grad if training else torch.no_grad
  with context():
    for batch in loader:
      camera = batch["camera"].to(device, non_blocking=True)
      command = batch["nominal_command"].to(device, non_blocking=True)
      labels = batch["teacher_labels"].to(device, non_blocking=True)
      logits = model(camera, command)
      losses = safe_velocity_student_loss(logits, labels, loss_cfg)

      if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()

      batch_size = camera.shape[0]
      sample_count += batch_size
      for name in totals:
        totals[name] += float(losses[name].detach()) * batch_size

      predictions = decode_safe_velocity_student(logits.detach())
      alpha_error = predictions[:, 2] - labels[:, 2]
      alpha_absolute_error += float(alpha_error.abs().sum())
      unsafe_alpha_count += int((alpha_error > 0.05).sum())
      predicted_stop = predictions[:, 3] >= 0.5
      target_stop = labels[:, 3] >= 0.5
      false_stop_count += int((predicted_stop & ~target_stop).sum())
      negative_stop_count += int((~target_stop).sum())
      missed_stop_count += int((~predicted_stop & target_stop).sum())
      positive_stop_count += int(target_stop.sum())

  metrics = {name: value / sample_count for name, value in totals.items()}
  metrics.update(
    {
      "alpha_mae": alpha_absolute_error / sample_count,
      "unsafe_alpha_rate": unsafe_alpha_count / sample_count,
      "false_stop_rate": false_stop_count / max(negative_stop_count, 1),
      "missed_stop_rate": missed_stop_count / max(positive_stop_count, 1),
    }
  )
  return metrics


def _save_checkpoint(
  path: Path,
  *,
  model: SafeVelocityVisionStudent,
  optimizer: AdamW,
  epoch: int,
  loss_cfg: SafeVelocityStudentLossCfg,
  history_length: int,
  datasets: dict[str, list[str]],
) -> None:
  temporary_path = path.with_suffix(path.suffix + ".tmp")
  torch.save(
    {
      "model": model.state_dict(),
      "model_version": SAFE_VELOCITY_STUDENT_MODEL_VERSION,
      "optimizer": optimizer.state_dict(),
      "epoch": epoch,
      "history_length": history_length,
      "loss_config": asdict(loss_cfg),
      "datasets": datasets,
    },
    temporary_path,
  )
  temporary_path.replace(path)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train-dataset", type=Path, action="append", required=True)
  parser.add_argument("--validation-dataset", type=Path, action="append", required=True)
  parser.add_argument("--test-dataset", type=Path, action="append", default=[])
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--unsafe-alpha-weight", type=float, default=4.0)
  parser.add_argument("--stop-positive-weight", type=float, default=None)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", type=str, default=None)
  args = parser.parse_args()
  if args.epochs < 1 or args.batch_size < 1:
    parser.error("--epochs and --batch-size must be positive")
  if args.learning_rate <= 0 or args.weight_decay < 0:
    parser.error("Invalid optimizer parameters")
  if args.unsafe_alpha_weight < 1:
    parser.error("--unsafe-alpha-weight must be at least 1")
  if args.stop_positive_weight is not None and args.stop_positive_weight <= 0:
    parser.error("--stop-positive-weight must be positive")
  return args


def main() -> None:
  args = _parse_args()
  if args.output_dir.exists() and any(args.output_dir.iterdir()):
    raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  _seed_everything(args.seed)
  device = torch.device(
    args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  )

  train_dataset = SafeVelocityVisionDataset(
    args.train_dataset, allowed_splits=frozenset({"train"})
  )
  validation_dataset = SafeVelocityVisionDataset(
    args.validation_dataset, allowed_splits=frozenset({"validation"})
  )
  test_dataset = (
    SafeVelocityVisionDataset(args.test_dataset, allowed_splits=frozenset({"test"}))
    if args.test_dataset
    else None
  )
  history_length = train_dataset.camera.shape[1]
  if validation_dataset.camera.shape[1:] != train_dataset.camera.shape[1:]:
    raise ValueError("Train and validation camera shapes differ")
  if (
    test_dataset is not None
    and test_dataset.camera.shape[1:] != train_dataset.camera.shape[1:]
  ):
    raise ValueError("Train and test camera shapes differ")

  positives = train_dataset.stop_positive_count
  negatives = len(train_dataset) - positives
  if positives == 0:
    raise ValueError("Training data contains no positive stop labels")
  stop_positive_weight = args.stop_positive_weight or negatives / positives
  loss_cfg = SafeVelocityStudentLossCfg(
    unsafe_alpha_weight=args.unsafe_alpha_weight,
    stop_positive_weight=stop_positive_weight,
  )
  model = SafeVelocityVisionStudent(history_length=history_length).to(device)
  optimizer = AdamW(
    model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
  )
  datasets = {
    "train": [str(path.resolve()) for path in args.train_dataset],
    "validation": [str(path.resolve()) for path in args.validation_dataset],
    "test": [str(path.resolve()) for path in args.test_dataset],
  }
  run_config: dict[str, Any] = {
    "seed": args.seed,
    "device": str(device),
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
    "weight_decay": args.weight_decay,
    "history_length": history_length,
    "model_version": SAFE_VELOCITY_STUDENT_MODEL_VERSION,
    "loss_config": asdict(loss_cfg),
    "datasets": datasets,
    "samples": {
      "train": len(train_dataset),
      "validation": len(validation_dataset),
      "test": len(test_dataset) if test_dataset is not None else 0,
    },
  }
  (args.output_dir / "config.json").write_text(
    json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )

  train_loader = _loader(
    train_dataset, batch_size=args.batch_size, shuffle=True, device=device
  )
  validation_loader = _loader(
    validation_dataset, batch_size=args.batch_size, shuffle=False, device=device
  )
  test_loader = (
    _loader(test_dataset, batch_size=args.batch_size, shuffle=False, device=device)
    if test_dataset is not None
    else None
  )

  history: list[dict[str, Any]] = []
  best_validation_loss = float("inf")
  for epoch in range(1, args.epochs + 1):
    train_metrics = _run_epoch(
      model, train_loader, loss_cfg, device, optimizer=optimizer
    )
    validation_metrics = _run_epoch(
      model, validation_loader, loss_cfg, device, optimizer=None
    )
    epoch_metrics = {
      "epoch": epoch,
      "train": train_metrics,
      "validation": validation_metrics,
    }
    history.append(epoch_metrics)
    (args.output_dir / "metrics.json").write_text(
      json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _save_checkpoint(
      args.output_dir / "last.pt",
      model=model,
      optimizer=optimizer,
      epoch=epoch,
      loss_cfg=loss_cfg,
      history_length=history_length,
      datasets=datasets,
    )
    if validation_metrics["loss"] < best_validation_loss:
      best_validation_loss = validation_metrics["loss"]
      _save_checkpoint(
        args.output_dir / "best.pt",
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        loss_cfg=loss_cfg,
        history_length=history_length,
        datasets=datasets,
      )
    print(
      f"epoch={epoch}/{args.epochs} "
      f"train_loss={train_metrics['loss']:.5f} "
      f"val_loss={validation_metrics['loss']:.5f} "
      f"val_alpha_mae={validation_metrics['alpha_mae']:.5f} "
      f"val_false_stop={validation_metrics['false_stop_rate']:.4f} "
      f"val_missed_stop={validation_metrics['missed_stop_rate']:.4f}"
    )

  if test_loader is not None:
    best_checkpoint = torch.load(
      args.output_dir / "best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = _run_epoch(model, test_loader, loss_cfg, device, optimizer=None)
    scenario_metrics = {}
    assert test_dataset is not None
    for scenario_id, start, end in test_dataset.sources:
      scenario_loader = _loader(
        Subset(test_dataset, range(start, end)),
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
      )
      scenario_metrics[scenario_id] = _run_epoch(
        model, scenario_loader, loss_cfg, device, optimizer=None
      )
    test_report = {"combined": test_metrics, "scenarios": scenario_metrics}
    (args.output_dir / "test_metrics.json").write_text(
      json.dumps(test_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
      f"test_best_epoch={best_checkpoint['epoch']} "
      f"test_loss={test_metrics['loss']:.5f} "
      f"test_alpha_mae={test_metrics['alpha_mae']:.5f} "
      f"test_false_stop={test_metrics['false_stop_rate']:.4f} "
      f"test_missed_stop={test_metrics['missed_stop_rate']:.4f}"
    )
    for scenario_id, metrics in scenario_metrics.items():
      print(
        f"test_scenario={scenario_id} "
        f"alpha_mae={metrics['alpha_mae']:.5f} "
        f"false_stop={metrics['false_stop_rate']:.4f} "
        f"missed_stop={metrics['missed_stop_rate']:.4f}"
      )


if __name__ == "__main__":
  main()
