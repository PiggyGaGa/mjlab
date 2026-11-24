"""Export semantic projections and prediction-blind manual labeling samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def sample_region(
  coordinates: np.ndarray, bounds: np.ndarray, count: int, seed: int
) -> np.ndarray:
  if coordinates.ndim != 2 or coordinates.shape[1] != 3:
    raise ValueError("Coordinates must have shape (N, 3).")
  if bounds.shape != (6,) or not np.isfinite(bounds).all():
    raise ValueError("Bounds must contain six finite values.")
  if np.any(bounds[:3] >= bounds[3:]) or count <= 0:
    raise ValueError("Bounds must increase and sample count must be positive.")
  selected = np.flatnonzero(
    np.isfinite(coordinates).all(axis=1)
    & (coordinates >= bounds[:3]).all(axis=1)
    & (coordinates <= bounds[3:]).all(axis=1)
  )
  if not len(selected):
    raise ValueError("No finite coordinates inside bounds.")
  return np.sort(
    np.random.default_rng(seed).choice(
      selected, size=min(count, len(selected)), replace=False
    )
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input_root", type=Path)
  parser.add_argument("raw_labels", type=Path)
  parser.add_argument("gated_labels", type=Path)
  parser.add_argument("class_names", type=Path)
  parser.add_argument("output_dir", type=Path)
  parser.add_argument("--bounds", nargs=6, type=float, required=True)
  parser.add_argument("--samples", type=int, default=48)
  parser.add_argument("--preview-points", type=int, default=30000)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  coordinates = np.load(args.input_root / "coord.npy", allow_pickle=False)
  colors = np.load(args.input_root / "color.npy", allow_pickle=False)
  raw = np.load(args.raw_labels, allow_pickle=False)
  gated = np.load(args.gated_labels, allow_pickle=False)
  names = json.loads(args.class_names.read_text())
  if (
    not isinstance(names, list)
    or not names
    or not all(isinstance(name, str) for name in names)
  ):
    raise ValueError("Class names must be a non-empty JSON string list.")
  for labels in (raw, gated):
    if labels.shape != (len(coordinates),) or labels.dtype.kind not in "iu":
      raise ValueError("Labels must contain one integer per coordinate row.")
    if np.any((labels < -1) | (labels >= len(names))):
      raise ValueError("Label IDs must be -1 or valid class indices.")
  if colors.shape != coordinates.shape or not np.isfinite(colors).all():
    raise ValueError("Colors must be a finite (N, 3) array in [0, 255].")
  if np.any((colors < 0) | (colors > 255)):
    raise ValueError("Colors must be in [0, 255].")
  bounds = np.asarray(args.bounds)
  samples = sample_region(coordinates, bounds, args.samples, args.seed)
  preview = sample_region(coordinates, bounds, args.preview_points, args.seed)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  annotation_path = args.output_dir / "manual_labels.csv"
  if annotation_path.exists():
    raise FileExistsError(f"Refusing to overwrite manual labels: {annotation_path}")
  palette = plt.get_cmap("hsv")(np.mod(np.arange(len(names)) * 0.61803398875, 1))[:, :3]
  projections = ((0, 1), (0, 2), (1, 2))
  figure, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
  for row, labels in enumerate((None, raw, gated)):
    display_colors = colors[preview] / 255
    if labels is not None:
      display_colors = palette[np.maximum(labels[preview], 0)].copy()
      display_colors[labels[preview] == -1] = 0.65
    for column, (horizontal, vertical) in enumerate(projections):
      axis = axes[row, column]
      axis.scatter(
        coordinates[preview, horizontal],
        coordinates[preview, vertical],
        c=display_colors,
        s=1,
      )
      axis.set_aspect("equal")
      axis.set_xlabel("xyz"[horizontal] + " (m)")
      axis.set_ylabel("xyz"[vertical] + " (m)")
      axis.set_title(("Original RGB", "Raw prediction", "Gated prediction")[row])
  figure.savefig(args.output_dir / "semantic_projections.png", dpi=150)
  plt.close(figure)
  figure = plt.figure(figsize=(14, 10))
  figure.legend(
    handles=[Patch(color="0.65", label="-1: unknown")]
    + [
      Patch(color=palette[index], label=f"{index}: {name}")
      for index, name in enumerate(names)
    ],
    loc="center",
    ncol=4,
    fontsize=9,
  )
  figure.savefig(args.output_dir / "class_legend.png", dpi=150)
  plt.close(figure)
  figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
  for axis, (horizontal, vertical) in zip(axes, projections, strict=True):
    axis.scatter(
      coordinates[preview, horizontal],
      coordinates[preview, vertical],
      c=colors[preview] / 255,
      s=2,
    )
    for index, source_id in enumerate(samples):
      axis.annotate(
        str(index),
        (coordinates[source_id, horizontal], coordinates[source_id, vertical]),
        fontsize=7,
        color="red",
      )
    axis.set_aspect("equal")
    axis.set_xlabel("xyz"[horizontal] + " (m)")
    axis.set_ylabel("xyz"[vertical] + " (m)")
  figure.savefig(args.output_dir / "blind_samples.png", dpi=180)
  plt.close(figure)
  with annotation_path.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
      ("sample_id", "source_id", "x", "y", "z", "human_class_id", "notes")
    )
    for index, source_id in enumerate(samples):
      writer.writerow((index, source_id, *coordinates[source_id], "", ""))
  metadata = {
    "bounds": args.bounds,
    "seed": args.seed,
    "sample_count": len(samples),
    "sampling": "uniform original Gaussian rows within bounds, not area-weighted",
    "limitation": "projections can occlude; abstain instead of guessing human labels",
    "classes": [
      {"id": index, "name": name, "rgb": palette[index].tolist()}
      for index, name in enumerate(names)
    ],
    "inputs": {
      str(path): hashlib.sha256(path.read_bytes()).hexdigest()
      for path in (
        args.input_root / "coord.npy",
        args.input_root / "color.npy",
        args.raw_labels,
        args.gated_labels,
        args.class_names,
      )
    },
  }
  (args.output_dir / "inspection.json").write_text(
    json.dumps(metadata, indent=2) + "\n"
  )


if __name__ == "__main__":
  main()
