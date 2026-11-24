"""Generate collision with optional semantic hints and an explicit geometric fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mjlab.scene.gaussian_splat import _read_ply_vertices


def semantic_rejection_reasons(
  baseline: dict[str, Any], candidate: dict[str, Any], maximum_error_ratio: float
) -> list[str]:
  if not np.isfinite(maximum_error_ratio) or maximum_error_ratio < 1:
    raise ValueError("Maximum error ratio must be finite and at least 1.")
  reasons = []
  coverages = np.array(
    [
      baseline["statistics"]["mapped_gaussian_fraction"],
      candidate["statistics"]["mapped_gaussian_fraction"],
    ],
    dtype=np.float64,
  )
  if not np.isfinite(coverages).all() or np.any((coverages < 0) | (coverages > 1)):
    raise ValueError("Source mapping fractions must be finite and in [0, 1].")
  if baseline["input"]["sha256"] != candidate["input"]["sha256"]:
    raise ValueError("Guard reports must refer to the same input PLY.")
  if (
    baseline["statistics"]["filtered_gaussian_count"]
    != candidate["statistics"]["filtered_gaussian_count"]
  ):
    raise ValueError("Guard reports must use the same filtered source population.")
  if (
    candidate["statistics"]["mapped_gaussian_fraction"]
    < baseline["statistics"]["mapped_gaussian_fraction"]
  ):
    reasons.append("semantic_source_mapping_coverage_decreased")
  for key in ("surface_distance_mean", "surface_distance_p95", "surface_distance_max"):
    baseline_value = float(baseline["source_surface_evaluation"][key])
    candidate_value = float(candidate["source_surface_evaluation"][key])
    if (
      not np.isfinite([baseline_value, candidate_value]).all()
      or min(baseline_value, candidate_value) < 0
    ):
      raise ValueError("Guard requires finite nonnegative source-distance metrics.")
    if candidate_value > baseline_value * maximum_error_ratio + 1e-9:
      reasons.append(f"semantic_{key}_increased")
  return reasons


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input_ply", type=Path)
  parser.add_argument("semantic_labels", type=Path)
  parser.add_argument("output_dir", type=Path)
  parser.add_argument("--maximum-unknown-fraction", type=float, default=0.5)
  parser.add_argument("--maximum-error-ratio", type=float, default=1.05)
  args = parser.parse_args()
  if not 0 <= args.maximum_unknown_fraction <= 1:
    raise ValueError("Maximum unknown fraction must be in [0, 1].")
  if not np.isfinite(args.maximum_error_ratio) or args.maximum_error_ratio < 1:
    raise ValueError("Maximum error ratio must be finite and at least 1.")
  labels = np.load(args.semantic_labels, allow_pickle=False)
  if (
    labels.ndim != 1
    or not len(labels)
    or labels.dtype.kind not in "iu"
    or np.any(labels < -1)
  ):
    raise ValueError(
      "Semantic labels must be a non-empty integer vector with unknown=-1."
    )
  input_ply = args.input_ply.resolve()
  semantic_labels = args.semantic_labels.resolve()
  output_dir = args.output_dir.resolve()
  for source in (input_ply, semantic_labels):
    if source == output_dir or source.is_relative_to(output_dir):
      raise ValueError("Inputs must not reside within the output directory.")
  if len(_read_ply_vertices(input_ply)["x"]) != len(labels):
    raise ValueError("Semantic label count must match the original PLY row count.")
  generator = Path(__file__).with_name("generate_3dgs_collision.py")

  def generate(name: str, with_semantics: bool) -> dict[str, Any]:
    folder = output_dir / name
    command = [
      sys.executable,
      str(generator),
      str(input_ply),
      str(folder / "collision.xml"),
      "--report",
      str(folder / "report.json"),
      "--primitive-mode",
      "auto",
      "--evaluate-source-surface",
    ]
    if with_semantics:
      command.extend(("--semantic-labels", str(semantic_labels)))
    subprocess.run(command, check=True)
    return json.loads((folder / "report.json").read_text())

  baseline = generate("geometry", False)
  selected = "geometry"
  unknown_fraction = float(np.mean(labels == -1))
  reasons = []
  candidate = None
  if unknown_fraction > args.maximum_unknown_fraction:
    reasons.append("semantic_unknown_fraction_exceeds_limit")
  else:
    candidate = generate("semantic", True)
    reasons = semantic_rejection_reasons(baseline, candidate, args.maximum_error_ratio)
    if not reasons:
      selected = "semantic"
  for name in ("collision.xml", "report.json", "report.source_ids.npz"):
    shutil.copyfile(output_dir / selected / name, output_dir / name)
  guard_report = {
    "schema_version": 1,
    "selected": selected,
    "semantic_used": selected == "semantic",
    "rejection_reasons": reasons,
    "unknown_fraction": unknown_fraction,
    "maximum_unknown_fraction": args.maximum_unknown_fraction,
    "maximum_error_ratio": args.maximum_error_ratio,
    "limitations": [
      "Coverage/error gates are engineering safeguards, not calibrated semantic confidence.",
      "Source-surface distances are not independent geometric or semantic ground truth.",
      "Choosing geometry does not infer labels or treat unknown categories as free space.",
      "This guard does not establish support correctness, task success or collision safety.",
    ],
    "input": {
      "ply_sha256": hashlib.sha256(input_ply.read_bytes()).hexdigest(),
      "labels_sha256": hashlib.sha256(semantic_labels.read_bytes()).hexdigest(),
    },
    "baseline": baseline,
    "semantic_candidate": candidate,
    "output_mjcf_sha256": hashlib.sha256(
      (output_dir / "collision.xml").read_bytes()
    ).hexdigest(),
  }
  (output_dir / "guard_report.json").write_text(
    json.dumps(guard_report, indent=2) + "\n"
  )
  print(f"Selected {selected} collision; semantic rejection reasons: {reasons}")


if __name__ == "__main__":
  main()
