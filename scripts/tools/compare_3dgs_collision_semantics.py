"""Compare geometry-only and semantic-priority proxies at one geom budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from mjlab.asset_zoo.gaussian_collision import (
  PrimitiveFitConfig,
  VoxelBoxConfig,
  build_collision_mjcf,
  evaluate_collision_proxy,
  fit_collision_primitives,
  generate_voxel_boxes,
  select_collision_primitive_budget,
)
from mjlab.scene.gaussian_splat import _read_ply_vertices
from mjlab.scripts.benchmark_collision_mjcf import benchmark_collision_mjcf


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_priorities(path: Path) -> dict[int, float]:
  raw = json.loads(path.read_text())
  if not isinstance(raw, dict):
    raise ValueError("--semantic-priorities must contain a JSON object.")
  try:
    priorities = {int(label): float(weight) for label, weight in raw.items()}
  except (TypeError, ValueError) as exc:
    raise ValueError("Semantic priorities must map integer labels to numbers.") from exc
  if any(not np.isfinite(value) or value <= 0 for value in priorities.values()):
    raise ValueError("Semantic priority weights must be finite and positive.")
  return priorities


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input_ply", type=Path)
  parser.add_argument("semantic_labels", type=Path)
  parser.add_argument("semantic_priorities", type=Path)
  parser.add_argument("output_dir", type=Path)
  parser.add_argument("--max-primitives", type=int, required=True)
  parser.add_argument("--evaluation-points", type=Path)
  parser.add_argument("--voxel-size", type=float, default=0.1)
  parser.add_argument("--opacity-threshold", type=float, default=0.5)
  parser.add_argument("--lower-quantile", type=float, default=0.001)
  parser.add_argument("--upper-quantile", type=float, default=0.999)
  parser.add_argument("--min-points-per-voxel", type=int, default=3)
  parser.add_argument("--min-component-voxels", type=int, default=8)
  parser.add_argument("--min-box-volume", type=float, default=0.1)
  parser.add_argument("--warmup-steps", type=int, default=1_000)
  parser.add_argument("--measured-steps", type=int, default=10_000)
  parser.add_argument("--probe-grid", type=int, default=3)
  parser.add_argument("--probe-spacing", type=float, default=0.4)
  parser.add_argument("--probe-z", type=float, default=1.0)
  parser.add_argument("--probe-radius", type=float, default=0.08)
  return parser.parse_args()


def _evaluation_arrays(
  path: Path | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
  if path is None:
    return None, None
  with np.load(path, allow_pickle=False) as arrays:
    supported = {"surface_points", "free_points"}
    unexpected = set(arrays.files) - supported
    if unexpected or not arrays.files:
      raise ValueError(
        "Evaluation NPZ must contain only surface_points and/or free_points; "
        f"unexpected={sorted(unexpected)}."
      )
    surface = arrays["surface_points"].copy() if "surface_points" in arrays else None
    free = arrays["free_points"].copy() if "free_points" in arrays else None
  return surface, free


def main() -> None:
  args = _parse_args()
  if args.max_primitives <= 0:
    raise ValueError("--max-primitives must be positive.")
  vertices = _read_ply_vertices(args.input_ply)
  missing = [name for name in ("x", "y", "z", "opacity") if name not in vertices]
  if missing:
    raise ValueError(f"3DGS PLY is missing required fields: {missing}.")
  points = np.column_stack([vertices[axis] for axis in ("x", "y", "z")])
  labels = np.load(args.semantic_labels, allow_pickle=False)
  if labels.ndim != 1 or len(labels) != len(points) or labels.dtype.kind not in "iu":
    raise ValueError("Semantic labels must be one integer per original PLY vertex.")
  priorities = _load_priorities(args.semantic_priorities)
  surface_points, free_points = _evaluation_arrays(args.evaluation_points)
  config = VoxelBoxConfig(
    voxel_size=args.voxel_size,
    opacity_threshold=args.opacity_threshold,
    lower_quantile=args.lower_quantile,
    upper_quantile=args.upper_quantile,
    min_points_per_voxel=args.min_points_per_voxel,
    min_component_voxels=args.min_component_voxels,
    min_box_volume=args.min_box_volume,
  )
  fit_config = PrimitiveFitConfig()
  args.output_dir.mkdir(parents=True, exist_ok=True)

  variants: dict[str, dict[str, object]] = {}
  paths: list[Path] = []
  for name, variant_labels, variant_priorities in (
    ("geometry_only", None, None),
    ("semantic_priority", labels, priorities),
  ):
    started = time.perf_counter()
    voxel_result = generate_voxel_boxes(
      points,
      vertices["opacity"],
      config,
      semantic_labels=variant_labels,
    )
    all_primitives = fit_collision_primitives(
      points,
      voxel_result,
      config,
      fit_config,
    )
    budgeted = select_collision_primitive_budget(
      all_primitives,
      voxel_result,
      max_primitives=args.max_primitives,
      semantic_priorities=variant_priorities,
    )
    generation_seconds = time.perf_counter() - started
    if len(budgeted.primitives) != args.max_primitives:
      raise ValueError(
        f"{name} produced only {len(budgeted.primitives)} primitives before the "
        f"requested fixed budget of {args.max_primitives}; lower the budget."
      )
    output_path = args.output_dir / f"{name}.xml"
    output_path.write_text(
      build_collision_mjcf(
        voxel_result,
        config,
        model_name=f"p8_{name}",
        primitive_result=budgeted,
      )
    )
    paths.append(output_path)
    variants[name] = {
      "generation_seconds": generation_seconds,
      "primitive_count_before_budget": len(all_primitives.primitives),
      "primitive_count": len(budgeted.primitives),
      "quality": evaluate_collision_proxy(
        voxel_result,
        config,
        primitive_result=budgeted,
        reference_surface_points=surface_points,
        free_space_points=free_points,
      ),
      "output": {"path": str(output_path), "sha256": _sha256(output_path)},
    }

  for path, variant in zip(paths, variants.values(), strict=True):
    variant["physics"] = benchmark_collision_mjcf(
      path,
      warmup_steps=args.warmup_steps,
      measured_steps=args.measured_steps,
      probe_grid=args.probe_grid,
      probe_spacing=args.probe_spacing,
      probe_z=args.probe_z,
      probe_radius=args.probe_radius,
    )
  report = {
    "schema_version": 1,
    "tool_version": version("mjlab"),
    "comparison_contract": {
      "fixed_primitive_budget": args.max_primitives,
      "same_input": True,
      "same_voxel_and_fit_config": True,
      "same_physics_probe_config": True,
      "semantic_variant_difference": (
        "integer labels split candidates and priorities rank the fixed budget"
      ),
      "visual_throughput": None,
      "visual_throughput_limitation": (
        "Requires a CUDA-visible 3DGS renderer and is intentionally not inferred "
        "from this rendering-independent CPU benchmark."
      ),
    },
    "input": {
      "path": str(args.input_ply),
      "sha256": _sha256(args.input_ply),
      "gaussian_count": len(points),
      "semantic_labels_path": str(args.semantic_labels),
      "semantic_labels_sha256": _sha256(args.semantic_labels),
      "semantic_priorities_path": str(args.semantic_priorities),
      "semantic_priorities_sha256": _sha256(args.semantic_priorities),
      "evaluation_points_path": (
        str(args.evaluation_points) if args.evaluation_points is not None else None
      ),
      "evaluation_points_sha256": (
        _sha256(args.evaluation_points) if args.evaluation_points is not None else None
      ),
    },
    "physics_settings": {
      "warmup_steps": args.warmup_steps,
      "measured_steps": args.measured_steps,
      "probe_grid": args.probe_grid,
      "probe_spacing": args.probe_spacing,
      "probe_z": args.probe_z,
      "probe_radius": args.probe_radius,
    },
    "variants": variants,
  }
  report_path = args.output_dir / "semantic_ablation_report.json"
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"Wrote fixed-budget semantic ablation to {report_path}")


if __name__ == "__main__":
  main()
