"""Generate deterministic voxel/primitive collision proxies from 3DGS centers."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from mjlab.asset_zoo.gaussian_collision import (
  PrimitiveFitConfig,
  VoxelBoxConfig,
  build_collision_mjcf,
  config_dict,
  evaluate_collision_proxy,
  fit_collision_primitives,
  gaussian_covariances,
  generate_voxel_boxes,
  primitive_fit_config_dict,
  select_collision_primitive_budget,
)
from mjlab.asset_zoo.gaussian_collision_surfaces import (
  HorizontalSurfaceConfig,
  consolidate_horizontal_surfaces,
)
from mjlab.scene.gaussian_splat import _read_ply_vertices


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _write_source_map(path: Path, arrays: dict[str, np.ndarray]) -> None:
  """Write an NPZ with fixed metadata so identical arrays have identical bytes."""
  with zipfile.ZipFile(path, "w") as archive:
    for name, array in arrays.items():
      buffer = io.BytesIO()
      np.save(buffer, array, allow_pickle=False)
      info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
      info.compress_type = zipfile.ZIP_DEFLATED
      info.external_attr = 0o600 << 16
      archive.writestr(info, buffer.getvalue(), compresslevel=9)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input_ply", type=Path)
  parser.add_argument("output_mjcf", type=Path)
  parser.add_argument("--report", type=Path, required=True)
  parser.add_argument("--source-map", type=Path)
  parser.add_argument("--model-name", default="gaussian_collision")
  parser.add_argument("--voxel-size", type=float, default=0.1)
  parser.add_argument("--opacity-threshold", type=float, default=0.5)
  parser.add_argument(
    "--opacity-encoding", choices=("logit", "linear"), default="logit"
  )
  parser.add_argument("--scale-encoding", choices=("log", "linear"), default="log")
  parser.add_argument("--max-gaussian-scale", type=float)
  parser.add_argument("--lower-quantile", type=float, default=0.001)
  parser.add_argument("--upper-quantile", type=float, default=0.999)
  parser.add_argument("--min-points-per-voxel", type=int, default=3)
  parser.add_argument("--min-component-voxels", type=int, default=8)
  parser.add_argument("--min-box-fill", type=float, default=0.2)
  parser.add_argument("--max-surface-expansion", type=float, default=0.4)
  parser.add_argument("--min-box-volume", type=float, default=0.1)
  parser.add_argument("--max-grid-voxels", type=int, default=50_000_000)
  parser.add_argument("--floor-z", type=float)
  parser.add_argument(
    "--semantic-labels",
    type=Path,
    help="Optional .npy integer label per original PLY vertex.",
  )
  parser.add_argument("--unknown-semantic-label", type=int, default=-1)
  parser.add_argument(
    "--evaluation-points",
    type=Path,
    help="Optional .npz with independent surface_points and/or free_points arrays.",
  )
  parser.add_argument(
    "--evaluate-source-surface",
    action="store_true",
    help="Report distance from retained high-confidence Gaussian centers to proxy.",
  )
  parser.add_argument(
    "--known-free-points",
    type=Path,
    help="Optional .npy array of finite world-space XYZ points boxes must not cover.",
  )
  parser.add_argument("--max-planar-merge-residual", type=float)
  parser.add_argument("--planar-merge-eigenvalue-ratio", type=float, default=0.08)
  parser.add_argument("--primitive-mode", choices=("boxes", "auto"), default="boxes")
  parser.add_argument("--primitive-min-points", type=int, default=24)
  parser.add_argument("--primitive-max-error", type=float, default=0.12)
  parser.add_argument("--linear-eigenvalue-ratio", type=float, default=0.35)
  parser.add_argument("--planar-eigenvalue-ratio", type=float, default=0.08)
  parser.add_argument("--isotropic-eigenvalue-ratio", type=float, default=0.45)
  parser.add_argument("--radial-eigenvalue-ratio", type=float, default=0.55)
  parser.add_argument("--min-planar-fill", type=float, default=0.2)
  parser.add_argument("--min-round-error-improvement", type=float, default=0.3)
  parser.add_argument("--planar-slab-thickness", type=float)
  parser.add_argument("--planar-free-side-max-distance", type=float, default=0.5)
  parser.add_argument("--planar-free-side-min-agreement", type=float, default=0.9)
  parser.add_argument("--use-gaussian-covariance", action="store_true")
  parser.add_argument("--merge-horizontal-surfaces", action="store_true")
  parser.add_argument("--horizontal-level-tolerance", type=float, default=0.04)
  parser.add_argument("--horizontal-surface-residual", type=float, default=0.06)
  parser.add_argument("--horizontal-slab-thickness", type=float, default=0.1)
  parser.add_argument("--horizontal-max-tilt-degrees", type=float, default=5.0)
  parser.add_argument(
    "--max-primitives",
    type=int,
    help="Optional deterministic output geom budget for auto mode.",
  )
  parser.add_argument(
    "--semantic-priorities",
    type=Path,
    help="Optional JSON object mapping integer semantic labels to positive weights.",
  )
  parser.add_argument(
    "--point-importance",
    type=Path,
    help="Optional .npy non-negative importance value per original PLY vertex.",
  )
  parser.add_argument(
    "--debug-color-by-type",
    action="store_true",
    help="Color auto primitives by type for viewer inspection; physics is unchanged.",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.merge_horizontal_surfaces and args.primitive_mode != "auto":
    raise ValueError("--merge-horizontal-surfaces requires --primitive-mode auto.")
  if args.debug_color_by_type and args.primitive_mode != "auto":
    raise ValueError("--debug-color-by-type requires --primitive-mode auto.")
  if args.max_primitives is not None and args.primitive_mode != "auto":
    raise ValueError("--max-primitives requires --primitive-mode auto.")
  if (args.semantic_priorities is not None or args.point_importance is not None) and (
    args.max_primitives is None
  ):
    raise ValueError("Priority inputs require --max-primitives.")
  config = VoxelBoxConfig(
    voxel_size=args.voxel_size,
    opacity_threshold=args.opacity_threshold,
    opacity_encoding=args.opacity_encoding,
    scale_encoding=args.scale_encoding,
    max_gaussian_scale=args.max_gaussian_scale,
    lower_quantile=args.lower_quantile,
    upper_quantile=args.upper_quantile,
    min_points_per_voxel=args.min_points_per_voxel,
    min_component_voxels=args.min_component_voxels,
    min_box_fill=args.min_box_fill,
    max_surface_expansion=args.max_surface_expansion,
    min_box_volume=args.min_box_volume,
    max_grid_voxels=args.max_grid_voxels,
    floor_z=args.floor_z,
    max_planar_merge_residual=args.max_planar_merge_residual,
    planar_merge_eigenvalue_ratio=args.planar_merge_eigenvalue_ratio,
    unknown_semantic_label=args.unknown_semantic_label,
  )
  vertices = _read_ply_vertices(args.input_ply)
  missing = [name for name in ("x", "y", "z", "opacity") if name not in vertices]
  if missing:
    raise ValueError(f"3DGS PLY is missing required fields: {missing}.")
  points = np.column_stack([vertices[axis] for axis in ("x", "y", "z")])
  scale_fields = tuple(f"scale_{axis}" for axis in range(3))
  rotation_fields = tuple(f"rot_{axis}" for axis in range(4))
  needs_scales = args.max_gaussian_scale is not None or args.use_gaussian_covariance
  if needs_scales:
    required_shape_fields = (
      (*scale_fields, *rotation_fields)
      if args.use_gaussian_covariance
      else scale_fields
    )
    missing_shape_fields = [
      name for name in required_shape_fields if name not in vertices
    ]
    if missing_shape_fields:
      raise ValueError(
        f"Gaussian scale/covariance options require PLY fields: {missing_shape_fields}."
      )
  encoded_scales = (
    np.column_stack([vertices[name] for name in scale_fields]) if needs_scales else None
  )
  point_covariances = (
    gaussian_covariances(
      encoded_scales,
      np.column_stack([vertices[name] for name in rotation_fields]),
      scale_encoding=args.scale_encoding,
    )
    if args.use_gaussian_covariance and encoded_scales is not None
    else None
  )
  if args.known_free_points is not None and args.known_free_points.suffix != ".npy":
    raise ValueError("--known-free-points must reference a .npy array.")
  if args.semantic_labels is not None and args.semantic_labels.suffix != ".npy":
    raise ValueError("--semantic-labels must reference a .npy array.")
  if args.evaluation_points is not None and args.evaluation_points.suffix != ".npz":
    raise ValueError("--evaluation-points must reference a .npz archive.")
  known_free_points: np.ndarray | None = (
    np.load(args.known_free_points, allow_pickle=False)
    if args.known_free_points is not None
    else None
  )
  semantic_labels: np.ndarray | None = (
    np.load(args.semantic_labels, allow_pickle=False)
    if args.semantic_labels is not None
    else None
  )
  if args.point_importance is not None and args.point_importance.suffix != ".npy":
    raise ValueError("--point-importance must reference a .npy array.")
  point_importance: np.ndarray | None = (
    np.load(args.point_importance, allow_pickle=False)
    if args.point_importance is not None
    else None
  )
  if point_importance is not None and (
    point_importance.ndim != 1 or len(point_importance) != len(points)
  ):
    raise ValueError(
      "--point-importance must contain one value per original PLY vertex."
    )
  if args.semantic_priorities is not None and semantic_labels is None:
    raise ValueError("--semantic-priorities requires --semantic-labels.")
  semantic_priorities: dict[int, float] | None = None
  if args.semantic_priorities is not None:
    priority_data = json.loads(args.semantic_priorities.read_text())
    if not isinstance(priority_data, dict):
      raise ValueError("--semantic-priorities must contain a JSON object.")
    try:
      semantic_priorities = {
        int(label): float(weight) for label, weight in priority_data.items()
      }
    except (TypeError, ValueError) as exc:
      raise ValueError(
        "--semantic-priorities keys must be integers and values numeric."
      ) from exc
  result = generate_voxel_boxes(
    points,
    vertices["opacity"],
    config,
    known_free_points=known_free_points,
    gaussian_scales=encoded_scales,
    semantic_labels=semantic_labels,
  )
  known_free_point_count = (
    int(known_free_points.shape[0]) if known_free_points is not None else 0
  )
  fit_config = PrimitiveFitConfig(
    min_points=args.primitive_min_points,
    max_normalized_error=args.primitive_max_error,
    linear_eigenvalue_ratio=args.linear_eigenvalue_ratio,
    planar_eigenvalue_ratio=args.planar_eigenvalue_ratio,
    isotropic_eigenvalue_ratio=args.isotropic_eigenvalue_ratio,
    radial_eigenvalue_ratio=args.radial_eigenvalue_ratio,
    min_planar_fill=args.min_planar_fill,
    min_round_error_improvement=args.min_round_error_improvement,
    planar_slab_thickness=args.planar_slab_thickness,
    planar_free_side_max_distance=args.planar_free_side_max_distance,
    planar_free_side_min_agreement=args.planar_free_side_min_agreement,
    use_gaussian_covariance=args.use_gaussian_covariance,
  )
  primitive_result = (
    fit_collision_primitives(
      points,
      result,
      config,
      fit_config,
      point_covariances=point_covariances,
    )
    if args.primitive_mode == "auto"
    else None
  )
  horizontal_config = None
  horizontal_count_before = None
  if primitive_result is not None and args.merge_horizontal_surfaces:
    horizontal_config = HorizontalSurfaceConfig(
      level_tolerance=args.horizontal_level_tolerance,
      surface_residual=args.horizontal_surface_residual,
      slab_thickness=args.horizontal_slab_thickness,
      max_tilt_degrees=args.horizontal_max_tilt_degrees,
    )
    horizontal_count_before = len(primitive_result.primitives)
    primitive_result = consolidate_horizontal_surfaces(
      points, result, config, primitive_result, horizontal_config
    )
  primitive_count_before_budget = (
    len(primitive_result.primitives) if primitive_result is not None else None
  )
  if primitive_result is not None and args.max_primitives is not None:
    primitive_result = select_collision_primitive_budget(
      primitive_result,
      result,
      max_primitives=args.max_primitives,
      semantic_priorities=semantic_priorities,
      point_importance=point_importance,
    )
  evaluation: dict[str, float | int | None] | None = None
  evaluation_input: dict[str, object] | None = None
  if args.evaluation_points is not None:
    with np.load(args.evaluation_points, allow_pickle=False) as evaluation_arrays:
      supported_keys = {"surface_points", "free_points"}
      unexpected_keys = set(evaluation_arrays.files) - supported_keys
      if unexpected_keys:
        raise ValueError(f"Unsupported evaluation array(s): {sorted(unexpected_keys)}.")
      if not evaluation_arrays.files:
        raise ValueError("--evaluation-points archive must not be empty.")
      surface_points = (
        evaluation_arrays["surface_points"]
        if "surface_points" in evaluation_arrays
        else None
      )
      free_points = (
        evaluation_arrays["free_points"] if "free_points" in evaluation_arrays else None
      )
      evaluation = evaluate_collision_proxy(
        result,
        config,
        primitive_result=primitive_result,
        reference_surface_points=surface_points,
        free_space_points=free_points,
      )
      evaluation_input = {
        "path": str(args.evaluation_points),
        "sha256": _sha256(args.evaluation_points),
        "arrays": sorted(evaluation_arrays.files),
      }
  source_surface_evaluation = (
    evaluate_collision_proxy(
      result,
      config,
      primitive_result=primitive_result,
      reference_surface_points=points[result.filtered_source_ids],
    )
    if args.evaluate_source_surface
    else None
  )

  source_map = args.source_map or args.report.with_suffix(".source_ids.npz")
  for path in (args.output_mjcf, args.report, source_map):
    path.parent.mkdir(parents=True, exist_ok=True)
  args.output_mjcf.write_text(
    build_collision_mjcf(
      result,
      config,
      model_name=args.model_name,
      primitive_result=primitive_result,
      debug_color_by_type=args.debug_color_by_type,
    )
  )
  if primitive_result is None:
    source_arrays = {
      "filtered_source_gaussian_ids": result.filtered_source_ids,
      "box_source_id_offsets": result.box_source_id_offsets,
      "box_source_gaussian_ids": result.box_source_ids,
    }
    if result.box_semantic_labels is not None:
      source_arrays["box_semantic_labels"] = result.box_semantic_labels
    mapping_semantics = "CSR box index to original zero-based PLY vertex IDs"
    primitive_counts = {"box": len(result.boxes)}
    mapped_gaussian_count = len(result.box_source_ids)
    output_semantic_labels = result.box_semantic_labels
  else:
    source_arrays = {
      "filtered_source_gaussian_ids": result.filtered_source_ids,
      "primitive_source_id_offsets": primitive_result.primitive_source_id_offsets,
      "primitive_source_gaussian_ids": primitive_result.primitive_source_ids,
    }
    output_semantic_labels = (
      np.asarray(
        [
          result.candidate_semantic_labels[primitive.source_candidate_index]
          for primitive in primitive_result.primitives
        ],
        dtype=np.int64,
      )
      if result.candidate_semantic_labels is not None
      else None
    )
    if output_semantic_labels is not None:
      source_arrays["primitive_semantic_labels"] = output_semantic_labels
    mapping_semantics = "CSR primitive index to original zero-based PLY vertex IDs"
    primitive_counts = {
      geom_type: sum(
        primitive.geom_type == geom_type for primitive in primitive_result.primitives
      )
      for geom_type in ("box", "sphere", "cylinder", "capsule")
    }
    mapped_gaussian_count = len(primitive_result.primitive_source_ids)
  semantic_output_counts = (
    {
      str(int(label)): int(np.count_nonzero(output_semantic_labels == label))
      for label in np.unique(output_semantic_labels)
    }
    if output_semantic_labels is not None
    else None
  )
  _write_source_map(source_map, source_arrays)
  report = {
    "schema_version": 1,
    "algorithm": (
      "center_only_voxel_primitives"
      if primitive_result is not None
      else "center_only_voxel_boxes"
    ),
    "algorithm_version": (
      8 if horizontal_config is not None else 7 if primitive_result is not None else 3
    ),
    "tool_version": version("mjlab"),
    "seed": None,
    "coordinate_system": {
      "handedness": "right",
      "up_axis": "z",
      "length_unit": "meter",
      "transform_applied": False,
    },
    "input": {
      "path": str(args.input_ply),
      "sha256": _sha256(args.input_ply),
      "gaussian_count": len(points),
    },
    "known_free_input": (
      {
        "path": str(args.known_free_points),
        "sha256": _sha256(args.known_free_points),
        "point_count": known_free_point_count,
      }
      if args.known_free_points is not None
      else None
    ),
    "semantic_input": (
      {
        "path": str(args.semantic_labels),
        "sha256": _sha256(args.semantic_labels),
        "label_count": int(semantic_labels.shape[0]),
        "unique_labels": np.unique(semantic_labels).tolist(),
        "unknown_label": args.unknown_semantic_label,
      }
      if args.semantic_labels is not None and semantic_labels is not None
      else None
    ),
    "priority_input": {
      "max_primitives": args.max_primitives,
      "primitive_count_before_budget": primitive_count_before_budget,
      "semantic_priorities": semantic_priorities,
      "semantic_priorities_path": (
        str(args.semantic_priorities) if args.semantic_priorities is not None else None
      ),
      "semantic_priorities_sha256": (
        _sha256(args.semantic_priorities)
        if args.semantic_priorities is not None
        else None
      ),
      "point_importance_path": (
        str(args.point_importance) if args.point_importance is not None else None
      ),
      "point_importance_sha256": (
        _sha256(args.point_importance) if args.point_importance is not None else None
      ),
    },
    "evaluation_input": evaluation_input,
    "evaluation": evaluation,
    "source_surface_evaluation": source_surface_evaluation,
    "config": config_dict(config),
    "primitive_mode": args.primitive_mode,
    "primitive_fit_config": primitive_fit_config_dict(fit_config),
    "debug_color_by_type": args.debug_color_by_type,
    "horizontal_surfaces": (
      {
        "level_tolerance": horizontal_config.level_tolerance,
        "surface_residual": horizontal_config.surface_residual,
        "slab_thickness": horizontal_config.slab_thickness,
        "max_tilt_degrees": horizontal_config.max_tilt_degrees,
        "primitive_count_before": horizontal_count_before,
        "aligned_primitive_count": sum(
          primitive.horizontal_surface_level is not None
          for primitive in primitive_result.primitives
        )
        if primitive_result is not None
        else 0,
      }
      if horizontal_config is not None
      else None
    ),
    "bounds": {"lower": result.lower.tolist(), "upper": result.upper.tolist()},
    "representation": {
      "type": "primitives" if primitive_result is not None else "axis_aligned_boxes",
      "mesh_vertex_count": 0,
      "mesh_face_count": 0,
    },
    "statistics": {
      "filtered_gaussian_count": len(result.filtered_source_ids),
      "grid_shape": list(result.grid_shape),
      "occupied_voxel_count": result.occupied_voxel_count,
      "connected_component_count": result.connected_component_count,
      "known_free_voxel_count": result.known_free_voxel_count,
      "merged_box_count": result.merged_box_count,
      "retained_box_count": len(result.boxes),
      "discarded_box_count": result.merged_box_count - len(result.boxes),
      "primitive_counts": primitive_counts,
      "semantic_output_counts": semantic_output_counts,
      "mapped_gaussian_count": mapped_gaussian_count,
      "mapped_gaussian_fraction": (
        mapped_gaussian_count / len(result.filtered_source_ids)
      ),
      "primitive_per_occupied_voxel": (
        sum(primitive_counts.values()) / result.occupied_voxel_count
      ),
    },
    "outputs": {
      "mjcf": {"path": str(args.output_mjcf), "sha256": _sha256(args.output_mjcf)},
      "source_map": {
        "path": str(source_map),
        "sha256": _sha256(source_map),
        "semantics": mapping_semantics,
      },
    },
  }
  if primitive_result is not None:
    report["primitive_diagnostics"] = [
      {
        "index": index,
        "type": primitive.geom_type,
        "source_candidate_index": primitive.source_candidate_index,
        "source_candidate_indices": list(primitive.source_candidate_indices),
        "horizontal_surface_level": primitive.horizontal_surface_level,
        "source_gaussian_count": int(
          primitive_result.primitive_source_id_offsets[index + 1]
          - primitive_result.primitive_source_id_offsets[index]
        ),
        "semantic_label": (
          int(result.candidate_semantic_labels[primitive.source_candidate_index])
          if result.candidate_semantic_labels is not None
          else None
        ),
        "normalized_error": primitive.normalized_error,
        "baseline_box_error": primitive.baseline_box_error,
        "relative_error_improvement": primitive.relative_error_improvement,
        "rejected_round_type": primitive.rejected_round_type,
        "rejected_round_error": primitive.rejected_round_error,
        "rejected_round_error_improvement": (
          primitive.rejected_round_error_improvement
        ),
        "known_free_rejected_fit": primitive.known_free_rejected_fit,
        "free_side_anchored": primitive.free_side_anchored,
        "free_side_sign": primitive.free_side_sign,
        "center": primitive.center.tolist(),
        "quaternion_wxyz": primitive.quaternion_wxyz.tolist(),
        "size": list(primitive.size),
      }
      for index, primitive in enumerate(primitive_result.primitives)
    ]
  args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(
    f"Generated {sum(primitive_counts.values())} primitives {primitive_counts} from "
    f"{result.occupied_voxel_count} "
    f"occupied voxels; report: {args.report}"
  )


if __name__ == "__main__":
  main()
