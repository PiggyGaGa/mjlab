"""Compare primitive, occupancy-mesh, TSDF, and heightfield collision proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import trimesh

from mjlab.asset_zoo.gaussian_collision import (
  PrimitiveFitConfig,
  PrimitiveFitResult,
  VoxelBoxConfig,
  build_collision_mjcf,
  evaluate_collision_proxy,
  fit_collision_primitives,
  generate_voxel_boxes,
)
from mjlab.asset_zoo.gaussian_collision_mesh import (
  SurfaceMesh,
  build_heightfield_mjcf,
  downsample_occupancy,
  heightfield_surface,
  marching_tetrahedra_surface,
  occupancy_surface_mesh,
  repair_surface_mesh,
  surface_mesh_obj,
  truncated_signed_distance,
)
from mjlab.scene.gaussian_splat import _read_ply_vertices


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _surface_distances(mesh: SurfaceMesh, points: np.ndarray) -> dict[str, float]:
  triangle_mesh = trimesh.Trimesh(
    vertices=mesh.vertices,
    faces=mesh.faces,
    process=False,
  )
  _, distances, _ = trimesh.proximity.closest_point(triangle_mesh, points)
  return {
    "mean": float(distances.mean()),
    "p95": float(np.quantile(distances, 0.95)),
    "max": float(distances.max()),
  }


def _mesh_report(mesh: SurfaceMesh, points: np.ndarray) -> dict[str, object]:
  _, diagnostics = repair_surface_mesh(mesh)
  return {
    "vertex_count": diagnostics.vertex_count,
    "face_count": diagnostics.face_count,
    "degenerate_face_count": diagnostics.degenerate_face_count,
    "duplicate_face_count": diagnostics.duplicate_face_count,
    "boundary_edge_count": diagnostics.boundary_edge_count,
    "nonmanifold_edge_count": diagnostics.nonmanifold_edge_count,
    "watertight": diagnostics.watertight,
    "winding_consistent": diagnostics.winding_consistent,
    "source_surface_distance": _surface_distances(mesh, points),
  }


def _residual_primitives(
  primitive_result: PrimitiveFitResult,
  points: np.ndarray,
  heightfield_heights: np.ndarray,
  heightfield_lower_xy: np.ndarray,
  cell_size: float,
) -> PrimitiveFitResult:
  retained_primitives = []
  retained_source_ids: list[np.ndarray] = []
  offsets = [0]
  shape = np.asarray(heightfield_heights.shape)
  for index, primitive in enumerate(primitive_result.primitives):
    start = primitive_result.primitive_source_id_offsets[index]
    end = primitive_result.primitive_source_id_offsets[index + 1]
    source_ids = primitive_result.primitive_source_ids[start:end]
    source_points = points[source_ids]
    xy_indices = np.floor(
      (source_points[:, :2] - heightfield_lower_xy) / cell_size
    ).astype(np.int64)
    xy_indices = np.clip(xy_indices, 0, shape - 1)
    ground_heights = heightfield_heights[tuple(xy_indices.T)]
    if not np.any(source_points[:, 2] > ground_heights + cell_size / 2):
      continue
    retained_primitives.append(primitive)
    retained_source_ids.append(source_ids)
    offsets.append(offsets[-1] + len(source_ids))
  return PrimitiveFitResult(
    primitives=tuple(retained_primitives),
    primitive_source_id_offsets=np.asarray(offsets, dtype=np.int64),
    primitive_source_ids=(
      np.concatenate(retained_source_ids)
      if retained_source_ids
      else np.empty(0, dtype=np.int64)
    ),
  )


def _mixed_heightfield_mjcf(
  heightfield_xml: str,
  primitive_xml: str,
) -> str:
  model = ET.fromstring(heightfield_xml)
  worldbody = model.find("worldbody")
  primitive_worldbody = ET.fromstring(primitive_xml).find("worldbody")
  if worldbody is None or primitive_worldbody is None:
    raise ValueError("Generated MJCF is missing worldbody.")
  for geom in primitive_worldbody.findall("geom"):
    worldbody.append(geom)
  ET.indent(model, space="  ")
  return ET.tostring(model, encoding="unicode") + "\n"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input_ply", type=Path)
  parser.add_argument("output_dir", type=Path)
  parser.add_argument(
    "--bounds", type=float, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1")
  )
  parser.add_argument("--voxel-size", type=float, default=0.1)
  parser.add_argument("--opacity-threshold", type=float, default=0.5)
  parser.add_argument("--tsdf-truncation", type=float, default=0.3)
  parser.add_argument("--simplify-cell-size", type=float, default=0.2)
  parser.add_argument("--heightfield-max-fill-distance", type=float, default=0.2)
  parser.add_argument("--max-evaluation-points", type=int, default=20_000)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  vertices = _read_ply_vertices(args.input_ply)
  missing = [name for name in ("x", "y", "z", "opacity") if name not in vertices]
  if missing:
    raise ValueError(f"3DGS PLY is missing required fields: {missing}.")
  points = np.column_stack([vertices[axis] for axis in ("x", "y", "z")])
  opacity = vertices["opacity"]
  if args.bounds is not None:
    lower_bound = np.asarray(args.bounds[:3])
    upper_bound = np.asarray(args.bounds[3:])
    if np.any(lower_bound >= upper_bound):
      raise ValueError("--bounds lower coordinates must be below upper coordinates.")
    bounded = ((points >= lower_bound) & (points <= upper_bound)).all(axis=1)
    points = points[bounded]
    opacity = opacity[bounded]
  if not len(points):
    raise ValueError("No Gaussian centers remain inside the requested bounds.")

  config = VoxelBoxConfig(
    voxel_size=args.voxel_size,
    opacity_threshold=args.opacity_threshold,
    lower_quantile=0,
    upper_quantile=1,
    min_points_per_voxel=1,
    min_component_voxels=1,
    min_box_volume=0,
    max_planar_merge_residual=args.voxel_size * 0.4,
  )
  timings: dict[str, float] = {}
  started = time.perf_counter()
  result = generate_voxel_boxes(points, opacity, config)
  primitives = fit_collision_primitives(points, result, config, PrimitiveFitConfig())
  timings["voxel_and_primitive_seconds"] = time.perf_counter() - started
  occupied = np.zeros(result.grid_shape, dtype=bool)
  occupied[tuple(result.occupied_voxel_indices.T)] = True
  evaluation_ids = np.linspace(
    0,
    len(result.filtered_source_ids) - 1,
    min(args.max_evaluation_points, len(result.filtered_source_ids)),
    dtype=np.int64,
  )
  evaluation_points = points[result.filtered_source_ids[evaluation_ids]]

  started = time.perf_counter()
  occupancy_mesh = occupancy_surface_mesh(occupied, result.lower, config.voxel_size)
  timings["occupancy_mesh_seconds"] = time.perf_counter() - started
  started = time.perf_counter()
  tsdf = truncated_signed_distance(
    occupied,
    config.voxel_size,
    args.tsdf_truncation,
  )
  tsdf_mesh = marching_tetrahedra_surface(
    tsdf,
    result.lower + config.voxel_size / 2,
    config.voxel_size,
  )
  timings["tsdf_mesh_seconds"] = time.perf_counter() - started
  simplification_factor = max(
    2,
    int(round(args.simplify_cell_size / config.voxel_size)),
  )
  started = time.perf_counter()
  simplified_occupancy = downsample_occupancy(occupied, simplification_factor)
  simplified_voxel_size = config.voxel_size * simplification_factor
  simplified_occupancy = np.pad(simplified_occupancy, 1)
  simplified_tsdf = truncated_signed_distance(
    simplified_occupancy,
    simplified_voxel_size,
    max(args.tsdf_truncation, simplified_voxel_size),
  )
  simplified_mesh = marching_tetrahedra_surface(
    simplified_tsdf,
    result.lower - simplified_voxel_size / 2,
    simplified_voxel_size,
  )
  timings["mesh_simplification_seconds"] = time.perf_counter() - started
  started = time.perf_counter()
  heightfield = heightfield_surface(
    occupied,
    result.lower,
    config.voxel_size,
    max_fill_distance=args.heightfield_max_fill_distance,
    surface_mode="lowest",
  )
  timings["heightfield_seconds"] = time.perf_counter() - started

  args.output_dir.mkdir(parents=True, exist_ok=True)
  occupancy_path = args.output_dir / "occupancy_surface.obj"
  tsdf_path = args.output_dir / "tsdf_surface.obj"
  simplified_path = args.output_dir / "tsdf_surface_simplified.obj"
  occupancy_path.write_text(surface_mesh_obj(occupancy_mesh))
  tsdf_path.write_text(surface_mesh_obj(tsdf_mesh))
  simplified_path.write_text(surface_mesh_obj(simplified_mesh))
  primitive_path = args.output_dir / "primitives.xml"
  primitive_path.write_text(
    build_collision_mjcf(
      result,
      config,
      model_name="gaussian_primitives",
      primitive_result=primitives,
    )
  )

  heightfield_path: Path | None = None
  heightfield_compile: dict[str, int] | None = None
  mixed_path: Path | None = None
  mixed_compile: dict[str, int] | None = None
  residual_primitive_count = 0
  if heightfield.observed.all():
    complete_heightfield_path = args.output_dir / "heightfield.xml"
    heightfield_xml = build_heightfield_mjcf(
      heightfield,
      model_name="gaussian_heightfield",
    )
    complete_heightfield_path.write_text(heightfield_xml)
    heightfield_path = complete_heightfield_path
    model = mujoco.MjModel.from_xml_path(str(complete_heightfield_path))
    heightfield_compile = {"ngeom": model.ngeom, "nhfield": model.nhfield}
    residual = _residual_primitives(
      primitives,
      points,
      heightfield.heights,
      heightfield.lower_xy,
      heightfield.cell_size,
    )
    residual_primitive_count = len(residual.primitives)
    primitive_xml = build_collision_mjcf(
      result,
      config,
      model_name="gaussian_heightfield_mixed",
      primitive_result=residual,
    )
    mixed_xml = _mixed_heightfield_mjcf(heightfield_xml, primitive_xml)
    complete_mixed_path = args.output_dir / "heightfield_mixed.xml"
    complete_mixed_path.write_text(mixed_xml)
    mixed_path = complete_mixed_path
    mixed_model = mujoco.MjModel.from_xml_path(str(complete_mixed_path))
    mixed_compile = {
      "ngeom": mixed_model.ngeom,
      "nhfield": mixed_model.nhfield,
    }

  primitive_metrics = evaluate_collision_proxy(
    result,
    config,
    primitive_result=primitives,
    reference_surface_points=evaluation_points,
  )
  observed_heights = heightfield.heights[heightfield.observed]
  report = {
    "schema_version": 1,
    "input": {
      "path": str(args.input_ply),
      "sha256": _sha256(args.input_ply),
      "bounded_gaussian_count": len(points),
    },
    "config": {
      "bounds": args.bounds,
      "voxel_size": args.voxel_size,
      "opacity_threshold": args.opacity_threshold,
      "tsdf_truncation": args.tsdf_truncation,
      "simplify_cell_size": args.simplify_cell_size,
      "heightfield_max_fill_distance": args.heightfield_max_fill_distance,
      "evaluation_point_count": len(evaluation_points),
    },
    "timings": timings,
    "representations": {
      "primitives": {
        "count": len(primitives.primitives),
        "source_surface_evaluation": primitive_metrics,
        "path": str(primitive_path),
        "sha256": _sha256(primitive_path),
      },
      "occupancy_mesh": {
        **_mesh_report(occupancy_mesh, evaluation_points),
        "path": str(occupancy_path),
        "sha256": _sha256(occupancy_path),
      },
      "tsdf_mesh": {
        **_mesh_report(tsdf_mesh, evaluation_points),
        "path": str(tsdf_path),
        "sha256": _sha256(tsdf_path),
      },
      "simplified_tsdf_mesh": {
        **_mesh_report(simplified_mesh, evaluation_points),
        "path": str(simplified_path),
        "sha256": _sha256(simplified_path),
        "voxel_size": simplified_voxel_size,
      },
      "heightfield": {
        "shape": list(heightfield.heights.shape),
        "observed_fraction": float(heightfield.observed.mean()),
        "ambiguous_column_count": heightfield.ambiguous_column_count,
        "minimum_height": float(observed_heights.min()),
        "maximum_height": float(observed_heights.max()),
        "path": str(heightfield_path) if heightfield_path is not None else None,
        "compile": heightfield_compile,
        "limitation": "single-valued 2.5D surface; ambiguous columns lose overhangs",
      },
      "heightfield_mixed": {
        "residual_primitive_count": residual_primitive_count,
        "path": str(mixed_path) if mixed_path is not None else None,
        "compile": mixed_compile,
        "semantics": "lowest occupied run as terrain plus above-surface primitives",
      },
      "mesh_collision_limitation": (
        "MuJoCo user mesh collision uses its convex hull; exported concave OBJ files "
        "are geometry baselines, not valid single-geom room collisions."
      ),
    },
  }
  report_path = args.output_dir / "representation_report.json"
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"Wrote representation comparison to {report_path}")


if __name__ == "__main__":
  main()
