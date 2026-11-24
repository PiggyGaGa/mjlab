"""Convert a triangle collision mesh into component-wise convex MuJoCo geoms."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import trimesh

from mjlab.asset_zoo.gaussian_collision_mesh import (
  SurfaceMesh,
  convex_component_hulls,
  surface_mesh_obj,
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input_mesh", type=Path)
  parser.add_argument("output_dir", type=Path)
  parser.add_argument("--model-name", default="component_convex_collision")
  parser.add_argument("--max-distance-points", type=int, default=20_000)
  parser.add_argument(
    "--decomposition",
    choices=("component-hulls", "vhacd"),
    default="component-hulls",
  )
  parser.add_argument("--max-convex-hulls", type=int, default=64)
  parser.add_argument("--vhacd-resolution", type=int, default=200_000)
  parser.add_argument("--max-vertices-per-hull", type=int, default=64)
  parser.add_argument(
    "--free-points",
    type=Path,
    help="Optional independent (N, 3) NPY points that should remain outside hulls.",
  )
  return parser.parse_args()


def _build_mjcf(mesh_files: list[Path], model_name: str) -> str:
  model = ET.Element("mujoco", model=model_name)
  ET.SubElement(model, "compiler", meshdir="meshes")
  asset = ET.SubElement(model, "asset")
  worldbody = ET.SubElement(model, "worldbody")
  for index, path in enumerate(mesh_files):
    name = f"component_{index:04d}"
    ET.SubElement(asset, "mesh", name=name, file=path.name)
    ET.SubElement(
      worldbody,
      "geom",
      name=name,
      type="mesh",
      mesh=name,
      group="3",
      contype="1",
      conaffinity="1",
      condim="3",
      friction="1.0 0.005 0.0001",
      rgba="0.95 0.55 0.10 0.3",
    )
  ET.indent(model, space="  ")
  return ET.tostring(model, encoding="unicode") + "\n"


def main() -> None:
  args = _parse_args()
  if args.max_distance_points <= 0:
    raise ValueError("--max-distance-points must be positive.")
  if (
    args.max_convex_hulls <= 0
    or args.vhacd_resolution <= 0
    or args.max_vertices_per_hull < 4
  ):
    raise ValueError("V-HACD budgets must be positive and allow at least 4 vertices.")
  loaded = trimesh.load(args.input_mesh, force="scene")
  if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
    raise ValueError("Input must contain at least one triangle geometry.")
  triangle_mesh = loaded.to_geometry()
  if not isinstance(triangle_mesh, trimesh.Trimesh):
    raise ValueError("Input scene did not resolve to a triangle mesh.")
  source_mesh = SurfaceMesh(
    np.asarray(triangle_mesh.vertices, dtype=np.float64),
    np.asarray(triangle_mesh.faces, dtype=np.int64),
  )
  source_edge_counts = np.bincount(triangle_mesh.edges_unique_inverse)
  component_diagnostics = None
  if args.decomposition == "component-hulls":
    hulls, component_diagnostics = convex_component_hulls(source_mesh)
  else:
    decomposed = triangle_mesh.convex_decomposition(
      maxConvexHulls=args.max_convex_hulls,
      resolution=args.vhacd_resolution,
      minimumVolumePercentErrorAllowed=1.0,
      maxRecursionDepth=10,
      maxNumVerticesPerCH=args.max_vertices_per_hull,
      asyncACD=False,
    )
    decomposed.sort(
      key=lambda hull: (
        *np.asarray(hull.bounds[0], dtype=np.float64).tolist(),
        *np.asarray(hull.bounds[1], dtype=np.float64).tolist(),
      )
    )
    hulls = tuple(
      SurfaceMesh(
        np.asarray(hull.vertices, dtype=np.float64),
        np.asarray(hull.faces, dtype=np.int64),
      )
      for hull in decomposed
    )
  if not hulls:
    raise ValueError("Input mesh has no connected triangle components.")

  mesh_dir = args.output_dir / "meshes"
  mesh_dir.mkdir(parents=True, exist_ok=True)
  mesh_paths: list[Path] = []
  for index, hull in enumerate(hulls):
    path = mesh_dir / f"component_{index:04d}.obj"
    path.write_text(surface_mesh_obj(hull))
    mesh_paths.append(path)
  mjcf_path = args.output_dir / "collision.xml"
  mjcf_path.write_text(_build_mjcf(mesh_paths, args.model_name))
  model = mujoco.MjModel.from_xml_path(str(mjcf_path))

  combined_hulls = trimesh.util.concatenate(
    [
      trimesh.Trimesh(vertices=hull.vertices, faces=hull.faces, process=False)
      for hull in hulls
    ]
  )
  sample_ids = np.linspace(
    0,
    len(source_mesh.vertices) - 1,
    min(args.max_distance_points, len(source_mesh.vertices)),
    dtype=np.int64,
  )
  _, hull_distances, _ = trimesh.proximity.closest_point(
    combined_hulls,
    source_mesh.vertices[sample_ids],
  )
  free_space_report = None
  if args.free_points is not None:
    free_points = np.load(args.free_points, allow_pickle=False)
    if (
      free_points.ndim != 2
      or free_points.shape[1] != 3
      or not len(free_points)
      or not np.isfinite(free_points).all()
    ):
      raise ValueError("--free-points must contain a finite non-empty (N, 3) array.")
    occupied_free_points = np.zeros(len(free_points), dtype=bool)
    for hull in hulls:
      hull_mesh = trimesh.Trimesh(
        vertices=hull.vertices,
        faces=hull.faces,
        process=False,
      )
      occupied_free_points |= hull_mesh.contains(free_points)
    free_space_report = {
      "path": str(args.free_points),
      "sha256": _sha256(args.free_points),
      "point_count": len(free_points),
      "false_occupied_count": int(occupied_free_points.sum()),
      "false_occupied_fraction": float(occupied_free_points.mean()),
      "false_occupied_indices": np.flatnonzero(occupied_free_points).tolist(),
    }
  source_volume = (
    float(abs(triangle_mesh.volume)) if triangle_mesh.is_watertight else None
  )
  hull_volumes = np.asarray(
    [
      abs(
        trimesh.Trimesh(
          vertices=hull.vertices,
          faces=hull.faces,
          process=False,
        ).volume
      )
      for hull in hulls
    ]
  )
  report = {
    "schema_version": 1,
    "conversion": args.decomposition,
    "config": {
      "max_convex_hulls": args.max_convex_hulls,
      "vhacd_resolution": args.vhacd_resolution,
      "max_vertices_per_hull": args.max_vertices_per_hull,
    },
    "collision_semantics": ("Each output piece becomes one MuJoCo convex mesh geom."),
    "limitation": (
      "Convex pieces can still fill free space; inspect volume and free-space probes "
      "before task use."
    ),
    "input": {
      "path": str(args.input_mesh),
      "sha256": _sha256(args.input_mesh),
      "vertex_count": len(source_mesh.vertices),
      "face_count": len(source_mesh.faces),
      "watertight": bool(triangle_mesh.is_watertight),
      "winding_consistent": bool(triangle_mesh.is_winding_consistent),
      "boundary_edge_count": int(np.count_nonzero(source_edge_counts == 1)),
      "nonmanifold_edge_count": int(np.count_nonzero(source_edge_counts > 2)),
    },
    "piece_count": len(hulls),
    "pieces": [
      {
        "index": index,
        **(
          asdict(component_diagnostics[index])
          if component_diagnostics is not None
          else {
            "hull_vertex_count": len(hulls[index].vertices),
            "hull_face_count": len(hulls[index].faces),
            "hull_volume": float(hull_volumes[index]),
          }
        ),
        "mesh": {
          "path": str(mesh_paths[index]),
          "sha256": _sha256(mesh_paths[index]),
        },
      }
      for index in range(len(hulls))
    ],
    "volume": {
      "source": source_volume,
      "sum_hulls": float(hull_volumes.sum()),
      "sum_hulls_to_source_ratio": (
        float(hull_volumes.sum() / source_volume)
        if source_volume is not None and source_volume > 1e-12
        else None
      ),
    },
    "source_vertex_to_hull_surface_distance": {
      "point_count": len(hull_distances),
      "mean": float(hull_distances.mean()),
      "p95": float(np.quantile(hull_distances, 0.95)),
      "max": float(hull_distances.max()),
    },
    "free_space_evaluation": free_space_report,
    "output": {
      "mjcf": {"path": str(mjcf_path), "sha256": _sha256(mjcf_path)},
      "compiled_ngeom": model.ngeom,
      "compiled_nmesh": model.nmesh,
    },
  }
  report_path = args.output_dir / "conversion_report.json"
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"Converted mesh to {len(hulls)} convex pieces at {mjcf_path}")


if __name__ == "__main__":
  main()
