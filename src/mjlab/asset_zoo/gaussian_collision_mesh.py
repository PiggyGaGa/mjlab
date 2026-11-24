"""Surface reconstruction baselines for voxelized Gaussian collision evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
from scipy import ndimage


@dataclass(frozen=True)
class SurfaceMesh:
  """A deterministic triangle mesh in world coordinates."""

  vertices: np.ndarray
  faces: np.ndarray


@dataclass(frozen=True)
class MeshDiagnostics:
  """Topology and numeric diagnostics after mesh repair."""

  vertex_count: int
  face_count: int
  degenerate_face_count: int
  duplicate_face_count: int
  boundary_edge_count: int
  nonmanifold_edge_count: int
  watertight: bool
  winding_consistent: bool


@dataclass(frozen=True)
class HeightfieldSurface:
  """A single-valued Z surface sampled on a regular XY grid."""

  heights: np.ndarray
  observed: np.ndarray
  lower_xy: np.ndarray
  cell_size: float
  ambiguous_column_count: int


@dataclass(frozen=True)
class ConvexComponentDiagnostics:
  """Source-component and convex-hull counts and volume diagnostics."""

  source_vertex_count: int
  source_face_count: int
  source_watertight: bool
  source_volume: float | None
  hull_vertex_count: int
  hull_face_count: int
  hull_volume: float
  volume_ratio: float | None


_CUBE_CORNERS = np.array(
  [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
  ],
  dtype=np.int8,
)

_TETRAHEDRA = (
  (0, 5, 1, 6),
  (0, 1, 2, 6),
  (0, 2, 3, 6),
  (0, 3, 7, 6),
  (0, 7, 4, 6),
  (0, 4, 5, 6),
)

_TETRAHEDRON_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

_MARCHING_TETRAHEDRA_TRIANGLES: tuple[tuple[int, ...], ...] = (
  (),
  (0, 2, 1),
  (0, 3, 4),
  (3, 4, 1, 1, 4, 2),
  (3, 1, 5),
  (0, 2, 5, 0, 5, 3),
  (0, 1, 5, 0, 5, 4),
  (5, 4, 2),
  (2, 4, 5),
  (4, 5, 0, 5, 1, 0),
  (3, 5, 0, 5, 2, 0),
  (5, 1, 3),
  (2, 4, 1, 1, 4, 3),
  (4, 3, 0),
  (1, 2, 0),
  (),
)


def _validate_grid(occupied: np.ndarray, voxel_size: float) -> np.ndarray:
  occupied = np.asarray(occupied, dtype=bool)
  if occupied.ndim != 3 or not occupied.any():
    raise ValueError("occupied must be a non-empty three-dimensional boolean grid.")
  if not np.isfinite(voxel_size) or voxel_size <= 0:
    raise ValueError("voxel_size must be finite and positive.")
  return occupied


def occupancy_surface_mesh(
  occupied: np.ndarray, lower: np.ndarray, voxel_size: float
) -> SurfaceMesh:
  """Extract the exact exposed faces of an occupancy grid as triangles."""
  occupied = _validate_grid(occupied, voxel_size)
  lower = np.asarray(lower, dtype=np.float64)
  if lower.shape != (3,) or not np.isfinite(lower).all():
    raise ValueError("lower must contain three finite coordinates.")

  vertices: list[np.ndarray] = []
  faces: list[tuple[int, int, int]] = []
  face_corners = {
    (0, -1): (0, 4, 7, 3),
    (0, 1): (1, 2, 6, 5),
    (1, -1): (0, 1, 5, 4),
    (1, 1): (3, 7, 6, 2),
    (2, -1): (0, 3, 2, 1),
    (2, 1): (4, 5, 6, 7),
  }
  shape = np.asarray(occupied.shape)
  for index in np.argwhere(occupied):
    for axis in range(3):
      for direction in (-1, 1):
        neighbour = index.copy()
        neighbour[axis] += direction
        exposed = bool(np.any(neighbour < 0) or np.any(neighbour >= shape))
        if not exposed:
          exposed = not bool(occupied[tuple(neighbour)])
        if not exposed:
          continue
        start = len(vertices)
        corners = face_corners[(axis, direction)]
        vertices.extend(
          lower + (index + _CUBE_CORNERS[corner]) * voxel_size for corner in corners
        )
        faces.extend(((start, start + 1, start + 2), (start, start + 2, start + 3)))
  return repair_surface_mesh(
    SurfaceMesh(np.asarray(vertices), np.asarray(faces, dtype=np.int64)),
    merge_tolerance=voxel_size * 1e-8,
  )[0]


def truncated_signed_distance(
  occupied: np.ndarray, voxel_size: float, truncation_distance: float
) -> np.ndarray:
  """Build a center-sampled TSDF from binary occupancy evidence."""
  occupied = _validate_grid(occupied, voxel_size)
  if not np.isfinite(truncation_distance) or truncation_distance <= 0:
    raise ValueError("truncation_distance must be finite and positive.")
  outside = cast(np.ndarray, ndimage.distance_transform_edt(~occupied)) * voxel_size
  inside = cast(np.ndarray, ndimage.distance_transform_edt(occupied)) * voxel_size
  signed = outside - inside
  return np.clip(signed, -truncation_distance, truncation_distance)


def _interpolate_zero(
  point_a: np.ndarray,
  point_b: np.ndarray,
  value_a: float,
  value_b: float,
) -> np.ndarray:
  fraction = value_a / (value_a - value_b)
  return point_a + fraction * (point_b - point_a)


def marching_tetrahedra_surface(
  values: np.ndarray,
  lower: np.ndarray,
  voxel_size: float,
  *,
  merge_tolerance: float | None = None,
) -> SurfaceMesh:
  """Extract a zero isosurface without an optional scikit-image dependency."""
  values = np.asarray(values, dtype=np.float64)
  lower = np.asarray(lower, dtype=np.float64)
  if values.ndim != 3 or min(values.shape) < 2 or not np.isfinite(values).all():
    raise ValueError(
      "values must be a finite 3D grid with at least two samples per axis."
    )
  if lower.shape != (3,) or not np.isfinite(lower).all():
    raise ValueError("lower must contain three finite coordinates.")
  if not np.isfinite(voxel_size) or voxel_size <= 0:
    raise ValueError("voxel_size must be finite and positive.")
  if not (np.any(values < 0) and np.any(values > 0)):
    raise ValueError("values must contain samples on both sides of zero.")

  cell_corners = np.stack(
    [
      values[:-1, :-1, :-1],
      values[1:, :-1, :-1],
      values[1:, 1:, :-1],
      values[:-1, 1:, :-1],
      values[:-1, :-1, 1:],
      values[1:, :-1, 1:],
      values[1:, 1:, 1:],
      values[:-1, 1:, 1:],
    ]
  )
  cell_min = cell_corners.min(axis=0)
  cell_max = cell_corners.max(axis=0)
  active_cells = np.argwhere((cell_min < 0) & (cell_max > 0))
  vertices: list[np.ndarray] = []
  faces: list[tuple[int, int, int]] = []
  for cell in active_cells:
    corner_indices = cell + _CUBE_CORNERS
    corner_values = values[tuple(corner_indices.T)]
    corner_points = lower + corner_indices * voxel_size
    for tetrahedron in _TETRAHEDRA:
      tetrahedron_indices = np.asarray(tetrahedron)
      tetra_values = corner_values[tetrahedron_indices]
      tetra_points = corner_points[tetrahedron_indices]
      case = sum(
        (1 << vertex_index)
        for vertex_index, value in enumerate(tetra_values)
        if value < 0
      )
      triangle_edges = _MARCHING_TETRAHEDRA_TRIANGLES[case]
      if not triangle_edges:
        continue
      intersections: dict[int, np.ndarray] = {}
      for edge_index in set(triangle_edges):
        start, end = _TETRAHEDRON_EDGES[edge_index]
        if (tetra_values[start] < 0) == (tetra_values[end] < 0):
          raise RuntimeError("Invalid marching-tetrahedra case table.")
        intersections[edge_index] = _interpolate_zero(
          tetra_points[start],
          tetra_points[end],
          float(tetra_values[start]),
          float(tetra_values[end]),
        )
      for triangle_start in range(0, len(triangle_edges), 3):
        start = len(vertices)
        vertices.extend(
          intersections[edge]
          for edge in triangle_edges[triangle_start : triangle_start + 3]
        )
        faces.append((start, start + 1, start + 2))
  if not faces:
    raise ValueError("The zero isosurface is empty.")
  tolerance = merge_tolerance or voxel_size * 1e-7
  return repair_surface_mesh(
    SurfaceMesh(np.asarray(vertices), np.asarray(faces, dtype=np.int64)),
    merge_tolerance=tolerance,
  )[0]


def _raw_mesh_diagnostics(mesh: SurfaceMesh) -> tuple[int, int]:
  triangles = mesh.vertices[mesh.faces]
  areas = np.linalg.norm(
    np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
    axis=1,
  )
  degenerate = int(np.count_nonzero(areas <= 1e-14))
  canonical_faces = np.sort(mesh.faces, axis=1)
  duplicate = len(canonical_faces) - len(np.unique(canonical_faces, axis=0))
  return degenerate, duplicate


def repair_surface_mesh(
  mesh: SurfaceMesh,
  *,
  merge_tolerance: float = 1e-8,
) -> tuple[SurfaceMesh, MeshDiagnostics]:
  """Remove invalid topology, merge vertices, and consistently orient normals."""
  vertices = np.asarray(mesh.vertices, dtype=np.float64)
  faces = np.asarray(mesh.faces, dtype=np.int64)
  if (
    vertices.ndim != 2
    or vertices.shape[1] != 3
    or faces.ndim != 2
    or faces.shape[1] != 3
    or not len(vertices)
    or not len(faces)
    or not np.isfinite(vertices).all()
    or np.any(faces < 0)
    or np.any(faces >= len(vertices))
  ):
    raise ValueError("mesh must contain finite vertices and valid triangle indices.")
  if not np.isfinite(merge_tolerance) or merge_tolerance <= 0:
    raise ValueError("merge_tolerance must be finite and positive.")
  raw = SurfaceMesh(vertices, faces)
  raw_degenerate, raw_duplicate = _raw_mesh_diagnostics(raw)
  digits = max(0, int(np.ceil(-np.log10(merge_tolerance))))
  repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
  repaired.merge_vertices(digits_vertex=digits)
  repaired.update_faces(repaired.unique_faces())
  repaired.update_faces(repaired.nondegenerate_faces())
  repaired.remove_unreferenced_vertices()
  trimesh.repair.fix_winding(repaired)
  trimesh.repair.fix_normals(repaired, multibody=True)
  edge_counts = np.bincount(repaired.edges_unique_inverse)
  result = SurfaceMesh(
    vertices=np.asarray(repaired.vertices, dtype=np.float64),
    faces=np.asarray(repaired.faces, dtype=np.int64),
  )
  diagnostics = MeshDiagnostics(
    vertex_count=len(result.vertices),
    face_count=len(result.faces),
    degenerate_face_count=raw_degenerate,
    duplicate_face_count=raw_duplicate,
    boundary_edge_count=int(np.count_nonzero(edge_counts == 1)),
    nonmanifold_edge_count=int(np.count_nonzero(edge_counts > 2)),
    watertight=bool(repaired.is_watertight),
    winding_consistent=bool(repaired.is_winding_consistent),
  )
  return result, diagnostics


def simplify_vertex_clusters(mesh: SurfaceMesh, cell_size: float) -> SurfaceMesh:
  """Deterministically simplify a mesh by clustering nearby vertices."""
  if not np.isfinite(cell_size) or cell_size <= 0:
    raise ValueError("cell_size must be finite and positive.")
  vertices = np.asarray(mesh.vertices, dtype=np.float64)
  lower = vertices.min(axis=0)
  clusters = np.floor((vertices - lower) / cell_size).astype(np.int64)
  unique_clusters, inverse = np.unique(clusters, axis=0, return_inverse=True)
  simplified_vertices = np.zeros((len(unique_clusters), 3), dtype=np.float64)
  np.add.at(simplified_vertices, inverse, vertices)
  counts = np.bincount(inverse)
  simplified_vertices /= counts[:, None]
  simplified_faces = inverse[np.asarray(mesh.faces, dtype=np.int64)]
  keep = (
    (simplified_faces[:, 0] != simplified_faces[:, 1])
    & (simplified_faces[:, 1] != simplified_faces[:, 2])
    & (simplified_faces[:, 0] != simplified_faces[:, 2])
  )
  if not keep.any():
    raise ValueError("cell_size collapsed every mesh face.")
  return repair_surface_mesh(
    SurfaceMesh(simplified_vertices, simplified_faces[keep]),
    merge_tolerance=cell_size * 1e-8,
  )[0]


def downsample_occupancy(
  occupied: np.ndarray,
  factor: int,
  *,
  minimum_fill: float = 0.125,
) -> np.ndarray:
  """Coarsen occupancy by deterministic block voting for manifold remeshing."""
  occupied = np.asarray(occupied, dtype=bool)
  if occupied.ndim != 3 or not occupied.any():
    raise ValueError("occupied must be a non-empty three-dimensional boolean grid.")
  if factor <= 1:
    raise ValueError("factor must be greater than one.")
  if not 0 < minimum_fill <= 1:
    raise ValueError("minimum_fill must be in (0, 1].")
  padding = tuple((0, (-size) % factor) for size in occupied.shape)
  padded = np.pad(occupied, padding)
  blocked = padded.reshape(
    padded.shape[0] // factor,
    factor,
    padded.shape[1] // factor,
    factor,
    padded.shape[2] // factor,
    factor,
  )
  result = blocked.mean(axis=(1, 3, 5)) >= minimum_fill
  if not result.any():
    raise ValueError("downsampling removed every occupied block.")
  return result


def heightfield_surface(
  occupied: np.ndarray,
  lower: np.ndarray,
  voxel_size: float,
  *,
  max_fill_distance: float = 0.0,
  surface_mode: Literal["highest", "lowest"] = "highest",
) -> HeightfieldSurface:
  """Extract one occupied surface per XY column with bounded hole filling."""
  occupied = _validate_grid(occupied, voxel_size)
  lower = np.asarray(lower, dtype=np.float64)
  if lower.shape != (3,) or not np.isfinite(lower).all():
    raise ValueError("lower must contain three finite coordinates.")
  if not np.isfinite(max_fill_distance) or max_fill_distance < 0:
    raise ValueError("max_fill_distance must be finite and non-negative.")
  if surface_mode not in {"highest", "lowest"}:
    raise ValueError("surface_mode must be 'highest' or 'lowest'.")
  observed = occupied.any(axis=2)
  if not observed.any():
    raise ValueError("No occupied XY columns are available for a heightfield.")
  if surface_mode == "highest":
    reversed_top = np.argmax(occupied[:, :, ::-1], axis=2)
    top_index = occupied.shape[2] - 1 - reversed_top
  else:
    run_ends = occupied & ~np.pad(occupied[:, :, 1:], ((0, 0), (0, 0), (0, 1)))
    top_index = np.argmax(run_ends, axis=2)
  heights = lower[2] + (top_index + 1) * voxel_size
  heights = heights.astype(np.float64)
  heights[~observed] = np.nan
  if max_fill_distance > 0 and not observed.all():
    distances, nearest = cast(
      tuple[np.ndarray, np.ndarray],
      ndimage.distance_transform_edt(
        ~observed,
        return_distances=True,
        return_indices=True,
      ),
    )
    fill = (~observed) & (distances * voxel_size <= max_fill_distance + 1e-12)
    heights[fill] = heights[tuple(nearest[:, fill])]
    observed = observed | fill
  occupied_transitions = np.diff(
    np.pad(occupied.astype(np.int8), ((0, 0), (0, 0), (1, 1))), axis=2
  )
  run_count = np.count_nonzero(occupied_transitions == 1, axis=2)
  return HeightfieldSurface(
    heights=heights,
    observed=observed,
    lower_xy=lower[:2].copy(),
    cell_size=voxel_size,
    ambiguous_column_count=int(np.count_nonzero(run_count > 1)),
  )


def surface_mesh_obj(mesh: SurfaceMesh) -> str:
  """Serialize a triangle mesh as deterministic Wavefront OBJ text."""
  vertices = np.asarray(mesh.vertices, dtype=np.float64)
  faces = np.asarray(mesh.faces, dtype=np.int64)
  if (
    not np.isfinite(vertices).all()
    or np.any(faces < 0)
    or np.any(faces >= len(vertices))
  ):
    raise ValueError("mesh contains invalid vertices or face indices.")
  lines = ["# mjlab deterministic collision reconstruction"]
  lines.extend(
    "v " + " ".join(f"{value:.9f}" for value in vertex) for vertex in vertices
  )
  lines.extend("f " + " ".join(str(int(index) + 1) for index in face) for face in faces)
  return "\n".join(lines) + "\n"


def convex_component_hulls(
  mesh: SurfaceMesh,
) -> tuple[tuple[SurfaceMesh, ...], tuple[ConvexComponentDiagnostics, ...]]:
  """Split a triangle mesh and replace every connected component by its hull."""
  vertices = np.asarray(mesh.vertices, dtype=np.float64)
  faces = np.asarray(mesh.faces, dtype=np.int64)
  if (
    vertices.ndim != 2
    or vertices.shape[1] != 3
    or faces.ndim != 2
    or faces.shape[1] != 3
    or not len(vertices)
    or not len(faces)
    or not np.isfinite(vertices).all()
    or np.any(faces < 0)
    or np.any(faces >= len(vertices))
  ):
    raise ValueError("mesh must contain finite vertices and valid triangle indices.")
  triangle_mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=False,
  )
  components = list(triangle_mesh.split(only_watertight=False))
  components.sort(
    key=lambda component: (
      *np.asarray(component.bounds[0], dtype=np.float64).tolist(),
      *np.asarray(component.bounds[1], dtype=np.float64).tolist(),
      len(component.vertices),
      len(component.faces),
    )
  )
  hulls: list[SurfaceMesh] = []
  diagnostics: list[ConvexComponentDiagnostics] = []
  for component in components:
    hull = component.convex_hull
    hull_mesh, hull_diagnostics = repair_surface_mesh(
      SurfaceMesh(
        np.asarray(hull.vertices, dtype=np.float64),
        np.asarray(hull.faces, dtype=np.int64),
      )
    )
    hull_triangle_mesh = trimesh.Trimesh(
      vertices=hull_mesh.vertices,
      faces=hull_mesh.faces,
      process=False,
    )
    source_watertight = bool(component.is_watertight)
    source_volume = float(abs(component.volume)) if source_watertight else None
    hull_volume = float(abs(hull_triangle_mesh.volume))
    volume_ratio = (
      hull_volume / source_volume
      if source_volume is not None and source_volume > 1e-12
      else None
    )
    hulls.append(hull_mesh)
    diagnostics.append(
      ConvexComponentDiagnostics(
        source_vertex_count=len(component.vertices),
        source_face_count=len(component.faces),
        source_watertight=source_watertight,
        source_volume=source_volume,
        hull_vertex_count=hull_diagnostics.vertex_count,
        hull_face_count=hull_diagnostics.face_count,
        hull_volume=hull_volume,
        volume_ratio=volume_ratio,
      )
    )
  return tuple(hulls), tuple(diagnostics)


def build_heightfield_mjcf(
  surface: HeightfieldSurface,
  *,
  model_name: str,
  base_depth: float = 0.1,
) -> str:
  """Serialize a complete height grid as a self-contained collision-only MJCF."""
  heights = np.asarray(surface.heights, dtype=np.float64)
  observed = np.asarray(surface.observed, dtype=bool)
  if (
    heights.ndim != 2
    or min(heights.shape) < 2
    or observed.shape != heights.shape
    or not observed.all()
    or not np.isfinite(heights).all()
  ):
    raise ValueError("Heightfield export requires a finite, fully observed 2D grid.")
  if not np.isfinite(base_depth) or base_depth <= 0:
    raise ValueError("base_depth must be finite and positive.")
  minimum = float(heights.min())
  height_range = float(heights.max() - minimum)
  vertical_scale = max(height_range, surface.cell_size * 1e-6)
  normalized = (heights - minimum) / vertical_scale
  x_half = (heights.shape[0] - 1) * surface.cell_size / 2
  y_half = (heights.shape[1] - 1) * surface.cell_size / 2
  center_xy = (
    surface.lower_xy
    + np.array([heights.shape[0], heights.shape[1]], dtype=np.float64)
    * surface.cell_size
    / 2
  )

  model = ET.Element("mujoco", model=model_name)
  asset = ET.SubElement(model, "asset")
  ET.SubElement(
    asset,
    "hfield",
    name="surface",
    nrow=str(heights.shape[1]),
    ncol=str(heights.shape[0]),
    size=f"{x_half:.9f} {y_half:.9f} {vertical_scale:.9f} {base_depth:.9f}",
    elevation=" ".join(f"{value:.9f}" for value in normalized.T.ravel()),
  )
  worldbody = ET.SubElement(model, "worldbody")
  ET.SubElement(
    worldbody,
    "geom",
    name="heightfield",
    type="hfield",
    hfield="surface",
    pos=f"{center_xy[0]:.9f} {center_xy[1]:.9f} {minimum:.9f}",
    group="3",
    contype="1",
    conaffinity="1",
    condim="3",
    friction="1.0 0.005 0.0001",
    rgba="0.15 0.75 0.25 0.35",
  )
  ET.indent(model, space="  ")
  return ET.tostring(model, encoding="unicode") + "\n"
