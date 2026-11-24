"""Opt-in consolidation of finite horizontal collision patches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mjlab.asset_zoo.gaussian_collision import (
  CollisionPrimitive,
  PrimitiveFitResult,
  VoxelBoxConfig,
  VoxelBoxResult,
  _primitive_contains_points,
  _principal_axes,
  _quaternion_wxyz_to_matrix,
)


@dataclass(frozen=True, kw_only=True)
class HorizontalSurfaceConfig:
  level_tolerance: float = 0.04
  surface_residual: float = 0.06
  slab_thickness: float = 0.1
  max_tilt_degrees: float = 5.0

  def __post_init__(self) -> None:
    for value in (self.level_tolerance, self.surface_residual, self.slab_thickness):
      if not np.isfinite(value) or value <= 0:
        raise ValueError(
          "Horizontal tolerances and thickness must be finite and positive."
        )
    if not np.isfinite(self.max_tilt_degrees) or not 0 < self.max_tilt_degrees < 45:
      raise ValueError("Horizontal maximum tilt must be in (0, 45) degrees.")


def _rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
  remaining = mask.copy()
  rectangles = []
  while remaining.any():
    start_x, start_y = np.argwhere(remaining)[0]
    width = 0
    while start_y + width < remaining.shape[1] and remaining[start_x, start_y + width]:
      width += 1
    best_area = width
    end_x, end_y = start_x + 1, start_y + width
    for row in range(start_x + 1, remaining.shape[0]):
      available = 0
      while available < width and remaining[row, start_y + available]:
        available += 1
      width = min(width, available)
      if not width:
        break
      area = (row - start_x + 1) * width
      if area > best_area:
        best_area = area
        end_x, end_y = row + 1, start_y + width
    remaining[start_x:end_x, start_y:end_y] = False
    rectangles.append((int(start_x), int(end_x), int(start_y), int(end_y)))
  return rectangles


def consolidate_horizontal_surfaces(
  points: np.ndarray,
  result: VoxelBoxResult,
  voxel_config: VoxelBoxConfig,
  fitted: PrimitiveFitResult,
  config: HorizontalSurfaceConfig,
) -> PrimitiveFitResult:
  """Align nearby supported levels and retile their exact candidate-footprint union."""
  candidates: list[tuple[int, float, int | None]] = []
  source_groups = [
    fitted.primitive_source_ids[start:end]
    for start, end in zip(
      fitted.primitive_source_id_offsets[:-1],
      fitted.primitive_source_id_offsets[1:],
      strict=True,
    )
  ]
  for index, primitive in enumerate(fitted.primitives):
    source = points[source_groups[index]]
    if (
      primitive.geom_type != "box"
      or primitive.free_side_anchored
      or primitive.horizontal_surface_level is not None
      or len(source) < 24
    ):
      continue
    span = np.ptp(source, axis=0)
    if min(span[:2]) < 2 * voxel_config.voxel_size:
      continue
    _, eigenvalues, axes = _principal_axes(source)
    if eigenvalues[1] <= 1e-12 or eigenvalues[2] / eigenvalues[1] > 0.08:
      continue
    if abs(axes[2, 2]) < np.cos(np.deg2rad(config.max_tilt_degrees)):
      continue
    level = float(np.median(source[:, 2]))
    if np.quantile(np.abs(source[:, 2] - level), 0.95) > config.surface_residual:
      continue
    label = (
      int(result.candidate_semantic_labels[primitive.source_candidate_index])
      if result.candidate_semantic_labels is not None
      else None
    )
    rotation = _quaternion_wxyz_to_matrix(primitive.quaternion_wxyz)
    vertical_axis = int(np.argmax(np.abs(rotation[2])))
    top_level = float(
      primitive.center[2]
      + primitive.size[vertical_axis] / abs(rotation[2, vertical_axis])
    )
    candidates.append((index, top_level, label))
  candidates.sort(
    key=lambda entry: (entry[2] if entry[2] is not None else -1, entry[1], entry[0])
  )
  groups: list[list[tuple[int, float, int | None]]] = []
  for candidate in candidates:
    if (
      not groups
      or candidate[2] != groups[-1][0][2]
      or candidate[1] - groups[-1][0][1] > config.level_tolerance
    ):
      groups.append([])
    groups[-1].append(candidate)
  replacements: dict[int, list[tuple[CollisionPrimitive, np.ndarray]]] = {}
  removed: set[int] = set()
  for group in groups:
    if len(group) < 2:
      continue
    indices = [entry[0] for entry in group]
    source_ids = np.sort(np.concatenate([source_groups[index] for index in indices]))
    source_level = float(np.median(points[source_ids, 2]))
    level = float(np.median([entry[1] for entry in group]))
    if (
      np.quantile(np.abs(points[source_ids, 2] - source_level), 0.95)
      > config.surface_residual
    ):
      continue
    mask = np.zeros(result.grid_shape[:2], dtype=bool)
    for index in indices:
      candidate_index = fitted.primitives[index].source_candidate_index
      lower, upper = result.candidate_boxes[candidate_index]
      mask[lower[0] : upper[0], lower[1] : upper[1]] = True
    rectangles = _rectangles(mask)
    if len(rectangles) > len(indices):
      continue
    source_xy = np.floor(
      (points[source_ids, :2] - result.lower[:2]) / voxel_config.voxel_size
    ).astype(int)
    outputs = []
    rejected = False
    representative = min(
      fitted.primitives[index].source_candidate_index for index in indices
    )
    for start_x, end_x, start_y, end_y in rectangles:
      selected = (
        (source_xy[:, 0] >= start_x)
        & (source_xy[:, 0] < end_x)
        & (source_xy[:, 1] >= start_y)
        & (source_xy[:, 1] < end_y)
      )
      if not selected.any():
        continue
      primitive = CollisionPrimitive(
        geom_type="box",
        center=np.array(
          [
            result.lower[0] + (start_x + end_x) * voxel_config.voxel_size / 2,
            result.lower[1] + (start_y + end_y) * voxel_config.voxel_size / 2,
            level - config.slab_thickness / 2,
          ]
        ),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        size=(
          (end_x - start_x) * voxel_config.voxel_size / 2,
          (end_y - start_y) * voxel_config.voxel_size / 2,
          config.slab_thickness / 2,
        ),
        normalized_error=float(
          np.sqrt(np.mean((points[source_ids[selected], 2] - level) ** 2))
          / max((end_x - start_x), (end_y - start_y), 1)
          / voxel_config.voxel_size
        ),
        source_candidate_index=representative,
        horizontal_surface_level=level,
        source_candidate_indices=tuple(
          sorted(fitted.primitives[index].source_candidate_index for index in indices)
        ),
      )
      constraints = result.known_free_points
      if group[0][2] is not None and result.filtered_semantic_labels is not None:
        constraints = np.concatenate(
          [
            constraints,
            points[
              result.filtered_source_ids[result.filtered_semantic_labels != group[0][2]]
            ],
          ]
        )
      if _primitive_contains_points(primitive, constraints).any():
        rejected = True
        break
      outputs.append((primitive, source_ids[selected]))
    if rejected or sum(len(source) for _, source in outputs) != len(source_ids):
      continue
    first = min(indices)
    replacements[first] = outputs
    removed.update(indices)
  primitives = []
  mapping = []
  offsets = [0]
  for index, primitive in enumerate(fitted.primitives):
    outputs = (
      replacements.get(index, [])
      if index in removed
      else [(primitive, source_groups[index])]
    )
    for output, source in outputs:
      primitives.append(output)
      mapping.append(source)
      offsets.append(offsets[-1] + len(source))
  return PrimitiveFitResult(
    primitives=tuple(primitives),
    primitive_source_id_offsets=np.array(offsets, dtype=np.int64),
    primitive_source_ids=np.concatenate(mapping)
    if mapping
    else np.empty(0, dtype=np.int64),
  )
