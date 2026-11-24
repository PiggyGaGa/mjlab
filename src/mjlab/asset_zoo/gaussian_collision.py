"""Scene-independent center-only collision generation for 3DGS assets."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Literal, cast
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


@dataclass(frozen=True, kw_only=True)
class VoxelBoxConfig:
  """Parameters for the deterministic center-only voxel/box baseline."""

  voxel_size: float = 0.1
  opacity_threshold: float = 0.5
  opacity_encoding: Literal["linear", "logit"] = "logit"
  scale_encoding: Literal["linear", "log"] = "log"
  max_gaussian_scale: float | None = None
  lower_quantile: float = 0.001
  upper_quantile: float = 0.999
  min_points_per_voxel: int = 3
  min_component_voxels: int = 8
  min_box_fill: float = 0.2
  max_surface_expansion: float = 0.4
  min_box_volume: float = 0.1
  max_grid_voxels: int = 50_000_000
  floor_z: float | None = None
  max_planar_merge_residual: float | None = None
  planar_merge_eigenvalue_ratio: float = 0.08
  unknown_semantic_label: int = -1

  def __post_init__(self) -> None:
    if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
      raise ValueError("voxel_size must be finite and positive.")
    if not 0 <= self.opacity_threshold <= 1:
      raise ValueError("opacity_threshold must be between 0 and 1.")
    if self.opacity_encoding not in {"linear", "logit"}:
      raise ValueError("opacity_encoding must be 'linear' or 'logit'.")
    if self.scale_encoding not in {"linear", "log"}:
      raise ValueError("scale_encoding must be 'linear' or 'log'.")
    if self.max_gaussian_scale is not None and (
      not np.isfinite(self.max_gaussian_scale) or self.max_gaussian_scale <= 0
    ):
      raise ValueError("max_gaussian_scale must be finite and positive when set.")
    if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
      raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1.")
    if self.min_points_per_voxel <= 0 or self.min_component_voxels <= 0:
      raise ValueError("voxel and component minimums must be positive.")
    if not 0 < self.min_box_fill <= 1:
      raise ValueError("min_box_fill must be in (0, 1].")
    if self.max_surface_expansion < 0 or not np.isfinite(self.max_surface_expansion):
      raise ValueError("max_surface_expansion must be finite and non-negative.")
    if self.min_box_volume < 0 or not np.isfinite(self.min_box_volume):
      raise ValueError("min_box_volume must be finite and non-negative.")
    if self.max_grid_voxels <= 0:
      raise ValueError("max_grid_voxels must be positive.")
    if self.max_planar_merge_residual is not None and (
      not np.isfinite(self.max_planar_merge_residual)
      or self.max_planar_merge_residual <= 0
    ):
      raise ValueError(
        "max_planar_merge_residual must be finite and positive when set."
      )
    if not 0 < self.planar_merge_eigenvalue_ratio <= 1:
      raise ValueError("planar_merge_eigenvalue_ratio must be in (0, 1].")
    if self.floor_z is not None and not np.isfinite(self.floor_z):
      raise ValueError("floor_z must be finite when specified.")


@dataclass(frozen=True)
class VoxelBoxResult:
  """Generated grid, boxes, and provenance mapping."""

  lower: np.ndarray
  upper: np.ndarray
  grid_shape: tuple[int, int, int]
  occupied_voxel_indices: np.ndarray
  candidate_boxes: tuple[tuple[np.ndarray, np.ndarray], ...]
  candidate_source_id_offsets: np.ndarray
  candidate_source_ids: np.ndarray
  candidate_semantic_labels: np.ndarray | None
  boxes: tuple[tuple[np.ndarray, np.ndarray], ...]
  filtered_source_ids: np.ndarray
  filtered_semantic_labels: np.ndarray | None
  known_free_points: np.ndarray
  box_source_id_offsets: np.ndarray
  box_source_ids: np.ndarray
  box_semantic_labels: np.ndarray | None
  occupied_voxel_count: int
  connected_component_count: int
  known_free_voxel_count: int
  merged_box_count: int


PrimitiveType = Literal["box", "sphere", "cylinder", "capsule"]

_DEBUG_RGBA_BY_TYPE: dict[PrimitiveType, str] = {
  "box": "0.15 0.75 0.25 0.18",
  "sphere": "0.95 0.15 0.85 0.85",
  "cylinder": "1.00 0.35 0.05 0.90",
  "capsule": "0.05 0.65 1.00 0.90",
}


@dataclass(frozen=True, kw_only=True)
class PrimitiveFitConfig:
  """Shape classification and fit thresholds for automatic primitives."""

  min_points: int = 24
  max_normalized_error: float = 0.12
  linear_eigenvalue_ratio: float = 0.35
  planar_eigenvalue_ratio: float = 0.08
  isotropic_eigenvalue_ratio: float = 0.45
  radial_eigenvalue_ratio: float = 0.55
  min_planar_fill: float = 0.2
  min_round_error_improvement: float = 0.3
  planar_slab_thickness: float | None = None
  planar_free_side_max_distance: float = 0.5
  planar_free_side_min_agreement: float = 0.9
  use_gaussian_covariance: bool = False

  def __post_init__(self) -> None:
    if self.min_points < 4:
      raise ValueError("min_points must be at least 4.")
    for name in (
      "max_normalized_error",
      "linear_eigenvalue_ratio",
      "planar_eigenvalue_ratio",
      "isotropic_eigenvalue_ratio",
      "radial_eigenvalue_ratio",
      "min_planar_fill",
    ):
      value = getattr(self, name)
      if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1].")
    if not 0 <= self.min_round_error_improvement <= 1:
      raise ValueError("min_round_error_improvement must be in [0, 1].")
    if self.planar_slab_thickness is not None and (
      not np.isfinite(self.planar_slab_thickness) or self.planar_slab_thickness <= 0
    ):
      raise ValueError("planar_slab_thickness must be finite and positive when set.")
    if (
      not np.isfinite(self.planar_free_side_max_distance)
      or self.planar_free_side_max_distance <= 0
    ):
      raise ValueError("planar_free_side_max_distance must be finite and positive.")
    if not 0.5 < self.planar_free_side_min_agreement <= 1:
      raise ValueError("planar_free_side_min_agreement must be in (0.5, 1].")


@dataclass(frozen=True)
class CollisionPrimitive:
  """One fitted MuJoCo primitive and its diagnostic fit error."""

  geom_type: PrimitiveType
  center: np.ndarray
  quaternion_wxyz: np.ndarray
  size: tuple[float, ...]
  normalized_error: float
  source_candidate_index: int
  baseline_box_error: float | None = None
  relative_error_improvement: float | None = None
  rejected_round_type: Literal["sphere", "cylinder", "capsule"] | None = None
  rejected_round_error: float | None = None
  rejected_round_error_improvement: float | None = None
  known_free_rejected_fit: bool = False
  free_side_anchored: bool = False
  free_side_sign: int | None = None
  horizontal_surface_level: float | None = None
  source_candidate_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class PrimitiveFitResult:
  """Fitted primitives and CSR mapping to original PLY vertex IDs."""

  primitives: tuple[CollisionPrimitive, ...]
  primitive_source_id_offsets: np.ndarray
  primitive_source_ids: np.ndarray


def select_collision_primitive_budget(
  primitive_result: PrimitiveFitResult,
  result: VoxelBoxResult,
  *,
  max_primitives: int,
  semantic_priorities: dict[int, float] | None = None,
  point_importance: np.ndarray | None = None,
) -> PrimitiveFitResult:
  """Select a deterministic primitive subset under a task-aware geom budget."""
  if max_primitives <= 0:
    raise ValueError("max_primitives must be positive.")
  priorities = semantic_priorities or {}
  if any(not np.isfinite(value) or value <= 0 for value in priorities.values()):
    raise ValueError("semantic priority weights must be finite and positive.")
  if point_importance is not None:
    point_importance = np.asarray(point_importance, dtype=np.float64)
    source_count = (
      int(primitive_result.primitive_source_ids.max()) + 1
      if len(primitive_result.primitive_source_ids)
      else 0
    )
    if (
      point_importance.ndim != 1
      or len(point_importance) < source_count
      or not np.isfinite(point_importance).all()
      or np.any(point_importance < 0)
    ):
      raise ValueError(
        "point_importance must cover every source ID with finite non-negative values."
      )
  if len(primitive_result.primitives) <= max_primitives:
    return primitive_result

  ranked: list[tuple[float, int, int]] = []
  for index, primitive in enumerate(primitive_result.primitives):
    start = primitive_result.primitive_source_id_offsets[index]
    end = primitive_result.primitive_source_id_offsets[index + 1]
    source_ids = primitive_result.primitive_source_ids[start:end]
    semantic_label: int | None = None
    if result.candidate_semantic_labels is not None:
      semantic_label = int(
        result.candidate_semantic_labels[primitive.source_candidate_index]
      )
    elif result.box_semantic_labels is not None and index < len(
      result.box_semantic_labels
    ):
      semantic_label = int(result.box_semantic_labels[index])
    semantic_weight = (
      priorities.get(semantic_label, 1.0) if semantic_label is not None else 1.0
    )
    importance_weight = (
      float(point_importance[source_ids].mean())
      if point_importance is not None and len(source_ids)
      else 1.0
    )
    score = semantic_weight * importance_weight * max(len(source_ids), 1)
    ranked.append((score, len(source_ids), index))
  retained = {
    index
    for _, _, index in sorted(
      ranked,
      key=lambda item: (-item[0], -item[1], item[2]),
    )[:max_primitives]
  }
  primitives = []
  source_groups: list[np.ndarray] = []
  offsets = [0]
  for index, primitive in enumerate(primitive_result.primitives):
    if index not in retained:
      continue
    start = primitive_result.primitive_source_id_offsets[index]
    end = primitive_result.primitive_source_id_offsets[index + 1]
    source_ids = primitive_result.primitive_source_ids[start:end]
    primitives.append(primitive)
    source_groups.append(source_ids)
    offsets.append(offsets[-1] + len(source_ids))
  return PrimitiveFitResult(
    primitives=tuple(primitives),
    primitive_source_id_offsets=np.asarray(offsets, dtype=np.int64),
    primitive_source_ids=np.concatenate(source_groups),
  )


def _activated_opacity(
  values: np.ndarray, encoding: Literal["linear", "logit"]
) -> np.ndarray:
  if encoding == "linear":
    return values
  positive = values >= 0
  result = np.empty_like(values, dtype=np.float64)
  result[positive] = 1 / (1 + np.exp(-values[positive]))
  exp_values = np.exp(values[~positive])
  result[~positive] = exp_values / (1 + exp_values)
  return result


def _activated_scales(
  values: np.ndarray, encoding: Literal["linear", "log"]
) -> np.ndarray:
  if encoding == "linear":
    return values
  with np.errstate(over="ignore", invalid="ignore"):
    return np.exp(values)


def gaussian_covariances(
  encoded_scales: np.ndarray,
  rotations_wxyz: np.ndarray,
  *,
  scale_encoding: Literal["linear", "log"] = "log",
) -> np.ndarray:
  """Convert Graphdeco scales and quaternions to world-space covariances."""
  encoded_scales = np.asarray(encoded_scales, dtype=np.float64)
  rotations_wxyz = np.asarray(rotations_wxyz, dtype=np.float64)
  if encoded_scales.ndim != 2 or encoded_scales.shape[1] != 3:
    raise ValueError("encoded_scales must have shape (N, 3).")
  if rotations_wxyz.shape != (len(encoded_scales), 4):
    raise ValueError("rotations_wxyz must have shape (N, 4).")
  scales = _activated_scales(encoded_scales, scale_encoding)
  norms = np.linalg.norm(rotations_wxyz, axis=1)
  if (
    not np.isfinite(scales).all()
    or np.any(scales <= 0)
    or not np.isfinite(rotations_wxyz).all()
    or np.any(norms <= 1e-12)
  ):
    raise ValueError("Gaussian scales and rotations must be finite and non-degenerate.")
  quaternion = rotations_wxyz / norms[:, None]
  w, x, y, z = quaternion.T
  rotation = np.empty((len(quaternion), 3, 3), dtype=np.float64)
  rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
  rotation[:, 0, 1] = 2 * (x * y - z * w)
  rotation[:, 0, 2] = 2 * (x * z + y * w)
  rotation[:, 1, 0] = 2 * (x * y + z * w)
  rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
  rotation[:, 1, 2] = 2 * (y * z - x * w)
  rotation[:, 2, 0] = 2 * (x * z - y * w)
  rotation[:, 2, 1] = 2 * (y * z + x * w)
  rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
  scaled_rotation = rotation * scales[:, None, :]
  return scaled_rotation @ scaled_rotation.transpose(0, 2, 1)


def merge_surface_boxes(
  occupied: np.ndarray,
  config: VoxelBoxConfig,
  known_free: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
  """Partition occupied voxels into bounded axis-aligned boxes."""
  if known_free is None:
    known_free = np.zeros_like(occupied)
  else:
    known_free = np.asarray(known_free, dtype=bool)
    if known_free.shape != occupied.shape:
      raise ValueError("known_free must have the same shape as occupied.")
    if np.any(known_free & occupied):
      raise ValueError("known_free and occupied voxels must not overlap.")
  distance = (
    cast(np.ndarray, ndimage.distance_transform_edt(~occupied)) * config.voxel_size
  )
  boxes: list[tuple[np.ndarray, np.ndarray]] = []

  def planar_residual_is_bounded(indices: np.ndarray) -> bool:
    threshold = config.max_planar_merge_residual
    if threshold is None or len(indices) < 3:
      return True
    centered = indices - indices.mean(axis=0)
    covariance = np.cov(centered, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[1] <= 1e-12:
      return True
    if eigenvalues[0] / eigenvalues[1] > config.planar_merge_eigenvalue_ratio:
      return True
    residual = np.abs(centered @ eigenvectors[:, 0]) * config.voxel_size
    return bool(residual.max() <= threshold + 1e-12)

  def partition(indices: np.ndarray) -> None:
    if not len(indices):
      return
    lower = indices.min(axis=0)
    upper = indices.max(axis=0) + 1
    region = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper, strict=True))
    volume = int(np.prod(upper - lower))
    if (
      len(indices) / volume >= config.min_box_fill
      and distance[region].max() <= config.max_surface_expansion + 1e-9
      and not known_free[region].any()
      and planar_residual_is_bounded(indices)
    ):
      boxes.append((lower, upper))
      return

    best_score: tuple[int, int, int, int] | None = None
    best_axis = best_cut = 0
    for axis in range(3):
      for cut in range(int(lower[axis]) + 1, int(upper[axis])):
        left = indices[indices[:, axis] < cut]
        right = indices[indices[:, axis] >= cut]
        if not len(left) or not len(right):
          continue
        left_lower = left.min(axis=0)
        left_upper = left.max(axis=0) + 1
        right_lower = right.min(axis=0)
        right_upper = right.max(axis=0) + 1
        left_region = tuple(
          slice(int(a), int(b)) for a, b in zip(left_lower, left_upper, strict=True)
        )
        right_region = tuple(
          slice(int(a), int(b)) for a, b in zip(right_lower, right_upper, strict=True)
        )
        left_free = int(known_free[left_region].sum())
        right_free = int(known_free[right_region].sum())
        child_volume = int(
          np.prod(left_upper - left_lower) + np.prod(right_upper - right_lower)
        )
        score = (
          int(left_free > 0) + int(right_free > 0),
          left_free + right_free,
          child_volume,
          abs(len(left) - len(right)),
        )
        if best_score is None or score < best_score:
          best_score = score
          best_axis, best_cut = axis, cut
    if best_score is None:
      boxes.append((lower, upper))
      return
    partition(indices[indices[:, best_axis] < best_cut])
    partition(indices[indices[:, best_axis] >= best_cut])

  partition(np.argwhere(occupied))
  return boxes


def generate_voxel_boxes(
  points: np.ndarray,
  opacity: np.ndarray,
  config: VoxelBoxConfig,
  *,
  known_free_points: np.ndarray | None = None,
  gaussian_scales: np.ndarray | None = None,
  semantic_labels: np.ndarray | None = None,
) -> VoxelBoxResult:
  """Generate boxes and source-point mappings from Gaussian centers."""
  points = np.asarray(points, dtype=np.float64)
  opacity = np.asarray(opacity, dtype=np.float64)
  if points.ndim != 2 or points.shape[1] != 3:
    raise ValueError("points must have shape (N, 3).")
  if opacity.shape != (len(points),):
    raise ValueError("opacity must have shape (N,).")
  if gaussian_scales is None:
    if config.max_gaussian_scale is not None:
      raise ValueError("max_gaussian_scale requires gaussian_scales.")
  else:
    gaussian_scales = np.asarray(gaussian_scales, dtype=np.float64)
    if gaussian_scales.shape != (len(points), 3):
      raise ValueError("gaussian_scales must have shape (N, 3).")
  if semantic_labels is not None:
    semantic_labels = np.asarray(semantic_labels)
    if semantic_labels.shape != (len(points),):
      raise ValueError("semantic_labels must have shape (N,).")
    if not np.issubdtype(semantic_labels.dtype, np.integer):
      raise ValueError("semantic_labels must contain integer labels.")
    semantic_labels = semantic_labels.astype(np.int64, copy=False)
  if known_free_points is None:
    known_free_points = np.empty((0, 3), dtype=np.float64)
  else:
    known_free_points = np.asarray(known_free_points, dtype=np.float64)
    if known_free_points.ndim != 2 or known_free_points.shape[1] != 3:
      raise ValueError("known_free_points must have shape (N, 3).")
    if not np.isfinite(known_free_points).all():
      raise ValueError("known_free_points must contain only finite coordinates.")

  source_ids = np.arange(len(points), dtype=np.int64)
  keep = np.isfinite(points).all(axis=1) & np.isfinite(opacity)
  keep &= _activated_opacity(opacity, config.opacity_encoding) >= (
    config.opacity_threshold
  )
  if gaussian_scales is not None:
    activated_scales = _activated_scales(gaussian_scales, config.scale_encoding)
    keep &= np.isfinite(activated_scales).all(axis=1)
    keep &= (activated_scales > 0).all(axis=1)
    if config.max_gaussian_scale is not None:
      keep &= activated_scales.max(axis=1) <= config.max_gaussian_scale
  points = points[keep]
  source_ids = source_ids[keep]
  if not len(points):
    raise ValueError("No finite Gaussian centers remain after opacity filtering.")

  lower, upper = np.quantile(
    points, [config.lower_quantile, config.upper_quantile], axis=0
  )
  lower = (np.floor(lower / config.voxel_size) - 1) * config.voxel_size
  upper = (np.ceil(upper / config.voxel_size) + 1) * config.voxel_size
  if config.floor_z is not None:
    lower[2] = config.floor_z + config.voxel_size
  shape = np.rint((upper - lower) / config.voxel_size).astype(int)
  if np.any(shape <= 0):
    raise ValueError("Filtered Gaussian bounds do not define a non-empty voxel grid.")
  grid_voxels = math.prod(int(value) for value in shape)
  if grid_voxels > config.max_grid_voxels:
    raise ValueError(
      f"Voxel grid needs {grid_voxels:,} cells, exceeding max_grid_voxels="
      f"{config.max_grid_voxels:,}; increase voxel_size or tighten bounds."
    )

  indices = np.floor((points - lower) / config.voxel_size).astype(int)
  in_bounds = ((indices >= 0) & (indices < shape)).all(axis=1)
  indices = indices[in_bounds]
  source_ids = source_ids[in_bounds]
  counts = np.zeros(tuple(shape), dtype=np.int32)
  np.add.at(counts, tuple(indices.T), 1)
  occupied = counts >= config.min_points_per_voxel
  labels, _component_count = cast(
    tuple[NDArray[np.int32], int], ndimage.label(occupied)
  )
  component_sizes = np.bincount(labels.ravel())
  retained_components = component_sizes[1:] >= config.min_component_voxels
  occupied &= component_sizes[labels] >= config.min_component_voxels
  if not occupied.any():
    raise ValueError("No occupied voxels remain after density and component filtering.")

  known_free = np.zeros(tuple(shape), dtype=bool)
  known_free_indices = np.floor((known_free_points - lower) / config.voxel_size).astype(
    int
  )
  known_free_in_bounds = ((known_free_indices >= 0) & (known_free_indices < shape)).all(
    axis=1
  )
  known_free_indices = known_free_indices[known_free_in_bounds]
  bounded_known_free_points = known_free_points[known_free_in_bounds]
  if len(known_free_indices):
    known_free[tuple(known_free_indices.T)] = True
  overlap_count = int(np.count_nonzero(known_free & occupied))
  if overlap_count:
    raise ValueError(
      f"known_free_points conflict with {overlap_count} occupied voxel(s)."
    )

  point_is_occupied = occupied[tuple(indices.T)]
  indices = indices[point_is_occupied]
  filtered_source_ids = source_ids[point_is_occupied]
  filtered_semantic_labels = (
    semantic_labels[filtered_source_ids] if semantic_labels is not None else None
  )
  candidate_semantic_labels: np.ndarray | None = None
  if filtered_semantic_labels is None:
    merged = merge_surface_boxes(occupied, config, known_free)
  else:
    best_label_count = np.zeros(tuple(shape), dtype=np.int32)
    voxel_semantic_labels = np.full(
      tuple(shape), config.unknown_semantic_label, dtype=np.int64
    )
    unique_labels = np.unique(filtered_semantic_labels)
    for label in unique_labels:
      label_counts = np.zeros(tuple(shape), dtype=np.int32)
      label_points = filtered_semantic_labels == label
      np.add.at(label_counts, tuple(indices[label_points].T), 1)
      replace_label = label_counts > best_label_count
      replace_label |= (label_counts == best_label_count) & (
        label < voxel_semantic_labels
      )
      replace_label &= label_counts > 0
      voxel_semantic_labels[replace_label] = label
      best_label_count[replace_label] = label_counts[replace_label]
    semantic_boxes: list[tuple[np.ndarray, np.ndarray]] = []
    merged_labels: list[int] = []
    for label in unique_labels:
      label_occupied = occupied & (voxel_semantic_labels == label)
      other_semantic_occupied = occupied & (voxel_semantic_labels != label)
      label_boxes = merge_surface_boxes(
        label_occupied,
        config,
        known_free | other_semantic_occupied,
      )
      semantic_boxes.extend(label_boxes)
      merged_labels.extend([int(label)] * len(label_boxes))
    merged = semantic_boxes
    candidate_semantic_labels = np.asarray(merged_labels, dtype=np.int64)

  candidate_mapped_ids: list[np.ndarray] = []
  candidate_offsets = [0]
  for candidate_index, (start, end) in enumerate(merged):
    point_in_candidate = ((indices >= start) & (indices < end)).all(axis=1)
    if filtered_semantic_labels is not None and candidate_semantic_labels is not None:
      point_in_candidate &= (
        filtered_semantic_labels == candidate_semantic_labels[candidate_index]
      )
    ids = filtered_source_ids[point_in_candidate]
    candidate_mapped_ids.append(ids)
    candidate_offsets.append(candidate_offsets[-1] + len(ids))
  candidate_source_ids = np.concatenate(candidate_mapped_ids)

  retained_candidate_indices = [
    index
    for index, (start, end) in enumerate(merged)
    if np.prod(end - start) * config.voxel_size**3 >= config.min_box_volume - 1e-12
  ]
  boxes = [merged[index] for index in retained_candidate_indices]
  box_semantic_labels = (
    candidate_semantic_labels[retained_candidate_indices]
    if candidate_semantic_labels is not None
    else None
  )

  mapped_ids: list[np.ndarray] = []
  offsets = [0]
  for box_index, (start, end) in enumerate(boxes):
    point_in_box = ((indices >= start) & (indices < end)).all(axis=1)
    if filtered_semantic_labels is not None and box_semantic_labels is not None:
      point_in_box &= filtered_semantic_labels == box_semantic_labels[box_index]
    ids = filtered_source_ids[point_in_box]
    mapped_ids.append(ids)
    offsets.append(offsets[-1] + len(ids))
  box_source_ids = (
    np.concatenate(mapped_ids) if mapped_ids else np.empty(0, dtype=np.int64)
  )
  return VoxelBoxResult(
    lower=lower,
    upper=upper,
    grid_shape=(int(shape[0]), int(shape[1]), int(shape[2])),
    occupied_voxel_indices=np.argwhere(occupied),
    candidate_boxes=tuple(merged),
    candidate_source_id_offsets=np.asarray(candidate_offsets, dtype=np.int64),
    candidate_source_ids=candidate_source_ids,
    candidate_semantic_labels=candidate_semantic_labels,
    boxes=tuple(boxes),
    filtered_source_ids=filtered_source_ids,
    filtered_semantic_labels=filtered_semantic_labels,
    known_free_points=bounded_known_free_points,
    box_source_id_offsets=np.asarray(offsets, dtype=np.int64),
    box_source_ids=box_source_ids,
    box_semantic_labels=box_semantic_labels,
    occupied_voxel_count=int(occupied.sum()),
    connected_component_count=int(np.count_nonzero(retained_components)),
    known_free_voxel_count=int(known_free.sum()),
    merged_box_count=len(merged),
  )


def _canonical_frame(frame: np.ndarray) -> np.ndarray:
  """Choose deterministic axis signs and return a right-handed frame."""
  result = np.array(frame, dtype=np.float64, copy=True)
  for axis_index in range(3):
    axis = result[:, axis_index]
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
      result[:, axis_index] *= -1
  if np.linalg.det(result) < 0:
    result[:, 2] *= -1
  return result


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
  """Convert a right-handed rotation matrix to a canonical quaternion."""
  trace = float(np.trace(matrix))
  quaternion = np.empty(4, dtype=np.float64)
  if trace > 0:
    scale = math.sqrt(trace + 1.0) * 2
    quaternion[0] = 0.25 * scale
    quaternion[1] = (matrix[2, 1] - matrix[1, 2]) / scale
    quaternion[2] = (matrix[0, 2] - matrix[2, 0]) / scale
    quaternion[3] = (matrix[1, 0] - matrix[0, 1]) / scale
  else:
    diagonal = np.diag(matrix)
    index = int(np.argmax(diagonal))
    next_index = (index + 1) % 3
    last_index = (index + 2) % 3
    scale = (
      math.sqrt(
        1.0
        + matrix[index, index]
        - matrix[next_index, next_index]
        - matrix[last_index, last_index]
      )
      * 2
    )
    quaternion[0] = (
      matrix[last_index, next_index] - matrix[next_index, last_index]
    ) / scale
    quaternion[index + 1] = 0.25 * scale
    quaternion[next_index + 1] = (
      matrix[next_index, index] + matrix[index, next_index]
    ) / scale
    quaternion[last_index + 1] = (
      matrix[last_index, index] + matrix[index, last_index]
    ) / scale
  quaternion /= np.linalg.norm(quaternion)
  if quaternion[0] < 0:
    quaternion *= -1
  return quaternion


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
  """Convert a unit wxyz quaternion into a local-to-world rotation matrix."""
  w, x, y, z = quaternion
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
  )


def _primitive_contains_points(
  primitive: CollisionPrimitive, points: np.ndarray
) -> np.ndarray:
  """Return whether points lie strictly inside a fitted primitive."""
  if not len(points):
    return np.zeros(0, dtype=bool)
  local = (points - primitive.center) @ _quaternion_wxyz_to_matrix(
    primitive.quaternion_wxyz
  )
  tolerance = 1e-12
  if primitive.geom_type == "box":
    return (np.abs(local) < np.asarray(primitive.size) - tolerance).all(axis=1)
  if primitive.geom_type == "sphere":
    return np.linalg.norm(local, axis=1) < primitive.size[0] - tolerance
  radial = np.linalg.norm(local[:, :2], axis=1)
  if primitive.geom_type == "cylinder":
    return (radial < primitive.size[0] - tolerance) & (
      np.abs(local[:, 2]) < primitive.size[1] - tolerance
    )
  axial_excess = np.maximum(np.abs(local[:, 2]) - primitive.size[1], 0)
  return np.hypot(radial, axial_excess) < primitive.size[0] - tolerance


def _primitive_signed_distance(
  primitive: CollisionPrimitive, points: np.ndarray
) -> np.ndarray:
  local = (points - primitive.center) @ _quaternion_wxyz_to_matrix(
    primitive.quaternion_wxyz
  )
  if primitive.geom_type == "box":
    return _box_signed_distance(local, np.asarray(primitive.size))
  if primitive.geom_type == "sphere":
    return np.linalg.norm(local, axis=1) - primitive.size[0]
  radial = np.linalg.norm(local[:, :2], axis=1)
  if primitive.geom_type == "cylinder":
    return _cylinder_signed_distance(
      radial, local[:, 2], primitive.size[0], primitive.size[1]
    )
  axial_excess = np.maximum(np.abs(local[:, 2]) - primitive.size[1], 0)
  return np.hypot(radial, axial_excess) - primitive.size[0]


def _result_primitives(
  result: VoxelBoxResult,
  config: VoxelBoxConfig,
  primitive_result: PrimitiveFitResult | None,
) -> tuple[CollisionPrimitive, ...]:
  if primitive_result is not None:
    return primitive_result.primitives
  return tuple(
    CollisionPrimitive(
      geom_type="box",
      center=result.lower + (start + end) * config.voxel_size / 2,
      quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
      size=tuple(float(value) for value in (end - start) * config.voxel_size / 2),
      normalized_error=0.0,
      source_candidate_index=index,
    )
    for index, (start, end) in enumerate(result.boxes)
  )


def evaluate_collision_proxy(
  result: VoxelBoxResult,
  config: VoxelBoxConfig,
  *,
  primitive_result: PrimitiveFitResult | None = None,
  reference_surface_points: np.ndarray | None = None,
  free_space_points: np.ndarray | None = None,
) -> dict[str, float | int | None]:
  """Evaluate surface distance and false occupancy against independent probes."""
  primitives = _result_primitives(result, config, primitive_result)
  if not primitives:
    raise ValueError("Cannot evaluate an empty collision proxy.")

  def validate_points(name: str, values: np.ndarray | None) -> np.ndarray:
    if values is None:
      return np.empty((0, 3), dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
      raise ValueError(f"{name} must have shape (N, 3) with finite coordinates.")
    return values

  surface = validate_points("reference_surface_points", reference_surface_points)
  free = validate_points("free_space_points", free_space_points)
  if len(surface):
    distances = np.full(len(surface), np.inf, dtype=np.float64)
    for primitive in primitives:
      distances = np.minimum(
        distances, np.abs(_primitive_signed_distance(primitive, surface))
      )
    surface_mean: float | None = float(distances.mean())
    surface_p95: float | None = float(np.quantile(distances, 0.95))
    surface_max: float | None = float(distances.max())
  else:
    surface_mean = surface_p95 = surface_max = None
  if len(free):
    false_occupied = np.zeros(len(free), dtype=bool)
    for primitive in primitives:
      false_occupied |= _primitive_contains_points(primitive, free)
    false_occupied_count = int(false_occupied.sum())
    false_occupied_fraction: float | None = float(false_occupied.mean())
  else:
    false_occupied_count = 0
    false_occupied_fraction = None
  return {
    "reference_surface_point_count": len(surface),
    "surface_distance_mean": surface_mean,
    "surface_distance_p95": surface_p95,
    "surface_distance_max": surface_max,
    "free_space_point_count": len(free),
    "false_occupied_count": false_occupied_count,
    "false_occupied_fraction": false_occupied_fraction,
  }


def _principal_axes(
  points: np.ndarray, covariances: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  center = points.mean(axis=0)
  covariance = np.cov(points - center, rowvar=False, bias=True)
  if covariances is not None:
    covariance = covariance + covariances.mean(axis=0)
  eigenvalues, eigenvectors = np.linalg.eigh(covariance)
  order = np.argsort(eigenvalues)[::-1]
  return center, eigenvalues[order], _canonical_frame(eigenvectors[:, order])


def _normalized_rms(values: np.ndarray, scale: float) -> float:
  return float(np.sqrt(np.mean(np.square(values))) / max(scale, 1e-12))


def _fit_sphere(
  points: np.ndarray,
  center: np.ndarray,
  voxel_size: float,
  candidate_index: int,
) -> CollisionPrimitive | None:
  distances = np.linalg.norm(points - center, axis=1)
  radius = float(np.median(distances))
  if radius < voxel_size / 2:
    return None
  error = _normalized_rms(distances - radius, radius)
  return CollisionPrimitive(
    geom_type="sphere",
    center=center,
    quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    size=(radius + voxel_size / 2,),
    normalized_error=error,
    source_candidate_index=candidate_index,
  )


def _cylinder_signed_distance(
  radial: np.ndarray, axial: np.ndarray, radius: float, half_height: float
) -> np.ndarray:
  radial_delta = radial - radius
  axial_delta = np.abs(axial) - half_height
  outside = np.linalg.norm(
    np.column_stack([np.maximum(radial_delta, 0), np.maximum(axial_delta, 0)]),
    axis=1,
  )
  inside = np.minimum(np.maximum(radial_delta, axial_delta), 0)
  return outside + inside


def _box_signed_distance(local: np.ndarray, half_size: np.ndarray) -> np.ndarray:
  delta = np.abs(local) - half_size
  outside = np.linalg.norm(np.maximum(delta, 0), axis=1)
  inside = np.minimum(np.max(delta, axis=1), 0)
  return outside + inside


def _fit_oriented_box(
  points: np.ndarray,
  center: np.ndarray,
  principal_axes: np.ndarray,
  voxel_size: float,
  candidate_index: int,
) -> CollisionPrimitive:
  frame = _canonical_frame(principal_axes)
  local = (points - center) @ frame
  lower = local.min(axis=0)
  upper = local.max(axis=0)
  midpoint = (lower + upper) / 2
  raw_half_size = (upper - lower) / 2
  residual = _box_signed_distance(local - midpoint, raw_half_size)
  error = _normalized_rms(residual, max(float(raw_half_size.max()), voxel_size))
  half_size = np.maximum(
    raw_half_size + voxel_size / 2,
    voxel_size / 2,
  )
  return CollisionPrimitive(
    geom_type="box",
    center=center + frame @ midpoint,
    quaternion_wxyz=_matrix_to_quaternion_wxyz(frame),
    size=tuple(float(value) for value in half_size),
    normalized_error=error,
    source_candidate_index=candidate_index,
    baseline_box_error=error,
    relative_error_improvement=0.0,
  )


def _fit_round_axis_primitive(
  points: np.ndarray,
  center: np.ndarray,
  principal_axes: np.ndarray,
  voxel_size: float,
  candidate_index: int,
) -> CollisionPrimitive | None:
  frame = _canonical_frame(
    np.column_stack([principal_axes[:, 1], principal_axes[:, 2], principal_axes[:, 0]])
  )
  local = (points - center) @ frame
  radial_center = np.median(local[:, :2], axis=0)
  axial_min, axial_max = np.quantile(local[:, 2], [0.01, 0.99])
  axial_center = float((axial_min + axial_max) / 2)
  local[:, :2] -= radial_center
  local[:, 2] -= axial_center
  radial = np.linalg.norm(local[:, :2], axis=1)
  radius = float(np.median(radial))
  half_height = float((axial_max - axial_min) / 2)
  if radius < voxel_size / 2 or half_height < voxel_size:
    return None

  cylinder_error = _normalized_rms(
    _cylinder_signed_distance(radial, local[:, 2], radius, half_height),
    max(radius, half_height),
  )
  capsule_half_length = max(half_height - radius, 0.0)
  capsule_distance = (
    np.sqrt(
      np.square(radial)
      + np.square(np.maximum(np.abs(local[:, 2]) - capsule_half_length, 0))
    )
    - radius
  )
  capsule_error = _normalized_rms(capsule_distance, max(radius, half_height))
  primitive_type: Literal["cylinder", "capsule"]
  if cylinder_error <= capsule_error * 1.05:
    primitive_type = "cylinder"
    error = cylinder_error
    axial_size = half_height + voxel_size / 2
  else:
    primitive_type = "capsule"
    error = capsule_error
    axial_size = capsule_half_length + voxel_size / 2

  local_center = np.array([radial_center[0], radial_center[1], axial_center])
  return CollisionPrimitive(
    geom_type=primitive_type,
    center=center + frame @ local_center,
    quaternion_wxyz=_matrix_to_quaternion_wxyz(frame),
    size=(radius + voxel_size / 2, axial_size),
    normalized_error=error,
    source_candidate_index=candidate_index,
  )


def _fit_planar_box(
  points: np.ndarray,
  center: np.ndarray,
  principal_axes: np.ndarray,
  voxel_config: VoxelBoxConfig,
  fit_config: PrimitiveFitConfig,
  candidate_index: int,
  known_free_points: np.ndarray,
) -> CollisionPrimitive | None:
  frame = _canonical_frame(principal_axes)
  local = (points - center) @ frame
  lower = local.min(axis=0)
  upper = local.max(axis=0)
  planar_span = upper[:2] - lower[:2]
  grid_shape = np.maximum(
    np.ceil(planar_span / voxel_config.voxel_size).astype(int) + 1, 1
  )
  projected_indices = np.floor(
    (local[:, :2] - lower[:2]) / voxel_config.voxel_size
  ).astype(int)
  projected_indices = np.minimum(projected_indices, grid_shape - 1)
  support = np.zeros(tuple(grid_shape), dtype=bool)
  support[tuple(projected_indices.T)] = True
  if support.mean() < fit_config.min_planar_fill:
    return None
  gap_distance = cast(np.ndarray, ndimage.distance_transform_edt(~support))
  if (
    gap_distance.max() * voxel_config.voxel_size
    > voxel_config.max_surface_expansion + 1e-9
  ):
    return None

  box = _fit_oriented_box(
    points,
    center,
    principal_axes,
    voxel_config.voxel_size,
    candidate_index,
  )
  normal_residual = local[:, 2] - np.median(local[:, 2])
  error = _normalized_rms(
    normal_residual, max(float(planar_span.max()), voxel_config.voxel_size)
  )
  thickness = fit_config.planar_slab_thickness
  if thickness is None or not len(known_free_points):
    return replace(box, normalized_error=error, baseline_box_error=error)

  free_local = (known_free_points - center) @ frame
  plane_offset = float(np.median(local[:, 2]))
  footprint_margin = voxel_config.voxel_size / 2
  within_footprint = (
    (free_local[:, :2] >= lower[:2] - footprint_margin)
    & (free_local[:, :2] <= upper[:2] + footprint_margin)
  ).all(axis=1)
  signed_distance = free_local[:, 2] - plane_offset
  eligible = within_footprint & (
    np.abs(signed_distance) >= voxel_config.voxel_size / 2 - 1e-12
  )
  eligible &= (
    np.abs(signed_distance) <= fit_config.planar_free_side_max_distance + 1e-12
  )
  signs = np.sign(signed_distance[eligible])
  if not len(signs):
    return replace(box, normalized_error=error, baseline_box_error=error)
  positive_count = int(np.count_nonzero(signs > 0))
  negative_count = int(np.count_nonzero(signs < 0))
  majority_count = max(positive_count, negative_count)
  agreement = majority_count / len(signs)
  if agreement < fit_config.planar_free_side_min_agreement:
    return replace(box, normalized_error=error, baseline_box_error=error)
  free_side_sign = 1 if positive_count > negative_count else -1
  midpoint = (lower + upper) / 2
  midpoint[2] = plane_offset - free_side_sign * thickness / 2
  half_size = np.maximum((upper - lower) / 2 + footprint_margin, footprint_margin)
  half_size[2] = thickness / 2
  return replace(
    box,
    center=center + frame @ midpoint,
    size=tuple(float(value) for value in half_size),
    normalized_error=error,
    baseline_box_error=error,
    free_side_anchored=True,
    free_side_sign=free_side_sign,
  )


def _fit_candidate_primitive(
  points: np.ndarray,
  voxel_config: VoxelBoxConfig,
  fit_config: PrimitiveFitConfig,
  candidate_index: int,
  known_free_points: np.ndarray,
  gaussian_covariance: np.ndarray | None,
) -> tuple[CollisionPrimitive | None, bool]:
  if len(points) < fit_config.min_points:
    return None, False
  center, eigenvalues, principal_axes = _principal_axes(points, gaussian_covariance)
  if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 1e-12:
    return None, False

  box = _fit_oriented_box(
    points, center, principal_axes, voxel_config.voxel_size, candidate_index
  )
  allow_small_candidate = False
  primitive: CollisionPrimitive | None = None
  if eigenvalues[2] / eigenvalues[0] >= fit_config.isotropic_eigenvalue_ratio:
    primitive = _fit_sphere(points, center, voxel_config.voxel_size, candidate_index)
  elif (
    eigenvalues[1] / eigenvalues[0] <= fit_config.linear_eigenvalue_ratio
    and eigenvalues[2] / max(eigenvalues[1], 1e-12)
    >= fit_config.radial_eigenvalue_ratio
  ):
    primitive = _fit_round_axis_primitive(
      points, center, principal_axes, voxel_config.voxel_size, candidate_index
    )
  elif (
    eigenvalues[2] / max(eigenvalues[1], 1e-12) <= fit_config.planar_eigenvalue_ratio
  ):
    allow_small_candidate = True
    box = _fit_planar_box(
      points,
      center,
      principal_axes,
      voxel_config,
      fit_config,
      candidate_index,
      known_free_points,
    )
  if (
    primitive is not None
    and primitive.normalized_error <= fit_config.max_normalized_error
  ):
    improvement = 1 - primitive.normalized_error / max(
      box.normalized_error if box is not None else 0.0, 1e-12
    )
    if improvement >= fit_config.min_round_error_improvement:
      return (
        replace(
          primitive,
          baseline_box_error=box.normalized_error if box is not None else None,
          relative_error_improvement=improvement,
        ),
        True,
      )
    if box is not None:
      return (
        replace(
          box,
          rejected_round_type=primitive.geom_type,
          rejected_round_error=primitive.normalized_error,
          rejected_round_error_improvement=improvement,
        ),
        True,
      )
  if box is not None and box.normalized_error <= fit_config.max_normalized_error:
    return box, allow_small_candidate
  return None, False


def fit_collision_primitives(
  points: np.ndarray,
  result: VoxelBoxResult,
  voxel_config: VoxelBoxConfig,
  fit_config: PrimitiveFitConfig | None = None,
  *,
  point_covariances: np.ndarray | None = None,
) -> PrimitiveFitResult:
  """Fit simple primitives per merged voxel candidate, falling back to boxes."""
  points = np.asarray(points, dtype=np.float64)
  if points.ndim != 2 or points.shape[1] != 3:
    raise ValueError("points must have shape (N, 3).")
  fit_config = fit_config or PrimitiveFitConfig()
  if point_covariances is not None:
    point_covariances = np.asarray(point_covariances, dtype=np.float64)
    if point_covariances.shape != (len(points), 3, 3):
      raise ValueError("point_covariances must have shape (N, 3, 3).")
    if not np.isfinite(point_covariances).all():
      raise ValueError("point_covariances must contain only finite values.")
  if fit_config.use_gaussian_covariance and point_covariances is None:
    raise ValueError("use_gaussian_covariance requires point_covariances.")
  primitives: list[CollisionPrimitive] = []
  mapped_ids: list[np.ndarray] = []
  offsets = [0]

  for index, (start, end) in enumerate(result.candidate_boxes):
    map_start = result.candidate_source_id_offsets[index]
    map_end = result.candidate_source_id_offsets[index + 1]
    source_ids = result.candidate_source_ids[map_start:map_end]
    primitive, allow_small_candidate = _fit_candidate_primitive(
      points[source_ids],
      voxel_config,
      fit_config,
      index,
      result.known_free_points,
      (
        point_covariances[source_ids]
        if fit_config.use_gaussian_covariance and point_covariances is not None
        else None
      ),
    )
    constraint_points = result.known_free_points
    if (
      result.filtered_semantic_labels is not None
      and result.candidate_semantic_labels is not None
    ):
      other_label = (
        result.filtered_semantic_labels != result.candidate_semantic_labels[index]
      )
      constraint_points = np.concatenate(
        [constraint_points, points[result.filtered_source_ids[other_label]]]
      )
    known_free_rejected_fit = primitive is not None and bool(
      _primitive_contains_points(primitive, constraint_points).any()
    )
    if known_free_rejected_fit:
      primitive = None
    candidate_volume = float(np.prod(end - start) * voxel_config.voxel_size**3)
    if (
      primitive is not None
      and not allow_small_candidate
      and candidate_volume < voxel_config.min_box_volume - 1e-12
    ):
      continue
    if primitive is None:
      if candidate_volume < voxel_config.min_box_volume - 1e-12:
        continue
      center = result.lower + (start + end) * voxel_config.voxel_size / 2
      half_size = (end - start) * voxel_config.voxel_size / 2
      primitive = CollisionPrimitive(
        geom_type="box",
        center=center,
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        size=tuple(float(value) for value in half_size),
        normalized_error=0.0,
        source_candidate_index=index,
        known_free_rejected_fit=known_free_rejected_fit,
      )
    primitives.append(primitive)
    mapped_ids.append(source_ids)
    offsets.append(offsets[-1] + len(source_ids))

  primitive_source_ids = (
    np.concatenate(mapped_ids) if mapped_ids else np.empty(0, dtype=np.int64)
  )
  return PrimitiveFitResult(
    primitives=tuple(primitives),
    primitive_source_id_offsets=np.asarray(offsets, dtype=np.int64),
    primitive_source_ids=primitive_source_ids,
  )


def build_collision_mjcf(
  result: VoxelBoxResult,
  config: VoxelBoxConfig,
  *,
  model_name: str,
  primitive_result: PrimitiveFitResult | None = None,
  debug_color_by_type: bool = False,
) -> str:
  """Serialize generated boxes or fitted primitives as collision-only MJCF."""
  model = ET.Element("mujoco", model=model_name)
  defaults = ET.SubElement(model, "default")
  ET.SubElement(
    defaults,
    "geom",
    group="3",
    contype="1",
    conaffinity="1",
    condim="3",
    friction="1.0 0.005 0.0001",
    rgba="0.15 0.75 0.25 1",
  )
  world = ET.SubElement(model, "worldbody")

  def add_box(
    name: str, centre: np.ndarray, half_size: np.ndarray, rgba: str | None = None
  ) -> None:
    attributes = {
      "name": name,
      "type": "box",
      "pos": " ".join(f"{value:.6f}" for value in centre),
      "size": " ".join(f"{value:.6f}" for value in half_size),
    }
    if rgba is not None:
      attributes["rgba"] = rgba
    ET.SubElement(world, "geom", attributes)

  if config.floor_z is not None:
    centre = (result.lower + result.upper) / 2
    centre[2] = config.floor_z - config.voxel_size / 2
    half_size = (result.upper - result.lower) / 2
    half_size[2] = config.voxel_size / 2
    ground_rgba = "0.50 0.50 0.50 0.12" if debug_color_by_type else None
    add_box("ground", centre, half_size, ground_rgba)
  if primitive_result is not None:
    for index, primitive in enumerate(primitive_result.primitives):
      expected_size_count = {
        "box": 3,
        "sphere": 1,
        "cylinder": 2,
        "capsule": 2,
      }[primitive.geom_type]
      size = np.asarray(primitive.size, dtype=np.float64)
      quaternion = np.asarray(primitive.quaternion_wxyz, dtype=np.float64)
      if (
        not np.isfinite(primitive.center).all()
        or not np.isfinite(size).all()
        or len(size) != expected_size_count
        or np.any(size <= 0)
        or quaternion.shape != (4,)
        or not np.isfinite(quaternion).all()
        or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6)
      ):
        raise ValueError(
          f"Primitive {index} ({primitive.geom_type}) has invalid pose or size."
        )
      attributes: dict[str, str] = {
        "name": f"{primitive.geom_type}_{index:04d}",
        "type": primitive.geom_type,
        "pos": " ".join(f"{value:.6f}" for value in primitive.center),
        "size": " ".join(f"{value:.6f}" for value in primitive.size),
      }
      if primitive.geom_type != "sphere":
        attributes["quat"] = " ".join(
          f"{value:.8f}" for value in primitive.quaternion_wxyz
        )
      if debug_color_by_type:
        attributes["rgba"] = _DEBUG_RGBA_BY_TYPE[primitive.geom_type]
      ET.SubElement(world, "geom", attributes)
    ET.indent(model, space="  ")
    return ET.tostring(model, encoding="unicode") + "\n"
  for index, (start, end) in enumerate(result.boxes):
    centre = result.lower + (start + end) * config.voxel_size / 2
    half_size = (end - start) * config.voxel_size / 2
    add_box(f"surface_{index:04d}", centre, half_size)

  ET.indent(model, space="  ")
  return ET.tostring(model, encoding="unicode") + "\n"


def config_dict(config: VoxelBoxConfig) -> dict[str, object]:
  """Return a JSON-serializable configuration dictionary."""
  return asdict(config)


def primitive_fit_config_dict(config: PrimitiveFitConfig) -> dict[str, object]:
  """Return a JSON-serializable primitive-fit configuration dictionary."""
  return asdict(config)
