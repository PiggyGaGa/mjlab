"""Tests for the scene-independent center-only collision baseline."""

from dataclasses import replace
from typing import cast

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.gaussian_collision import (
  CollisionPrimitive,
  PrimitiveFitConfig,
  PrimitiveFitResult,
  VoxelBoxConfig,
  VoxelBoxResult,
  build_collision_mjcf,
  evaluate_collision_proxy,
  fit_collision_primitives,
  gaussian_covariances,
  generate_voxel_boxes,
  select_collision_primitive_budget,
)
from mjlab.asset_zoo.gaussian_collision import (
  _primitive_contains_points as primitive_contains_points,
)
from mjlab.asset_zoo.gaussian_collision import (
  _principal_axes as principal_axes,
)
from mjlab.asset_zoo.gaussian_collision_mesh import (
  SurfaceMesh,
  build_heightfield_mjcf,
  convex_component_hulls,
  heightfield_surface,
  marching_tetrahedra_surface,
  occupancy_surface_mesh,
  repair_surface_mesh,
  simplify_vertex_clusters,
  surface_mesh_obj,
  truncated_signed_distance,
)
from mjlab.asset_zoo.gaussian_semantics import decode_semantic_features
from mjlab.scripts.benchmark_collision_mjcf import benchmark_collision_mjcf


def _auto_config(voxel_size: float = 0.05) -> VoxelBoxConfig:
  return VoxelBoxConfig(
    voxel_size=voxel_size,
    lower_quantile=0,
    upper_quantile=1,
    min_points_per_voxel=1,
    min_component_voxels=1,
    min_box_volume=0.1,
  )


def _fit_types(points: np.ndarray) -> tuple[str, ...]:
  config = _auto_config()
  result = generate_voxel_boxes(points, np.zeros(len(points)), config)
  fitted = fit_collision_primitives(
    points,
    result,
    config,
    PrimitiveFitConfig(min_points=12, max_normalized_error=0.15),
  )
  return tuple(primitive.geom_type for primitive in fitted.primitives)


def _box_contains_point(
  result: VoxelBoxResult, config: VoxelBoxConfig, point: np.ndarray
) -> bool:
  return any(
    np.all(result.lower + start * config.voxel_size <= point)
    and np.all(point < result.lower + end * config.voxel_size)
    for start, end in result.boxes
  )


def test_voxel_boxes_preserve_doorway_and_source_mapping() -> None:
  voxel_size = 0.1
  voxel_indices = np.array(
    [
      (x, y, z)
      for x in range(1)
      for y in range(30)
      for z in range(30)
      if not (10 <= y < 20 and z < 22)
    ]
  )
  points = np.repeat((voxel_indices + 0.25) * voxel_size, 3, axis=0)
  opacity = np.zeros(len(points))
  config = VoxelBoxConfig(
    voxel_size=voxel_size,
    lower_quantile=0,
    upper_quantile=1,
    min_component_voxels=1,
    min_box_volume=0,
  )

  result = generate_voxel_boxes(points, opacity, config)
  bounded_result = generate_voxel_boxes(
    points, opacity, replace(config, max_planar_merge_residual=0.04)
  )
  fitted = fit_collision_primitives(points, result, config)

  assert len(result.boxes) > 1
  assert len(bounded_result.boxes) > 1
  assert len(result.box_source_ids) == len(points)
  assert np.array_equal(np.sort(result.box_source_ids), np.arange(len(points)))
  assert result.box_source_id_offsets[-1] == len(points)
  for start, end in result.boxes:
    assert not (start[1] <= 16 < end[1] and start[2] <= 11 < end[2])
  for start, end in bounded_result.boxes:
    assert not (start[1] <= 16 < end[1] and start[2] <= 11 < end[2])
  assert len(fitted.primitives) > 1
  assert all(primitive.geom_type == "box" for primitive in fitted.primitives)


def test_planar_merge_residual_preserves_stair_treads_and_free_space() -> None:
  voxel_size = 0.1
  voxel_indices: list[tuple[int, int, int]] = []
  voxel_indices.extend((x, y, 0) for x in range(0, 4) for y in range(10))
  voxel_indices.extend((4, y, z) for y in range(10) for z in range(0, 3))
  voxel_indices.extend((x, y, 2) for x in range(4, 8) for y in range(10))
  voxel_indices.extend((8, y, z) for y in range(10) for z in range(2, 5))
  voxel_indices.extend((x, y, 4) for x in range(8, 12) for y in range(10))
  unique_indices = np.unique(np.asarray(voxel_indices), axis=0)
  points = (unique_indices + 0.25) * voxel_size
  config = VoxelBoxConfig(
    voxel_size=voxel_size,
    lower_quantile=0,
    upper_quantile=1,
    min_points_per_voxel=1,
    min_component_voxels=1,
    min_box_volume=0,
    max_planar_merge_residual=0.04,
  )

  result = generate_voxel_boxes(points, np.zeros(len(points)), config)

  free_probes = np.array([[0.225, 0.525, 0.25], [0.625, 0.525, 0.45]])
  free_indices = np.floor((free_probes - result.lower) / voxel_size).astype(int)
  for probe in free_indices:
    assert not any(
      ((start <= probe) & (probe < end)).all() for start, end in result.boxes
    )

  support_xy = np.array([[0.225, 0.525], [0.625, 0.525], [1.025, 0.525]])
  support_heights = []
  for xy in support_xy:
    tops = [
      result.lower[2] + end[2] * voxel_size
      for start, end in result.boxes
      if np.all(result.lower[:2] + start[:2] * voxel_size <= xy)
      and np.all(xy < result.lower[:2] + end[:2] * voxel_size)
    ]
    support_heights.append(max(tops))
  np.testing.assert_allclose(support_heights, [0.1, 0.3, 0.5], atol=1e-9)
  assert len(result.boxes) <= 8
  metrics = evaluate_collision_proxy(
    result,
    config,
    reference_surface_points=points,
    free_space_points=free_probes,
  )
  assert metrics["false_occupied_count"] == 0
  assert metrics["surface_distance_max"] == pytest.approx(0.075)


@pytest.mark.parametrize(("free_z", "expected_sign"), [(0.325, 1), (0.125, -1)])
def test_planar_slab_thickness_extends_away_from_known_free_side(
  free_z: float, expected_sign: int
) -> None:
  xy = (np.arange(5) + 0.25) * 0.1
  x_grid, y_grid = np.meshgrid(xy, xy, indexing="ij")
  points = np.column_stack(
    [x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, 0.225)]
  )
  free_points = np.column_stack(
    [x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, free_z)]
  )
  config = VoxelBoxConfig(
    voxel_size=0.1,
    lower_quantile=0,
    upper_quantile=1,
    min_points_per_voxel=1,
    min_component_voxels=1,
    min_box_volume=0,
  )
  result = generate_voxel_boxes(
    points,
    np.zeros(len(points)),
    config,
    known_free_points=free_points,
  )
  fitted = fit_collision_primitives(
    points,
    result,
    config,
    PrimitiveFitConfig(min_points=4, planar_slab_thickness=0.2),
  )

  assert len(fitted.primitives) == 1
  primitive = fitted.primitives[0]
  assert primitive.geom_type == "box"
  assert primitive.free_side_anchored
  assert primitive.free_side_sign == expected_sign
  assert primitive.size[2] == pytest.approx(0.1)
  free_face_z = primitive.center[2] + expected_sign * primitive.size[2]
  assert free_face_z == pytest.approx(0.225)
  assert not primitive_contains_points(primitive, free_points).any()


def test_planar_slab_does_not_guess_when_free_side_is_ambiguous() -> None:
  xy = (np.arange(5) + 0.25) * 0.1
  x_grid, y_grid = np.meshgrid(xy, xy, indexing="ij")
  points = np.column_stack(
    [x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, 0.225)]
  )
  free_points = np.concatenate(
    [
      np.column_stack([x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, free_z)])
      for free_z in (0.125, 0.325)
    ]
  )
  config = VoxelBoxConfig(
    voxel_size=0.1,
    lower_quantile=0,
    upper_quantile=1,
    min_points_per_voxel=1,
    min_component_voxels=1,
    min_box_volume=0,
  )
  result = generate_voxel_boxes(
    points,
    np.zeros(len(points)),
    config,
    known_free_points=free_points,
  )
  fitted = fit_collision_primitives(
    points,
    result,
    config,
    PrimitiveFitConfig(min_points=4, planar_slab_thickness=0.2),
  )

  assert len(fitted.primitives) == 1
  primitive = fitted.primitives[0]
  assert not primitive.free_side_anchored
  assert primitive.free_side_sign is None
  assert primitive.center[2] == pytest.approx(0.225)


@pytest.mark.parametrize(
  ("voxel_indices", "free_indices", "max_boxes"),
  [
    (
      [
        *((x, y, 4) for x in range(9) for y in range(9)),
        *((x, y, z) for x in (0, 8) for y in (0, 8) for z in range(4)),
      ],
      [(x, y, z) for x in (2, 4, 6) for y in (2, 4, 6) for z in (1, 2, 3)],
      12,
    ),
    (
      [
        (x, y, 0)
        for x in range(9)
        for y in range(9)
        if not (3 <= x <= 5 and 3 <= y <= 5)
      ],
      [(x, y, 0) for x in range(3, 6) for y in range(3, 6)],
      6,
    ),
    (
      [(x, y, z) for x in range(9) for y in range(9) for z in (0, 4)],
      [(x, y, z) for x in range(9) for y in range(9) for z in (1, 2, 3)],
      2,
    ),
  ],
  ids=("table", "pit", "multilevel"),
)
def test_known_free_points_preserve_synthetic_voids(
  voxel_indices: list[tuple[int, int, int]],
  free_indices: list[tuple[int, int, int]],
  max_boxes: int,
) -> None:
  config = VoxelBoxConfig(
    voxel_size=0.1,
    lower_quantile=0,
    upper_quantile=1,
    min_points_per_voxel=1,
    min_component_voxels=1,
    min_box_volume=0,
  )
  points = (np.unique(np.asarray(voxel_indices), axis=0) + 0.25) * config.voxel_size
  free_points = (np.asarray(free_indices) + 0.25) * config.voxel_size
  unconstrained = generate_voxel_boxes(points, np.zeros(len(points)), config)
  constrained = generate_voxel_boxes(
    points,
    np.zeros(len(points)),
    config,
    known_free_points=free_points,
  )

  assert all(_box_contains_point(unconstrained, config, point) for point in free_points)
  assert not any(
    _box_contains_point(constrained, config, point) for point in free_points
  )
  assert constrained.known_free_voxel_count == len(free_points)
  assert len(constrained.boxes) <= max_boxes
  unconstrained_metrics = evaluate_collision_proxy(
    unconstrained,
    config,
    free_space_points=free_points,
  )
  constrained_metrics = evaluate_collision_proxy(
    constrained,
    config,
    free_space_points=free_points,
  )
  assert unconstrained_metrics["false_occupied_count"] == len(free_points)
  assert constrained_metrics["false_occupied_count"] == 0
  fitted = fit_collision_primitives(
    points,
    constrained,
    config,
    PrimitiveFitConfig(min_points=4),
  )
  assert not any(
    primitive_contains_points(primitive, free_points).any()
    for primitive in fitted.primitives
  )


def test_semantic_labels_preserve_multilevel_boundary_and_unknown_residual() -> None:
  lower_level = [(x, y, 0) for x in range(7) for y in range(7)]
  upper_level = [(x, y, 4) for x in range(7) for y in range(7)]
  points = (np.asarray([*lower_level, *upper_level]) + 0.25) * 0.1
  labels = np.concatenate([np.full(len(lower_level), 3), np.full(len(upper_level), -1)])
  config = VoxelBoxConfig(
    voxel_size=0.1,
    lower_quantile=0,
    upper_quantile=1,
    min_points_per_voxel=1,
    min_component_voxels=1,
    min_box_volume=0,
  )
  unconstrained = generate_voxel_boxes(points, np.zeros(len(points)), config)
  semantic = generate_voxel_boxes(
    points,
    np.zeros(len(points)),
    config,
    semantic_labels=labels,
  )
  gap_point = np.array([0.325, 0.325, 0.225])

  assert _box_contains_point(unconstrained, config, gap_point)
  assert not _box_contains_point(semantic, config, gap_point)
  assert semantic.candidate_semantic_labels is not None
  assert set(semantic.candidate_semantic_labels.tolist()) == {-1, 3}
  assert semantic.filtered_semantic_labels is not None
  assert -1 in semantic.filtered_semantic_labels
  assert np.array_equal(
    np.sort(semantic.box_source_ids), np.arange(len(points), dtype=np.int64)
  )


def test_semantic_priority_improves_free_space_at_same_geom_budget() -> None:
  lower_level = [(x, y, 0) for x in range(5) for y in range(5)]
  upper_level = [(x, y, 4) for x in range(5) for y in range(5)]
  distractor = [(10 + x, y, z) for x in range(2) for y in range(2) for z in range(2)]
  voxel_indices = np.asarray([*lower_level, *upper_level, *distractor])
  points = (voxel_indices + 0.25) * 0.1
  labels = np.concatenate(
    [
      np.full(len(lower_level), 10),
      np.full(len(upper_level), 11),
      np.zeros(len(distractor), dtype=np.int64),
    ]
  )
  config = VoxelBoxConfig(
    voxel_size=0.1,
    lower_quantile=0,
    upper_quantile=1,
    min_points_per_voxel=1,
    min_component_voxels=1,
    min_box_volume=0,
  )
  geometry_result = generate_voxel_boxes(points, np.zeros(len(points)), config)
  semantic_result = generate_voxel_boxes(
    points,
    np.zeros(len(points)),
    config,
    semantic_labels=labels,
  )
  fit_config = PrimitiveFitConfig(min_points=4)
  geometry_fit = fit_collision_primitives(points, geometry_result, config, fit_config)
  semantic_fit = fit_collision_primitives(points, semantic_result, config, fit_config)
  geometry_budget = select_collision_primitive_budget(
    geometry_fit,
    geometry_result,
    max_primitives=2,
  )
  semantic_budget = select_collision_primitive_budget(
    semantic_fit,
    semantic_result,
    max_primitives=2,
    semantic_priorities={0: 0.1, 10: 10.0, 11: 10.0},
  )
  free_probe = np.array([[0.225, 0.225, 0.225]])

  geometry_metrics = evaluate_collision_proxy(
    geometry_result,
    config,
    primitive_result=geometry_budget,
    free_space_points=free_probe,
  )
  semantic_metrics = evaluate_collision_proxy(
    semantic_result,
    config,
    primitive_result=semantic_budget,
    free_space_points=free_probe,
  )

  assert len(geometry_budget.primitives) == 2
  assert len(semantic_budget.primitives) == 2
  assert geometry_metrics["false_occupied_count"] == 1
  assert semantic_metrics["false_occupied_count"] == 0


def test_point_importance_budget_is_deterministic_and_rebuilds_mapping() -> None:
  voxel_indices = np.asarray(
    [
      (x, y, z)
      for x_offset in (0, 10, 20)
      for x in range(x_offset, x_offset + 2)
      for y in range(2)
      for z in range(2)
    ]
  )
  points = (voxel_indices + 0.25) * 0.1
  config = VoxelBoxConfig(
    voxel_size=0.1,
    lower_quantile=0,
    upper_quantile=1,
    min_points_per_voxel=1,
    min_component_voxels=1,
    min_box_fill=0.9,
    max_surface_expansion=0,
    min_box_volume=0,
  )
  result = generate_voxel_boxes(points, np.zeros(len(points)), config)
  fitted = fit_collision_primitives(
    points,
    result,
    config,
    PrimitiveFitConfig(min_points=4),
  )
  importance = np.ones(len(points))
  importance[points[:, 0] > 1.5] = 100

  first = select_collision_primitive_budget(
    fitted,
    result,
    max_primitives=1,
    point_importance=importance,
  )
  second = select_collision_primitive_budget(
    fitted,
    result,
    max_primitives=1,
    point_importance=importance,
  )

  assert len(first.primitives) == 1
  assert first.primitives[0].center[0] > 1.5
  assert np.array_equal(first.primitive_source_id_offsets, np.array([0, 8]))
  assert np.array_equal(first.primitive_source_ids, second.primitive_source_ids)
  assert np.array_equal(
    first.primitive_source_ids,
    np.flatnonzero(points[:, 0] > 1.5),
  )


def test_budget_rejects_invalid_priority_inputs() -> None:
  points = np.asarray([(x + 0.25, 0.25, 0.25) for x in range(4)]) * 0.1
  config = VoxelBoxConfig(
    voxel_size=0.1,
    lower_quantile=0,
    upper_quantile=1,
    min_points_per_voxel=1,
    min_component_voxels=1,
    min_box_volume=0,
  )
  result = generate_voxel_boxes(points, np.zeros(len(points)), config)
  fitted = fit_collision_primitives(points, result, config)

  with pytest.raises(ValueError, match="max_primitives"):
    select_collision_primitive_budget(fitted, result, max_primitives=0)
  with pytest.raises(ValueError, match="priority weights"):
    select_collision_primitive_budget(
      fitted,
      result,
      max_primitives=1,
      semantic_priorities={1: 0},
    )
  with pytest.raises(ValueError, match="point_importance"):
    select_collision_primitive_budget(
      fitted,
      result,
      max_primitives=1,
      point_importance=np.array([1.0]),
    )


def test_semantic_feature_decoder_preserves_order_and_unknowns() -> None:
  features = np.array(
    [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.7, 0.7, 0.0],
      [-1.0, 0.0, 0.0],
    ]
  )
  embeddings = np.eye(3)

  decoded = decode_semantic_features(
    features,
    embeddings,
    class_ids=np.array([10, 20, 30]),
    minimum_similarity=0.5,
    minimum_margin=0.1,
    unknown_label=-1,
    block_size=2,
  )

  assert np.array_equal(decoded.labels, np.array([10, 20, -1, -1]))
  np.testing.assert_allclose(decoded.similarities[:2], 1)
  assert decoded.margins[2] == pytest.approx(0)


def test_semantic_feature_decoder_rejects_invalid_contracts() -> None:
  with pytest.raises(ValueError, match="same feature dimension"):
    decode_semantic_features(np.ones((2, 3)), np.ones((2, 4)))
  with pytest.raises(ValueError, match="zero vectors"):
    decode_semantic_features(np.zeros((2, 3)), np.ones((2, 3)))
  with pytest.raises(ValueError, match="unique integer"):
    decode_semantic_features(
      np.ones((2, 3)),
      np.ones((2, 3)),
      class_ids=np.array([1, 1]),
    )


def test_floor_is_opt_in_and_serialized_below_surface() -> None:
  points = np.repeat(np.array([[0.05, 0.05, 0.15], [0.15, 0.05, 0.15]]), 3, axis=0)
  config = VoxelBoxConfig(
    floor_z=0,
    lower_quantile=0,
    upper_quantile=1,
    min_component_voxels=1,
    min_box_volume=0,
  )

  result = generate_voxel_boxes(points, np.zeros(len(points)), config)
  xml = build_collision_mjcf(result, config, model_name="test_collision")

  assert 'model="test_collision"' in xml
  assert 'name="ground"' in xml
  assert 'pos="0.100000 0.050000 -0.050000"' in xml
  assert 'name="surface_0000"' in xml


def test_scale_filter_and_covariance_aware_axes_are_opt_in() -> None:
  points = np.array(
    [
      [0.05, 0.05, 0.05],
      [0.15, 0.05, 0.05],
      [0.25, 0.05, 0.05],
    ]
  )
  encoded_scales = np.log(
    np.array([[0.05, 0.05, 0.05], [1.0, 1.0, 1.0], [0.05, 0.05, 0.05]])
  )
  config = VoxelBoxConfig(
    voxel_size=0.1,
    lower_quantile=0,
    upper_quantile=1,
    min_points_per_voxel=1,
    min_component_voxels=1,
    min_box_volume=0,
    max_gaussian_scale=0.1,
  )
  result = generate_voxel_boxes(
    points,
    np.zeros(len(points)),
    config,
    gaussian_scales=encoded_scales,
  )

  assert result.filtered_source_ids.tolist() == [0, 2]

  identity = np.array([[1.0, 0.0, 0.0, 0.0]])
  covariance = gaussian_covariances(
    np.array([[1.0, 2.0, 3.0]]), identity, scale_encoding="linear"
  )
  np.testing.assert_allclose(covariance[0], np.diag([1.0, 4.0, 9.0]), atol=1e-12)

  line = np.column_stack([np.linspace(-1, 1, 9), np.zeros((9, 2))])
  _, center_only_values, center_only_frame = principal_axes(line)
  point_covariances = np.repeat(np.diag([0.0, 4.0, 0.0])[None, :, :], 9, axis=0)
  _, covariance_values, covariance_frame = principal_axes(line, point_covariances)
  assert center_only_values[0] < covariance_values[0]
  assert abs(center_only_frame[0, 0]) == pytest.approx(1.0)
  assert abs(covariance_frame[1, 0]) == pytest.approx(1.0)


def test_invalid_or_empty_input_fails_with_context() -> None:
  config = VoxelBoxConfig()
  with pytest.raises(ValueError, match="max_planar_merge_residual"):
    VoxelBoxConfig(max_planar_merge_residual=0)
  with pytest.raises(ValueError, match="max_planar_merge_residual"):
    VoxelBoxConfig(max_planar_merge_residual=np.inf)
  with pytest.raises(ValueError, match="planar_merge_eigenvalue_ratio"):
    VoxelBoxConfig(planar_merge_eigenvalue_ratio=0)
  with pytest.raises(ValueError, match="planar_slab_thickness"):
    PrimitiveFitConfig(planar_slab_thickness=0)
  with pytest.raises(ValueError, match="max_gaussian_scale requires"):
    generate_voxel_boxes(
      np.zeros((3, 3)),
      np.zeros(3),
      replace(
        config,
        min_points_per_voxel=1,
        min_component_voxels=1,
        max_gaussian_scale=0.1,
      ),
    )
  with pytest.raises(ValueError, match="integer labels"):
    generate_voxel_boxes(
      np.zeros((3, 3)),
      np.zeros(3),
      replace(config, min_points_per_voxel=1, min_component_voxels=1),
      semantic_labels=np.zeros(3, dtype=np.float64),
    )
  with pytest.raises(ValueError, match="shape"):
    generate_voxel_boxes(np.zeros((2, 2)), np.zeros(2), config)
  with pytest.raises(ValueError, match="known_free_points"):
    generate_voxel_boxes(
      np.zeros((2, 3)),
      np.zeros(2),
      config,
      known_free_points=np.zeros((2, 2)),
    )
  occupied_config = replace(
    config,
    lower_quantile=0,
    upper_quantile=1,
    min_points_per_voxel=1,
    min_component_voxels=1,
    min_box_volume=0,
  )
  with pytest.raises(ValueError, match="conflict with 1 occupied voxel"):
    generate_voxel_boxes(
      np.array([[0.05, 0.05, 0.05]]),
      np.zeros(1),
      occupied_config,
      known_free_points=np.array([[0.05, 0.05, 0.05]]),
    )
  with pytest.raises(ValueError, match="No finite Gaussian centers"):
    generate_voxel_boxes(np.full((2, 3), np.nan), np.zeros(2), config)


def test_voxel_grid_limit_fails_before_allocation() -> None:
  points = np.repeat(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), 3, axis=0)
  config = VoxelBoxConfig(
    voxel_size=0.01,
    lower_quantile=0,
    upper_quantile=1,
    max_grid_voxels=100,
  )

  with pytest.raises(ValueError, match="increase voxel_size or tighten bounds"):
    generate_voxel_boxes(points, np.zeros(len(points)), config)


def test_auto_mode_fits_sphere_instead_of_box() -> None:
  phi = np.linspace(0.1, np.pi - 0.1, 24)
  theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
  phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")
  radius = 0.3
  points = radius * np.column_stack(
    [
      np.sin(phi_grid).ravel() * np.cos(theta_grid).ravel(),
      np.sin(phi_grid).ravel() * np.sin(theta_grid).ravel(),
      np.cos(phi_grid).ravel(),
    ]
  )

  assert _fit_types(points) == ("sphere",)

  config = _auto_config()
  result = generate_voxel_boxes(points, np.zeros(len(points)), config)
  conservative = fit_collision_primitives(
    points,
    result,
    config,
    PrimitiveFitConfig(
      min_points=12,
      max_normalized_error=0.15,
      min_round_error_improvement=1.0,
    ),
  )

  assert len(conservative.primitives) == 1
  assert conservative.primitives[0].geom_type == "box"
  assert conservative.primitives[0].rejected_round_type == "sphere"
  assert len(conservative.primitive_source_ids) == len(points)


def test_auto_mode_distinguishes_cylinder_and_capsule() -> None:
  theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
  height = np.linspace(-0.8, 0.8, 33)
  theta_grid, height_grid = np.meshgrid(theta, height, indexing="ij")
  radius = 0.2
  cylinder_side = np.column_stack(
    [
      radius * np.cos(theta_grid).ravel(),
      radius * np.sin(theta_grid).ravel(),
      height_grid.ravel(),
    ]
  )
  cap_radius = np.linspace(0, radius, 8)
  cap_theta, cap_radius_grid = np.meshgrid(theta, cap_radius, indexing="ij")
  cap_xy = np.column_stack(
    [
      cap_radius_grid.ravel() * np.cos(cap_theta).ravel(),
      cap_radius_grid.ravel() * np.sin(cap_theta).ravel(),
    ]
  )
  cylinder_caps = np.concatenate(
    [
      np.column_stack([cap_xy, np.full(len(cap_xy), -0.8)]),
      np.column_stack([cap_xy, np.full(len(cap_xy), 0.8)]),
    ]
  )
  cylinder = np.concatenate([cylinder_side, cylinder_caps])

  segment = np.linspace(-0.5, 0.5, 21)
  side_theta, side_segment = np.meshgrid(theta, segment, indexing="ij")
  capsule_side = np.column_stack(
    [
      radius * np.cos(side_theta).ravel(),
      radius * np.sin(side_theta).ravel(),
      side_segment.ravel(),
    ]
  )
  polar = np.linspace(0, np.pi / 2, 12)
  hemi_theta, hemi_polar = np.meshgrid(theta, polar, indexing="ij")
  hemi_xy = np.column_stack(
    [
      radius * np.sin(hemi_polar).ravel() * np.cos(hemi_theta).ravel(),
      radius * np.sin(hemi_polar).ravel() * np.sin(hemi_theta).ravel(),
    ]
  )
  capsule_ends = np.concatenate(
    [
      np.column_stack([hemi_xy, 0.5 + radius * np.cos(hemi_polar).ravel()]),
      np.column_stack([hemi_xy, -0.5 - radius * np.cos(hemi_polar).ravel()]),
    ]
  )
  capsule = np.concatenate([capsule_side, capsule_ends])

  assert _fit_types(cylinder) == ("cylinder",)
  assert _fit_types(capsule) == ("capsule",)


def test_auto_mode_keeps_box_when_round_fit_does_not_improve_error() -> None:
  x = np.linspace(-0.8, 0.8, 33)
  edge = np.linspace(-0.2, 0.2, 17)
  x_grid, edge_grid = np.meshgrid(x, edge, indexing="ij")
  faces = []
  for fixed_axis in (1, 2):
    for sign in (-1, 1):
      coordinates = np.zeros((x_grid.size, 3))
      coordinates[:, 0] = x_grid.ravel()
      coordinates[:, fixed_axis] = sign * 0.2
      coordinates[:, 3 - fixed_axis] = edge_grid.ravel()
      faces.append(coordinates)
  points = np.concatenate(faces)

  assert _fit_types(points) == ("box",)


def test_auto_mode_serializes_primitive_types_and_source_mapping() -> None:
  phi = np.linspace(0.1, np.pi - 0.1, 16)
  theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
  phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")
  points = 0.3 * np.column_stack(
    [
      np.sin(phi_grid).ravel() * np.cos(theta_grid).ravel(),
      np.sin(phi_grid).ravel() * np.sin(theta_grid).ravel(),
      np.cos(phi_grid).ravel(),
    ]
  )
  config = _auto_config()
  result = generate_voxel_boxes(points, np.zeros(len(points)), config)
  fitted = fit_collision_primitives(points, result, config)

  xml = build_collision_mjcf(
    result, config, model_name="auto_collision", primitive_result=fitted
  )

  assert 'type="sphere"' in xml
  assert len(fitted.primitive_source_ids) == len(points)
  assert fitted.primitive_source_id_offsets.tolist() == [0, len(points)]


def test_mjcf_export_rejects_non_finite_or_degenerate_primitives() -> None:
  points = np.repeat(np.array([[0.05, 0.05, 0.05]]), 3, axis=0)
  config = VoxelBoxConfig(
    lower_quantile=0,
    upper_quantile=1,
    min_component_voxels=1,
    min_box_volume=0,
  )
  result = generate_voxel_boxes(points, np.zeros(len(points)), config)
  invalid = PrimitiveFitResult(
    primitives=(
      CollisionPrimitive(
        geom_type="sphere",
        center=np.zeros(3),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        size=(0.0,),
        normalized_error=0.0,
        source_candidate_index=0,
      ),
    ),
    primitive_source_id_offsets=np.array([0, 3]),
    primitive_source_ids=np.arange(3),
  )

  with pytest.raises(ValueError, match="invalid pose or size"):
    build_collision_mjcf(
      result,
      config,
      model_name="invalid_collision",
      primitive_result=invalid,
    )


def test_debug_colors_distinguish_primitive_types_without_changing_physics() -> None:
  points = np.array(
    [
      [0.0, 0.0, 0.1],
      [0.1, 0.0, 0.1],
      [0.0, 0.1, 0.1],
      [0.1, 0.1, 0.1],
    ]
  )
  config = VoxelBoxConfig(
    floor_z=0,
    lower_quantile=0,
    upper_quantile=1,
    min_points_per_voxel=1,
    min_component_voxels=1,
    min_box_volume=0,
  )
  result = generate_voxel_boxes(points, np.zeros(len(points)), config)
  fitted = fit_collision_primitives(
    points,
    result,
    config,
    PrimitiveFitConfig(min_points=4),
  )

  normal_xml = build_collision_mjcf(
    result, config, model_name="normal", primitive_result=fitted
  )
  debug_xml = build_collision_mjcf(
    result,
    config,
    model_name="debug",
    primitive_result=fitted,
    debug_color_by_type=True,
  )

  assert 'rgba="0.50 0.50 0.50 0.12"' not in normal_xml
  assert 'rgba="0.50 0.50 0.50 0.12"' in debug_xml
  assert 'rgba="0.15 0.75 0.25 0.18"' in debug_xml
  assert 'contype="1" conaffinity="1"' in normal_xml
  assert 'contype="1" conaffinity="1"' in debug_xml


def test_collision_benchmark_reports_physics_and_contact_cost(tmp_path) -> None:
  collision_mjcf = tmp_path / "collision.xml"
  collision_mjcf.write_text(
    """<mujoco model="benchmark">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
  </worldbody>
</mujoco>
"""
  )

  report = benchmark_collision_mjcf(
    collision_mjcf,
    warmup_steps=1,
    measured_steps=5,
    probe_grid=1,
    probe_spacing=0.4,
    probe_z=0.08,
    probe_radius=0.08,
  )

  model = cast(dict[str, int], report["model"])
  assert model["nbody"] == 2
  assert model["ngeom"] == 2
  assert cast(float, report["physics_steps_per_second"]) > 0
  assert report["contacts"] == {"mean": 1.0, "max": 1}

  with pytest.raises(ValueError, match="measured_steps"):
    benchmark_collision_mjcf(
      collision_mjcf,
      warmup_steps=0,
      measured_steps=0,
      probe_grid=1,
      probe_spacing=0.4,
      probe_z=0.08,
      probe_radius=0.08,
    )


def test_collision_benchmark_opt_in_accepts_only_bundle_local_meshes(tmp_path) -> None:
  mesh_dir = tmp_path / "meshes"
  mesh_dir.mkdir()
  mesh_path = mesh_dir / "tetra.obj"
  mesh_path.write_text(
    "v 0 0 0\nv 0.2 0 0\nv 0 0.2 0\nv 0 0 0.2\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n"
  )
  collision_mjcf = tmp_path / "collision.xml"
  collision_mjcf.write_text(
    """<mujoco model="mesh_benchmark">
  <compiler meshdir="meshes"/>
  <asset><mesh name="tetra" file="tetra.obj"/></asset>
  <worldbody><geom type="mesh" mesh="tetra"/></worldbody>
</mujoco>
"""
  )

  with pytest.raises(ValueError, match="allow_file_assets"):
    benchmark_collision_mjcf(
      collision_mjcf,
      warmup_steps=0,
      measured_steps=2,
      probe_grid=1,
      probe_spacing=0.4,
      probe_z=0.4,
      probe_radius=0.05,
    )
  report = benchmark_collision_mjcf(
    collision_mjcf,
    warmup_steps=0,
    measured_steps=2,
    probe_grid=1,
    probe_spacing=0.4,
    probe_z=0.4,
    probe_radius=0.05,
    allow_file_assets=True,
  )

  dependencies = cast(list[dict[str, str]], report["dependencies"])
  assert len(dependencies) == 1
  assert dependencies[0]["path"] == str(mesh_path)
  assert len(dependencies[0]["sha256"]) == 64


def test_component_convex_hulls_report_concavity_inflation() -> None:
  occupied = np.zeros((4, 4, 3), dtype=bool)
  occupied[1:3, 1, 1] = True
  occupied[1, 2, 1] = True
  mesh = occupancy_surface_mesh(occupied, np.zeros(3), 0.1)

  hulls, diagnostics = convex_component_hulls(mesh)

  assert len(hulls) == 1
  assert len(diagnostics) == 1
  assert diagnostics[0].source_watertight
  assert diagnostics[0].source_volume == pytest.approx(0.003)
  assert diagnostics[0].hull_volume > cast(float, diagnostics[0].source_volume)
  assert cast(float, diagnostics[0].volume_ratio) > 1


def test_occupancy_and_tsdf_surface_reconstruction_are_watertight() -> None:
  occupied = np.zeros((5, 5, 5), dtype=bool)
  occupied[1:4, 1:4, 1:4] = True
  lower = np.array([-0.25, -0.25, -0.25])

  occupancy_mesh = occupancy_surface_mesh(occupied, lower, 0.1)
  _, occupancy_diagnostics = repair_surface_mesh(occupancy_mesh)
  tsdf = truncated_signed_distance(occupied, 0.1, 0.3)
  tsdf_mesh = marching_tetrahedra_surface(
    tsdf,
    lower + 0.05,
    0.1,
  )
  _, tsdf_diagnostics = repair_surface_mesh(tsdf_mesh)

  assert occupancy_diagnostics.watertight
  assert occupancy_diagnostics.winding_consistent
  assert occupancy_diagnostics.boundary_edge_count == 0
  assert occupancy_diagnostics.nonmanifold_edge_count == 0
  assert tsdf_diagnostics.watertight
  assert tsdf_diagnostics.winding_consistent
  assert tsdf_diagnostics.boundary_edge_count == 0
  assert tsdf_diagnostics.nonmanifold_edge_count == 0
  np.testing.assert_allclose(
    occupancy_mesh.vertices.mean(axis=0),
    tsdf_mesh.vertices.mean(axis=0),
    atol=1e-12,
  )


def test_mesh_repair_and_vertex_clustering_remove_invalid_faces() -> None:
  vertices = np.array(
    [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 1.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0],
    ]
  )
  faces = np.array([[0, 1, 2], [0, 2, 3], [4, 1, 2], [0, 0, 1]])

  repaired, diagnostics = repair_surface_mesh(SurfaceMesh(vertices, faces))
  simplified = simplify_vertex_clusters(repaired, 0.4)

  assert diagnostics.degenerate_face_count == 1
  assert len(repaired.faces) == 2
  assert len(simplified.faces) == 2
  assert np.isfinite(simplified.vertices).all()


def test_heightfield_reports_overhangs_and_only_fills_bounded_holes(tmp_path) -> None:
  occupied = np.zeros((5, 5, 6), dtype=bool)
  occupied[:, :, 0] = True
  occupied[2, 2, 4] = True
  occupied[0, 0, :] = False

  surface = heightfield_surface(
    occupied,
    np.zeros(3),
    0.1,
    max_fill_distance=0.11,
  )

  assert surface.observed.all()
  assert surface.heights[0, 0] == pytest.approx(0.1)
  assert surface.heights[2, 2] == pytest.approx(0.5)
  assert surface.ambiguous_column_count == 1
  lowest_surface = heightfield_surface(
    occupied,
    np.zeros(3),
    0.1,
    max_fill_distance=0.11,
    surface_mode="lowest",
  )
  assert lowest_surface.heights[2, 2] == pytest.approx(0.1)
  xml = build_heightfield_mjcf(surface, model_name="heightfield_test")
  model = mujoco.MjModel.from_xml_string(xml)
  assert model.nhfield == 1
  assert model.ngeom == 1
  heightfield_path = tmp_path / "heightfield.xml"
  heightfield_path.write_text(xml)
  benchmark = benchmark_collision_mjcf(
    heightfield_path,
    warmup_steps=1,
    measured_steps=2,
    probe_grid=1,
    probe_spacing=0.4,
    probe_z=0.2,
    probe_radius=0.08,
  )
  assert cast(dict[str, int], benchmark["model"])["ngeom"] == 2

  obj = surface_mesh_obj(
    SurfaceMesh(
      vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
      faces=np.array([[0, 1, 2]]),
    )
  )
  assert obj.endswith("f 1 2 3\n")
