"""Horizontal consolidation preserves finite holes, levels and source mappings."""

from dataclasses import replace

import mujoco
import numpy as np

from mjlab.asset_zoo.gaussian_collision import (
  CollisionPrimitive,
  PrimitiveFitResult,
  VoxelBoxConfig,
  build_collision_mjcf,
  generate_voxel_boxes,
)
from mjlab.asset_zoo.gaussian_collision_surfaces import (
  HorizontalSurfaceConfig,
  consolidate_horizontal_surfaces,
)


def _patches(rectangles: list[tuple[int, int, int, int, float]]):
  config = VoxelBoxConfig(
    voxel_size=0.1,
    lower_quantile=0,
    upper_quantile=1,
    min_points_per_voxel=1,
    min_component_voxels=1,
    min_box_volume=0,
  )
  groups = [
    np.array(
      [
        ((horizontal + offset_x) * 0.1, (vertical + offset_y) * 0.1, height)
        for horizontal in range(start_x, end_x)
        for vertical in range(start_y, end_y)
        for offset_x in (0.1, 0.35, 0.65, 0.9)
        for offset_y in (0.1, 0.35, 0.65, 0.9)
      ]
    )
    for start_x, end_x, start_y, end_y, height in rectangles
  ]
  points = np.concatenate(groups)
  result = generate_voxel_boxes(points, np.zeros(len(points)), config)
  candidate_boxes = []
  primitives = []
  offsets = np.cumsum([0, *(len(group) for group in groups)])
  for index, (start_x, end_x, start_y, end_y, height) in enumerate(rectangles):
    lower = np.rint(
      (np.array([start_x * 0.1, start_y * 0.1, height]) - result.lower) / 0.1
    ).astype(int)
    upper = lower + np.array([end_x - start_x, end_y - start_y, 1])
    candidate_boxes.append((lower, upper))
    primitives.append(
      CollisionPrimitive(
        geom_type="box",
        center=np.array([(start_x + end_x) * 0.05, (start_y + end_y) * 0.05, height]),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        size=((end_x - start_x) * 0.05, (end_y - start_y) * 0.05, 0.01),
        normalized_error=0,
        source_candidate_index=index,
      )
    )
  result = replace(
    result,
    candidate_boxes=tuple(candidate_boxes),
    candidate_source_id_offsets=offsets,
    candidate_source_ids=np.arange(len(points)),
  )
  fitted = PrimitiveFitResult(
    primitives=tuple(primitives),
    primitive_source_id_offsets=offsets,
    primitive_source_ids=np.arange(len(points)),
  )
  return points, result, fitted, config


def test_adjacent_horizontal_patches_share_one_top_and_preserve_mapping() -> None:
  points, result, fitted, voxel = _patches([(0, 3, 0, 3, 0), (3, 6, 0, 3, 0.01)])
  merged = consolidate_horizontal_surfaces(
    points, result, voxel, fitted, HorizontalSurfaceConfig()
  )
  assert len(merged.primitives) == 1
  assert np.isclose(merged.primitives[0].horizontal_surface_level, 0.015)
  assert merged.primitives[0].source_candidate_indices == (0, 1)
  assert np.isclose(
    merged.primitives[0].center[2] + merged.primitives[0].size[2], 0.015
  )
  np.testing.assert_array_equal(
    np.sort(merged.primitive_source_ids), np.arange(len(points))
  )
  repeated = consolidate_horizontal_surfaces(
    points, result, voxel, fitted, HorizontalSurfaceConfig()
  )
  assert build_collision_mjcf(
    result, voxel, model_name="test", primitive_result=merged
  ) == build_collision_mjcf(result, voxel, model_name="test", primitive_result=repeated)
  mujoco.MjModel.from_xml_string(
    build_collision_mjcf(result, voxel, model_name="test", primitive_result=merged)
  )


def test_horizontal_union_preserves_pit_and_outer_boundary() -> None:
  rectangles = [
    (horizontal * 3, horizontal * 3 + 3, vertical * 3, vertical * 3 + 3, 0.0)
    for horizontal in range(3)
    for vertical in range(3)
    if (horizontal, vertical) != (1, 1)
  ]
  points, result, fitted, voxel = _patches(rectangles)
  merged = consolidate_horizontal_surfaces(
    points, result, voxel, fitted, HorizontalSurfaceConfig()
  )
  assert len(merged.primitives) < len(fitted.primitives)
  for primitive in merged.primitives:
    assert np.isclose(primitive.center[2] + primitive.size[2], 0.01)
    for position in ((0.45, 0.45, -0.02), (0.95, 0.45, -0.02)):
      assert not (
        np.abs(np.array(position) - primitive.center) < np.array(primitive.size)
      ).all()
  np.testing.assert_array_equal(
    np.sort(merged.primitive_source_ids), np.arange(len(points))
  )


def test_horizontal_levels_and_known_free_constraints_are_not_overridden() -> None:
  points, result, fitted, voxel = _patches([(0, 3, 0, 3, 0), (3, 6, 0, 3, 0.1)])
  unchanged = consolidate_horizontal_surfaces(
    points, result, voxel, fitted, HorizontalSurfaceConfig()
  )
  assert all(
    primitive.horizontal_surface_level is None for primitive in unchanged.primitives
  )
  points, result, fitted, voxel = _patches([(0, 3, 0, 3, 0), (3, 6, 0, 3, 0.01)])
  constrained = replace(result, known_free_points=np.array([[0.15, 0.15, -0.07]]))
  unchanged = consolidate_horizontal_surfaces(
    points, constrained, voxel, fitted, HorizontalSurfaceConfig()
  )
  assert all(
    primitive.horizontal_surface_level is None for primitive in unchanged.primitives
  )
  distinct_labels = replace(result, candidate_semantic_labels=np.array([1, 2]))
  unchanged = consolidate_horizontal_surfaces(
    points, distinct_labels, voxel, fitted, HorizontalSurfaceConfig()
  )
  assert all(
    primitive.horizontal_surface_level is None for primitive in unchanged.primitives
  )


def test_horizontal_consolidation_does_not_flatten_a_ramp() -> None:
  points, result, fitted, voxel = _patches([(0, 3, 0, 3, 0), (3, 6, 0, 3, 0)])
  points[:, 2] = points[:, 0] * 0.25
  unchanged = consolidate_horizontal_surfaces(
    points, result, voxel, fitted, HorizontalSurfaceConfig()
  )
  assert all(
    primitive.horizontal_surface_level is None for primitive in unchanged.primitives
  )
