"""Regression checks for bounded merging of apartment collision surfaces."""

from importlib import import_module

import numpy as np
from scipy import ndimage

_generator = import_module(
  "mjlab.asset_zoo.3dgs.apartment.collision.generate_collision"
)


def _merged_mask(source: np.ndarray) -> tuple[np.ndarray, int]:
  boxes = _generator.merge_surface_boxes(source)
  result = np.zeros_like(source)
  for lower, upper in boxes:
    region = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper, strict=True))
    result[region] = True
  return result, len(boxes)


def test_wall_scan_holes_merge_into_one_box() -> None:
  source = np.ones((1, 30, 30), dtype=bool)
  source[0, 5::6, 5::6] = False

  result, count = _merged_mask(source)

  assert count == 1
  assert result.all()


def test_surface_merging_preserves_doorway_and_coverage() -> None:
  source = np.zeros((3, 30, 30), dtype=bool)
  source[1] = True
  source[1, 10:20, :22] = False  # One-metre-wide, 2.2-metre-high doorway.
  source[1, 4, 4] = False

  result, _ = _merged_mask(source)

  assert result[source].all()
  assert not result[:, 15, :18].any()
  distances = ndimage.distance_transform_edt(~source) * _generator.VOXEL_SIZE
  assert distances[result].max() <= _generator.MAX_SURFACE_EXPANSION + 1e-9


def test_empty_surface_has_no_boxes() -> None:
  assert _generator.merge_surface_boxes(np.zeros((3, 3, 3), dtype=bool)) == []


def test_pruning_removes_fragments_but_preserves_thin_wall_at_threshold() -> None:
  origin = np.zeros(3, dtype=int)
  fragment = (origin, np.array([1, 9, 11]))  # 0.099 m³.
  thin_wall = (origin, np.array([1, 10, 10]))  # Exactly 0.1 m³.
  large_wall = (origin, np.array([1, 30, 30]))

  result = _generator.prune_small_surface_boxes([fragment, thin_wall, large_wall])

  assert len(result) == 2
  assert all(np.array_equal(a, b) for a, b in zip(result[0], thin_wall, strict=True))
  assert all(np.array_equal(a, b) for a, b in zip(result[1], large_wall, strict=True))
