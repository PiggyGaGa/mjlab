"""Build approximate static box collisions from apartment Gaussian centres.

Run from the repository root with uv run and this script's path.
"""

import json
from pathlib import Path
from typing import cast
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from mjlab.asset_zoo.gaussian_collision import (
  VoxelBoxConfig,
)
from mjlab.asset_zoo.gaussian_collision import (
  merge_surface_boxes as merge_voxel_boxes,
)
from mjlab.scene.gaussian_splat import _read_ply_vertices

VOXEL_SIZE = 0.1
FLOOR_Z = 0.0
MIN_BOX_FILL = 0.20
MAX_SURFACE_EXPANSION = 0.4
MIN_SURFACE_BOX_VOLUME = 0.1  # Cubic metres, using full box dimensions.
VISIBLE_PILLARS = (
  ((1.6, -0.45, 0.8), (0.90, 0.20, 0.10, 1.0)),
  ((1.6, 0.45, 0.8), (0.10, 0.40, 0.90, 1.0)),
)


def merge_surface_boxes(occupied: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
  """Partition surfaces into boxes, allowing bounded filling of scan gaps.

  Bounds use voxel indices with exclusive upper endpoints. Every original voxel
  remains covered. Added cells must be within MAX_SURFACE_EXPANSION of the source
  occupancy; the density threshold prevents bounding whole sparse structures.
  """
  config = VoxelBoxConfig(
    voxel_size=VOXEL_SIZE,
    min_box_fill=MIN_BOX_FILL,
    max_surface_expansion=MAX_SURFACE_EXPANSION,
    min_box_volume=MIN_SURFACE_BOX_VOLUME,
  )
  return merge_voxel_boxes(occupied, config)


def prune_small_surface_boxes(
  boxes: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
  """Discard small merged fragments; retained boxes keep their original bounds."""
  return [
    (start, end)
    for start, end in boxes
    if np.prod(end - start) * VOXEL_SIZE**3 >= MIN_SURFACE_BOX_VOLUME - 1e-12
  ]


def main() -> None:
  root = Path(__file__).resolve().parents[1]
  manifest_path = root / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  ply_path = root / "scene_transform_p5.ply"
  vertices = _read_ply_vertices(ply_path)
  points = np.column_stack([vertices[axis] for axis in ("x", "y", "z")])
  # The source stores opacity as logits: zero corresponds to alpha=0.5.
  points = points[np.isfinite(points).all(axis=1) & (vertices["opacity"] >= 0)]
  lower, upper = np.quantile(points, [0.001, 0.999], axis=0)
  lower = (np.floor(lower / VOXEL_SIZE) - 1) * VOXEL_SIZE
  upper = (np.ceil(upper / VOXEL_SIZE) + 1) * VOXEL_SIZE
  # A separate solid slab represents the floor; discard its noisy surface points.
  lower[2] = FLOOR_Z + VOXEL_SIZE
  shape = np.rint((upper - lower) / VOXEL_SIZE).astype(int)
  indices = np.floor((points - lower) / VOXEL_SIZE).astype(int)
  indices = indices[((indices >= 0) & (indices < shape)).all(axis=1)]
  counts = np.zeros(tuple(shape), dtype=np.int32)
  np.add.at(counts, tuple(indices.T), 1)
  occupied = counts >= 3
  labels, _ = cast(tuple[NDArray[np.int32], int], ndimage.label(occupied))
  component_sizes = np.bincount(labels.ravel())
  occupied &= component_sizes[labels] >= 8
  voxel_count = int(occupied.sum())

  model = ET.Element("mujoco", model="apartment_collision")
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
  world.append(ET.Comment("Merged surface boxes; <=0.4 m expansion. See README.md."))

  def add_box(name: str, centre: np.ndarray, half_size: np.ndarray) -> None:
    ET.SubElement(
      world,
      "geom",
      name=name,
      type="box",
      pos=" ".join(f"{value:.4f}" for value in centre),
      size=" ".join(f"{value:.4f}" for value in half_size),
    )

  centre = (lower + upper) / 2
  centre[2] = FLOOR_Z - 0.05
  half_size = (upper - lower) / 2
  half_size[2] = 0.05
  add_box("ground", centre, half_size)

  world.append(
    ET.Comment("Opaque camera-visible obstacles with matching collision geometry.")
  )
  for index, (position, rgba) in enumerate(VISIBLE_PILLARS):
    ET.SubElement(
      world,
      "geom",
      name=f"pillar_{index:02d}",
      type="cylinder",
      pos=" ".join(f"{value:.4f}" for value in position),
      size="0.1800 0.8000",
      group="1",
      contype="1",
      conaffinity="1",
      condim="3",
      friction="1.0 0.005 0.0001",
      rgba=" ".join(f"{value:.2f}" for value in rgba),
    )

  merged = merge_surface_boxes(occupied)
  boxes = prune_small_surface_boxes(merged)
  for box_count, (start, end) in enumerate(boxes):
    centre = lower + (start + end) * VOXEL_SIZE / 2
    half_size = (end - start) * VOXEL_SIZE / 2
    add_box(f"surface_{box_count:04d}", centre, half_size)

  xml_path = root / "collision/collision.xml"
  ET.indent(model, space="  ")
  xml_path.write_text(ET.tostring(model, encoding="unicode") + "\n")
  manifest["visual"]["path"] = ply_path.name
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  print(f"Bounds: {lower} to {upper}; floor z={FLOOR_Z}")
  print(f"Merged {voxel_count} occupied voxels into {len(boxes)} surface boxes")
  print(f"Discarded {len(merged) - len(boxes)} boxes below {MIN_SURFACE_BOX_VOLUME} m³")


if __name__ == "__main__":
  main()
