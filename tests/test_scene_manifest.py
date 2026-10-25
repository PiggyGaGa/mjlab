"""Tests for versioned 3DGS visual/collision scene manifests."""

import hashlib
import json
from pathlib import Path

import pytest

from mjlab.scene import (
  GaussianSplatAssetCfg,
  GaussianSplatSceneCfg,
  SceneCfg,
  apply_scene_manifest,
  load_scene_manifest,
)
from mjlab.terrains import TerrainEntityCfg

_ROOM1_MANIFEST = (
  Path(__file__).parents[1] / "src/mjlab/asset_zoo/3dgs/room1/manifest.json"
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest_bundle(tmp_path: Path) -> Path:
  visual = tmp_path / "scene.ply"
  collision = tmp_path / "collision.xml"
  dependency = tmp_path / "collision.obj"
  visual.write_bytes(b"test splat payload")
  collision.write_text(
    "<mujoco><asset><mesh name='collision' file='collision.obj'/></asset>"
    "<worldbody><geom type='mesh' mesh='collision'/></worldbody></mujoco>"
  )
  dependency.write_text("v 0 0 0\n")
  manifest = {
    "schema_version": 1,
    "coordinate_system": {
      "handedness": "right",
      "up_axis": "z",
      "length_unit": "meter",
    },
    "visual": {
      "type": "3dgs_ply",
      "path": visual.name,
      "sha256": _sha256(visual),
    },
    "collision": {
      "type": "mjcf",
      "path": collision.name,
      "sha256": _sha256(collision),
      "geom_group": 4,
      "dependencies": [{"path": dependency.name, "sha256": _sha256(dependency)}],
    },
    "world_from_scene": {
      "position": [1.0, 2.0, 3.0],
      "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    },
    "spawn_points": [
      {
        "name": "default",
        "position": [0.5, 0.0, 0.0],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
      }
    ],
  }
  path = tmp_path / "manifest.json"
  path.write_text(json.dumps(manifest))
  return path


def test_load_scene_manifest_resolves_and_verifies_assets(tmp_path: Path) -> None:
  path = _write_manifest_bundle(tmp_path)

  manifest = load_scene_manifest(path)

  assert manifest.visual.path == (tmp_path / "scene.ply").resolve()
  assert manifest.collision.path == (tmp_path / "collision.xml").resolve()
  assert manifest.collision.geom_group == 4
  assert (
    manifest.collision.dependencies[0].path == (tmp_path / "collision.obj").resolve()
  )
  assert manifest.world_position == (1.0, 2.0, 3.0)
  assert manifest.spawn_points[0].name == "default"


def test_apply_scene_manifest_uses_one_transform_for_visual_and_collision(
  tmp_path: Path,
) -> None:
  path = _write_manifest_bundle(tmp_path)
  scene_cfg = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    gaussian_splats=GaussianSplatSceneCfg(
      assets=(GaussianSplatAssetCfg(path="old.ply", opacity_threshold=0.25),)
    ),
  )

  apply_scene_manifest(scene_cfg, path)

  assert scene_cfg.gaussian_splats is not None
  visual = scene_cfg.gaussian_splats.assets[0]
  assert visual.path == str((tmp_path / "scene.ply").resolve())
  assert visual.pos == (1.0, 2.0, 3.0)
  assert visual.opacity_threshold == 0.25
  assert scene_cfg.terrain is not None
  assert scene_cfg.terrain.terrain_type == "mjcf"
  assert scene_cfg.terrain.mjcf_pos == visual.pos
  assert scene_cfg.terrain.mjcf_quat == visual.quat
  assert scene_cfg.terrain.mjcf_geom_group == 4
  assert not scene_cfg.terrain.mjcf_keep_plane
  assert scene_cfg.spawn_position == (1.5, 2.0, 3.0)
  assert scene_cfg.spawn_quaternion_wxyz == (1.0, 0.0, 0.0, 0.0)


def test_scene_manifest_requires_every_mjcf_dependency(tmp_path: Path) -> None:
  path = _write_manifest_bundle(tmp_path)
  data = json.loads(path.read_text())
  data["collision"]["dependencies"] = []
  path.write_text(json.dumps(data))

  with pytest.raises(ValueError, match="missing files.*collision.obj"):
    load_scene_manifest(path)


def test_scene_manifest_rejects_unused_mjcf_dependency(tmp_path: Path) -> None:
  path = _write_manifest_bundle(tmp_path)
  collision = tmp_path / "collision.xml"
  collision.write_text(
    "<mujoco><worldbody><geom type='box' size='1 1 .1'/></worldbody></mujoco>"
  )
  data = json.loads(path.read_text())
  data["collision"]["sha256"] = _sha256(collision)
  path.write_text(json.dumps(data))

  with pytest.raises(ValueError, match="not referenced.*collision.obj"):
    load_scene_manifest(path)


def test_scene_manifest_resolves_compiler_mesh_directory(tmp_path: Path) -> None:
  path = _write_manifest_bundle(tmp_path)
  meshes = tmp_path / "meshes"
  meshes.mkdir()
  dependency = tmp_path / "collision.obj"
  nested_dependency = meshes / dependency.name
  dependency.rename(nested_dependency)
  collision = tmp_path / "collision.xml"
  collision.write_text(
    "<mujoco><compiler meshdir='meshes'/><asset>"
    "<mesh name='collision' file='collision.obj'/></asset>"
    "<worldbody><geom type='mesh' mesh='collision'/></worldbody></mujoco>"
  )
  data = json.loads(path.read_text())
  data["collision"]["sha256"] = _sha256(collision)
  data["collision"]["dependencies"] = [
    {"path": "meshes/collision.obj", "sha256": _sha256(nested_dependency)}
  ]
  path.write_text(json.dumps(data))

  manifest = load_scene_manifest(path)

  assert manifest.collision.dependencies[0].path == nested_dependency.resolve()


def test_apply_scene_manifest_selects_and_transforms_spawn(tmp_path: Path) -> None:
  path = _write_manifest_bundle(tmp_path)
  data = json.loads(path.read_text())
  data["world_from_scene"] = {
    "position": [10.0, 20.0, 0.0],
    "quaternion_wxyz": [2**-0.5, 0.0, 0.0, 2**-0.5],
  }
  data["spawn_points"].append(
    {
      "name": "door",
      "position": [2.0, 0.0, 0.5],
      "quaternion_wxyz": [2**-0.5, 0.0, 0.0, 2**-0.5],
    }
  )
  path.write_text(json.dumps(data))
  scene_cfg = SceneCfg()

  apply_scene_manifest(scene_cfg, path, spawn_point="door")

  assert scene_cfg.spawn_position == pytest.approx((10.0, 22.0, 0.5))
  assert scene_cfg.spawn_quaternion_wxyz == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_apply_scene_manifest_rejects_unknown_spawn(tmp_path: Path) -> None:
  path = _write_manifest_bundle(tmp_path)

  with pytest.raises(ValueError, match="Unknown scene spawn point.*missing"):
    apply_scene_manifest(SceneCfg(), path, spawn_point="missing")


def test_scene_manifest_rejects_hash_mismatch(tmp_path: Path) -> None:
  path = _write_manifest_bundle(tmp_path)
  (tmp_path / "scene.ply").write_bytes(b"modified")

  with pytest.raises(ValueError, match="SHA-256 mismatch"):
    load_scene_manifest(path)


def test_scene_manifest_rejects_asset_outside_bundle(tmp_path: Path) -> None:
  path = _write_manifest_bundle(tmp_path)
  data = json.loads(path.read_text())
  data["visual"]["path"] = "../outside.ply"
  path.write_text(json.dumps(data))

  with pytest.raises(ValueError, match="must stay inside"):
    load_scene_manifest(path, verify_hashes=False)


def test_room1_scene_manifest_is_current() -> None:
  manifest = load_scene_manifest(_ROOM1_MANIFEST)

  assert manifest.schema_version == 1
  assert manifest.visual.path.name == "point_cloud_2_transform.ply"
  assert manifest.collision.path.name == "collision.xml"
