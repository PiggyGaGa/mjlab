"""Tests for play-specific command-line configuration."""

from pathlib import Path

import pytest

from mjlab.scripts.play import (
  PlayConfig,
  _apply_play_scene_inputs,
  _enable_collision_visibility,
  _override_collision_mjcf_file,
  _override_gaussian_splat_file,
)
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.g1_3dgs import (
  G1_3DGS_VIDEO_SENSOR,
  enable_g1_3dgs_video_sensor,
)
from mjlab.tasks.registry import load_env_cfg


def test_override_gaussian_splat_file(tmp_path: Path) -> None:
  ply_path = tmp_path / "scene.ply"
  ply_path.touch()
  cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-3DGS", play=True)

  _override_gaussian_splat_file(cfg, str(ply_path))

  assert cfg.scene.gaussian_splats is not None
  assert cfg.scene.gaussian_splats.assets[0].path == str(ply_path)


def test_override_gaussian_splat_file_rejects_non_3dgs_task(
  tmp_path: Path,
) -> None:
  ply_path = tmp_path / "scene.ply"
  ply_path.touch()
  cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=True)

  with pytest.raises(ValueError, match="requires a task"):
    _override_gaussian_splat_file(cfg, str(ply_path))


def test_override_gaussian_splat_file_requires_existing_path() -> None:
  cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-3DGS", play=True)

  with pytest.raises(FileNotFoundError, match="PLY not found"):
    _override_gaussian_splat_file(cfg, "/does/not/exist.ply")


def test_override_collision_mjcf_file_replaces_plane(tmp_path: Path) -> None:
  collision_path = tmp_path / "collision.xml"
  collision_path.write_text("<mujoco/>")
  cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-3DGS", play=True)

  _override_collision_mjcf_file(cfg, str(collision_path), keep_plane=True)

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "mjcf"
  assert cfg.scene.terrain.mjcf_path == str(collision_path)
  assert cfg.scene.terrain.mjcf_keep_plane
  assert cfg.scene.terrain.terrain_generator is None


def test_override_collision_mjcf_file_requires_existing_path() -> None:
  cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-3DGS", play=True)

  with pytest.raises(FileNotFoundError, match="Collision MJCF not found"):
    _override_collision_mjcf_file(cfg, "/does/not/exist.xml")


def test_collision_play_options_default_to_safe_values() -> None:
  cfg = PlayConfig()

  assert cfg.collision_mjcf_file is None
  assert not cfg.collision_keep_plane
  assert not cfg.collision_visible
  assert cfg.scene_manifest is None
  assert cfg.scene_spawn_point is None


def test_scene_manifest_and_split_asset_options_are_mutually_exclusive() -> None:
  env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-3DGS", play=True)
  cfg = PlayConfig(scene_manifest="scene.json", gaussian_splat_file="scene.ply")

  with pytest.raises(ValueError, match="cannot be combined"):
    _apply_play_scene_inputs(env_cfg, cfg)


def test_scene_spawn_point_requires_manifest() -> None:
  env_cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1-3DGS", play=True)

  with pytest.raises(ValueError, match="scene-spawn-point requires"):
    _apply_play_scene_inputs(env_cfg, PlayConfig(scene_spawn_point="door"))


def test_enable_collision_visibility_only_enables_group_three() -> None:
  cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-3DGS", play=True)
  original = cfg.viewer.geom_group

  _enable_collision_visibility(cfg)

  assert cfg.viewer.geom_group[3] == 1
  assert cfg.viewer.geom_group[:3] == original[:3]
  assert cfg.viewer.geom_group[4:] == original[4:]


def test_g1_3dgs_video_sensor_is_enabled_on_demand() -> None:
  cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-3DGS", play=True)

  assert not any(
    isinstance(sensor, CameraSensorCfg) for sensor in cfg.scene.sensors or ()
  )

  enable_g1_3dgs_video_sensor(cfg)

  camera = next(
    sensor
    for sensor in cfg.scene.sensors or ()
    if isinstance(sensor, CameraSensorCfg) and sensor.name == G1_3DGS_VIDEO_SENSOR
  )
  assert camera.camera_name == "robot/tracking"
  assert camera.enabled_geom_groups == (2,)

  enable_g1_3dgs_video_sensor(cfg)
  assert (
    sum(sensor.name == G1_3DGS_VIDEO_SENSOR for sensor in cfg.scene.sensors or ()) == 1
  )
