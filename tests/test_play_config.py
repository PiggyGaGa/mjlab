"""Tests for play-specific command-line configuration."""

from pathlib import Path

import pytest

from mjlab.scripts.play import _override_gaussian_splat_file
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
