"""Configuration tests for the G1 3DGS depth-training task."""

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.g1_3dgs import G1_3DGS_DEPTH_SENSOR
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


def test_g1_3dgs_depth_task_uses_yam_vision_pipeline() -> None:
  cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth")

  assert cfg.scene.gaussian_splats is not None
  camera = next(
    sensor for sensor in cfg.scene.sensors or () if sensor.name == G1_3DGS_DEPTH_SENSOR
  )
  assert isinstance(camera, CameraSensorCfg)
  assert camera.parent_body == "robot/torso_link"
  assert camera.data_types == ("depth",)
  assert (camera.height, camera.width) == (48, 64)
  assert camera.enabled_geom_groups == (2,)

  camera_obs = cfg.observations["camera"]
  assert tuple(camera_obs.terms) == ("depth",)
  assert camera_obs.concatenate_terms
  assert camera_obs.nan_policy == "error"
  assert "foot_height" not in cfg.observations["critic"].terms
  assert "foot_clearance" not in cfg.rewards
  assert "foot_swing_height" not in cfg.rewards
  assert not any(
    sensor.name == "foot_height_scan" for sensor in cfg.scene.sensors or ()
  )


def test_g1_3dgs_depth_runner_uses_camera_cnn() -> None:
  cfg = load_rl_cfg("Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth")

  assert isinstance(cfg, RslRlOnPolicyRunnerCfg)
  assert cfg.obs_groups == {
    "actor": ("actor", "camera"),
    "critic": ("critic", "camera"),
  }
  assert cfg.actor.class_name.endswith(":SpatialSoftmaxCNNModel")
  assert cfg.actor.cnn_cfg is not None
  assert cfg.critic.class_name.endswith(":SpatialSoftmaxCNNModel")
  assert cfg.critic.cnn_cfg is not None
  assert cfg.experiment_name == "g1_velocity_3dgs_depth"


def test_proprioceptive_g1_task_does_not_create_camera_workload() -> None:
  cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")

  assert cfg.scene.gaussian_splats is None
  assert "camera" not in cfg.observations
  assert not any(
    isinstance(sensor, CameraSensorCfg) for sensor in cfg.scene.sensors or ()
  )


@pytest.mark.slow
def test_g1_3dgs_depth_observation_runtime(device: str) -> None:
  cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth")
  cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    observations, _ = env.reset()
    depth = observations["camera"]
    assert isinstance(depth, torch.Tensor)
    assert depth.shape == (1, 1, 48, 64)
    assert depth.dtype == torch.float32
    assert torch.isfinite(depth).all()
    assert torch.all((depth >= 0.0) & (depth <= 1.0))
    assert depth.max() > depth.min()
  finally:
    env.close()
