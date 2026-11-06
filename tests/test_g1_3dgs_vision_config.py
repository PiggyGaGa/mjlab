"""Configuration tests for the G1 3DGS depth-training task."""

import pytest
import torch
from conftest import get_test_device
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel
from mjlab.sensor import CameraSensor, CameraSensorCfg
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
  assert camera_obs.terms["depth"].params["zero_depth_as_far"] is True
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


def test_validity_task_keeps_baseline_and_ppo_settings():
  baseline = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth")
  validity = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity")
  assert tuple(baseline.observations["camera"].terms) == ("depth",)
  assert tuple(validity.observations["camera"].terms) == ("depth", "validity")
  assert validity.observations["camera"].concatenate_dim == 0
  assert baseline.scene.sensors == validity.scene.sensors
  single = load_rl_cfg("Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth")
  paired = load_rl_cfg("Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity")
  assert isinstance(single, RslRlOnPolicyRunnerCfg)
  assert isinstance(paired, RslRlOnPolicyRunnerCfg)
  assert single.algorithm == paired.algorithm
  assert single.actor == paired.actor and single.critic == paired.critic
  assert single.experiment_name != paired.experiment_name


def test_validity_checkpoint_cannot_be_silently_loaded_as_depth_only():
  config = load_rl_cfg("Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth")
  assert isinstance(config, RslRlOnPolicyRunnerCfg)
  assert config.actor.cnn_cfg is not None
  models = []
  for channels in (1, 2):
    obs = TensorDict(
      {
        "actor": torch.zeros(2, 99),
        "camera": torch.zeros(2, channels, 48, 64),
      },
      batch_size=[2],
    )
    models.append(
      SpatialSoftmaxCNNModel(
        obs=obs,
        obs_groups={"actor": ["actor", "camera"]},
        obs_set="actor",
        output_dim=29,
        cnn_cfg=config.actor.cnn_cfg,
        hidden_dims=list(config.actor.hidden_dims),
        obs_normalization=True,
      )
    )
  for source, target in ((models[0], models[1]), (models[1], models[0])):
    with pytest.raises(RuntimeError, match="size mismatch"):
      target.load_state_dict(source.state_dict(), strict=True)


@pytest.mark.slow
@pytest.mark.parametrize("suffix,channels", [("", 1), ("-Validity", 2)])
def test_g1_3dgs_depth_observation_runtime(
  device: str, suffix: str, channels: int
) -> None:
  cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth" + suffix)
  cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    observations, _ = env.reset()
    depth = observations["camera"]
    assert isinstance(depth, torch.Tensor)
    assert depth.shape == (1, channels, 48, 64)
    assert depth.dtype == torch.float32
    assert torch.isfinite(depth).all()
    assert torch.all((depth >= 0.0) & (depth <= 1.0))
    assert depth.max() > depth.min()
    if channels == 2:
      sensor = env.scene[G1_3DGS_DEPTH_SENSOR]
      assert isinstance(sensor, CameraSensor)
      raw = sensor.data.depth
      assert raw is not None
      expected = (torch.isfinite(raw) & (raw > 0)).permute(0, 3, 1, 2).float()
      torch.testing.assert_close(depth[:, 1:2], expected, rtol=0, atol=0)
      after, *_ = env.step(torch.zeros((1, 29), device=device))
      raw = sensor.data.depth
      assert raw is not None
      expected = (torch.isfinite(raw) & (raw > 0)).permute(0, 3, 1, 2).float()
      after_camera = after["camera"]
      assert isinstance(after_camera, torch.Tensor)
      torch.testing.assert_close(after_camera[:, 1:2], expected, rtol=0, atol=0)
  finally:
    env.close()
