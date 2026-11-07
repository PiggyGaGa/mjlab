"""Configuration tests for the G1 3DGS depth-training task."""

from typing import Any, cast

import pytest
import torch
from conftest import get_test_device
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel
from mjlab.rl.temporal_spatial_softmax import TemporalSpatialSoftmaxCNNModel
from mjlab.sensor import CameraSensor, CameraSensorCfg
from mjlab.tasks.g1_3dgs import G1_3DGS_DEPTH_SENSOR
from mjlab.tasks.g1_3dgs_vision import camera_depth_validity_robust
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
  assert camera.enabled_geom_groups == (1, 2)

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


def test_temporal_validity_task_reuses_observation_history() -> None:
  task = "Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity-History"
  env_cfg = load_env_cfg(task)
  camera = env_cfg.observations["camera"]

  assert tuple(camera.terms) == ("depth", "validity")
  assert camera.history_length == 4
  assert camera.flatten_history_dim is False
  assert camera.concatenate_dim == 1

  rl_cfg = load_rl_cfg(task)
  assert isinstance(rl_cfg, RslRlOnPolicyRunnerCfg)
  assert rl_cfg.actor.class_name.endswith(":TemporalSpatialSoftmaxCNNModel")
  assert rl_cfg.critic.class_name.endswith(":TemporalSpatialSoftmaxCNNModel")
  assert rl_cfg.experiment_name == "g1_velocity_3dgs_depth_validity_history"


def test_temporal_spatial_softmax_stacks_history_as_channels() -> None:
  task = "Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity-History"
  config = load_rl_cfg(task)
  assert isinstance(config, RslRlOnPolicyRunnerCfg)
  assert config.actor.cnn_cfg is not None
  obs = TensorDict(
    {
      "actor": torch.zeros(2, 99),
      "camera": torch.zeros(2, 4, 2, 48, 64),
    },
    batch_size=[2],
  )
  model = TemporalSpatialSoftmaxCNNModel(
    obs=obs,
    obs_groups={"actor": ["actor", "camera"]},
    obs_set="actor",
    output_dim=29,
    cnn_cfg=config.actor.cnn_cfg,
    hidden_dims=list(config.actor.hidden_dims),
    obs_normalization=True,
  )

  assert model(obs).shape == (2, 29)
  first_conv_weight = model.state_dict()["cnns.camera.encoder.cnn.0.weight"]
  assert first_conv_weight.shape[1] == 8


def test_robust_temporal_task_configures_stride_delay_and_noise() -> None:
  task = "Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity-History-Robust"
  env_cfg = load_env_cfg(task)
  camera = env_cfg.observations["camera"]

  assert camera.history_length == 7
  assert camera.flatten_history_dim is False
  assert tuple(camera.terms) == ("depth_validity",)
  term = camera.terms["depth_validity"]
  assert term.func is camera_depth_validity_robust
  assert (term.delay_min_lag, term.delay_max_lag) == (1, 1)
  assert term.params["update_period"] == 2
  assert term.params["enable_corruption"] is True
  assert term.params["range_noise_std"] == 0.01
  assert term.params["blur_kernel_size"] == 3
  assert term.params["structured_dropout_prob"] == 0.1
  assert term.params["frame_dropout_prob"] == 0.02

  play_cfg = load_env_cfg(task, play=True)
  play_term = play_cfg.observations["camera"].terms["depth_validity"]
  assert play_term.params["enable_corruption"] is False

  rl_cfg = load_rl_cfg(task)
  assert isinstance(rl_cfg, RslRlOnPolicyRunnerCfg)
  assert rl_cfg.actor.cnn_cfg is not None
  assert rl_cfg.actor.cnn_cfg["temporal_frames"] == 4
  assert rl_cfg.actor.cnn_cfg["temporal_stride"] == 2


def test_asymmetric_robust_task_keeps_camera_actor_only() -> None:
  task = "Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity-History-Robust-Asymmetric"
  rl_cfg = load_rl_cfg(task)

  assert isinstance(rl_cfg, RslRlOnPolicyRunnerCfg)
  assert rl_cfg.obs_groups["actor"] == ("actor", "camera")
  assert rl_cfg.obs_groups["critic"] == ("critic",)
  assert rl_cfg.actor.class_name.endswith(":TemporalSpatialSoftmaxCNNModel")
  assert rl_cfg.critic.class_name == "MLPModel"
  assert rl_cfg.critic.cnn_cfg is None


def test_temporal_spatial_softmax_selects_strided_history() -> None:
  task = "Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity-History-Robust"
  config = load_rl_cfg(task)
  assert isinstance(config, RslRlOnPolicyRunnerCfg)
  assert config.actor.cnn_cfg is not None
  obs = TensorDict(
    {
      "actor": torch.zeros(2, 99),
      "camera": torch.zeros(2, 7, 2, 48, 64),
    },
    batch_size=[2],
  )
  model = TemporalSpatialSoftmaxCNNModel(
    obs=obs,
    obs_groups={"actor": ["actor", "camera"]},
    obs_set="actor",
    output_dim=29,
    cnn_cfg=config.actor.cnn_cfg,
    hidden_dims=list(config.actor.hidden_dims),
    obs_normalization=True,
  )

  assert model(obs).shape == (2, 29)
  first_conv_weight = model.state_dict()["cnns.camera.encoder.cnn.0.weight"]
  assert first_conv_weight.shape[1] == 8

  exported = cast(Any, model.as_onnx())
  dummy_inputs = exported.get_dummy_inputs()
  assert dummy_inputs[1].shape == (1, 8, 48, 64)
  assert exported(*dummy_inputs).shape == (1, 29)


def test_robust_depth_validity_frame_dropout_stays_aligned() -> None:
  observation = torch.cat(
    (torch.full((2, 1, 8, 8), 0.5), torch.ones((2, 1, 8, 8))), dim=1
  )
  corrupted = camera_depth_validity_robust._corrupt(
    observation,
    range_noise_std=0.0,
    blur_kernel_size=1,
    structured_dropout_prob=0.0,
    frame_dropout_prob=1.0,
    dropout_size_range=(0.1, 0.3),
  )

  assert torch.all(corrupted[:, :1] == 1)
  assert torch.all(corrupted[:, 1:] == 0)


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


@pytest.mark.slow
def test_g1_3dgs_depth_history_runtime(device: str) -> None:
  task = "Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity-History"
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    observations, _ = env.reset()
    history = observations["camera"]
    assert isinstance(history, torch.Tensor)
    assert history.shape == (1, 4, 2, 48, 64)
    torch.testing.assert_close(history[:, 1:], history[:, :1].expand(-1, 3, -1, -1, -1))

    observations, *_ = env.step(torch.zeros((1, 29), device=device))
    history = observations["camera"]
    assert isinstance(history, torch.Tensor)
    sensor = env.scene[G1_3DGS_DEPTH_SENSOR]
    assert isinstance(sensor, CameraSensor)
    raw = sensor.data.depth
    assert raw is not None
    expected_validity = (torch.isfinite(raw) & (raw > 0)).permute(0, 3, 1, 2)
    torch.testing.assert_close(
      history[:, -1, 1:2], expected_validity.float(), rtol=0, atol=0
    )
  finally:
    env.close()


@pytest.mark.slow
def test_g1_3dgs_robust_history_runtime(device: str) -> None:
  task = "Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity-History-Robust"
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    observations, _ = env.reset()
    history = observations["camera"]
    assert isinstance(history, torch.Tensor)
    assert history.shape == (1, 7, 2, 48, 64)
    initial_validity = history[:, -1, 1:2].clone()
    assert torch.all((history[:, :, :1] >= 0.0) & (history[:, :, :1] <= 1.0))

    observations, *_ = env.step(torch.zeros((1, 29), device=device))
    history = observations["camera"]
    assert isinstance(history, torch.Tensor)
    torch.testing.assert_close(history[:, -1, 1:2], initial_validity, rtol=0, atol=0)

    rl_cfg = load_rl_cfg(task)
    assert isinstance(rl_cfg, RslRlOnPolicyRunnerCfg)
    assert rl_cfg.actor.cnn_cfg is not None
    actor_obs = observations["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    model_obs = TensorDict({"actor": actor_obs, "camera": history}, batch_size=[1])
    model = TemporalSpatialSoftmaxCNNModel(
      obs=model_obs,
      obs_groups={"actor": ["actor", "camera"]},
      obs_set="actor",
      output_dim=29,
      cnn_cfg=rl_cfg.actor.cnn_cfg,
      hidden_dims=list(rl_cfg.actor.hidden_dims),
      obs_normalization=True,
    ).to(device)
    assert model(model_obs).shape == (1, 29)
  finally:
    env.close()
