"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

_VISION_CNN_CFG = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax": True,
  "spatial_softmax_temperature": 1.0,
}
_VISION_MODEL_CLS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
_TEMPORAL_VISION_MODEL_CLS = (
  "mjlab.rl.temporal_spatial_softmax:TemporalSpatialSoftmaxCNNModel"
)


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_3dgs_depth_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the YAM-style CNN runner for G1 3DGS depth observations."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.actor.class_name = _VISION_MODEL_CLS
  cfg.actor.cnn_cfg = _VISION_CNN_CFG
  cfg.critic.class_name = _VISION_MODEL_CLS
  cfg.critic.cnn_cfg = _VISION_CNN_CFG
  cfg.obs_groups = {
    "actor": ("actor", "camera"),
    "critic": ("critic", "camera"),
  }
  cfg.experiment_name = "g1_velocity_3dgs_depth"
  return cfg


def unitree_g1_3dgs_depth_validity_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Use identical PPO/CNN settings with a distinct two-channel experiment name."""
  cfg = unitree_g1_3dgs_depth_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_3dgs_depth_validity"
  return cfg


def unitree_g1_3dgs_depth_validity_history_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Use the temporal CNN adapter for four depth-validity frames."""
  cfg = unitree_g1_3dgs_depth_ppo_runner_cfg()
  cfg.actor.class_name = _TEMPORAL_VISION_MODEL_CLS
  cfg.critic.class_name = _TEMPORAL_VISION_MODEL_CLS
  cfg.experiment_name = "g1_velocity_3dgs_depth_validity_history"
  return cfg


def unitree_g1_3dgs_depth_validity_history_robust_ppo_runner_cfg(
  history_frames: int = 4,
  history_stride: int = 2,
) -> RslRlOnPolicyRunnerCfg:
  """Select strided temporal frames for the robust observation task."""
  cfg = unitree_g1_3dgs_depth_validity_history_ppo_runner_cfg()
  assert cfg.actor.cnn_cfg is not None
  assert cfg.critic.cnn_cfg is not None
  temporal_cfg = {
    "temporal_frames": history_frames,
    "temporal_stride": history_stride,
  }
  cfg.actor.cnn_cfg = {**cfg.actor.cnn_cfg, **temporal_cfg}
  cfg.critic.cnn_cfg = {**cfg.critic.cnn_cfg, **temporal_cfg}
  cfg.experiment_name = "g1_velocity_3dgs_depth_validity_history_robust"
  return cfg


def unitree_g1_3dgs_depth_validity_history_robust_asymmetric_ppo_runner_cfg() -> (
  RslRlOnPolicyRunnerCfg
):
  """Keep camera input actor-only for an asymmetric critic baseline."""
  cfg = unitree_g1_3dgs_depth_validity_history_robust_ppo_runner_cfg()
  cfg.critic.class_name = "MLPModel"
  cfg.critic.cnn_cfg = None
  cfg.obs_groups["critic"] = ("critic",)
  cfg.experiment_name = "g1_velocity_3dgs_depth_validity_history_robust_asymmetric"
  return cfg
