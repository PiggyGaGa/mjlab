"""Configuration and protocol tests for safe perceptive velocity tracking."""

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import load_scene_manifest
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.g1_3dgs import G1_3DGS_DEPTH_SENSOR
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.g1.safe_velocity import (
  SAFE_VELOCITY_ADAPTER_TASK_ID,
  SAFE_VELOCITY_ALLOWED_CONTACT_GEOMS,
  SAFE_VELOCITY_BLIND_TASK_ID,
  SAFE_VELOCITY_COLLISION_SENSOR_PREFIX,
  SAFE_VELOCITY_LOCAL_GEOMETRY_SENSOR_PREFIX,
  SAFE_VELOCITY_PRIVILEGED_TASK_ID,
  SAFE_VELOCITY_TASK_ID,
  unitree_g1_safe_velocity_3dgs_env_cfg,
  unitree_g1_safe_velocity_adapter_3dgs_env_cfg,
)
from mjlab.tasks.velocity.config.g1.safe_velocity_baseline import (
  SafeVelocityLocalGeometryTrackingReward,
  SafeVelocityTrackingReward,
  adapted_velocity_command,
  safe_velocity_base_obstacle_collision_cost,
  safe_velocity_local_geometry,
  safe_velocity_twist_tracking_reward,
)
from mjlab.tasks.velocity.config.g1.safe_velocity_scenarios import (
  SAFE_VELOCITY_SCENARIO_CATALOG,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


def test_safe_velocity_catalog_freezes_scenarios_and_splits() -> None:
  catalog = SAFE_VELOCITY_SCENARIO_CATALOG
  scenarios = {scenario.id: scenario for scenario in catalog.scenarios}

  assert set(scenarios) == {
    "free",
    "left_open",
    "right_open",
    "both_open",
    "dead_end",
    "false_alarm",
  }
  assert {scenario.split for scenario in scenarios.values()} == {
    "train",
    "validation",
    "test",
  }
  assert all(scenario.review_status == "approved" for scenario in scenarios.values())
  assert all(scenario.min_clearance_m == 0.6 for scenario in scenarios.values())
  assert catalog.thresholds.search_trigger_distance_m == 1.0
  assert catalog.thresholds.search_time_budget_s == 18.0
  assert all(scenario.search_time_budget_s == 18.0 for scenario in scenarios.values())
  assert scenarios["dead_end"].safe_exit_sectors_deg == ()
  assert len(scenarios["both_open"].safe_exit_sectors_deg) == 2
  assert scenarios["free"].expected_outcome == "continuous_tracking"
  assert scenarios["false_alarm"].evaluation_time_s == 10.0
  assert scenarios["left_open"].expected_outcome == "avoidance_resume"
  assert scenarios["right_open"].evaluation_time_s == 22.0
  assert scenarios["both_open"].expected_outcome == "avoidance_resume"
  assert scenarios["dead_end"].expected_outcome == "safe_blocked_stop"
  assert scenarios["dead_end"].evaluation_time_s > (
    scenarios["dead_end"].search_time_budget_s + 1.0
  )


def test_safe_velocity_catalog_references_scene_collision_geoms() -> None:
  catalog = SAFE_VELOCITY_SCENARIO_CATALOG
  spec = mujoco.MjSpec.from_file(
    str(catalog.scene_manifest.parent / "collision/collision.xml")
  )
  geom_names = {geom.name for geom in spec.geoms}

  for scenario in catalog.scenarios:
    assert set(scenario.blocking_geoms) <= geom_names


@pytest.mark.parametrize(
  "scenario_id",
  ["free", "left_open", "right_open", "both_open", "dead_end", "false_alarm"],
)
def test_safe_velocity_factory_applies_deterministic_spawn(scenario_id: str) -> None:
  catalog = SAFE_VELOCITY_SCENARIO_CATALOG
  scenario = catalog.scenario(scenario_id)
  manifest = load_scene_manifest(catalog.scene_manifest)
  spawn = next(
    item for item in manifest.spawn_points if item.name == scenario.spawn_point
  )
  cfg = unitree_g1_safe_velocity_3dgs_env_cfg(scenario_id=scenario_id)

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "mjcf"
  assert cfg.scene.terrain.mjcf_path == str(
    catalog.scene_manifest.parent / "collision/collision.xml"
  )
  assert scenario.spawn_point.startswith("p7_")

  collision_sensors = tuple(
    sensor
    for sensor in cfg.scene.sensors or ()
    if sensor.name.startswith(SAFE_VELOCITY_COLLISION_SENSOR_PREFIX)
  )
  spec = mujoco.MjSpec.from_file(
    str(catalog.scene_manifest.parent / "collision/collision.xml")
  )
  obstacle_geom_names = tuple(
    geom.name
    for geom in spec.geoms
    if geom.name not in SAFE_VELOCITY_ALLOWED_CONTACT_GEOMS
  )
  assert len(collision_sensors) == len(obstacle_geom_names)
  assert all(isinstance(sensor, ContactSensorCfg) for sensor in collision_sensors)
  assert all(
    isinstance(sensor, ContactSensorCfg)
    and sensor.primary.mode == "body"
    and sensor.primary.pattern == "pelvis"
    for sensor in collision_sensors
  )
  assert (
    tuple(
      sensor.secondary.pattern
      for sensor in collision_sensors
      if isinstance(sensor, ContactSensorCfg) and sensor.secondary
    )
    == obstacle_geom_names
  )
  assert cfg.scene.spawn_position == spawn.position
  assert cfg.scene.spawn_quaternion_wxyz == spawn.quaternion_wxyz
  assert cfg.events["reset_base"].params["pose_range"] == {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.02, 0.02),
    "yaw": (0.0, 0.0),
  }


def test_safe_velocity_task_reuses_velocity_action_and_camera() -> None:
  cfg = load_env_cfg(SAFE_VELOCITY_TASK_ID)
  runner = load_rl_cfg(SAFE_VELOCITY_TASK_ID)

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  assert not twist.heading_command
  assert twist.ranges.heading is None
  assert twist.rel_world_envs == 0.0
  assert twist.rel_forward_envs == 0.8
  assert twist.ranges.lin_vel_x == (0.0, 0.8)
  assert twist.ranges.lin_vel_y == (0.0, 0.0)
  assert twist.ranges.ang_vel_z == (-0.6, 0.6)
  assert "command_vel" not in cfg.curriculum
  assert isinstance(cfg.actions["joint_pos"], JointPositionActionCfg)
  assert "push_robot" not in cfg.events
  assert cfg.scene.env_spacing == 12.0
  assert any(sensor.name == G1_3DGS_DEPTH_SENSOR for sensor in cfg.scene.sensors or ())
  assert tuple(cfg.observations["camera"].terms) == ("depth_validity",)
  forbidden_actor_terms = {
    "base_pos_w",
    "base_yaw_w",
    "goal_position",
    "goal_direction",
    "collision_geometry",
    "safe_exit",
  }
  assert forbidden_actor_terms.isdisjoint(cfg.observations["actor"].terms)
  assert "camera" not in runner.obs_groups["critic"]
  assert runner.experiment_name == "g1_3dgs_safe_velocity"

  assert cfg.rewards["track_linear_velocity"].weight == 0.0
  assert cfg.rewards["track_angular_velocity"].weight == 0.0
  safe_tracking = cfg.rewards["safe_velocity_tracking"]
  assert safe_tracking.func is SafeVelocityTrackingReward
  assert safe_tracking.weight == 2.0
  assert safe_tracking.params["sensor_name"] == G1_3DGS_DEPTH_SENSOR
  collision = cfg.rewards["base_obstacle_collision"]
  assert collision.func is safe_velocity_base_obstacle_collision_cost
  assert collision.weight < 0.0
  assert all(
    name.startswith(SAFE_VELOCITY_COLLISION_SENSOR_PREFIX)
    for name in collision.params["sensor_names"]
  )


def test_safe_velocity_tracking_reward_prefers_safe_deviation_near_obstacle() -> None:
  stopped = torch.zeros((1, 3))
  unsafe_forward = torch.tensor([[0.5, 0.0, 0.0]])
  safe_stop = torch.zeros((1, 3))

  safe_reward = safe_velocity_twist_tracking_reward(
    stopped,
    stopped,
    safe_stop,
    linear_std=0.5,
    angular_std=0.5,
  )
  unsafe_reward = safe_velocity_twist_tracking_reward(
    unsafe_forward,
    stopped,
    safe_stop,
    linear_std=0.5,
    angular_std=0.5,
  )

  assert safe_reward.item() == pytest.approx(2.0)
  assert unsafe_reward.item() < safe_reward.item()


def test_safe_velocity_does_not_change_existing_flat_g1_defaults() -> None:
  cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
  twist = cfg.commands["twist"]

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert cfg.scene.gaussian_splats is None
  assert isinstance(twist, UniformVelocityCommandCfg)
  assert twist.heading_command
  assert "push_robot" in cfg.events
  assert cfg.rewards["track_linear_velocity"].weight == 2.0
  assert cfg.rewards["track_angular_velocity"].weight == 2.0
  assert "safe_velocity_tracking" not in cfg.rewards


def test_safe_velocity_adapter_keeps_camera_outside_blind_policy() -> None:
  cfg = load_env_cfg(SAFE_VELOCITY_ADAPTER_TASK_ID)
  runner = load_rl_cfg(SAFE_VELOCITY_ADAPTER_TASK_ID)

  assert isinstance(runner, RslRlOnPolicyRunnerCfg)
  assert "camera" not in cfg.observations
  assert cfg.observations["actor"].terms["command"].func is adapted_velocity_command
  assert any(sensor.name == G1_3DGS_DEPTH_SENSOR for sensor in cfg.scene.sensors or ())
  assert runner.actor.class_name == "MLPModel"
  assert runner.obs_groups == {
    "actor": ("actor",),
    "critic": ("critic",),
  }


def test_safe_velocity_adapter_does_not_mutate_learning_task() -> None:
  adapter = unitree_g1_safe_velocity_adapter_3dgs_env_cfg()
  learning = unitree_g1_safe_velocity_3dgs_env_cfg()

  assert "camera" not in adapter.observations
  assert "camera" in learning.observations
  assert learning.observations["actor"].terms["command"].func is not (
    adapted_velocity_command
  )


def test_safe_velocity_blind_baseline_is_renderer_free() -> None:
  cfg = load_env_cfg(SAFE_VELOCITY_BLIND_TASK_ID)
  runner = load_rl_cfg(SAFE_VELOCITY_BLIND_TASK_ID)

  assert isinstance(runner, RslRlOnPolicyRunnerCfg)
  assert cfg.scene.gaussian_splats is None
  assert "camera" not in cfg.observations
  assert not any(isinstance(sensor, CameraSensorCfg) for sensor in cfg.scene.sensors)
  assert cfg.rewards["track_linear_velocity"].weight == 2.0
  assert cfg.rewards["track_angular_velocity"].weight == 2.0
  assert "safe_velocity_tracking" not in cfg.rewards
  assert runner.actor.class_name == "MLPModel"
  assert runner.experiment_name == "g1_3dgs_safe_velocity_blind"
  assert runner.actor.distribution_cfg == {
    "class_name": "GaussianDistribution",
    "init_std": 0.15,
    "std_type": "scalar",
    "std_range": (0.15, 0.15),
    "learn_std": False,
  }
  assert runner.algorithm.learning_rate == 1.0e-4
  assert runner.algorithm.entropy_coef == 0.0
  assert runner.algorithm.num_learning_epochs == 2


def test_safe_velocity_privileged_baseline_exposes_only_local_geometry() -> None:
  cfg = load_env_cfg(SAFE_VELOCITY_PRIVILEGED_TASK_ID)
  runner = load_rl_cfg(SAFE_VELOCITY_PRIVILEGED_TASK_ID)
  assert isinstance(runner, RslRlOnPolicyRunnerCfg)
  rays = tuple(
    sensor
    for sensor in cfg.scene.sensors
    if sensor.name.startswith(SAFE_VELOCITY_LOCAL_GEOMETRY_SENSOR_PREFIX)
  )

  assert cfg.scene.gaussian_splats is None
  assert "camera" not in cfg.observations
  assert len(rays) == 9
  assert all(isinstance(sensor, RayCastSensorCfg) for sensor in rays)
  assert all(
    isinstance(sensor, RayCastSensorCfg)
    and sensor.include_geom_groups == (1, 3)
    and isinstance(sensor.frame, ObjRef)
    and sensor.frame.name == "pelvis"
    for sensor in rays
  )
  actor_geometry = cfg.observations["actor"].terms["local_geometry"]
  critic_geometry = cfg.observations["critic"].terms["local_geometry"]
  assert actor_geometry.func is safe_velocity_local_geometry
  assert critic_geometry.func is safe_velocity_local_geometry
  assert actor_geometry.params["sensor_names"] == tuple(ray.name for ray in rays)
  assert cfg.rewards["track_linear_velocity"].weight == 0.0
  assert cfg.rewards["track_angular_velocity"].weight == 0.0
  safe_tracking = cfg.rewards["safe_velocity_tracking"]
  assert safe_tracking.func is SafeVelocityLocalGeometryTrackingReward
  assert safe_tracking.params["sensor_names"] == tuple(ray.name for ray in rays)
  assert runner.actor.class_name == "MLPModel"
  assert runner.experiment_name == "g1_3dgs_safe_velocity_privileged"
  assert runner.actor.distribution_cfg is not None
  assert runner.actor.distribution_cfg["std_range"] == (0.15, 0.15)
  assert runner.actor.distribution_cfg["learn_std"] is False
  assert runner.algorithm.learning_rate == 1.0e-4
  assert runner.algorithm.entropy_coef == 0.0
  assert runner.algorithm.num_learning_epochs == 2


@pytest.mark.slow
@pytest.mark.parametrize(
  ("task_id", "actor_shape", "critic_shape"),
  [
    (SAFE_VELOCITY_BLIND_TASK_ID, 99, 109),
    (SAFE_VELOCITY_PRIVILEGED_TASK_ID, 117, 127),
  ],
)
def test_safe_velocity_learning_baselines_reset_and_step(
  device: str,
  task_id: str,
  actor_shape: int,
  critic_shape: int,
) -> None:
  cfg = load_env_cfg(task_id)
  cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    observations, _ = env.reset(seed=0)
    actor = observations["actor"]
    critic = observations["critic"]
    assert isinstance(actor, torch.Tensor)
    assert isinstance(critic, torch.Tensor)
    assert actor.shape == (1, actor_shape)
    assert critic.shape == (1, critic_shape)
    assert torch.isfinite(actor).all()
    assert torch.isfinite(critic).all()
    if task_id == SAFE_VELOCITY_PRIVILEGED_TASK_ID:
      torch.testing.assert_close(actor[:, -9:], torch.ones_like(actor[:, -9:]))

    observations, rewards, *_ = env.step(torch.zeros((1, 29), device=device))
    actor = observations["actor"]
    assert isinstance(actor, torch.Tensor)
    assert torch.isfinite(actor).all()
    assert torch.isfinite(rewards).all()
  finally:
    env.close()


@pytest.mark.slow
@pytest.mark.parametrize("num_envs", [1, 2])
def test_safe_velocity_task_reset_and_step(device: str, num_envs: int) -> None:
  cfg = load_env_cfg(SAFE_VELOCITY_TASK_ID)
  cfg.scene.num_envs = num_envs

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    observations, _ = env.reset(seed=0)
    camera = observations["camera"]
    assert isinstance(camera, torch.Tensor)
    assert camera.shape == (num_envs, 7, 2, 48, 64)
    assert torch.isfinite(camera).all()

    observations, rewards, *_ = env.step(torch.zeros((num_envs, 29), device=device))
    actor = observations["actor"]
    camera = observations["camera"]
    assert isinstance(actor, torch.Tensor)
    assert isinstance(camera, torch.Tensor)
    assert actor.shape[0] == num_envs
    assert camera.shape == (num_envs, 7, 2, 48, 64)
    assert torch.isfinite(rewards).all()
    reward_term = env.reward_manager.get_term_cfg("safe_velocity_tracking").func
    assert isinstance(reward_term, SafeVelocityTrackingReward)
    assert reward_term.last_output is not None
  finally:
    env.close()


@pytest.mark.slow
@pytest.mark.parametrize("num_envs", [1, 2])
def test_safe_velocity_adapter_feeds_safe_command_to_actor(
  device: str, num_envs: int
) -> None:
  cfg = load_env_cfg(SAFE_VELOCITY_ADAPTER_TASK_ID)
  cfg.scene.num_envs = num_envs

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    observations, _ = env.reset(seed=0)
    actor = observations["actor"]
    assert isinstance(actor, torch.Tensor)
    assert actor.shape == (num_envs, 99)
    term_cfg = env.observation_manager.get_term_cfg("actor", "command")
    term = term_cfg.func
    assert isinstance(term, adapted_velocity_command)
    assert term.last_output is not None
    torch.testing.assert_close(actor[:, -3:], term.last_output.command)
    assert torch.isfinite(term.last_output.sector_clearance_m).all()

    observations, *_ = env.step(torch.zeros((num_envs, 29), device=device))
    actor = observations["actor"]
    assert isinstance(actor, torch.Tensor)
    assert term.last_output is not None
    torch.testing.assert_close(actor[:, -3:], term.last_output.command)
    assert term.adapter.intervention_count.shape == (num_envs,)
  finally:
    env.close()
