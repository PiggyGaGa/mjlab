"""Configuration and protocol tests for the P7 task skeleton."""

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import load_scene_manifest
from mjlab.tasks.g1_3dgs import G1_3DGS_DEPTH_SENSOR
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.g1.safe_velocity import (
  P7_ADAPTER_TASK_ID,
  P7_TASK_ID,
  load_p7_scenario_catalog,
  unitree_g1_safe_velocity_3dgs_env_cfg,
  unitree_g1_safe_velocity_adapter_3dgs_env_cfg,
)
from mjlab.tasks.velocity.config.g1.safe_velocity_baseline import (
  adapted_velocity_command,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


def test_p7_catalog_freezes_required_scenarios_and_splits() -> None:
  catalog = load_p7_scenario_catalog()
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
  assert scenarios["dead_end"].safe_exit_sectors_deg == ()
  assert len(scenarios["both_open"].safe_exit_sectors_deg) == 2


def test_p7_catalog_references_manifest_spawns_and_collision_geoms() -> None:
  catalog = load_p7_scenario_catalog()
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
def test_p7_scenario_factory_applies_deterministic_spawn(scenario_id: str) -> None:
  catalog = load_p7_scenario_catalog()
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
  assert cfg.scene.spawn_position == spawn.position
  assert cfg.scene.spawn_quaternion_wxyz == spawn.quaternion_wxyz
  assert cfg.events["reset_base"].params["pose_range"] == {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.02, 0.02),
    "yaw": (0.0, 0.0),
  }


def test_p7_registered_task_reuses_velocity_action_and_p6_camera() -> None:
  cfg = load_env_cfg(P7_TASK_ID)
  runner = load_rl_cfg(P7_TASK_ID)

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  assert not twist.heading_command
  assert twist.ranges.heading is None
  assert twist.rel_world_envs == 0.0
  assert twist.rel_forward_envs == 0.8
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


def test_p7_does_not_change_existing_flat_g1_defaults() -> None:
  cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
  twist = cfg.commands["twist"]

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert cfg.scene.gaussian_splats is None
  assert isinstance(twist, UniformVelocityCommandCfg)
  assert twist.heading_command
  assert "push_robot" in cfg.events


def test_p7_adapter_task_keeps_camera_outside_blind_policy() -> None:
  cfg = load_env_cfg(P7_ADAPTER_TASK_ID)
  runner = load_rl_cfg(P7_ADAPTER_TASK_ID)

  assert isinstance(runner, RslRlOnPolicyRunnerCfg)
  assert "camera" not in cfg.observations
  assert cfg.observations["actor"].terms["command"].func is adapted_velocity_command
  assert any(sensor.name == G1_3DGS_DEPTH_SENSOR for sensor in cfg.scene.sensors or ())
  assert runner.actor.class_name == "MLPModel"
  assert runner.obs_groups == {
    "actor": ("actor",),
    "critic": ("critic",),
  }


def test_p7_adapter_factory_does_not_mutate_learning_task() -> None:
  adapter = unitree_g1_safe_velocity_adapter_3dgs_env_cfg()
  learning = unitree_g1_safe_velocity_3dgs_env_cfg()

  assert "camera" not in adapter.observations
  assert "camera" in learning.observations
  assert learning.observations["actor"].terms["command"].func is not (
    adapted_velocity_command
  )


@pytest.mark.slow
@pytest.mark.parametrize("num_envs", [1, 2])
def test_p7_task_reset_and_step(device: str, num_envs: int) -> None:
  cfg = load_env_cfg(P7_TASK_ID)
  cfg.scene.num_envs = num_envs

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    observations, _ = env.reset(seed=0)
    camera = observations["camera"]
    assert isinstance(camera, torch.Tensor)
    assert camera.shape == (num_envs, 7, 2, 48, 64)
    assert torch.isfinite(camera).all()

    observations, *_ = env.step(torch.zeros((num_envs, 29), device=device))
    actor = observations["actor"]
    camera = observations["camera"]
    assert isinstance(actor, torch.Tensor)
    assert isinstance(camera, torch.Tensor)
    assert actor.shape[0] == num_envs
    assert camera.shape == (num_envs, 7, 2, 48, 64)
  finally:
    env.close()


@pytest.mark.slow
@pytest.mark.parametrize("num_envs", [1, 2])
def test_p7_adapter_task_feeds_safe_command_to_actor(
  device: str, num_envs: int
) -> None:
  cfg = load_env_cfg(P7_ADAPTER_TASK_ID)
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
