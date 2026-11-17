"""Hidden collision supports contact and supervision without entering depth."""

from pathlib import Path

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import (
  CameraSensor,
  CameraSensorCfg,
  ContactMatch,
  ContactSensor,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.sim import Simulation, SimulationCfg
from mjlab.tasks.velocity.config.g1.safe_velocity import (
  SAFE_VELOCITY_COLLISION_SENSOR_PREFIX,
  unitree_g1_safe_velocity_depth_stop_3dgs_env_cfg,
  unitree_g1_safe_velocity_vision_data_3dgs_env_cfg,
)
from mjlab.tasks.velocity.config.g1.safe_velocity_baseline import (
  adapted_velocity_command,
)
from mjlab.tasks.velocity.config.g1.safe_velocity_sensors import (
  CollisionTruthRaySensor,
  CollisionTruthRaySensorCfg,
)


@pytest.mark.parametrize("hidden_height", [0.5, 1.0])
def test_hidden_collision_changes_teacher_but_not_camera(hidden_height: float):
  xml = (
    '<mujoco><worldbody><site name="origin" pos="0 0 2"/>'
    '<geom type="plane" size="10 10 .1" group="1"/>'
    f'<body name="obstacle" pos="0 0 {hidden_height}">'
    '<joint name="height" type="slide" axis="0 0 1"/>'
    '<geom type="box" size="1 1 .1" group="3"/></body></worldbody></mujoco>'
  )
  device = get_test_device()
  scene = Scene(
    SceneCfg(
      num_envs=2,
      env_spacing=0.0,
      entities={"world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))},
      sensors=(
        CameraSensorCfg(
          name="depth",
          pos=(0, 0, 2),
          width=9,
          height=9,
          data_types=("depth",),
          enabled_geom_groups=(1,),
        ),
        CollisionTruthRaySensorCfg(
          name="teacher",
          frame=ObjRef(type="site", name="origin", entity="world"),
          pattern=GridPatternCfg(size=(0, 0), resolution=1.0),
          include_geom_groups=(3,),
          exclude_parent_body=False,
        ),
      ),
    ),
    device,
  )
  sim = Simulation(
    num_envs=2, cfg=SimulationCfg(), model=scene.compile(), device=device
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  assert scene.sensor_context is not None
  assert not scene.sensor_context.has_raycasts
  sim.set_sensor_context(scene.sensor_context)
  sim.forward()
  sim.sense()
  scene.update(0.0)
  camera = scene["depth"]
  teacher = scene["teacher"]
  assert isinstance(camera, CameraSensor)
  assert isinstance(teacher, CollisionTruthRaySensor)
  depth = camera.data.depth
  assert depth is not None
  torch.testing.assert_close(depth, torch.full_like(depth, 2.0))
  distances = teacher.data.distances
  torch.testing.assert_close(
    distances, torch.full_like(distances, 2.0 - hidden_height - 0.1)
  )
  # A reward may read before forward(), then an observation after forward(),
  # without scene.update() between them. Teacher queries must use the new pose.
  offsets = torch.tensor([0.1, 0.25], device=device)
  sim.data.qpos[:, 0] = offsets
  sim.forward()
  torch.testing.assert_close(
    teacher.data.distances[:, 0], 2.0 - hidden_height - 0.1 - offsets
  )
  # Querying the teacher must not add its hidden box to the next camera render.
  sim.sense()
  assert camera.data.depth is not None
  torch.testing.assert_close(camera.data.depth, depth)


def test_vision_data_isolation_default_and_explicit_legacy_mode():
  for isolated in (True, False):
    cfg = unitree_g1_safe_velocity_vision_data_3dgs_env_cfg(
      play=True, isolate_teacher_rays=isolated
    )
    rays = [
      sensor
      for sensor in cfg.scene.sensors
      if sensor.name.startswith("safe_velocity_local_geometry_ray_")
    ]
    assert len(rays) == 9
    assert all(
      sensor.build().requires_sensor_context is not isolated for sensor in rays
    )


def test_hidden_wall_contact_is_detected_without_entering_camera():
  robot_xml = (
    '<mujoco><worldbody><body name="pelvis" pos="-1 0 1">'
    '<joint name="forward" type="slide" axis="1 0 0"/>'
    '<geom type="box" size=".1 .1 .1" group="0" mass="1"/>'
    "</body></worldbody></mujoco>"
  )
  obstacle_xml = (
    '<mujoco><worldbody><geom type="plane" size="10 10 .1" group="1"/>'
    '<geom name="wall" type="box" pos="0 0 1" size=".1 1 1" group="3"/>'
    "</worldbody></mujoco>"
  )
  device = get_test_device()
  scene = Scene(
    SceneCfg(
      num_envs=2,
      env_spacing=0.0,
      entities={
        "robot": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml)),
        "obstacles": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(obstacle_xml)),
      },
      sensors=(
        CameraSensorCfg(
          name="depth",
          pos=(0, 0, 3),
          width=9,
          height=9,
          data_types=("depth",),
          enabled_geom_groups=(1, 2),
        ),
        ContactSensorCfg(
          name="base_obstacle_contact",
          primary=ContactMatch(mode="body", pattern="pelvis", entity="robot"),
          secondary=ContactMatch(mode="geom", pattern="wall", entity="obstacles"),
          fields=("found", "force", "dist"),
          reduce="maxforce",
          history_length=4,
        ),
      ),
    ),
    device,
  )
  sim = Simulation(
    num_envs=2, cfg=SimulationCfg(), model=scene.compile(), device=device
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  assert scene.sensor_context is not None
  assert not scene.sensor_context.has_raycasts
  sim.set_sensor_context(scene.sensor_context)
  camera = scene["depth"]
  contact = scene["base_obstacle_contact"]
  assert isinstance(camera, CameraSensor)
  assert isinstance(contact, ContactSensor)

  sim.forward()
  sim.sense()
  scene.update(0.005)
  before = camera.data.depth
  assert before is not None
  before = before.clone()
  torch.testing.assert_close(before, torch.full_like(before, 3.0))
  assert contact.data.found is not None
  assert torch.count_nonzero(contact.data.found) == 0

  # Only world 0 moves the pelvis 3 cm inside the hidden wall.
  sim.data.qpos[0, 0] = 0.83
  sim.forward()
  sim.sense()
  scene.update(0.005)
  data = contact.data
  assert data.found is not None and data.force is not None
  assert data.dist is not None and data.force_history is not None
  assert torch.any(data.found[0] > 0)
  assert torch.count_nonzero(data.found[1]) == 0
  assert torch.linalg.vector_norm(data.force[0], dim=-1).max() > 10.0
  assert data.dist[0].min() < 0.0
  assert camera.data.depth is not None
  torch.testing.assert_close(camera.data.depth, before)

  # Contact clears after separation, while history retains the short impact.
  sim.data.qpos[0, 0] = 0.0
  sim.forward()
  sim.sense()
  scene.update(0.005)
  data = contact.data
  assert data.found is not None and data.force_history is not None
  assert torch.count_nonzero(data.found) == 0
  assert torch.linalg.vector_norm(data.force_history[0], dim=-1).max() > 10.0
  assert torch.count_nonzero(data.force_history[1]) == 0
  assert camera.data.depth is not None
  torch.testing.assert_close(camera.data.depth, before)


@pytest.mark.parametrize("studio", [False, True])
def test_depth_stop_uses_visible_camera_and_contact_events_without_truth_rays(studio):
  manifest = (
    Path(__file__).parents[1]
    / "src/mjlab/asset_zoo/3dgs/ArtanimMotionCaptureStudio/manifest.json"
    if studio
    else None
  )
  cfg = unitree_g1_safe_velocity_depth_stop_3dgs_env_cfg(
    play=True, scene_manifest=manifest
  )
  assert not any(isinstance(s, RayCastSensorCfg) for s in cfg.scene.sensors)
  cameras = [s for s in cfg.scene.sensors if isinstance(s, CameraSensorCfg)]
  assert len(cameras) == 1
  assert cameras[0].enabled_geom_groups == (1, 2)
  command = cfg.observations["actor"].terms["command"]
  assert command.func is adapted_velocity_command
  assert command.params["sensor_name"] == cameras[0].name
  assert "safety_teacher" not in cfg.observations
  contacts = [
    s
    for s in cfg.scene.sensors
    if isinstance(s, ContactSensorCfg)
    and s.name.startswith(SAFE_VELOCITY_COLLISION_SENSOR_PREFIX)
  ]
  assert contacts
  assert cfg.rewards["base_obstacle_collision"].params["sensor_names"] == tuple(
    s.name for s in contacts
  )
  for sensor in contacts:
    assert sensor.primary == ContactMatch(mode="body", pattern="pelvis", entity="robot")
    assert sensor.secondary is not None and sensor.secondary.mode == "geom"
    assert sensor.history_length == cfg.decimation
