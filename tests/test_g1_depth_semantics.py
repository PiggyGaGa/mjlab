"""Depth units, missing samples, and the G1-specific observation convention."""

from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.observations import camera_depth, camera_depth_validity
from mjlab.scene import GaussianSplatAssetCfg, GaussianSplatSceneCfg, Scene, SceneCfg
from mjlab.sensor import CameraSensor, CameraSensorCfg
from mjlab.sim import Simulation, SimulationCfg


def test_zero_depth_as_far_preserves_legacy_behavior_and_nan():
  raw = torch.tensor([0.0, 0.02, 2.0, 8.0, float("nan")]).reshape(1, 1, 5, 1)
  sensor = MagicMock(spec=CameraSensor)
  sensor.data.depth = raw
  env = MagicMock(spec=ManagerBasedRlEnv)
  env.scene = {"depth": sensor}
  legacy = camera_depth(env, "depth", 5.0, 0.05)
  fixed = camera_depth(env, "depth", 5.0, 0.05, zero_depth_as_far=True)
  torch.testing.assert_close(legacy[0, 0, 0, :4], torch.tensor([0.01, 0.01, 0.4, 1]))
  torch.testing.assert_close(fixed[0, 0, 0, :4], torch.tensor([1, 0.01, 0.4, 1]))
  assert torch.isnan(fixed[0, 0, 0, 4])
  assert raw[0, 0, 0, 0] == 0, "Observation mapping must not mutate the sensor"


def test_validity_distinguishes_missing_from_positive_beyond_cutoff():
  raw = torch.tensor([0, -1, 0.02, 2, 8, float("nan"), float("inf")]).reshape(
    1, 1, 7, 1
  )
  sensor = MagicMock(spec=CameraSensor)
  sensor.data.depth = raw
  env = MagicMock(spec=ManagerBasedRlEnv)
  env.scene = {"depth": sensor}
  mask = camera_depth_validity(env, "depth")
  torch.testing.assert_close(mask.flatten(), torch.tensor([0, 0, 1, 1, 1, 0, 0.0]))
  depth = camera_depth(env, "depth", 5, 0.05, zero_depth_as_far=True)
  assert depth[0, 0, 0, 0] == depth[0, 0, 0, 4] == 1
  assert mask[0, 0, 0, 0] == 0 and mask[0, 0, 0, 4] == 1
  assert torch.isnan(depth[0, 0, 0, 5]), (
    "Depth NaNs must still fail observation validation"
  )
  assert raw[0, 0, 0, 0] == 0


@pytest.mark.parametrize("kind", ["plane", "empty", "opaque_splat", "thin_splat"])
def test_raw_depth_is_axial_and_requires_splat_opacity(tmp_path, kind):
  plane = '<geom type="plane" size="10 10 .1"/>' if kind == "plane" else ""
  xml = (
    '<mujoco><worldbody><geom type="sphere" size=".1" pos="100 100 0"/>'
    f"{plane}</worldbody></mujoco>"
  )
  splats = None
  if "splat" in kind:
    fields = (
      "x",
      "y",
      "z",
      "scale_0",
      "scale_1",
      "scale_2",
      "rot_0",
      "rot_1",
      "rot_2",
      "rot_3",
      "opacity",
      "f_dc_0",
      "f_dc_1",
      "f_dc_2",
    )
    opacity = 0.999999 if kind == "opaque_splat" else 0.1
    row = [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      float(np.log(opacity / (1 - opacity))),
      0,
      0,
      0,
    ]
    path = tmp_path / "splat.ply"
    path.write_text(
      "ply\nformat ascii 1.0\nelement vertex 1\n"
      + "".join(f"property float {field}\n" for field in fields)
      + "end_header\n"
      + " ".join(map(str, row))
      + "\n"
    )
    splats = GaussianSplatSceneCfg(assets=(GaussianSplatAssetCfg(path=str(path)),))
  sensor = CameraSensorCfg(
    name="depth",
    pos=(0, 0, 2),
    width=33,
    height=25,
    data_types=("depth",),
    enabled_geom_groups=(0,),
    fovy=60,
  )
  scene = Scene(
    SceneCfg(
      num_envs=1,
      entities={"world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))},
      sensors=(sensor,),
      gaussian_splats=splats,
    ),
    get_test_device(),
  )
  sim = Simulation(
    num_envs=1, cfg=SimulationCfg(), model=scene.compile(), device=get_test_device()
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  assert scene.sensor_context is not None
  sim.set_sensor_context(scene.sensor_context)
  sim.forward()
  sim.sense()
  camera = scene["depth"]
  assert isinstance(camera, CameraSensor)
  raw = camera.data.depth
  assert raw is not None
  if kind == "plane":
    # A flat plane is constant axial Z depth, not Euclidean distance along rays.
    torch.testing.assert_close(raw, torch.full_like(raw, 2), rtol=1e-5, atol=1e-5)
  elif kind == "opaque_splat":
    assert float(raw[0, 12, 16, 0]) == pytest.approx(2, abs=1e-5)
  else:
    assert torch.count_nonzero(raw) == 0


def test_multi_asset_two_camera_world_isolation(tmp_path):
  """Worlds 0/2 share an asset; moving world 0 must not affect 1 or 2."""
  paths = []
  fields = (
    "x",
    "y",
    "z",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "opacity",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
  )
  for index, z in enumerate((0, -2)):
    path = tmp_path / f"asset{index}.ply"
    path.write_text(
      "ply\nformat ascii 1.0\nelement vertex 1\n"
      + "".join(f"property float {field}\n" for field in fields)
      + f"end_header\n0 0 {z} 0 0 0 1 0 0 0 15 0 0 0\n"
    )
    paths.append(path)
  xml = (
    '<mujoco><worldbody><geom type="sphere" size=".1" pos="100 100 0"/>'
    '<body name="rig" pos="0 0 2"><freejoint/>'
    '<inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>'
    "</body></worldbody></mujoco>"
  )
  sensors = tuple(
    CameraSensorCfg(
      name=f"depth{index}",
      parent_body="world/rig",
      pos=(0, 0, index),
      width=33,
      height=25,
      data_types=("depth",),
      enabled_geom_groups=(0,),
    )
    for index in range(2)
  )
  scene = Scene(
    SceneCfg(
      num_envs=3,
      entities={"world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))},
      sensors=sensors,
      gaussian_splats=GaussianSplatSceneCfg(
        assets=tuple(GaussianSplatAssetCfg(path=str(path)) for path in paths),
        world_asset_ids=(0, 1, 0),
      ),
    ),
    get_test_device(),
  )
  sim = Simulation(
    num_envs=3, cfg=SimulationCfg(), model=scene.compile(), device=get_test_device()
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  assert scene.sensor_context is not None
  sim.set_sensor_context(scene.sensor_context)

  def sense():
    sim.forward()
    sim.sense()
    scene.update(0)
    result = []
    for index in range(2):
      camera = scene[f"depth{index}"]
      assert isinstance(camera, CameraSensor)
      raw = camera.data.depth
      assert raw is not None
      result.append(raw.clone())
    return result

  initial = sense()
  for index, frame in enumerate(initial):
    torch.testing.assert_close(
      frame[:, 12, 16, 0],
      torch.tensor([2 + index, 4 + index, 2 + index], device=frame.device).float(),
      rtol=1e-5,
      atol=1e-5,
    )
  sim.data.qpos[0, 2] += 0.5
  moved = sense()
  for index, frame in enumerate(moved):
    torch.testing.assert_close(frame[1:], initial[index][1:], rtol=0, atol=0)
    assert float(frame[0, 12, 16, 0]) == pytest.approx(2.5 + index, abs=1e-5)
  other_qpos = sim.data.qpos[1:].clone()
  scene.reset(torch.tensor([0], device=sim.data.qpos.device))
  sim.data.qpos[0, 2] = 2
  reset = sense()
  torch.testing.assert_close(sim.data.qpos[1:], other_qpos, rtol=0, atol=0)
  for index, frame in enumerate(reset):
    torch.testing.assert_close(frame, initial[index], rtol=0, atol=0)
