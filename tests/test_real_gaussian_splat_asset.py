"""Integration tests for a locally reconstructed 3DGS scene.

These tests use a real asset that is intentionally not checked into the repository.
Set ``MJLAB_3DGS_TEST_ASSET`` to test an equivalent Graphdeco-style PLY elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.scene import (
  GaussianSplatAssetCfg,
  GaussianSplatSceneCfg,
  Scene,
  SceneCfg,
  load_gaussian_splat_asset,
)
from mjlab.sensor import CameraSensorCfg
from mjlab.sim import Simulation, SimulationCfg

_DEFAULT_ASSET_PATH = Path(
  "/home/liana/SSD/Ubuntu/robot/exp-3DGS/1012/1012/point_cloud_2.ply"
)
_EXPECTED_SPLAT_COUNT = 398_978


@pytest.fixture(scope="module")
def real_splat_path() -> Path:
  path = Path(os.environ.get("MJLAB_3DGS_TEST_ASSET", _DEFAULT_ASSET_PATH))
  if not path.is_file():
    pytest.skip(
      f"Real 3DGS test asset not found at '{path}'. Set MJLAB_3DGS_TEST_ASSET "
      "to a Graphdeco-style PLY file."
    )
  return path


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


@pytest.mark.slow
def test_real_splat_asset_loads_and_normalizes(real_splat_path: Path) -> None:
  data = load_gaussian_splat_asset(GaussianSplatAssetCfg(path=str(real_splat_path)))
  splat_count = len(data.position)

  assert data.position.shape == (splat_count, 3)
  assert data.rotation.shape == (splat_count, 4)
  assert data.scale.shape == (splat_count, 3)
  assert data.rgba.shape == (splat_count, 4)
  assert data.position.dtype == np.float32
  assert data.rotation.dtype == np.float32
  assert data.scale.dtype == np.float32
  assert data.rgba.dtype == np.float32
  assert np.isfinite(data.position).all()
  assert np.isfinite(data.rotation).all()
  assert np.isfinite(data.scale).all()
  assert np.isfinite(data.rgba).all()
  assert np.all(data.scale > 0.0)
  assert np.all((data.rgba >= 0.0) & (data.rgba <= 1.0))
  np.testing.assert_allclose(
    np.linalg.norm(data.rotation, axis=1), 1.0, rtol=1.0e-6, atol=1.0e-6
  )
  np.testing.assert_array_equal(data.adr, [0, splat_count])

  if real_splat_path.resolve() == _DEFAULT_ASSET_PATH.resolve():
    assert splat_count == _EXPECTED_SPLAT_COUNT
    np.testing.assert_allclose(
      data.position.min(axis=0),
      [-20.0865478515625, -6.613313674926758, -4.496473789215088],
      atol=2.0e-5,
    )
    np.testing.assert_allclose(
      data.position.max(axis=0),
      [26.548572540283203, 19.23634910583496, 5.004644870758057],
      atol=2.0e-5,
    )


@pytest.mark.slow
def test_real_splat_asset_renders_with_native_mjwarp(
  real_splat_path: Path, device: str
) -> None:
  xml = """
    <mujoco>
      <worldbody>
        <camera name="camera" pos="3 -16 8" xyaxes="1 0 0 0 0.3 1"/>
        <geom type="sphere" pos="0 0 -100" size="0.01"
              contype="0" conaffinity="0" rgba="0 0 0 0"/>
      </worldbody>
    </mujoco>
  """
  camera = CameraSensorCfg(
    name="camera_sensor",
    camera_name="world/camera",
    width=160,
    height=120,
    data_types=("rgb", "depth"),
  )
  scene = Scene(
    SceneCfg(
      num_envs=1,
      entities={"world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))},
      sensors=(camera,),
      gaussian_splats=GaussianSplatSceneCfg(
        assets=(GaussianSplatAssetCfg(path=str(real_splat_path)),)
      ),
    ),
    device=device,
  )
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  assert scene.sensor_context is not None
  sim.set_sensor_context(scene.sensor_context)

  if real_splat_path.resolve() == _DEFAULT_ASSET_PATH.resolve():
    assert scene.sensor_context.render_context.splat_count == _EXPECTED_SPLAT_COUNT
  else:
    assert scene.sensor_context.render_context.splat_count > 0
  sim.forward()
  sim.sense()
  rgb = scene["camera_sensor"].data.rgb
  depth = scene["camera_sensor"].data.depth

  assert rgb is not None
  assert depth is not None
  rgb_array = rgb.cpu().numpy()
  depth_array = depth.cpu().numpy()
  assert np.ptp(rgb_array) > 0
  assert np.count_nonzero(depth_array) > 0
