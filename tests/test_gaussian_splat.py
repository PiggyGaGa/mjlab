"""Tests for static 3D Gaussian Splatting scene assets."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.scene import (
  GaussianSplatAssetCfg,
  GaussianSplatData,
  GaussianSplatSceneCfg,
  Scene,
  SceneCfg,
  gaussian_splat_covariances,
  load_gaussian_splat_asset,
  load_gaussian_splat_scene,
)
from mjlab.sensor import CameraSensorCfg
from mjlab.sim import Simulation, SimulationCfg

_SH_C0 = 0.28209479177387814
_PROPERTIES = (
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


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


def _write_ascii_ply(path: Path, rows: list[tuple[float, ...]]) -> None:
  header = ["ply", "format ascii 1.0", f"element vertex {len(rows)}"]
  header.extend(f"property float {name}" for name in _PROPERTIES)
  header.append("end_header")
  body = [" ".join(str(value) for value in row) for row in rows]
  path.write_text("\n".join((*header, *body, "")))


def _row(
  position: tuple[float, float, float],
  *,
  scale: float = 0.1,
  opacity_logit: float = 0.0,
  rgb: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> tuple[float, ...]:
  sh = tuple((channel - 0.5) / _SH_C0 for channel in rgb)
  return (
    *position,
    np.log(scale),
    np.log(scale),
    np.log(scale),
    1.0,
    0.0,
    0.0,
    0.0,
    opacity_logit,
    *sh,
  )


def test_load_graphdeco_ply_applies_filter_and_transform(tmp_path: Path) -> None:
  path = tmp_path / "scene.ply"
  _write_ascii_ply(
    path,
    [
      _row((1.0, 0.0, 0.0), opacity_logit=-2.0),
      _row((0.0, 0.0, 0.0), opacity_logit=0.0),
      _row((0.0, 1.0, 0.0), opacity_logit=2.0, rgb=(1.0, 0.0, 0.0)),
    ],
  )

  half_sqrt = np.sqrt(0.5)
  data = load_gaussian_splat_asset(
    GaussianSplatAssetCfg(
      path=str(path),
      pos=(1.0, 2.0, 3.0),
      quat=(half_sqrt, 0.0, 0.0, half_sqrt),
      scale=2.0,
      opacity_threshold=0.5,
      max_splats=1,
    )
  )

  np.testing.assert_allclose(data.position, [[-1.0, 2.0, 3.0]], atol=1.0e-6)
  np.testing.assert_allclose(data.scale, [[0.2, 0.2, 0.2]], atol=1.0e-6)
  np.testing.assert_allclose(
    data.rotation, [[half_sqrt, 0.0, 0.0, half_sqrt]], atol=1.0e-6
  )
  np.testing.assert_allclose(data.rgba[0, :3], [1.0, 0.0, 0.0], atol=1.0e-6)
  assert data.rgba[0, 3] == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))
  assert data.position.dtype == np.float32
  np.testing.assert_array_equal(data.adr, [0, 1])


def test_load_binary_little_endian_ply(tmp_path: Path) -> None:
  path = tmp_path / "binary.ply"
  dtype = np.dtype([(name, "<f4") for name in _PROPERTIES])
  vertices = np.zeros(1, dtype=dtype)
  values = _row((0.25, -0.5, 1.0), scale=0.2, rgb=(0.2, 0.4, 0.6))
  for name, value in zip(_PROPERTIES, values, strict=True):
    vertices[name] = value
  header = ["ply", "format binary_little_endian 1.0", "element vertex 1"]
  header.extend(f"property float {name}" for name in _PROPERTIES)
  header.append("end_header")
  path.write_bytes(("\n".join(header) + "\n").encode() + vertices.tobytes())

  data = load_gaussian_splat_asset(GaussianSplatAssetCfg(path=str(path)))

  np.testing.assert_allclose(data.position, [[0.25, -0.5, 1.0]])
  np.testing.assert_allclose(data.scale, [[0.2, 0.2, 0.2]], rtol=1.0e-6)
  np.testing.assert_allclose(data.rgba[0, :3], [0.2, 0.4, 0.6], atol=1.0e-6)


def test_scene_groups_assign_assets_per_world(tmp_path: Path) -> None:
  first = tmp_path / "first.ply"
  second = tmp_path / "second.ply"
  _write_ascii_ply(first, [_row((0.0, 0.0, 0.0))])
  _write_ascii_ply(
    second,
    [_row((1.0, 0.0, 0.0)), _row((2.0, 0.0, 0.0))],
  )
  cfg = GaussianSplatSceneCfg(
    assets=(
      GaussianSplatAssetCfg(path=str(first)),
      GaussianSplatAssetCfg(path=str(second)),
    ),
    world_asset_ids=(1, 0, 1),
  )

  data = load_gaussian_splat_scene(cfg, nworld=3)

  np.testing.assert_array_equal(data.adr, [0, 1, 3])
  np.testing.assert_array_equal(data.group_id, [1, 0, 1])
  assert data.position.shape == (3, 3)


def test_asset_config_validation() -> None:
  with pytest.raises(ValueError, match="bounds_min and bounds_max"):
    GaussianSplatAssetCfg(path="missing.ply", bounds_min=(0.0, 0.0, 0.0))


def test_gaussian_splat_covariances_include_rotation_and_scale() -> None:
  half_sqrt = np.float32(np.sqrt(0.5))
  data = GaussianSplatData(
    position=np.zeros((1, 3), dtype=np.float32),
    rotation=np.asarray([[half_sqrt, 0.0, 0.0, half_sqrt]], dtype=np.float32),
    scale=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
    rgba=np.ones((1, 4), dtype=np.float32),
    adr=np.asarray([0, 1], dtype=np.int32),
    group_id=np.zeros(1, dtype=np.int32),
  )

  covariance = gaussian_splat_covariances(data)

  np.testing.assert_allclose(
    covariance,
    np.asarray([[[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 9.0]]]),
    atol=1.0e-6,
  )
  assert covariance.dtype == np.float32


def test_invalid_world_mapping_is_rejected(tmp_path: Path) -> None:
  path = tmp_path / "scene.ply"
  _write_ascii_ply(path, [_row((0.0, 0.0, 0.0))])
  cfg = GaussianSplatSceneCfg(
    assets=(GaussianSplatAssetCfg(path=str(path)),), world_asset_ids=(0,)
  )

  with pytest.raises(ValueError, match="length 1; expected 2"):
    load_gaussian_splat_scene(cfg, nworld=2)


def test_camera_render_context_receives_splats(device: str, tmp_path: Path) -> None:
  path = tmp_path / "red_splat.ply"
  _write_ascii_ply(
    path,
    [
      _row(
        (0.0, 0.0, 0.6),
        scale=0.25,
        opacity_logit=4.0,
        rgb=(1.0, 0.0, 0.0),
      )
    ],
  )
  xml = """
    <mujoco>
      <worldbody>
        <camera name="camera" pos="0 -3 1" xyaxes="1 0 0 0 0.2 1"/>
        <geom type="plane" size="3 3 0.1" rgba="0.2 0.2 0.2 1"/>
      </worldbody>
    </mujoco>
  """
  camera = CameraSensorCfg(
    name="camera_sensor",
    camera_name="world/camera",
    width=48,
    height=48,
    data_types=("rgb", "depth"),
  )
  scene = Scene(
    SceneCfg(
      num_envs=1,
      entities={"world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))},
      sensors=(camera,),
      gaussian_splats=GaussianSplatSceneCfg(
        assets=(GaussianSplatAssetCfg(path=str(path)),)
      ),
    ),
    device=device,
  )
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  assert scene.sensor_context is not None
  sim.set_sensor_context(scene.sensor_context)

  assert scene.sensor_context.render_context.splat_count == 1
  sim.forward()
  sim.sense()
  rgb = scene["camera_sensor"].data.rgb
  depth = scene["camera_sensor"].data.depth

  assert rgb is not None
  assert depth is not None
  assert int(rgb[..., 0].max()) > int(rgb[..., 1].max())
  assert np.count_nonzero(depth.cpu().numpy()) > 0


def test_splats_are_available_without_camera(device: str, tmp_path: Path) -> None:
  path = tmp_path / "frontend_splat.ply"
  _write_ascii_ply(path, [_row((0.0, 0.0, 0.0))])
  cfg = SceneCfg(
    gaussian_splats=GaussianSplatSceneCfg(
      assets=(GaussianSplatAssetCfg(path=str(path)),)
    )
  )
  scene = Scene(cfg, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)

  scene.initialize(sim.mj_model, sim.model, sim.data)

  assert scene.sensor_context is None
  assert scene.gaussian_splat_data is not None
  assert len(scene.gaussian_splat_data.position) == 1
