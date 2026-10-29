"""Verify native PLY transport preserves Viser's world-space splat geometry."""

import importlib

import numpy as np
import pytest

from mjlab.scene import (
  GaussianSplatAssetCfg,
  GaussianSplatData,
  gaussian_splat_covariances,
  load_gaussian_splat_asset,
)


def test_native_ply_roundtrip_preserves_world_geometry_and_dc_color():
  try:
    bridge = importlib.import_module("mujoco.gaussian")
  except ModuleNotFoundError:
    pytest.skip("Requires the custom MuJoCo 3DGS wheel")
  angle = np.pi / 5
  source = GaussianSplatData(
    position=np.asarray([[1.5, -2.0, 0.7], [-0.3, 0.8, 2.4]], dtype=np.float32),
    rotation=np.asarray(
      [[np.cos(angle), 0, 0, np.sin(angle)], [1, 0, 0, 0]], dtype=np.float32
    ),
    scale=np.asarray([[0.02, 0.1, 0.3], [0.4, 0.2, 0.05]], dtype=np.float32),
    rgba=np.asarray([[0.1, 0.4, 0.8, 0.7], [0.9, 0.3, 0.2, 0.4]], dtype=np.float32),
    adr=np.asarray([0, 2], dtype=np.int32),
    group_id=np.asarray([0], dtype=np.int32),
  )
  assets = bridge.GaussianAssets(source)
  try:
    restored = load_gaussian_splat_asset(
      GaussianSplatAssetCfg(path=assets.path_for_world(0))
    )
    np.testing.assert_allclose(restored.position, source.position, atol=1e-7)
    np.testing.assert_allclose(restored.rgba, source.rgba, atol=1e-6)
    np.testing.assert_allclose(
      gaussian_splat_covariances(restored),
      gaussian_splat_covariances(source),
      atol=1e-7,
    )
  finally:
    assets.close()
