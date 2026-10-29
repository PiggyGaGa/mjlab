"""Native Gaussian asset selection without a display or physics stepping."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mjlab.viewer.native.viewer import NativeMujocoViewer


def make_viewer(data):
  env = MagicMock()
  env.cfg.viewer.env_idx = 0
  env.unwrapped.scene.gaussian_splat_data = data
  return NativeMujocoViewer(env, MagicMock())


def test_plain_native_does_not_load_bridge(monkeypatch):
  viewer = make_viewer(None)
  load = MagicMock(side_effect=AssertionError("unexpected bridge import"))
  monkeypatch.setattr("mjlab.viewer.native.viewer.importlib.import_module", load)
  assert viewer._prepare_gaussian_ply() is None
  load.assert_not_called()


def test_asset_switch_only_when_world_asset_changes(monkeypatch):
  viewer = make_viewer(SimpleNamespace(group_id=[0, 0, 1]))
  assets = MagicMock()
  assets.path_for_world.side_effect = lambda index: f"/tmp/world_{index}.ply"
  bridge = SimpleNamespace(GaussianAssets=MagicMock(return_value=assets))
  monkeypatch.setattr(
    "mjlab.viewer.native.viewer.importlib.import_module", lambda name: bridge
  )
  assert viewer._prepare_gaussian_ply() == "/tmp/world_0.ply"
  viewer.viewer = MagicMock()
  viewer.env_idx = 1
  viewer._sync_gaussian_asset()
  viewer.viewer.load_gaussian_ply.assert_not_called()
  viewer.env_idx = 2
  viewer._sync_gaussian_asset()
  viewer.viewer.load_gaussian_ply.assert_called_once_with("/tmp/world_2.ply")
  viewer._sync_gaussian_asset()
  assert viewer.viewer.load_gaussian_ply.call_count == 1
  viewer.close()
  assets.close.assert_called_once()


def test_missing_custom_wheel_has_actionable_error(monkeypatch):
  viewer = make_viewer(SimpleNamespace(group_id=[0]))
  load = MagicMock(side_effect=ModuleNotFoundError("mujoco.gaussian"))
  monkeypatch.setattr("mjlab.viewer.native.viewer.importlib.import_module", load)
  with pytest.raises(RuntimeError, match="custom MuJoCo 3DGS wheel"):
    viewer._prepare_gaussian_ply()
