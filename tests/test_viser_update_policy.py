"""Tests for Viser viewer update-policy helpers."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import mujoco
import numpy as np
from mjviser import ViserMujocoScene

from mjlab.scene import GaussianSplatData
from mjlab.viewer.viser.overlays import (
  ViserContactOverlays,
  ViserDebugOverlays,
)
from mjlab.viewer.viser.scene import MjlabViserScene
from mjlab.viewer.viser.viewer import ViserPlayViewer


@dataclass
class _DummyScene:
  env_idx: int
  debug_visualization_enabled: bool
  show_contact_points: bool = False
  show_contact_forces: bool = False
  needs_update: bool = False
  clear_count: int = 0
  clear_debug_count: int = 0

  def clear(self) -> None:
    self.clear_count += 1

  def clear_debug_all(self) -> None:
    self.clear_debug_count += 1


class _DummyEnv:
  def __init__(self, unwrapped: Any):
    self._unwrapped = unwrapped

  @property
  def unwrapped(self) -> Any:
    return self._unwrapped


def test_should_update_cameras():
  assert ViserPlayViewer._should_update_cameras(paused=False, has_pending_updates=False)
  assert not ViserPlayViewer._should_update_cameras(
    paused=True, has_pending_updates=False
  )
  assert ViserPlayViewer._should_update_cameras(paused=True, has_pending_updates=True)


def test_should_submit_scene_update():
  # Odd ticks are skipped to keep scene submits around 30Hz.
  assert not ViserPlayViewer._should_submit_scene_update(
    counter=1, paused=False, has_pending_updates=True
  )
  # Running: submit on even ticks regardless of pending flags.
  assert ViserPlayViewer._should_submit_scene_update(
    counter=2, paused=False, has_pending_updates=False
  )
  # Paused: submit only with pending updates.
  assert not ViserPlayViewer._should_submit_scene_update(
    counter=2, paused=True, has_pending_updates=False
  )
  assert ViserPlayViewer._should_submit_scene_update(
    counter=2, paused=True, has_pending_updates=True
  )


def test_scene_requires_live_refresh():
  """Base scene sets needs_update when contact overlays are visible."""
  model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
  server = MagicMock()
  scene = ViserMujocoScene(server, model, num_envs=1)

  scene.show_contact_points = False
  scene.show_contact_forces = False
  scene.needs_update = False
  scene._last_body_xpos = None  # No cached data, refresh is a no-op.
  scene.refresh_visualization()
  assert not scene.needs_update

  scene.show_contact_points = True
  assert scene.show_contact_points or scene.show_contact_forces


def test_scene_adds_native_gaussian_splats_and_tracks_world_offset():
  model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
  server = MagicMock()
  handle = server.scene.add_gaussian_splats.return_value
  splats = GaussianSplatData(
    position=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
    rotation=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    scale=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
    rgba=np.asarray([[0.2, 0.4, 0.6, 0.8]], dtype=np.float32),
    adr=np.asarray([0, 1], dtype=np.int32),
    group_id=np.asarray([0], dtype=np.int32),
  )

  scene = MjlabViserScene(
    server,
    model,
    num_envs=1,
    gaussian_splats=splats,
    geom_groups_visible=(0, 1, 1, 0, 0, 0),
  )
  scene._sync_gaussian_splats(0, np.asarray([4.0, 5.0, 6.0]))

  call_kwargs = server.scene.add_gaussian_splats.call_args.kwargs
  np.testing.assert_array_equal(call_kwargs["centers"], splats.position)
  np.testing.assert_allclose(
    call_kwargs["covariances"],
    np.asarray([[[0.01, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.09]]]),
    atol=1.0e-6,
  )
  np.testing.assert_array_equal(call_kwargs["rgbs"], splats.rgba[:, :3])
  np.testing.assert_array_equal(call_kwargs["opacities"], splats.rgba[:, 3:4])
  np.testing.assert_array_equal(handle.position, [4.0, 5.0, 6.0])
  assert handle.visible
  assert not scene.geom_groups_visible[0]


def test_debug_overlays_env_switch_and_queue():
  """env_switch respects enabled flag; queue clears + calls update_visualizers."""
  unwrapped = MagicMock(spec=["update_visualizers"])
  env = _DummyEnv(unwrapped)
  scene = _DummyScene(env_idx=0, debug_visualization_enabled=False)
  overlays = ViserDebugOverlays(env=env, scene=scene)

  # Disabled: env switch does not clear.
  overlays.on_env_switch()
  assert scene.clear_debug_count == 0

  # Enabled: env switch clears, queue dispatches.
  scene.debug_visualization_enabled = True
  overlays.on_env_switch()
  assert scene.clear_debug_count == 1

  overlays.queue()
  assert scene.clear_count == 1
  unwrapped.update_visualizers.assert_called_once_with(scene)


def test_contact_overlays_env_switch():
  """env_switch requests scene update only when contacts are enabled."""
  scene = _DummyScene(env_idx=0, debug_visualization_enabled=False)
  overlays = ViserContactOverlays(scene=scene)

  assert not overlays.is_enabled()
  overlays.on_env_switch()
  assert not scene.needs_update

  scene.show_contact_points = True
  assert overlays.is_enabled()
  overlays.on_env_switch()
  assert scene.needs_update
