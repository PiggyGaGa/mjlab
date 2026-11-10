"""Direct mesh transport for MuJoCo geometry, without GLB serialization."""

from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest
import trimesh
import trimesh.visual

from mjlab.viewer.viser.scene import MjlabViserScene
from mjlab.viewer.viser.simple_mesh import SimpleMeshSceneApi


def test_direct_mesh_preserves_colour_parts_and_transforms() -> None:
  red = trimesh.creation.box()
  assert isinstance(red.visual, trimesh.visual.ColorVisuals)
  red.visual.face_colors = [255, 0, 0, 25]
  blue = trimesh.creation.box()
  assert isinstance(blue.visual, trimesh.visual.ColorVisuals)
  blue.apply_translation([2, 0, 0])
  blue.visual.face_colors = [0, 0, 255, 255]
  merged = trimesh.util.concatenate([red, blue])
  api = MagicMock()
  children = [MagicMock(), MagicMock()]
  api.add_batched_meshes_simple.side_effect = children

  handle = SimpleMeshSceneApi(api).add_batched_meshes_trimesh(
    "/mesh",
    merged,
    batched_positions=np.zeros((2, 3)),
    batched_wxyzs=np.tile([1, 0, 0, 0], (2, 1)),
  )

  calls = api.add_batched_meshes_simple.call_args_list
  assert len(calls) == 2
  assert sum(len(call.kwargs["faces"]) for call in calls) == len(merged.faces)
  rgba = {tuple(c.kwargs["batched_colors"]): c.kwargs["opacity"] for c in calls}
  assert rgba[(255, 0, 0)] == pytest.approx(25 / 255)
  assert rgba[(0, 0, 255)] is None
  for call in calls:
    vertices, faces = call.kwargs["vertices"], call.kwargs["faces"]
    assert faces.max() < len(vertices)
  positions = np.asarray([[1, 2, 3], [4, 5, 6]])
  handle.batched_positions = positions
  handle.visible = False
  handle.remove()
  for child in children:
    np.testing.assert_array_equal(child.batched_positions, positions)
    assert child.visible is False
    child.remove.assert_called_once()
  api.add_batched_meshes_trimesh.assert_not_called()


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
def test_fixed_dynamic_and_sites_never_export_glb(monkeypatch, alpha: float) -> None:
  def fail_export(*args, **kwargs):
    pytest.fail("Viser model geometry must not export GLB")

  monkeypatch.setattr(trimesh.Trimesh, "export", fail_export)
  model = mujoco.MjModel.from_xml_string(
    f"""<mujoco>
      <asset><mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"
        face="0 2 1 0 1 3 0 3 2 1 2 3"/></asset>
      <worldbody>
        <geom type="cylinder" size="0.2 0.7" rgba="1 0 0 {alpha}"/>
        <site name="fixed_site" size="0.1" rgba="0 1 0 {alpha}"/>
        <body name="robot"><freejoint/>
          <geom type="mesh" mesh="tetra" rgba="0 0 1 {alpha}"/>
          <site name="body_site" size="0.1" rgba="0 1 0 {alpha}"/>
        </body>
      </worldbody></mujoco>"""
  )
  server = MagicMock()
  scene = MjlabViserScene(server, model, num_envs=2)
  assert server.scene.add_mesh_simple.call_count == 2
  body_calls = [
    call
    for call in server.scene.add_batched_meshes_simple.call_args_list
    if call.args[0].startswith("/bodies/")
  ]
  assert len(body_calls) == 2
  for calls in (server.scene.add_mesh_simple.call_args_list, body_calls):
    for call in calls:
      if alpha == 1:
        assert call.kwargs["opacity"] is None
      else:
        assert call.kwargs["opacity"] == pytest.approx(int(alpha * 255) / 255)
  scene.rebuild_visual_handles()
  server.scene.add_mesh_trimesh.assert_not_called()
  server.scene.add_batched_meshes_trimesh.assert_not_called()
  server.scene.add_glb.assert_not_called()


def test_textured_mesh_uses_base_colour_without_glb() -> None:
  mesh = trimesh.creation.box()
  mesh.visual = trimesh.visual.TextureVisuals(
    uv=np.zeros((len(mesh.vertices), 2)),
    material=trimesh.visual.material.PBRMaterial(baseColorFactor=[0.2, 0.4, 0.6, 0.5]),
  )
  api = MagicMock()
  SimpleMeshSceneApi(api).add_mesh_trimesh("/textured", mesh)

  kwargs = api.add_mesh_simple.call_args.kwargs
  np.testing.assert_array_equal(kwargs["color"], [51, 102, 153])
  assert kwargs["opacity"] == pytest.approx(128 / 255)
  assert "glb_data" not in kwargs
