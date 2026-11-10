"""Submit mjviser geometry through Viser's direct coloured-mesh API.

The adapter retains upstream geometry grouping and transform updates without
serializing GLB. Textures and PBR appearance are replaced by flat RGBA colours.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import trimesh
import viser
from trimesh.visual import ColorVisuals, TextureVisuals


class SimpleMeshHandle:
  """Forward a mesh group's transform/visibility updates to its colour parts."""

  def __init__(self, handles: list[Any]) -> None:
    object.__setattr__(self, "_handles", handles)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._handles[0], name)

  def __setattr__(self, name: str, value: Any) -> None:
    for handle in self._handles:
      setattr(handle, name, value)

  def remove(self) -> None:
    for handle in self._handles:
      handle.remove()


def _colour_parts(
  mesh: trimesh.Trimesh,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """Split flat face colours, compacting vertices without changing triangles."""
  visual = mesh.visual
  if isinstance(visual, TextureVisuals):
    # Keep the material's base colour, without sampling or transmitting textures.
    colours = np.tile(visual.material.main_color, (len(mesh.faces), 1))
  elif isinstance(visual, ColorVisuals):
    colours = np.asarray(visual.face_colors)
  else:
    colours = np.tile([102, 102, 102, 255], (len(mesh.faces), 1))
  unique, inverse = np.unique(colours, axis=0, return_inverse=True)
  for index, rgba in enumerate(unique):
    faces = np.asarray(mesh.faces)[inverse == index]
    vertex_ids, remap = np.unique(faces, return_inverse=True)
    yield np.asarray(mesh.vertices)[vertex_ids], remap.reshape(-1, 3), rgba


class SimpleMeshSceneApi:
  """Adapt only the two mesh submission calls made by mjviser."""

  def __init__(self, scene: viser.SceneApi) -> None:
    self._scene = scene

  def __getattr__(self, name: str) -> Any:
    return getattr(self._scene, name)

  def add_mesh_trimesh(
    self, name: str, mesh: trimesh.Trimesh, **kwargs: Any
  ) -> SimpleMeshHandle:
    handles = []
    for index, (vertices, faces, rgba) in enumerate(_colour_parts(mesh)):
      handles.append(
        self._scene.add_mesh_simple(
          f"{name}/colour_{index}",
          vertices=vertices,
          faces=faces,
          color=rgba[:3],
          opacity=float(rgba[3]) / 255 if rgba[3] < 255 else None,
          **kwargs,
        )
      )
    return SimpleMeshHandle(handles)

  def add_batched_meshes_trimesh(
    self, name: str, mesh: trimesh.Trimesh, **kwargs: Any
  ) -> SimpleMeshHandle:
    handles = []
    for index, (vertices, faces, rgba) in enumerate(_colour_parts(mesh)):
      handles.append(
        self._scene.add_batched_meshes_simple(
          f"{name}/colour_{index}",
          vertices=vertices,
          faces=faces,
          batched_colors=rgba[:3],
          opacity=float(rgba[3]) / 255 if rgba[3] < 255 else None,
          **kwargs,
        )
      )
    return SimpleMeshHandle(handles)


class SimpleMeshServer:
  """Keep the real server lifecycle while adapting mjviser's scene calls."""

  def __init__(self, server: viser.ViserServer) -> None:
    self._server = server
    self.scene = SimpleMeshSceneApi(server.scene)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._server, name)
