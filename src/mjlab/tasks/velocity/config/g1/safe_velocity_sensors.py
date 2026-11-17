"""Task-local collision supervision isolated from the visual render context."""

from dataclasses import dataclass

import mujoco_warp as mjwarp
import warp as wp

from mjlab.sensor.raycast_sensor import RayCastData, RayCastSensor, RayCastSensorCfg


@dataclass
class CollisionTruthRaySensorCfg(RayCastSensorCfg):
  """Use direct collision queries without adding hidden geoms to camera BVHs."""

  def build(self) -> RayCastSensor:
    return CollisionTruthRaySensor(self)


class CollisionTruthRaySensor(RayCastSensor):
  """Reuse ray frames and buffers with MJWarp's render-independent query path.

  Queries run lazily after sensing, against the current simulation state. This
  keeps teacher-only groups out of SensorContext's camera/raycast group union.
  Direct queries trade BVH acceleration for isolation and suit offline labels.
  """

  requires_sensor_context = False

  @property
  def data(self) -> RayCastData:
    # Rewards can query before forward(), then observations after forward().
    # This sensor is outside sense() and cannot rely on its cache invalidation.
    return self._compute_data()

  def _compute_data(self) -> RayCastData:
    assert self._model is not None and self._data is not None
    assert self._wp_device is not None
    assert self._ray_pnt is not None and self._ray_vec is not None
    assert self._ray_bodyexclude is not None
    assert self._ray_dist is not None and self._ray_geomid is not None
    assert self._ray_normal is not None
    with wp.ScopedDevice(self._wp_device):
      self.prepare_rays()
      mjwarp.rays(
        m=self._model.struct,  # type: ignore[attr-defined]
        d=self._data.struct,  # type: ignore[attr-defined]
        pnt=self._ray_pnt,  # type: ignore[invalid-argument-type]
        vec=self._ray_vec,  # type: ignore[invalid-argument-type]
        geomgroup=self._geomgroup,  # pyright: ignore[reportArgumentType]
        flg_static=True,
        bodyexclude=self._ray_bodyexclude,
        dist=self._ray_dist,  # type: ignore[invalid-argument-type]
        geomid=self._ray_geomid,  # type: ignore[invalid-argument-type]
        normal=self._ray_normal,  # type: ignore[invalid-argument-type]
      )
      self.postprocess_rays()

    assert self._distances is not None and self._normals_w is not None
    assert self._hit_pos_w is not None
    assert self._pos_w is not None and self._quat_w is not None
    assert self._frame_pos_w is not None and self._frame_quat_w is not None
    return RayCastData(
      distances=self._distances,
      normals_w=self._normals_w,
      hit_pos_w=self._hit_pos_w,
      pos_w=self._pos_w,
      quat_w=self._quat_w,
      frame_pos_w=self._frame_pos_w,
      frame_quat_w=self._frame_quat_w,
    )
