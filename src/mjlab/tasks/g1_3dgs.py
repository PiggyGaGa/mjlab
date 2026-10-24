"""Shared G1 3DGS play configuration."""

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import GaussianSplatAssetCfg, GaussianSplatSceneCfg
from mjlab.sensor import CameraSensorCfg, TerrainHeightSensorCfg

G1_3DGS_VIDEO_SENSOR = "g1_3dgs_video"
_DEFAULT_G1_3DGS_ASSET = (
  "/home/liana/SSD/Ubuntu/robot/exp-3DGS/1012/1012/point_cloud_2_transform.ply"
)


def apply_g1_3dgs_play_cfg(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Add native Viser 3DGS visualization without enabling MJWarp rendering."""
  asset_path = os.environ.get("MJLAB_G1_3DGS_ASSET", _DEFAULT_G1_3DGS_ASSET)
  cfg.scene.gaussian_splats = GaussianSplatSceneCfg(
    assets=(GaussianSplatAssetCfg(path=asset_path),)
  )

  # Group 0 contains the physical plane. Keep it active for contacts while hiding
  # it from Viser so the Gaussian scene remains visible.
  cfg.viewer.geom_group = (0, 1, 1, 0, 0, 0)

  # Cameras and ray sensors share one MJWarp render context. If a task contains the
  # critic-only foot height sensor, restrict it to the robot visual group so the
  # hidden plane is not merged back into the camera BVH.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.include_geom_groups = (2,)

  return cfg


def enable_g1_3dgs_video_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  """Enable the MJWarp 3DGS camera only for explicitly requested recording."""
  if cfg.scene.gaussian_splats is None:
    raise ValueError(
      f"Video sensor '{G1_3DGS_VIDEO_SENSOR}' requires a 3DGS play task."
    )
  if any(sensor.name == G1_3DGS_VIDEO_SENSOR for sensor in cfg.scene.sensors or ()):
    return
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    CameraSensorCfg(
      name=G1_3DGS_VIDEO_SENSOR,
      camera_name="robot/tracking",
      width=320,
      height=240,
      data_types=("rgb",),
      enabled_geom_groups=(2,),
      use_shadows=False,
      use_textures=True,
    ),
  )
