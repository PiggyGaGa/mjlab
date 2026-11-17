"""Compare 3DGS depth with offline hidden-collision reference projections.

The tool renders the same robot-mounted camera pose twice: once with only the
manifest's Gaussian asset and once with selected reference groups (default 3).
Group-1 synthetic visible meshes need not occur in the Gaussian reconstruction.
Reference depth is an offline diagnostic and is never a policy input. It writes review
overlays, a structured report, and optionally raw depth arrays without changing
runtime rendering defaults.
"""

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
from PIL import Image

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import load_scene_manifest
from mjlab.sensor import CameraSensor, CameraSensorCfg
from mjlab.tasks.g1_3dgs import G1_3DGS_DEPTH_SENSOR
from mjlab.tasks.velocity.config.g1.safe_velocity import (
  unitree_g1_safe_velocity_depth_stop_3dgs_env_cfg,
)


def depth_alignment_metrics(
  gaussian_depth_m: np.ndarray,
  collision_depth_m: np.ndarray,
  *,
  cutoff_distance_m: float,
  tolerance_m: float,
  minimum_collision_pixels: int,
  minimum_gaussian_coverage: float,
  minimum_aligned_fraction: float,
) -> dict:
  """Measure Gaussian depth agreement on pixels occupied by collision proxies."""
  if gaussian_depth_m.shape != collision_depth_m.shape:
    raise ValueError("Gaussian and collision depth arrays must have the same shape")
  if cutoff_distance_m <= 0 or tolerance_m <= 0:
    raise ValueError("Depth cutoff and alignment tolerance must be positive")

  gaussian_valid = (
    np.isfinite(gaussian_depth_m)
    & (gaussian_depth_m > 0)
    & (gaussian_depth_m < cutoff_distance_m)
  )
  collision_valid = (
    np.isfinite(collision_depth_m)
    & (collision_depth_m > 0)
    & (collision_depth_m < cutoff_distance_m)
  )
  collision_pixels = int(collision_valid.sum())
  jointly_valid = collision_valid & gaussian_valid
  jointly_valid_pixels = int(jointly_valid.sum())
  gaussian_coverage = (
    jointly_valid_pixels / collision_pixels if collision_pixels else 0.0
  )
  error = gaussian_depth_m[jointly_valid] - collision_depth_m[jointly_valid]
  absolute_error = np.abs(error)
  aligned_fraction = (
    float((absolute_error <= tolerance_m).mean()) if absolute_error.size else 0.0
  )
  accepted = (
    collision_pixels >= minimum_collision_pixels
    and gaussian_coverage >= minimum_gaussian_coverage
    and aligned_fraction >= minimum_aligned_fraction
  )
  return {
    "accepted": accepted,
    "collision_pixels": collision_pixels,
    "collision_pixel_fraction": float(collision_valid.mean()),
    "jointly_valid_pixels": jointly_valid_pixels,
    "gaussian_coverage": gaussian_coverage,
    "aligned_fraction": aligned_fraction,
    "signed_error_median_m": float(np.median(error)) if error.size else None,
    "absolute_error_median_m": (
      float(np.median(absolute_error)) if absolute_error.size else None
    ),
    "absolute_error_p90_m": (
      float(np.quantile(absolute_error, 0.9)) if absolute_error.size else None
    ),
  }


def _placeholder_entity() -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(
      '<mujoco><worldbody><geom type="sphere" size=".1" '
      'pos="1000 1000 1000" group="5" contype="0" conaffinity="0"/>'
      "</worldbody></mujoco>"
    )
  )


def _capture_depth(
  manifest_path: Path,
  spawn_point: str,
  *,
  gaussian_only: bool,
  seed: int,
  reference_geom_groups: tuple[int, ...] = (3,),
) -> tuple[np.ndarray, dict]:
  cfg = unitree_g1_safe_velocity_depth_stop_3dgs_env_cfg(
    play=True,
    scene_manifest=manifest_path,
    scene_spawn_point=spawn_point,
    nominal_command=(0.0, 0.0, 0.0),
  )
  cfg.scene.num_envs = 1
  cfg.seed = seed
  if gaussian_only:
    cfg.scene.entities["depth_placeholder"] = _placeholder_entity()
    enabled_groups = (5,)
  else:
    cfg.scene.gaussian_splats = None
    enabled_groups = reference_geom_groups
  cfg.scene.sensors = tuple(
    replace(sensor, enabled_geom_groups=enabled_groups)
    if isinstance(sensor, CameraSensorCfg) and sensor.name == G1_3DGS_DEPTH_SENSOR
    else sensor
    for sensor in cfg.scene.sensors or ()
  )

  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    env.reset(seed=seed)
    env.sim.forward()
    env.sim.sense()
    sensor = env.scene[G1_3DGS_DEPTH_SENSOR]
    if not isinstance(sensor, CameraSensor):
      raise TypeError("Alignment validation requires the G1 depth CameraSensor")
    sensor.update(0)
    raw = sensor.data.depth
    if raw is None:
      raise RuntimeError("Alignment validation camera returned no depth")
    depth = raw[0, :, :, 0].detach().cpu().numpy().copy()
    robot = env.scene["robot"]
    camera_index = sensor.camera_idx
    sim_data = env.sim.data
    metadata = {
      "root_position_w": robot.data.root_link_pos_w[0].detach().cpu().tolist(),
      "root_quaternion_wxyz": robot.data.root_link_quat_w[0].detach().cpu().tolist(),
      "camera_position_w": (sim_data.cam_xpos[0, camera_index].detach().cpu().tolist()),
      "camera_rotation_w": (
        sim_data.cam_xmat[0, camera_index].reshape(3, 3).detach().cpu().tolist()
      ),
    }
    return depth, metadata
  finally:
    env.close()


def _write_overlay(
  path: Path,
  gaussian_depth_m: np.ndarray,
  collision_depth_m: np.ndarray,
  *,
  cutoff_distance_m: float,
  tolerance_m: float,
  scale: int,
) -> None:
  gaussian_valid = (
    np.isfinite(gaussian_depth_m)
    & (gaussian_depth_m > 0)
    & (gaussian_depth_m < cutoff_distance_m)
  )
  collision_valid = (
    np.isfinite(collision_depth_m)
    & (collision_depth_m > 0)
    & (collision_depth_m < cutoff_distance_m)
  )
  normalized = np.clip(
    np.where(gaussian_valid, gaussian_depth_m, cutoff_distance_m),
    0.0,
    cutoff_distance_m,
  )
  grayscale = ((1.0 - normalized / cutoff_distance_m) * 255).astype(np.uint8)
  overlay = np.repeat(grayscale[:, :, None], 3, axis=2)
  jointly_valid = collision_valid & gaussian_valid
  aligned = jointly_valid & (
    np.abs(gaussian_depth_m - collision_depth_m) <= tolerance_m
  )
  mismatched = jointly_valid & ~aligned
  missing = collision_valid & ~gaussian_valid
  overlay[aligned] = (40, 220, 40)
  overlay[mismatched] = (240, 40, 40)
  overlay[missing] = (220, 40, 220)
  image = Image.fromarray(overlay)
  image.resize(
    (image.width * scale, image.height * scale),
    resample=Image.Resampling.NEAREST,
  ).save(path)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--scene-manifest", type=Path, required=True)
  parser.add_argument("--scene-spawn-prefix", required=True)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--reference-geom-groups",
    type=int,
    nargs="+",
    choices=range(6),
    default=[3],
    help="Offline reference groups; group 3 is hidden collision, group 1 visible mesh.",
  )
  parser.add_argument("--cutoff-distance", type=float, default=5.0)
  parser.add_argument("--tolerance", type=float, default=0.15)
  parser.add_argument("--minimum-collision-pixels", type=int, default=20)
  parser.add_argument("--minimum-gaussian-coverage", type=float, default=0.9)
  parser.add_argument("--minimum-aligned-fraction", type=float, default=0.8)
  parser.add_argument("--overlay-scale", type=int, default=8)
  parser.add_argument("--save-raw-depth", action="store_true")
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("/tmp/safe-velocity-scene-alignment"),
  )
  parser.add_argument("--require-pass", action="store_true")
  args = parser.parse_args()
  manifest_path = args.scene_manifest.expanduser().resolve()
  if not manifest_path.is_file():
    parser.error(f"Scene manifest not found: {manifest_path}")
  if not 0 <= args.minimum_gaussian_coverage <= 1:
    parser.error("--minimum-gaussian-coverage must be in [0, 1]")
  if not 0 <= args.minimum_aligned_fraction <= 1:
    parser.error("--minimum-aligned-fraction must be in [0, 1]")
  if args.minimum_collision_pixels < 1:
    parser.error("--minimum-collision-pixels must be positive")
  if args.overlay_scale < 1:
    parser.error("--overlay-scale must be positive")

  manifest = load_scene_manifest(manifest_path)
  spawn_points = tuple(
    spawn.name
    for spawn in manifest.spawn_points
    if spawn.name.startswith(args.scene_spawn_prefix)
  )
  if not spawn_points:
    parser.error(f"No spawn points match prefix: {args.scene_spawn_prefix}")
  args.output.mkdir(parents=True, exist_ok=True)

  results = []
  for spawn_point in spawn_points:
    print(f"[alignment] Capturing {spawn_point}", flush=True)
    gaussian_depth, gaussian_metadata = _capture_depth(
      manifest_path,
      spawn_point,
      gaussian_only=True,
      seed=args.seed,
    )
    collision_depth, collision_metadata = _capture_depth(
      manifest_path,
      spawn_point,
      gaussian_only=False,
      seed=args.seed,
      reference_geom_groups=tuple(args.reference_geom_groups),
    )
    np.testing.assert_allclose(
      gaussian_metadata["camera_position_w"],
      collision_metadata["camera_position_w"],
      atol=1.0e-6,
    )
    np.testing.assert_allclose(
      gaussian_metadata["camera_rotation_w"],
      collision_metadata["camera_rotation_w"],
      atol=1.0e-6,
    )
    np.testing.assert_allclose(
      gaussian_metadata["root_position_w"],
      collision_metadata["root_position_w"],
      atol=1.0e-6,
    )
    np.testing.assert_allclose(
      gaussian_metadata["root_quaternion_wxyz"],
      collision_metadata["root_quaternion_wxyz"],
      atol=1.0e-6,
    )
    metrics = depth_alignment_metrics(
      gaussian_depth,
      collision_depth,
      cutoff_distance_m=args.cutoff_distance,
      tolerance_m=args.tolerance,
      minimum_collision_pixels=args.minimum_collision_pixels,
      minimum_gaussian_coverage=args.minimum_gaussian_coverage,
      minimum_aligned_fraction=args.minimum_aligned_fraction,
    )
    stem = spawn_point.removeprefix("safe_velocity_")
    if args.save_raw_depth:
      np.save(args.output / f"{stem}-gaussian-depth.npy", gaussian_depth)
      np.save(args.output / f"{stem}-collision-depth.npy", collision_depth)
    _write_overlay(
      args.output / f"{stem}-overlay.png",
      gaussian_depth,
      collision_depth,
      cutoff_distance_m=args.cutoff_distance,
      tolerance_m=args.tolerance,
      scale=args.overlay_scale,
    )
    results.append(
      {
        "spawn_point": spawn_point,
        "camera": gaussian_metadata,
        **metrics,
      }
    )
    print(json.dumps(results[-1], indent=2), flush=True)

  summary = {
    "total": len(results),
    "accepted": sum(result["accepted"] for result in results),
    "all_accepted": all(result["accepted"] for result in results),
    "minimum_gaussian_coverage": min(result["gaussian_coverage"] for result in results),
    "minimum_aligned_fraction": min(result["aligned_fraction"] for result in results),
  }
  report = {
    "date": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "scope": "3DGS versus offline collision reference; excludes added visible meshes",
    "reference_geom_groups": args.reference_geom_groups,
    "reference_depth_is_policy_input": False,
    "policy_depth_sources": ["3dgs", "visible_mesh"],
    "scene_manifest": str(manifest_path),
    "scene_spawn_prefix": args.scene_spawn_prefix,
    "seed": args.seed,
    "criteria": {
      "cutoff_distance_m": args.cutoff_distance,
      "tolerance_m": args.tolerance,
      "minimum_collision_pixels": args.minimum_collision_pixels,
      "minimum_gaussian_coverage": args.minimum_gaussian_coverage,
      "minimum_aligned_fraction": args.minimum_aligned_fraction,
      "overlay_scale": args.overlay_scale,
      "raw_depth_saved": args.save_raw_depth,
    },
    "summary": summary,
    "results": results,
    "overlay_legend": {
      "green": "collision pixel with Gaussian depth within tolerance",
      "red": "collision pixel with Gaussian depth outside tolerance",
      "magenta": "collision pixel without valid Gaussian depth",
      "grayscale": "inverse Gaussian depth outside collision projection",
    },
  }
  report_path = args.output / "report.json"
  report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(summary, indent=2), flush=True)
  print(f"Wrote {report_path}", flush=True)
  if args.require_pass and not summary["all_accepted"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
