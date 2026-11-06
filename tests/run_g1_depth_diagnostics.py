"""Compare robot-only, Gaussian-only, and combined depth at identical poses."""

import argparse
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import torch
from PIL import Image

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import apply_scene_manifest
from mjlab.sensor import CameraSensor, CameraSensorCfg
from mjlab.tasks.g1_3dgs import G1_3DGS_DEPTH_SENSOR
from mjlab.tasks.registry import load_env_cfg


def statistics(raw: np.ndarray) -> dict:
  valid = raw[np.isfinite(raw) & (raw > 0)]
  return {
    "zero_fraction": float((raw == 0).mean()),
    "positive_near_fraction": float(((raw > 0) & (raw <= 0.05)).mean()),
    "beyond_cutoff_fraction": float((raw >= 5).mean()),
    "nonfinite_fraction": float((~np.isfinite(raw)).mean()),
    "positive_depth_quantiles_m": np.quantile(valid, [0, 0.1, 0.5, 0.9, 1]).tolist()
    if valid.size
    else [],
  }


def capture(env: ManagerBasedRlEnv, sensor: CameraSensor) -> np.ndarray:
  env.sim.forward()
  env.sim.sense()
  sensor.update(0)
  raw = sensor.data.depth
  assert raw is not None
  result = raw[0, :, :, 0].cpu().numpy().copy()
  assert np.isfinite(result).all()
  return result


def pose_sweep(anchor: torch.Tensor, model: mujoco.MjModel) -> dict[str, torch.Tensor]:
  """Controlled configurations, not a physically simulated motion trajectory."""
  poses = {"reset": anchor.clone()}
  for label, axis, degrees in (
    ("yaw_left", 2, 90),
    ("yaw_right", 2, -90),
    ("pitch_forward", 1, 15),
    ("pitch_back", 1, -15),
  ):
    pose = anchor.clone()
    half_angle = math.radians(degrees) / 2
    delta = np.zeros(4)
    delta[0] = math.cos(half_angle)
    delta[axis + 1] = math.sin(half_angle)
    result = np.empty(4)
    mujoco.mju_mulQuat(result, anchor[0, 3:7].cpu().numpy(), delta)
    pose[0, 3:7] = torch.as_tensor(result, device=anchor.device, dtype=anchor.dtype)
    poses[label] = pose
  for side in ("left", "right"):
    pose = anchor.clone()
    joint = model.joint(f"robot/{side}_shoulder_pitch_joint")
    pose[0, int(joint.qposadr[0])] = -math.pi / 2
    poses[f"{side}_arm_raised"] = pose
  return poses


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--scene-manifest",
    type=Path,
    default=Path("src/mjlab/asset_zoo/3dgs/room1/manifest.json"),
  )
  parser.add_argument("--output", type=Path, default=Path("/tmp/p6-depth-diagnostics"))
  parser.add_argument("--pose-sweep", action="store_true")
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  report: dict = {
    "date": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "cases": {},
    "scope": "Fixed reset pose; ray-cast Gaussian opacity depth",
  }
  anchor = None
  frames = {}
  sweep_frames = {}
  for case in ("combined", "robot_only", "gaussian_only"):
    print(f"Capturing {case}", flush=True)
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth", play=True)
    cfg.scene.num_envs = 1
    cfg.seed = 42
    apply_scene_manifest(cfg.scene, args.scene_manifest)
    if case == "robot_only":
      cfg.scene.gaussian_splats = None
    if case == "gaussian_only":
      # Avoid the installed renderer's empty mesh BVH crash with an offscreen geom.
      cfg.scene.entities["depth_placeholder"] = EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(
          '<mujoco><worldbody><geom type="sphere" size=".1" '
          'pos="1000 1000 1000" group="5" contype="0" conaffinity="0"/>'
          "</worldbody></mujoco>"
        )
      )
    cfg.scene.sensors = tuple(
      replace(s, enabled_geom_groups=(5,) if case == "gaussian_only" else (2,))
      if isinstance(s, CameraSensorCfg)
      else s
      for s in cfg.scene.sensors or ()
    )
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
      env.reset(seed=42)
      if anchor is None:
        anchor = env.sim.data.qpos.clone()
      env.sim.data.qpos.copy_(anchor)
      env.sim.data.qvel.zero_()
      sensor = env.scene[G1_3DGS_DEPTH_SENSOR]
      assert isinstance(sensor, CameraSensor)
      raw = capture(env, sensor)
      repeated = capture(env, sensor)
      np.testing.assert_array_equal(raw, repeated)
      np.save(args.output / f"{case}-raw.npy", raw)
      frames[case] = raw
      if args.pose_sweep:
        sweep_frames[case] = {}
        for label, state in pose_sweep(anchor, env.sim.mj_model).items():
          env.sim.data.qpos.copy_(state)
          frame = capture(env, sensor)
          np.testing.assert_array_equal(frame, capture(env, sensor))
          sweep_frames[case][label] = frame
          np.save(args.output / f"{case}-{label}-raw.npy", frame)
          Image.fromarray(
            (np.clip(np.where(frame == 0, 5, frame), 0.05, 5) / 5 * 255).astype(
              np.uint8
            )
          ).save(args.output / f"{case}-{label}.png")
        env.sim.data.qpos.copy_(anchor)
        capture(env, sensor)
      normalized = np.clip(np.where(raw == 0, 5, raw), 0.05, 5) / 5
      Image.fromarray((normalized * 255).astype(np.uint8)).save(
        args.output / f"{case}.png"
      )
      ids = sensor.camera_idx
      mjd = env.sim.mj_data
      mjd.qpos[:] = anchor[0].cpu().numpy()
      mujoco.mj_forward(env.sim.mj_model, mjd)
      rotation = mjd.cam_xmat[ids].reshape(3, 3)
      forward = -rotation[:, 2]
      robot = env.scene["robot"]
      pose = robot.data.root_link_pose_w.clone()
      pose[:, 0] += 0.5
      robot.write_root_link_pose_to_sim(pose)
      shifted = capture(env, sensor)
      report["cases"][case] = {
        **statistics(raw),
        "fixed_pose_repeat_max_abs_delta": float(np.abs(raw - repeated).max()),
        "shift_0_5m_max_abs_delta": float(np.abs(raw - shifted).max()),
        "shift_0_5m_changed_pixel_fraction": float(
          (np.abs(raw - shifted) > 1e-5).mean()
        ),
        "camera_position_world": mjd.cam_xpos[ids].tolist(),
        "camera_forward_world": forward.tolist(),
        "camera_below_horizon_degrees": float(np.degrees(np.arcsin(-forward[2]))),
      }
      print(json.dumps(report["cases"][case]), flush=True)
      if case in ("combined", "gaussian_only"):
        assert np.any(np.abs(raw - shifted) > 1e-5)
    finally:
      env.close()
  report["combined_vs_gaussian_only_max_abs_delta"] = float(
    np.abs(frames["combined"] - frames["gaussian_only"]).max()
  )
  report["gaussian_only_placeholder"] = (
    "Non-colliding group-5 sphere at (1000,1000,1000); avoids empty BVH crash"
  )
  if args.pose_sweep:
    report["pose_sweep"] = {
      label: {
        "combined": statistics(sweep_frames["combined"][label]),
        "robot_only_hit_fraction": float(
          (sweep_frames["robot_only"][label] > 0).mean()
        ),
        "combined_vs_gaussian_changed_pixel_fraction": float(
          (
            np.abs(
              sweep_frames["combined"][label] - sweep_frames["gaussian_only"][label]
            )
            > 1e-5
          ).mean()
        ),
      }
      for label in sweep_frames["combined"]
    }
    report["pose_sweep_scope"] = (
      "Seven controlled poses; not physical rollout or all-pose validation"
    )
  (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
