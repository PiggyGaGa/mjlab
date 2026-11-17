"""Play the depth-stop safety layer in a manifest-selected 3DGS map.

The environment configuration is built from the manifest before contact sensors
are created. This differs from the generic post-configuration scene override and
ensures base-link collision diagnostics refer to the selected map's MJCF geoms.
"""

import argparse
from pathlib import Path

from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls, register_mjlab_task
from mjlab.tasks.velocity.config.g1.safe_velocity import (
  SAFE_VELOCITY_DEPTH_STOP_TASK_ID,
  unitree_g1_safe_velocity_depth_stop_3dgs_env_cfg,
)

SAFE_VELOCITY_MAP_PLAY_TASK_ID = "Mjlab-Velocity-Safe-Depth-Stop-Map-Unitree-G1-3DGS"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-file", type=Path, required=True)
  parser.add_argument("--scene-manifest", type=Path, required=True)
  parser.add_argument("--scene-spawn-point", default=None)
  parser.add_argument(
    "--nominal-command",
    type=float,
    nargs=3,
    metavar=("VX", "VY", "WZ"),
    default=(0.5, 0.0, 0.0),
  )
  parser.add_argument("--device", default=None)
  parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
  parser.add_argument("--collision-visible", action="store_true")
  args = parser.parse_args()

  checkpoint = args.checkpoint_file.expanduser().resolve()
  manifest = args.scene_manifest.expanduser().resolve()
  if not checkpoint.is_file():
    parser.error(f"Checkpoint not found: {checkpoint}")
  if not manifest.is_file():
    parser.error(f"Scene manifest not found: {manifest}")

  nominal_command = tuple(args.nominal_command)
  env_cfg = unitree_g1_safe_velocity_depth_stop_3dgs_env_cfg(
    play=True,
    scene_manifest=manifest,
    scene_spawn_point=args.scene_spawn_point,
    nominal_command=nominal_command,
  )
  register_mjlab_task(
    task_id=SAFE_VELOCITY_MAP_PLAY_TASK_ID,
    env_cfg=env_cfg,
    play_env_cfg=env_cfg,
    rl_cfg=load_rl_cfg(SAFE_VELOCITY_DEPTH_STOP_TASK_ID),
    runner_cls=load_runner_cls(SAFE_VELOCITY_DEPTH_STOP_TASK_ID),
  )
  print(f"[INFO]: Safe-velocity scene configured before environment build: {manifest}")
  print(f"[INFO]: Nominal body-frame command: {nominal_command}")
  run_play(
    SAFE_VELOCITY_MAP_PLAY_TASK_ID,
    PlayConfig(
      checkpoint_file=str(checkpoint),
      num_envs=1,
      device=args.device,
      viewer=args.viewer,
      collision_visible=args.collision_visible,
    ),
  )


if __name__ == "__main__":
  main()
