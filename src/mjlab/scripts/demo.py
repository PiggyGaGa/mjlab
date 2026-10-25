"""Script to run a tracking demo with a pretrained policy.

This demo downloads a pretrained checkpoint and motion file from cloud storage
and launches an interactive viewer with a humanoid robot performing a cartwheel.
"""

from dataclasses import dataclass, replace

import tyro

import mjlab
from mjlab.scripts.gcs import ensure_default_checkpoint, ensure_default_motion
from mjlab.scripts.play import PlayConfig, run_play


@dataclass(frozen=True)
class DemoConfig(PlayConfig):
  """Configuration for the bundled pretrained tracking demo."""

  use_3dgs: bool = False
  """Run the pretrained tracking policy in the configured 3DGS scene."""


def _resolve_demo_args(args: DemoConfig) -> tuple[str, DemoConfig]:
  """Resolve task-specific demo defaults and discard inactive 3DGS options."""
  if not args.use_3dgs and args.scene_manifest is not None:
    raise ValueError("--scene-manifest requires --use-3dgs True")
  if args.num_envs is None:
    args = replace(args, num_envs=1 if args.use_3dgs else 8)
  if not args.use_3dgs and args.gaussian_splat_file is not None:
    args = replace(args, gaussian_splat_file=None)
  task_id = (
    "Mjlab-Tracking-Flat-Unitree-G1-3DGS"
    if args.use_3dgs
    else "Mjlab-Tracking-Flat-Unitree-G1"
  )
  return task_id, args


def main() -> None:
  """Run demo with pretrained tracking policy."""
  print("🎮 Setting up mjlab demo with pretrained tracking policy...")

  try:
    checkpoint_path = ensure_default_checkpoint()
    motion_path = ensure_default_motion()
  except RuntimeError as e:
    print(f"❌ Failed to download demo assets: {e}")
    print("Please check your internet connection and try again.")
    return

  args = tyro.cli(
    DemoConfig,
    default=DemoConfig(
      checkpoint_file=checkpoint_path,
      motion_file=motion_path,
      viewer="viser",
      _demo_mode=True,
    ),
    config=mjlab.TYRO_FLAGS,
  )
  task_id, args = _resolve_demo_args(args)
  run_play(task_id, args)


if __name__ == "__main__":
  main()
