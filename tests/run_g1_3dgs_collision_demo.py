"""Run the bundled G1 demo policy in the room1 3DGS collision test scene.

Run from the repository root with:

  uv run tests/run_g1_3dgs_collision_demo.py
"""

from pathlib import Path

from mjlab.scripts.demo import DemoConfig, _resolve_demo_args
from mjlab.scripts.gcs import ensure_default_checkpoint, ensure_default_motion
from mjlab.scripts.play import run_play

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROOM1_DIR = _REPO_ROOT / "src/mjlab/asset_zoo/3dgs/room1"
_SCENE_MANIFEST_PATH = _ROOM1_DIR / "manifest.json"


def main() -> None:
  """Load the standard demo policy with room1 visual and collision assets."""
  if not _SCENE_MANIFEST_PATH.is_file():
    raise FileNotFoundError(f"Required test asset not found: {_SCENE_MANIFEST_PATH}")

  checkpoint_path = ensure_default_checkpoint()
  motion_path = ensure_default_motion()
  args = DemoConfig(
    checkpoint_file=checkpoint_path,
    motion_file=motion_path,
    scene_manifest=str(_SCENE_MANIFEST_PATH),
    collision_visible=True,
    num_envs=1,
    viewer="viser",
    use_3dgs=True,
    _demo_mode=True,
  )
  task_id, args = _resolve_demo_args(args)
  run_play(task_id, args)


if __name__ == "__main__":
  main()
