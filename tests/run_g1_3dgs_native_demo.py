"""Exercise the installed custom MuJoCo viewer with the real G1 room1 demo.

uv run tests/run_g1_3dgs_native_demo.py --steps 200
uv run tests/run_g1_3dgs_native_demo.py --interactive

uv selects the released custom wheel on Linux x86_64 with Python 3.12/3.13.
"""

import argparse
import importlib
import time
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mjlab.scripts.demo import DemoConfig, _resolve_demo_args
from mjlab.scripts.gcs import ensure_default_checkpoint, ensure_default_motion
from mjlab.scripts.play import run_play
from mjlab.viewer import NativeMujocoViewer

_MANIFEST = (
  Path(__file__).resolve().parents[1] / "src/mjlab/asset_zoo/3dgs/room1/manifest.json"
)


def run_smoke(viewer: NativeMujocoViewer, steps: int, timeout: float) -> None:
  """Fail on rendering/stepping errors and always release native resources."""
  try:
    viewer.setup()
    handle = viewer.viewer
    assert handle is not None
    deadline = time.monotonic() + timeout
    while True:
      if not viewer.is_running():
        raise RuntimeError("Native window closed before the smoke test completed")
      viewer.sync_env_to_viewer()
      status = getattr(handle, "gaussian_status", "unavailable")
      if status == "ready":
        break
      if status.startswith("error:") or status == "unavailable":
        raise RuntimeError(f"Native 3DGS renderer: {status}")
      if time.monotonic() >= deadline:
        raise TimeoutError(f"Native 3DGS renderer did not become ready: {status}")
      time.sleep(0.02)

    print("PASS: native 3DGS renderer ready", flush=True)
    for _ in range(steps):
      if not viewer.is_running():
        raise RuntimeError("Native window closed before all steps completed")
      if not viewer._execute_step():
        raise RuntimeError(viewer._last_error)
      viewer.sync_env_to_viewer()
      if getattr(handle, "gaussian_status", None) != "ready":
        raise RuntimeError("Native 3DGS renderer stopped being ready")
      assert viewer.mjd is not None
      if not np.isfinite(viewer.mjd.qpos).all():
        raise AssertionError("Non-finite G1 joint positions")
    print(f"PASS: pretrained G1 policy, room1 collision scene, {steps} steps")
  finally:
    viewer.close()
    viewer.env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument("--timeout", type=float, default=60.0)
  parser.add_argument("--interactive", action="store_true")
  parser.add_argument("--collision-visible", action="store_true")
  args = parser.parse_args()
  if args.steps <= 0 or args.timeout <= 0:
    parser.error("--steps and --timeout must be positive")

  # Check the installed wheel before allocating the environment or downloading.
  importlib.import_module("mujoco.gaussian")
  print(f"Installed MuJoCo: {version('mujoco')}", flush=True)
  if not _MANIFEST.is_file():
    raise FileNotFoundError(_MANIFEST)
  config = DemoConfig(
    checkpoint_file=ensure_default_checkpoint(),
    motion_file=ensure_default_motion(),
    scene_manifest=str(_MANIFEST),
    collision_visible=args.collision_visible,
    num_envs=1,
    viewer="native",
    use_3dgs=True,
    _demo_mode=True,
  )
  task_id, config = _resolve_demo_args(config)
  if args.interactive:
    run_play(task_id, config)
  else:
    # Keep the production demo's environment and policy loading unchanged.
    with patch.object(
      NativeMujocoViewer,
      "run",
      lambda self: run_smoke(self, args.steps, args.timeout),
    ):
      run_play(task_id, config)


if __name__ == "__main__":
  main()
