"""Native viewer acceptance checks and a small, repeatable runtime comparison.

uv run tests/run_g1_3dgs_acceptance.py --checkpoint /path/to/model.pt
Metrics describe viewer loop submissions, not measured display refresh FPS.
"""

import argparse
import json
import platform
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.keys import KEY_ENTER, KEY_RIGHT, KEY_SPACE

_ROOT = Path(__file__).resolve().parents[1]


def legacy_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Match the supplied August velocity run's saved phase observation."""
  phase = env.episode_length_buf * env.step_dt % 0.6 / 0.6
  result = torch.stack(
    (torch.sin(phase * 2 * torch.pi), torch.cos(phase * 2 * torch.pi)), dim=1
  )
  command = env.command_manager.get_command("twist")
  assert command is not None
  standing = command.norm(dim=1) < 0.1
  return torch.where(standing[:, None], torch.zeros_like(result), result)


def legacy_config(task_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Explicit test-only adapter; never silently truncate checkpoint inputs."""
  cfg = load_env_cfg(task_id, play=play)
  terms = cfg.observations["actor"].terms
  terms["joint_pos"].params = {}  # Old run used unbiased relative joint positions.
  terms["phase"] = ObservationTermCfg(func=legacy_phase)
  cfg.observations["actor"].terms = {
    name: terms[name]
    for name in (
      "base_ang_vel",
      "projected_gravity",
      "command",
      "phase",
      "joint_pos",
      "joint_vel",
      "actions",
    )
  }
  return cfg


def gpu_memory() -> str:
  """Record device-wide usage, including other applications and graphics."""
  result = subprocess.run(
    [
      "nvidia-smi",
      "--query-gpu=name,driver_version,memory.used",
      "--format=csv,noheader",
    ],
    capture_output=True,
    text=True,
    check=True,
  )
  return result.stdout.strip()


def wait_ready(viewer: NativeMujocoViewer, timeout: float = 60.0) -> None:
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if not viewer.is_running():
      raise RuntimeError("Native window closed unexpectedly")
    viewer.sync_env_to_viewer()
    status = getattr(viewer.viewer, "gaussian_status", "unsupported")
    if status == "ready":
      return
    if status.startswith("error:") or status == "unsupported":
      raise RuntimeError(status)
    time.sleep(0.02)
  raise TimeoutError("3DGS renderer readiness timeout")


def check_controls(viewer: NativeMujocoViewer) -> dict:
  """Exercise the real callback and action queue without synthetic OS input."""
  assert viewer.mjd is not None and viewer.viewer is not None
  viewer._safe_key_callback(KEY_SPACE)
  viewer.tick()
  assert viewer.get_status().paused
  before = viewer.mjd.qpos.copy()
  count = viewer.get_status().step_count
  for _ in range(3):
    viewer.tick()
  np.testing.assert_array_equal(before, viewer.mjd.qpos)
  assert viewer.get_status().step_count == count
  viewer._safe_key_callback(KEY_RIGHT)
  viewer.tick()
  assert viewer.get_status().step_count == count + 1
  viewer._safe_key_callback(KEY_ENTER)
  viewer.tick()
  assert viewer.get_status().step_count == 0
  assert viewer._last_error is None
  with viewer.viewer.lock():
    azimuth = viewer.viewer.cam.azimuth
    viewer.viewer.cam.azimuth = azimuth + 15.0
  viewer.sync_env_to_viewer()
  assert viewer.viewer.cam.azimuth == azimuth + 15.0
  with viewer.viewer.lock():
    viewer.viewer.cam.azimuth = azimuth
  viewer._safe_key_callback(KEY_SPACE)
  viewer.tick()
  assert not viewer.get_status().paused
  return {
    "callback_pause_single_step_reset_resume": "passed",
    "camera_parameter_roundtrip": "passed",
    "physical_keyboard_and_mouse": "manual verification required",
  }


def check_collision_visibility(viewer: NativeMujocoViewer) -> dict:
  """Replay identical CPU states/controls with each visual mask setting."""
  model, state, handle = viewer.mjm, viewer.mjd, viewer.viewer
  assert model is not None and state is not None and handle is not None
  assert np.any(model.geom_group == 3), "No collision proxy in group 3"
  trajectories = []
  contact_counts = []
  original = int(handle.opt.geomgroup[3])
  try:
    for visible in (0, 1):
      with handle.lock():
        handle.opt.geomgroup[3] = visible
      viewer.sync_env_to_viewer()
      replay = mujoco.MjData(model)
      mujoco.mj_copyData(replay, model, state)
      rows, contacts = [], []
      for _ in range(100):
        mujoco.mj_step(model, replay)
        rows.append(np.concatenate((replay.qpos, replay.qvel)).copy())
        contacts.append(
          [(int(c.geom1), int(c.geom2), float(c.dist)) for c in replay.contact]
        )
      trajectories.append(np.stack(rows))
      contact_counts.append(contacts)
    np.testing.assert_array_equal(trajectories[0], trajectories[1])
    assert contact_counts[0] == contact_counts[1]
    total = sum(len(c) for c in contact_counts[0])
    assert total > 0, "Contact comparison would be vacuous"
    return {
      "cpu_replay_steps": 100,
      "contact_samples": total,
      "qpos_qvel_contact_pairs_and_distances": "identical",
      "scope": "CPU MuJoCo replay; excludes full MJWarp policy trajectory replay",
    }
  finally:
    with handle.lock():
      handle.opt.geomgroup[3] = original


def measure(viewer: NativeMujocoViewer, seconds: float) -> dict:
  """Warm up then sample the production paced tick loop."""
  for _ in range(10):
    if not viewer._execute_step():
      raise RuntimeError(viewer._last_error)
  viewer.sync_env_to_viewer()
  started = time.perf_counter()
  viewer._last_tick_time = started
  viewer._sim_budget = 0.0
  viewer._time_until_next_render = 0.0
  count = viewer.get_status().step_count
  frames = 0
  while time.perf_counter() - started < seconds:
    if not viewer.is_running():
      raise RuntimeError("Window closed during measurement")
    rendered = viewer.tick()
    frames += int(rendered)
    if viewer._last_error:
      raise RuntimeError(viewer._last_error)
    assert viewer.mjd is not None
    if not np.isfinite(viewer.mjd.qpos).all():
      raise AssertionError("Non-finite robot state")
    if rendered:
      np.testing.assert_array_equal(
        viewer.mjd.qpos,
        viewer.env.unwrapped.sim.data.qpos[viewer.env_idx].cpu().numpy(),
      )
      np.testing.assert_array_equal(
        viewer.mjd.qvel,
        viewer.env.unwrapped.sim.data.qvel[viewer.env_idx].cpu().numpy(),
      )
    time.sleep(0.001)
  elapsed = time.perf_counter() - started
  steps = viewer.get_status().step_count - count
  return {
    "wall_seconds": elapsed,
    "policy_steps": steps,
    "actual_rt": steps * viewer.env.unwrapped.step_dt / elapsed,
    "viewer_submissions_per_second": frames / elapsed,
    "device_wide_gpu_memory": gpu_memory(),
  }


def acceptance(source: NativeMujocoViewer, seconds: float, report: dict) -> None:
  """Reopen native windows with one already loaded environment and policy."""
  try:
    report["gpu_before_windows"] = gpu_memory()
    for enabled in (False, True, False, True):
      viewer = NativeMujocoViewer(source.env, source.policy)
      sample: dict = {"gaussians_enabled": enabled}
      report["samples"].append(sample)
      cached_path = None
      handle = None
      try:
        viewer.reset_environment()
        started = time.perf_counter()
        if enabled:
          viewer.setup()
          wait_ready(viewer)
          cached_path = viewer._prepare_gaussian_ply()
        else:
          with patch.object(viewer, "_prepare_gaussian_ply", return_value=None):
            viewer.setup()
          viewer.sync_env_to_viewer()
        sample["viewer_setup_seconds"] = time.perf_counter() - started
        handle = viewer.viewer
        sample["controls"] = check_controls(viewer)
        sample.update(measure(viewer, seconds))
        if enabled:
          sample["collision_visibility"] = check_collision_visibility(viewer)
        print(json.dumps(sample, ensure_ascii=False), flush=True)
      finally:
        viewer.close()
        if handle is not None:
          assert not handle.is_running(), "Viewer still running after close"
        if cached_path is not None:
          assert not Path(cached_path).exists(), "Exported PLY was not removed"
        sample["window_close_and_ply_cleanup"] = "passed"
    report["gpu_after_windows"] = gpu_memory()
  finally:
    source.env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--seconds", type=float, default=10.0)
  parser.add_argument("--legacy-phase-observations", action="store_true")
  parser.add_argument("--interactive", action="store_true")
  parser.add_argument("--viewer", choices=("native", "viser"), default="native")
  parser.add_argument("--collision-visible", action="store_true")
  parser.add_argument("--output", type=Path, default=Path("/tmp/p3-acceptance.json"))
  args = parser.parse_args()
  if args.seconds <= 0 or not args.checkpoint.is_file():
    parser.error("Provide an existing checkpoint and positive --seconds")
  if args.viewer == "viser" and not args.interactive:
    parser.error("--viewer viser requires --interactive")
  report = {
    "date": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "platform": platform.platform(),
    "mujoco": version("mujoco"),
    "torch": torch.__version__,
    "checkpoint": str(args.checkpoint.resolve()),
    "legacy_phase_observations": args.legacy_phase_observations,
    "method": "on/off viewer only; same loaded room1 scene; two trials each",
    "limitations": [
      "No rendered-frame capture API; no image/occlusion equivalence claim",
      "GPU memory is device-wide and is not a process allocation benchmark",
      "Viewer submissions are not display refresh FPS",
      "Viewer setup time excludes environment initialization and checkpoint load",
      "Sequential resets are not guaranteed identical policy trajectories",
    ],
    "samples": [],
  }
  try:
    with (
      patch(
        "mjlab.scripts.play.load_env_cfg",
        legacy_config if args.legacy_phase_observations else load_env_cfg,
      ),
      nullcontext()
      if args.interactive
      else patch.object(
        NativeMujocoViewer,
        "run",
        lambda self: acceptance(self, args.seconds, report),
      ),
    ):
      run_play(
        "Mjlab-Velocity-Flat-Unitree-G1-3DGS",
        PlayConfig(
          checkpoint_file=str(args.checkpoint),
          scene_manifest=str(_ROOT / "src/mjlab/asset_zoo/3dgs/room1/manifest.json"),
          num_envs=1,
          viewer=args.viewer,
          collision_visible=args.collision_visible,
        ),
      )
    report["status"] = "interactive_session_closed" if args.interactive else "passed"
  except Exception as exc:
    report["status"] = "failed"
    report["error"] = repr(exc)
    raise
  finally:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
  main()
