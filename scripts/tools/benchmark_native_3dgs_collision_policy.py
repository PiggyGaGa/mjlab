"""Benchmark G1 policy stepping with the native 3DGS viewer and collision variants."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from mjlab.scripts.benchmark_g1_collision_policy import (
  _sha256,
  collision_bundle_hashes,
  load_variant_manifest,
)
from mjlab.scripts.demo import DemoConfig, _resolve_demo_args
from mjlab.scripts.play import run_play
from mjlab.viewer import NativeMujocoViewer


def _mean(values: list[float]) -> float:
  return sum(values) / len(values)


def _run_viewer_benchmark(
  viewer: NativeMujocoViewer,
  *,
  warmup_steps: int,
  measured_steps: int,
  ready_timeout: float,
  result: dict[str, object],
) -> None:
  setup_start = time.perf_counter()
  try:
    viewer.setup()
    handle = viewer.viewer
    assert handle is not None
    deadline = time.monotonic() + ready_timeout
    while True:
      if not viewer.is_running():
        raise RuntimeError("Native window closed before the renderer became ready.")
      viewer.sync_env_to_viewer()
      gaussian_status = getattr(handle, "gaussian_status", "unavailable")
      if gaussian_status == "ready":
        break
      if gaussian_status == "unavailable" or gaussian_status.startswith("error:"):
        raise RuntimeError(f"Native 3DGS renderer: {gaussian_status}")
      if time.monotonic() >= deadline:
        raise TimeoutError(
          f"Native 3DGS renderer did not become ready: {gaussian_status}"
        )
      time.sleep(0.02)
    result["viewer_setup_and_renderer_ready_seconds"] = (
      time.perf_counter() - setup_start
    )

    for _ in range(warmup_steps):
      if not viewer._execute_step():
        raise RuntimeError(viewer._last_error)
      viewer.sync_env_to_viewer()
    viewer.env.reset()
    reset_policy = getattr(viewer.policy, "reset", None)
    if reset_policy is not None:
      reset_policy()

    contacts: list[float] = []
    constraints: list[float] = []
    solver_iterations: list[float] = []
    overflow_bitmask_or = 0
    measured_start = time.perf_counter()
    for _ in range(measured_steps):
      if not viewer.is_running():
        raise RuntimeError("Native window closed during the benchmark.")
      if not viewer._execute_step():
        raise RuntimeError(viewer._last_error)
      viewer.sync_env_to_viewer()
      if getattr(handle, "gaussian_status", None) != "ready":
        raise RuntimeError("Native 3DGS renderer stopped being ready.")
      assert viewer.mjd is not None
      if not np.isfinite(viewer.mjd.qpos).all():
        raise RuntimeError("Non-finite G1 qpos during native benchmark.")
      sim_data = viewer.env.unwrapped.sim.data
      contacts.append(float(sim_data.nacon[0].item()))
      constraints.append(float(sim_data.nefc[0].item()))
      solver_iterations.append(float(sim_data.solver_niter[0].sum().item()))
      overflow_bitmask_or |= int(sim_data.overflow[0].item())
    measured_seconds = time.perf_counter() - measured_start
    step_dt = float(viewer.env.unwrapped.step_dt)
    result.update(
      {
        "measured_steps": measured_steps,
        "measured_seconds": measured_seconds,
        "viewer_submissions_per_second": measured_steps / measured_seconds,
        "actual_realtime": measured_steps * step_dt / measured_seconds,
        "contacts_total_mean": _mean(contacts),
        "contacts_total_max": max(contacts),
        "constraints_mean": _mean(constraints),
        "solver_iterations_mean": _mean(solver_iterations),
        "overflow_bitmask_or": overflow_bitmask_or,
        "gaussian_status": "ready",
      }
    )
  finally:
    viewer.close()


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("variant_manifest", type=Path)
  parser.add_argument("--gaussian-splat-file", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--motion", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--warmup-steps", type=int, default=20)
  parser.add_argument("--measured-steps", type=int, default=500)
  parser.add_argument("--ready-timeout", type=float, default=60.0)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.warmup_steps < 0 or args.measured_steps <= 0 or args.ready_timeout <= 0:
    raise ValueError("Warmup must be non-negative; steps and timeout must be positive.")
  gaussian = args.gaussian_splat_file.expanduser().resolve()
  checkpoint = args.checkpoint.expanduser().resolve()
  motion = args.motion.expanduser().resolve()
  for label, path in (
    ("Gaussian splat", gaussian),
    ("checkpoint", checkpoint),
    ("motion", motion),
  ):
    if not path.is_file():
      raise FileNotFoundError(f"{label} not found: {path}")

  variants = load_variant_manifest(args.variant_manifest)
  report: dict[str, Any] = {
    "schema_version": 1,
    "protocol": {
      "device": args.device,
      "warmup_steps": args.warmup_steps,
      "measured_steps": args.measured_steps,
      "rendering": "native MuJoCo 3DGS viewer; one sync submission per policy step",
      "terminations_enabled": False,
      "limitation": (
        "Viewer submissions are not display refresh FPS; actual_realtime includes "
        "policy, MuJoCo Warp, CPU viewer synchronization, and native 3DGS submission."
      ),
    },
    "versions": {
      "mujoco": version("mujoco"),
      "torch": version("torch"),
    },
    "inputs": {
      "gaussian_splat": {"path": str(gaussian), "sha256": _sha256(gaussian)},
      "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
      "motion": {"path": str(motion), "sha256": _sha256(motion)},
    },
    "results": [],
  }

  for variant in variants:
    print(f"[INFO] Native 3DGS benchmark: {variant.name}", flush=True)
    result: dict[str, object] = {
      "name": variant.name,
      "geometry": variant.geometry,
      "collision": (
        collision_bundle_hashes(variant.collision_mjcf)
        if variant.collision_mjcf is not None
        else None
      ),
      "keep_plane": variant.keep_plane,
    }
    config = DemoConfig(
      checkpoint_file=str(checkpoint),
      motion_file=str(motion),
      gaussian_splat_file=str(gaussian),
      collision_mjcf_file=(
        str(variant.collision_mjcf) if variant.collision_mjcf is not None else None
      ),
      collision_keep_plane=variant.keep_plane,
      num_envs=1,
      device=args.device,
      viewer="native",
      no_terminations=True,
      use_3dgs=True,
      _demo_mode=False,
    )
    task_id, config = _resolve_demo_args(config)

    def run_benchmark(
      viewer: NativeMujocoViewer, result: dict[str, object] = result
    ) -> None:
      _run_viewer_benchmark(
        viewer,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        ready_timeout=args.ready_timeout,
        result=result,
      )

    with patch.object(NativeMujocoViewer, "run", run_benchmark):
      run_play(task_id, config)
    report["results"].append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

  print(f"[INFO] Wrote native 3DGS benchmark: {args.output}", flush=True)


if __name__ == "__main__":
  main()
