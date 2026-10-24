"""Tests for bundled demo configuration resolution."""

from mjlab.scripts.demo import DemoConfig, _resolve_demo_args


def test_demo_ignores_gaussian_splat_file_when_3dgs_is_disabled() -> None:
  task_id, args = _resolve_demo_args(
    DemoConfig(use_3dgs=False, gaussian_splat_file="/does/not/exist.ply")
  )

  assert task_id == "Mjlab-Tracking-Flat-Unitree-G1"
  assert args.gaussian_splat_file is None
  assert args.num_envs == 8


def test_demo_preserves_gaussian_splat_file_when_3dgs_is_enabled() -> None:
  task_id, args = _resolve_demo_args(
    DemoConfig(use_3dgs=True, gaussian_splat_file="/path/to/scene.ply")
  )

  assert task_id == "Mjlab-Tracking-Flat-Unitree-G1-3DGS"
  assert args.gaussian_splat_file == "/path/to/scene.ply"
  assert args.num_envs == 1
