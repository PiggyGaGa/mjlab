"""CPU regression for benchmark rollout/reset and autograd boundaries."""

import importlib
import json
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tensordict import TensorDict


def test_rollout_history_can_reset_and_ppo_can_backpropagate(monkeypatch):
  monkeypatch.syspath_prepend(
    str(Path(__file__).resolve().parents[1] / "scripts" / "benchmarks")
  )
  benchmark = importlib.import_module("measure_3dgs_throughput")
  weight = torch.nn.Parameter(torch.ones(1))
  history = torch.zeros(1, 3)
  observations = TensorDict({"camera": torch.ones(1, 1)}, batch_size=[1])
  step_count = 0

  def act(obs):
    assert not torch.is_grad_enabled()
    return obs["camera"] * weight

  def step(actions):
    nonlocal history, step_count
    assert not actions.requires_grad
    # ContactSensor._update_history replaces its persistent buffers using roll.
    history = history.roll(1, dims=1)
    history[:, 0] = actions[:, 0]
    step_count += 1
    return observations, torch.ones(1), torch.zeros(1), {}

  def compute_returns(obs):
    assert not torch.is_grad_enabled()

  runner: Any = SimpleNamespace(
    alg=SimpleNamespace(
      act=act,
      process_env_step=lambda *args: None,
      compute_returns=compute_returns,
    )
  )
  wrapped: Any = SimpleNamespace(step=step)
  for _ in range(2):
    observations = benchmark.collect_rollout(runner, wrapped, observations, 4)
    assert not torch.is_inference(history)
    # Reproduce the reset mutation from the user's traceback outside rollout.
    history[:] = 0
    assert torch.is_grad_enabled()
    weight.grad = None
    (weight.square().sum()).backward()
    assert weight.grad is not None
  assert step_count == 8


@pytest.fixture
def capture_manifest(tmp_path):
  root = Path(__file__).resolve().parents[1]
  template = root / "src/mjlab/asset_zoo/3dgs/apartment/manifest.json"
  bundle = json.loads(template.read_text())
  bundle["visual"]["path"] = "scene.ply"
  bundle["collision"]["path"] = "collision.xml"
  (tmp_path / "scene.ply").write_bytes(b"not rendered in this CPU config test")
  (tmp_path / "collision.xml").write_text("<mujoco><worldbody/></mujoco>")
  manifest = tmp_path / "manifest.json"
  manifest.write_text(json.dumps(bundle))
  return manifest


def test_camera_control_preserves_collision_and_policy(monkeypatch, capture_manifest):
  root = Path(__file__).resolve().parents[1]
  monkeypatch.syspath_prepend(str(root / "scripts" / "benchmarks"))
  benchmark = importlib.import_module("measure_3dgs_throughput")
  manifest = capture_manifest
  tmp_path = manifest.parent
  configs = []

  def stop_before_simulation(*, cfg, device):
    configs.append(cfg)
    raise InterruptedError("Stop before GPU allocation")

  monkeypatch.setattr(benchmark, "ManagerBasedRlEnv", stop_before_simulation)
  monkeypatch.setattr(benchmark, "configure_torch_backends", lambda: None)
  reports = []
  for disabled, cap, spawn, yaw in (
    (False, None, None, 0.0),
    (True, None, None, 0.0),
    (False, 50000, None, 0.0),
    (False, 200000, "p7_free", 90.0),
  ):
    report = {}
    with pytest.raises(InterruptedError, match="Stop before GPU"):
      benchmark.run(
        Namespace(
          scene_manifest=manifest,
          num_envs=4,
          width=128,
          height=96,
          steps=32,
          device="cuda:0",
          without_gaussians=disabled,
          max_splats=cap,
          spawn_point=spawn,
          view_yaw_degrees=yaw,
        ),
        report,
      )
    reports.append(report)
  assert configs[0].scene.gaussian_splats is not None
  assert configs[1].scene.gaussian_splats is None
  assert asdict(configs[0].scene.terrain) == asdict(configs[1].scene.terrain)
  assert configs[1].scene.terrain.mjcf_path == str(tmp_path / "collision.xml")
  assert reports[0]["agent_cfg"] == reports[1]["agent_cfg"]
  assert reports[0]["collision_sha256"] == reports[1]["collision_sha256"]
  assert reports[0]["input_scene_assets"] == reports[1]["input_scene_assets"]
  assert reports[0]["gaussians_enabled"] is True
  assert reports[1]["gaussians_enabled"] is False
  assert reports[1]["scene_assets"] is None
  assert configs[2].scene.gaussian_splats.assets[0].max_splats == 50000
  assert configs[0].scene.gaussian_splats.assets[0].max_splats is None
  assert asdict(configs[0].scene.terrain) == asdict(configs[2].scene.terrain)
  assert reports[0]["agent_cfg"] == reports[2]["agent_cfg"]
  assert reports[0]["collision_sha256"] == reports[2]["collision_sha256"]
  assert reports[0]["input_scene_assets"] == reports[2]["input_scene_assets"]
  assert reports[0]["camera_capture"] == reports[2]["camera_capture"]
  assert configs[3].scene.spawn_position == (0.0, -2.6, 0.0)
  assert configs[3].events["reset_base"].params["pose_range"]["yaw"] == (
    pytest.approx(torch.pi / 2),
    pytest.approx(torch.pi / 2),
  )
  assert reports[3]["requested_view"]["spawn_point"] == "p7_free"
  assert reports[3]["collision_sha256"] == reports[0]["collision_sha256"]
  assert (
    reports[0]["camera_capture"]
    == reports[1]["camera_capture"]
    == {
      "width": 128,
      "height": 96,
      "data_types": ["rgb", "depth"],
    }
  )


def test_static_capture_closes_without_constructing_ppo(monkeypatch, capture_manifest):
  root = Path(__file__).resolve().parents[1]
  monkeypatch.syspath_prepend(str(root / "scripts" / "benchmarks"))
  benchmark = importlib.import_module("measure_3dgs_throughput")
  operations = []
  env = SimpleNamespace(
    scene=SimpleNamespace(sensor_context=None), close=lambda: operations.append("close")
  )
  monkeypatch.setattr(benchmark, "ManagerBasedRlEnv", lambda **kwargs: env)
  monkeypatch.setattr(benchmark, "configure_torch_backends", lambda: None)
  monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)

  def capture(actual_env, output):
    assert actual_env is env
    operations.append("capture")
    return {"directory": str(output)}

  def forbidden_runner(*args, **kwargs):
    pytest.fail("Static capture must not construct a PPO runner or wrapper")

  monkeypatch.setattr(benchmark, "save_reset_capture", capture)
  monkeypatch.setattr(benchmark, "RslRlVecEnvWrapper", forbidden_runner)
  monkeypatch.setattr(benchmark, "MjlabOnPolicyRunner", forbidden_runner)
  report = {}
  benchmark.run(
    Namespace(
      scene_manifest=capture_manifest,
      num_envs=1,
      width=64,
      height=48,
      steps=32,
      device="cuda:0",
      without_gaussians=False,
      max_splats=200000,
      spawn_point="default",
      view_yaw_degrees=180.0,
      capture_only=True,
      save_capture=False,
      output=capture_manifest.parent / "view.json",
    ),
    report,
  )
  assert operations == ["capture", "close"]
  assert report["mode"] == "static_capture_only_no_ppo"
  assert "samples" not in report
