"""CPU checks for renderer fingerprints and matched-frame comparisons."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def audit_modules(monkeypatch):
  monkeypatch.syspath_prepend(
    str(Path(__file__).resolve().parents[1] / "scripts" / "benchmarks")
  )
  return (
    importlib.import_module("gaussian_build_identity"),
    importlib.import_module("compare_3dgs_captures"),
  )


def test_fingerprint_detects_native_change_ignores_bytecode(tmp_path, audit_modules):
  identity, _ = audit_modules
  (tmp_path / "libmujoco.so.3.16.0").write_bytes(b"first binary")
  (tmp_path / "renderer.py").write_text("pass\n")
  before = identity.fingerprint_tree(tmp_path)
  cache = tmp_path / "__pycache__"
  cache.mkdir()
  (cache / "renderer.pyc").write_bytes(b"runtime cache")
  assert identity.fingerprint_tree(tmp_path)["tree_sha256"] == before["tree_sha256"]
  (tmp_path / "libmujoco.so.3.16.0").write_bytes(b"second binary, same version")
  assert identity.fingerprint_tree(tmp_path)["tree_sha256"] != before["tree_sha256"]


@pytest.fixture
def capture_pair(tmp_path):
  reports = []
  for name in ("reference", "candidate"):
    directory = tmp_path / name
    directory.mkdir()
    np.savez(
      directory / "reset.npz",
      rgb=np.zeros((1, 2, 2, 3), dtype=np.uint8),
      depth=np.ones((1, 2, 2), dtype=np.float32),
      qpos=np.zeros((1, 7)),
    )
    report = {"status": "passed", "reset_capture": {"directory": str(directory)}}
    for key in (
      "camera_capture",
      "manifest_sha256",
      "source_sha256",
      "collision_sha256",
      "scene_assets",
      "loaded_splats_shared_across_worlds",
    ):
      report[key] = "matching fixture value"
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(report))
    reports.append(path)
  return reports


def test_comparison_exact_and_unsigned_rgb_difference(capture_pair, audit_modules):
  _, compare = audit_modules
  reference, candidate = capture_pair
  result = compare.compare_reports(reference, candidate)
  assert result["rgb_exact"] and result["depth_exact"]
  directory = Path(json.loads(candidate.read_text())["reset_capture"]["directory"])
  with np.load(directory / "reset.npz") as z:
    data = {key: z[key] for key in z.files}
  data["rgb"].fill(255)
  data["depth"].fill(0)
  np.savez(directory / "reset.npz", **data)
  result = compare.compare_reports(reference, candidate)
  assert result["rgb_mae_0_255"] == 255.0
  assert result["reference_valid_depth_lost_fraction"] == 1.0
  assert result["common_valid_depth_mae_m"] is None
  assert not result["rgb_exact"] and not result["depth_exact"]


@pytest.mark.parametrize("mismatch", ["qpos", "collision"])
def test_comparison_rejects_unmatched_inputs(capture_pair, audit_modules, mismatch):
  _, compare = audit_modules
  reference, candidate = capture_pair
  report = json.loads(candidate.read_text())
  if mismatch == "collision":
    report["collision_sha256"] = "different collision"
    candidate.write_text(json.dumps(report))
  else:
    path = Path(report["reset_capture"]["directory"]) / "reset.npz"
    with np.load(path) as z:
      data = {key: z[key] for key in z.files}
    data["qpos"][0, 0] = 1
    np.savez(path, **data)
  with pytest.raises(ValueError):
    compare.compare_reports(reference, candidate)
