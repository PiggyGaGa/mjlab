"""Tests for rejecting semantic hints without discarding geometry."""

import copy
import json
import runpy
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest


def test_collision_guard_rejects_worse_semantic_geometry() -> None:
  script = (
    Path(__file__).resolve().parents[1]
    / "scripts/tools/generate_3dgs_collision_guarded.py"
  )
  guard = cast(
    Callable[..., list[str]], runpy.run_path(str(script))["semantic_rejection_reasons"]
  )
  baseline: dict[str, Any] = {
    "input": {"sha256": "input"},
    "statistics": {"filtered_gaussian_count": 10, "mapped_gaussian_fraction": 0.8},
    "source_surface_evaluation": {
      "surface_distance_mean": 0.1,
      "surface_distance_p95": 0.2,
      "surface_distance_max": 0.5,
    },
  }
  assert guard(baseline, baseline, 1.05) == []
  candidate = copy.deepcopy(baseline)
  candidate["statistics"]["mapped_gaussian_fraction"] = 0.7
  candidate["source_surface_evaluation"]["surface_distance_p95"] = 0.3
  assert guard(baseline, candidate, 1.05) == [
    "semantic_source_mapping_coverage_decreased",
    "semantic_surface_distance_p95_increased",
  ]
  candidate["input"]["sha256"] = "other"
  with pytest.raises(ValueError, match="same input"):
    guard(baseline, candidate, 1.05)
  with pytest.raises(ValueError, match="at least 1"):
    guard(baseline, baseline, 0.9)


def test_unknown_only_labels_select_geometry_without_semantic_generation(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  script = (
    Path(__file__).resolve().parents[1]
    / "scripts/tools/generate_3dgs_collision_guarded.py"
  )
  source = tmp_path / "scene.ply"
  source.write_text(
    "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n"
    "property float y\nproperty float z\nend_header\n0 0 0\n"
  )
  labels = tmp_path / "labels.npy"
  np.save(labels, np.array([-1]))
  output = tmp_path / "result"
  calls: list[list[str]] = []

  def fake_generator(command: list[str], *, check: bool) -> None:
    assert check
    assert "--semantic-labels" not in command
    calls.append(command)
    folder = Path(command[3]).parent
    folder.mkdir(parents=True)
    (folder / "collision.xml").write_text("<mujoco/>")
    (folder / "report.json").write_text("{}")
    (folder / "report.source_ids.npz").write_bytes(b"source mapping")

  monkeypatch.setattr(subprocess, "run", fake_generator)
  monkeypatch.setattr(sys, "argv", [str(script), str(source), str(labels), str(output)])
  main = cast(Callable[[], None], runpy.run_path(str(script))["main"])
  main()
  assert len(calls) == 1
  report = json.loads((output / "guard_report.json").read_text())
  assert report["selected"] == "geometry"
  assert not report["semantic_used"]
  assert report["unknown_fraction"] == 1
  assert report["semantic_candidate"] is None
  assert (output / "collision.xml").read_bytes() == (
    output / "geometry/collision.xml"
  ).read_bytes()
  np.testing.assert_array_equal(np.load(labels), [-1])
