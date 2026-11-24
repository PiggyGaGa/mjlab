"""Tests for shared-obstacle normalization in collision replacement experiments."""

import hashlib
import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mujoco
import numpy as np
import pytest


def _prepare() -> Callable[..., Path]:
  script = (
    Path(__file__).resolve().parents[1]
    / "scripts/tools/prepare_collision_comparison.py"
  )
  return cast(Callable[..., Path], runpy.run_path(str(script))["prepare_comparison"])


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
  reference = tmp_path / "reference.xml"
  reference.write_text(
    '<mujoco><default><geom type="box" friction="2 0.02 0.003" '
    'condim="4" solref="0.03 1" group="1"/></default><worldbody>'
    '<geom name="ground" size="2 2 0.1" pos="0 0 -0.1"/>'
    '<geom name="common" size="0.1 0.2 0.5" pos="1 0 0.5"/>'
    "</worldbody></mujoco>"
  )
  automatic = tmp_path / "automatic.xml"
  automatic.write_text(
    '<mujoco><default><geom friction="0.3 0.001 0.0001" group="3"/>'
    '</default><worldbody><geom name="proxy" type="box" size="2 2 0.1" '
    'pos="0 0 -0.1"/></worldbody></mujoco>'
  )
  visual = tmp_path / "visual.ply"
  visual.write_bytes(b"visual identity only; no PLY parsing in preparation")
  return reference, automatic, visual


def test_common_geom_physics_and_visual_attributes_match_without_mutating_inputs(
  tmp_path: Path,
) -> None:
  reference, automatic, visual = _inputs(tmp_path)
  originals = [path.read_bytes() for path in (reference, automatic, visual)]
  output = tmp_path / "comparison"
  manifest = _prepare()(reference, automatic, visual, output, ["common"])
  assert manifest.is_file()
  models = [
    mujoco.MjModel.from_xml_path(str(output / name))
    for name in ("reference.xml", "automatic.xml")
  ]
  indices = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "common") for model in models
  ]
  for field in (
    "type",
    "pos",
    "quat",
    "size",
    "friction",
    "solref",
    "solimp",
    "rgba",
    "group",
    "contype",
    "conaffinity",
    "condim",
    "margin",
    "gap",
    "priority",
    "solmix",
  ):
    np.testing.assert_array_equal(
      getattr(models[0], f"geom_{field}")[indices[0]],
      getattr(models[1], f"geom_{field}")[indices[1]],
    )
  assert int(models[1].geom_type[indices[1]]) == int(mujoco.mjtGeom.mjGEOM_BOX)
  assert models[0].ngeom == models[1].ngeom == 2
  assert [path.read_bytes() for path in (reference, automatic, visual)] == originals
  report: dict[str, Any] = json.loads((output / "preparation.json").read_text())
  assert report["visual"]["sha256"] == hashlib.sha256(originals[2]).hexdigest()
  assert len(report["outputs"]) == 3


def test_preparation_rejects_missing_duplicate_and_conflicting_common_geoms(
  tmp_path: Path,
) -> None:
  reference, automatic, visual = _inputs(tmp_path)
  prepare = _prepare()
  with pytest.raises(ValueError, match="not found"):
    prepare(reference, automatic, visual, tmp_path / "missing", ["missing"])
  with pytest.raises(ValueError, match="unique"):
    prepare(reference, automatic, visual, tmp_path / "duplicate", ["common", "common"])
  automatic.write_text(reference.read_text())
  with pytest.raises(ValueError, match="already contains"):
    prepare(reference, automatic, visual, tmp_path / "conflict", ["common"])
  with pytest.raises(ValueError, match="outside"):
    prepare(reference, automatic, visual, tmp_path, [])


def test_preparation_rejects_nonprimitive_bundles(tmp_path: Path) -> None:
  reference, automatic, visual = _inputs(tmp_path)
  automatic.write_text("<mujoco><worldbody><body/></worldbody></mujoco>")
  with pytest.raises(ValueError, match="worldbody primitive"):
    _prepare()(reference, automatic, visual, tmp_path / "invalid", [])


def test_preparation_binds_generation_report_and_is_reproducible(
  tmp_path: Path,
) -> None:
  reference, automatic, visual = _inputs(tmp_path)
  report_path = tmp_path / "generation.json"
  provenance = {
    "input": {"sha256": hashlib.sha256(visual.read_bytes()).hexdigest()},
    "outputs": {"mjcf": {"sha256": hashlib.sha256(automatic.read_bytes()).hexdigest()}},
  }
  report_path.write_text(json.dumps(provenance))
  prepare = _prepare()
  for name in ("first", "second"):
    prepare(reference, automatic, visual, tmp_path / name, ["common"], report_path)
  for name in ("reference.xml", "automatic.xml", "variants.json", "preparation.json"):
    assert (tmp_path / "first" / name).read_bytes() == (
      tmp_path / "second" / name
    ).read_bytes()
  provenance["input"]["sha256"] = "incorrect"
  report_path.write_text(json.dumps(provenance))
  with pytest.raises(ValueError, match="hashes do not match"):
    prepare(
      reference, automatic, visual, tmp_path / "mismatch", ["common"], report_path
    )
