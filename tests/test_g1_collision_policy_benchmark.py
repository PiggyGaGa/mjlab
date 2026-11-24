"""Tests for the joint G1 collision-policy benchmark protocol."""

import json
from pathlib import Path
from typing import cast

import pytest

from mjlab.scripts.benchmark_g1_collision_policy import (
  _overflow_stats,
  collision_bundle_hashes,
  load_variant_manifest,
  summarize_runs,
)


def test_load_variant_manifest_resolves_relative_paths(tmp_path: Path) -> None:
  collision = tmp_path / "collision.xml"
  collision.write_text("<mujoco><worldbody><geom type='plane'/></worldbody></mujoco>")
  manifest = tmp_path / "variants.json"
  manifest.write_text(
    json.dumps(
      {
        "schema_version": 1,
        "variants": [
          {"name": "plane", "collision_mjcf": None},
          {
            "name": "proxy",
            "collision_mjcf": "collision.xml",
            "geometry": {"surface_distance_mean": 0.1},
          },
        ],
      }
    )
  )

  variants = load_variant_manifest(manifest)

  assert variants[0].collision_mjcf is None
  assert variants[1].collision_mjcf == collision
  assert variants[1].geometry == {"surface_distance_mean": 0.1}


def test_load_variant_manifest_rejects_duplicate_names(tmp_path: Path) -> None:
  manifest = tmp_path / "variants.json"
  manifest.write_text(
    json.dumps(
      {
        "schema_version": 1,
        "variants": [
          {"name": "same", "collision_mjcf": None},
          {"name": "same", "collision_mjcf": None},
        ],
      }
    )
  )

  with pytest.raises(ValueError, match="Duplicate variant name"):
    load_variant_manifest(manifest)


def test_collision_bundle_hashes_local_mesh_dependencies(tmp_path: Path) -> None:
  meshes = tmp_path / "meshes"
  meshes.mkdir()
  mesh = meshes / "piece.obj"
  mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
  collision = tmp_path / "collision.xml"
  collision.write_text(
    "<mujoco><compiler meshdir='meshes'/><asset>"
    "<mesh name='piece' file='piece.obj'/></asset><worldbody>"
    "<geom type='mesh' mesh='piece'/></worldbody></mujoco>"
  )

  report = collision_bundle_hashes(collision)

  dependencies = report["dependencies"]
  assert isinstance(dependencies, list)
  dependency = cast(dict[str, str], dependencies[0])
  assert isinstance(dependency, dict)
  assert dependency["path"] == "meshes/piece.obj"
  assert len(dependency["sha256"]) == 64


def test_summarize_runs_reports_population_mean_and_std() -> None:
  def run(value: float, success: bool) -> dict[str, object]:
    tracking = {
      "mpkpe": value,
      "r_mpkpe": value,
      "joint_vel_error": value,
      "ee_pos_error": value,
      "ee_ori_error": value,
    }
    samples = {"mean": value, "max": value, "samples": 1}
    return {
      "completion": {"success": success},
      "tracking": tracking,
      "contacts_total": samples,
      "constraints": samples,
      "solver_iterations": samples,
      "overflow": {"any": value != 0, "nonzero_steps": int(value != 0)},
      "throughput": {
        "control_steps_per_second": value,
        "physics_steps_per_second": value,
      },
    }

  summary = summarize_runs([run(1.0, True), run(3.0, False)])

  assert summary["run_count"] == 2
  assert summary["success"] == {"mean": 0.5, "std": 0.5}
  assert summary["mpkpe"] == {"mean": 2.0, "std": 1.0}


def test_overflow_stats_decodes_bitmask_flags() -> None:
  assert _overflow_stats([0.0, 1024.0, 0.0]) == {
    "any": True,
    "bitmask_or": 1024,
    "flags": ["LS_ITERATIONS"],
    "nonzero_steps": 1,
    "samples": 3,
  }
