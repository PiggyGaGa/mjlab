"""Optional Gaussian cache correctness, invalidation and atomic-write tests."""

import json
import os
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from mjlab.scene import gaussian_splat, gaussian_splat_cache
from mjlab.scene.gaussian_splat import (
  GaussianSplatAssetCfg,
  GaussianSplatSceneCfg,
  load_gaussian_splat_asset,
  load_gaussian_splat_scene,
)


def _write_ply(path: Path, first_x: float = 0.0) -> None:
  properties = (
    "x",
    "y",
    "z",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "opacity",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "red",
    "green",
    "blue",
  )
  header = ["ply", "format ascii 1.0", "element vertex 3"]
  header.extend(f"property float {name}" for name in properties)
  header.append("end_header")
  rows = [
    (
      first_x + index,
      0.0,
      0.0,
      0.1,
      0.1,
      0.1,
      1.0,
      0.0,
      0.0,
      0.0,
      0.2 + index * 0.2,
      0.0,
      0.0,
      0.0,
      128.0,
      64.0,
      32.0,
    )
    for index in range(3)
  ]
  path.write_text(
    "\n".join([*header, *(" ".join(str(value) for value in row) for row in rows), ""])
  )


def _configuration(tmp_path: Path) -> GaussianSplatAssetCfg:
  source = tmp_path / "scene.ply"
  _write_ply(source)
  return GaussianSplatAssetCfg(path=str(source), cache_dir=str(tmp_path / "cache"))


def test_cached_load_is_exact_and_skips_ply_parser(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  config = replace(
    _configuration(tmp_path),
    pos=(1.0, 2.0, 3.0),
    quat=(0.0, 0.0, 0.0, 1.0),
    scale=2.0,
    max_splats=2,
  )
  expected = load_gaussian_splat_asset(replace(config, cache_dir=None))
  first = load_gaussian_splat_asset(config)

  def unexpected_parse(_path: Path):
    raise AssertionError("Cache hit must not parse PLY vertices")

  monkeypatch.setattr(gaussian_splat, "_read_ply_vertices", unexpected_parse)
  cached = load_gaussian_splat_asset(config)
  for field in fields(expected):
    np.testing.assert_array_equal(
      getattr(first, field.name), getattr(expected, field.name)
    )
    np.testing.assert_array_equal(
      getattr(cached, field.name), getattr(expected, field.name)
    )
  cached.position[:] = 100
  np.testing.assert_array_equal(
    load_gaussian_splat_asset(config).position, expected.position
  )
  assert len(list((tmp_path / "cache").glob("*.splats.npz"))) == 1


def test_disabled_cache_does_not_hash_source_or_create_files(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  config = _configuration(tmp_path)

  def unexpected_hash(_path: Path):
    raise AssertionError("Disabled cache must not hash sources")

  monkeypatch.setattr(gaussian_splat_cache, "_source_hash", unexpected_hash)
  load_gaussian_splat_asset(replace(config, cache_dir=None))
  assert not (tmp_path / "cache").exists()


def test_source_content_invalidates_cache_even_if_size_and_mtime_match(
  tmp_path: Path,
) -> None:
  config = _configuration(tmp_path)
  first = load_gaussian_splat_asset(config)
  source = Path(config.path)
  previous = source.stat()
  _write_ply(source, first_x=1.0)
  assert source.stat().st_size == previous.st_size
  os.utime(source, ns=(previous.st_atime_ns, previous.st_mtime_ns))
  second = load_gaussian_splat_asset(config)
  np.testing.assert_array_equal(second.position[:, 0], first.position[:, 0] + 1)
  assert len(list((tmp_path / "cache").glob("*.splats.npz"))) == 2


@pytest.mark.parametrize(
  "changes",
  [
    {"pos": (1.0, 0.0, 0.0)},
    {"quat": (0.0, 0.0, 0.0, 1.0)},
    {"scale": 2.0},
    {"opacity_threshold": 0.55},
    {"max_splats": 1},
    {"bounds_min": (0.5, -1.0, -1.0), "bounds_max": (3.0, 1.0, 1.0)},
    {"color_encoding": "rgb"},
    {"opacity_encoding": "linear"},
    {"scale_encoding": "linear"},
  ],
)
def test_each_import_option_invalidates_cache(tmp_path: Path, changes: dict) -> None:
  config = _configuration(tmp_path)
  load_gaussian_splat_asset(config)
  changed = replace(config, **changes)
  actual = load_gaussian_splat_asset(changed)
  expected = load_gaussian_splat_asset(replace(changed, cache_dir=None))
  for field in fields(actual):
    np.testing.assert_array_equal(
      getattr(actual, field.name), getattr(expected, field.name)
    )
  assert len(list((tmp_path / "cache").glob("*.splats.npz"))) == 2


def test_schema_change_uses_a_new_cache(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  config = _configuration(tmp_path)
  load_gaussian_splat_asset(config)
  monkeypatch.setattr(gaussian_splat_cache, "CACHE_SCHEMA_VERSION", 2)
  load_gaussian_splat_asset(config)
  assert len(list((tmp_path / "cache").glob("*.splats.npz"))) == 2
  assert len(list((tmp_path / "cache").glob("v2-*.splats.npz"))) == 1


def test_corrupt_and_modified_payloads_rebuild_from_source(tmp_path: Path) -> None:
  config = _configuration(tmp_path)
  expected = load_gaussian_splat_asset(config)
  cache = next((tmp_path / "cache").glob("*.splats.npz"))
  cache.write_bytes(b"incomplete ZIP")
  with pytest.warns(RuntimeWarning, match="Ignoring invalid"):
    restored = load_gaussian_splat_asset(config)
  np.testing.assert_array_equal(restored.position, expected.position)
  with np.load(cache, allow_pickle=False) as archive:
    arrays = {name: archive[name] for name in archive.files}
  arrays["position"][0, 0] += 1
  np.savez(cache, **arrays)
  with pytest.warns(RuntimeWarning, match="checksums"):
    restored = load_gaussian_splat_asset(config)
  np.testing.assert_array_equal(restored.position, expected.position)


def test_failed_atomic_replace_preserves_existing_cache_and_cleans_temp(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  config = _configuration(tmp_path)
  expected = load_gaussian_splat_asset(config)
  cache = next((tmp_path / "cache").glob("*.splats.npz"))
  original = cache.read_bytes()
  monkeypatch.setattr(gaussian_splat_cache, "_read_cache", lambda *_args: None)

  def failed_replace(_source, _destination):
    raise OSError("simulated atomic rename failure")

  monkeypatch.setattr(gaussian_splat_cache.os, "replace", failed_replace)
  with pytest.warns(RuntimeWarning, match="using uncached data"):
    actual = load_gaussian_splat_asset(config)
  np.testing.assert_array_equal(actual.position, expected.position)
  assert cache.read_bytes() == original
  assert not list((tmp_path / "cache").glob("*.tmp"))


def test_concurrent_cache_writers_publish_one_valid_entry(tmp_path: Path) -> None:
  config = _configuration(tmp_path)
  with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(load_gaussian_splat_asset, [config, config]))
  np.testing.assert_array_equal(results[0].position, results[1].position)
  np.testing.assert_array_equal(
    load_gaussian_splat_asset(config).position, results[0].position
  )
  assert len(list((tmp_path / "cache").glob("*.splats.npz"))) == 1
  assert not list((tmp_path / "cache").glob("*.tmp"))


def test_changing_source_is_not_published(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  config = _configuration(tmp_path)
  original = gaussian_splat._read_ply_vertices
  calls = 0

  def changing_parser(path: Path):
    nonlocal calls
    result = original(path)
    calls += 1
    _write_ply(path, first_x=float(calls))
    return result

  monkeypatch.setattr(gaussian_splat, "_read_ply_vertices", changing_parser)
  with pytest.raises(ValueError, match="changed while preparing"):
    load_gaussian_splat_asset(config)
  assert not list((tmp_path / "cache").glob("*.splats.npz"))


def test_cached_assets_do_not_freeze_world_assignment(tmp_path: Path) -> None:
  config = _configuration(tmp_path)
  assets = (config, replace(config, pos=(1.0, 0.0, 0.0)))
  cached = load_gaussian_splat_scene(
    GaussianSplatSceneCfg(assets=assets, world_asset_ids=(1, 0, 1)), nworld=3
  )
  uncached = load_gaussian_splat_scene(
    GaussianSplatSceneCfg(
      assets=tuple(replace(asset, cache_dir=None) for asset in assets),
      world_asset_ids=(1, 0, 1),
    ),
    nworld=3,
  )
  for field in fields(cached):
    np.testing.assert_array_equal(
      getattr(cached, field.name), getattr(uncached, field.name)
    )
  np.testing.assert_array_equal(cached.group_id, [1, 0, 1])


def test_existing_cache_does_not_hide_a_missing_source(tmp_path: Path) -> None:
  config = _configuration(tmp_path)
  load_gaussian_splat_asset(config)
  Path(config.path).unlink()
  with pytest.raises(ValueError, match="Unable to read 3DGS PLY"):
    load_gaussian_splat_asset(config)


def test_loading_benchmark_checks_equivalence_and_preserves_source(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  config = _configuration(tmp_path)
  source = Path(config.path)
  original = source.read_bytes()
  script = (
    Path(__file__).resolve().parents[1]
    / "scripts/tools/benchmark_gaussian_splat_loading.py"
  )
  report = tmp_path / "benchmark.json"
  monkeypatch.setattr(
    sys,
    "argv",
    [
      str(script),
      str(source),
      str(report),
      "--cache-dir",
      str(tmp_path / "entries"),
      "--repeats",
      "1",
    ],
  )
  runpy.run_path(str(script), run_name="__main__")
  data = json.loads(report.read_text())
  assert data["arrays_exactly_equal"]
  assert data["splat_count"] == 3
  assert data["cache_bytes"] > 0
  assert all(len(values) == 1 for values in data["timings_seconds"].values())
  assert source.read_bytes() == original
  monkeypatch.setattr(
    sys,
    "argv",
    [str(script), str(source), str(source), "--cache-dir", str(tmp_path / "entries")],
  )
  with pytest.raises(ValueError, match="must not overwrite"):
    runpy.run_path(str(script), run_name="__main__")
  assert source.read_bytes() == original
