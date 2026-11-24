"""Compare uncached PLY preprocessing, cache creation and verified warm loads."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import fields, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np

from mjlab.scene.gaussian_splat import GaussianSplatAssetCfg, load_gaussian_splat_asset
from mjlab.scene.gaussian_splat_cache import CACHE_SCHEMA_VERSION, _source_hash


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input_ply", type=Path)
  parser.add_argument("report", type=Path)
  parser.add_argument("--cache-dir", type=Path, required=True)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--max-splats", type=int)
  args = parser.parse_args()
  if args.repeats <= 0:
    raise ValueError("Repeat count must be positive.")
  source = args.input_ply.expanduser().resolve()
  if args.report.resolve() == source:
    raise ValueError("The benchmark report must not overwrite the PLY.")
  source_hash = _source_hash(source)
  args.cache_dir.mkdir(parents=True, exist_ok=True)
  cache_directory = Path(tempfile.mkdtemp(prefix="load-benchmark-", dir=args.cache_dir))
  uncached_cfg = GaussianSplatAssetCfg(path=str(source), max_splats=args.max_splats)
  cached_cfg = replace(uncached_cfg, cache_dir=str(cache_directory))
  timings: dict[str, list[float]] = {"uncached": [], "cache_build": [], "cache_hit": []}
  expected = None
  for phase, config, count in (
    ("uncached", uncached_cfg, args.repeats),
    ("cache_build", cached_cfg, 1),
    ("cache_hit", cached_cfg, args.repeats),
  ):
    for _attempt in range(count):
      start = time.perf_counter()
      actual = load_gaussian_splat_asset(config)
      timings[phase].append(time.perf_counter() - start)
      if expected is None:
        expected = actual
      else:
        for field in fields(actual):
          if not np.array_equal(
            getattr(actual, field.name), getattr(expected, field.name)
          ):
            raise ValueError(
              f"Cached or repeated {field.name} output differs from uncached output."
            )
  if _source_hash(source) != source_hash:
    raise ValueError("Source changed during the benchmark.")
  assert expected is not None
  report = {
    "schema_version": 1,
    "cache_schema_version": CACHE_SCHEMA_VERSION,
    "input": {
      "path": str(source),
      "sha256": source_hash,
      "bytes": source.stat().st_size,
    },
    "cache_directory": str(cache_directory.resolve()),
    "cache_bytes": sum(
      path.stat().st_size for path in cache_directory.glob("*.splats.npz")
    ),
    "max_splats": args.max_splats,
    "splat_count": len(expected.position),
    "repeats": args.repeats,
    "timings_seconds": timings,
    "mean_seconds": {name: float(np.mean(values)) for name, values in timings.items()},
    "arrays_exactly_equal": True,
    "versions": {name: version(name) for name in ("mjlab", "numpy")},
    "limitations": [
      "Cache hits include full source hashing, NPZ reading, array validation and checksum verification.",
      "Cache build uses a new directory; uncached PLY repetitions may use the OS page cache.",
      "This measures CPU asset loading, not GPU BVH creation, rendering FPS or training throughput.",
      "Single-process repeated timings do not establish a statistically controlled speedup.",
    ],
  }
  args.report.parent.mkdir(parents=True, exist_ok=True)
  args.report.write_text(json.dumps(report, indent=2) + "\n")
  print(
    json.dumps(
      {
        "splat_count": report["splat_count"],
        "mean_seconds": report["mean_seconds"],
        "report": str(args.report),
      }
    )
  )


if __name__ == "__main__":
  main()
