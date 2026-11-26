# P10 optional Gaussian preprocessing cache

Date: 2026-11-25. Scope: repeated CPU asset loading, not rendering-frame throughput.

## Architecture and compatibility

The local main baseline has no reusable 3DGS preprocessing cache. The optional
implementation lives in `src/mjlab/scene/gaussian_splat_cache.py`. The only existing
runtime-file change is a thin wrapper and optional `cache_dir=None` field in
`GaussianSplatAssetCfg` in `gaussian_splat.py`. The original normalization code is
retained unchanged in the uncached helper. No generic Scene/sensor/trainer,
play/demo, manifest loader or manifest schema is changed. Existing asset configs
stay uncached and perform no content hashing or cache I/O. The interface addition
is backward compatible and isolated to the 3DGS asset loader.

The key binds schema/normalization version, resolved source path, source SHA-256
and every asset import/filter/transform field except path/cache directory (source
path is separately explicit). Cache files contain normalized single-asset arrays,
not GPU buffers, BVHs or per-world assignments. Schema version must be bumped
when normalization semantics or array contracts change. Asset world mapping is
recomputed by the existing scene loader.

Cache validation disables pickle and checks metadata identity, field set, dtypes,
shapes, finite values, positive scales, normalized quaternions, RGBA ranges,
single-asset group offsets and per-array checksums. Source content is rehashed on
every cache load, so same-size/same-mtime edits cannot silently reuse stale data.
The source must still exist. This is a local performance cache, not authenticated
distribution of untrusted assets or a new manifest integrity requirement.

Writes use same-directory unique temporary files, flush/fsync and atomic replace.
Failed writes warn and use the freshly parsed arrays, preserving existing entries.
Concurrent writers may repeat work but publish complete entries. Source stat
snapshots ignore read-access-time changes; source mutation during preprocessing is
retried once, then rejected without publication. Source paths are pinned before
loading to avoid symlink-retargeting mixing a different source with its key.
Old keys are retained, not automatically deleted.

## Tests and current evidence

Cache tests cover exact uncached equivalence, bypassing the PLY parser on hits,
mutable-return isolation, disabled-path no hashing, content invalidation despite
unchanged size/mtime, every current import/transform field, schema invalidation,
truncated ZIP and modified finite payload recovery, atomic-replace failure,
concurrent writers, source mutation, source removal, and per-world remapping.
The CLI benchmark also checks all arrays and protects its source from report writes.

Focused cache/loader/native transport/manifest tests: 37 passed, 3 deselected.
The two GPU-render-context integration tests and the known unrelated Studio pillar
alpha assertion were not included. This is not a full repository test claim.
Final validation completed: `UV_NO_SYNC=1 make check` passes (Ruff format/lint,
ty and Pyright: zero errors), and `UV_NO_SYNC=1 make docs` succeeds. `git diff
--check` passes. The existing environment was used without package synchronization;
the newly refreshed remote wheel checksum was preserved but the wheel was not
installed as part of this CPU-cache validation. No GPU render test is implied.

## Real-asset CPU measurements

Commands run sequentially with OMP/OpenBLAS threads set to 4, each using a new
persistent cache subdirectory and three uncached plus three warm repetitions:

```bash
uv run scripts/tools/benchmark_gaussian_splat_loading.py \
  src/mjlab/asset_zoo/3dgs/apartment/scene_transform_p5.ply \
  artifacts/p10-splat-cache/apartment.json \
  --cache-dir artifacts/p10-splat-cache/entries --repeats 3

uv run scripts/tools/benchmark_gaussian_splat_loading.py \
  src/mjlab/asset_zoo/3dgs/ArtanimMotionCaptureStudio/scene_transform_20p.ply \
  artifacts/p10-splat-cache/artanim.json \
  --cache-dir artifacts/p10-splat-cache/entries --repeats 3
```

| Asset | Splats | Uncached mean (s) | Cache build (s) | Verified warm mean (s) |
|---|---:|---:|---:|---:|
| apartment | 374,410 | 0.288313 | 0.510121 | 0.067949 |
| Artanim | 659,010 | 0.222752 | 0.621773 | 0.086982 |

All six normalized arrays are exactly equal in every measured phase. First cache
creation is slower than uncached loading here. Warm timing includes full source
hashing, NPZ reads, validation and array checksums. Python startup, GPU uploads,
BVHs, browser transport and rendering are excluded. OS page-cache effects and host
load are uncontrolled; these means are not a statistically established speedup.
No improvement to Actual RT, rendering FPS or training SPS is inferred.

## Acceptance boundary

This optional preprocessing-cache subtask is complete. P8 remains closed; no collision
asset is regenerated. P10 sensor/training throughput matrices, full-suite validation
and release/PR preparation remain separate unfinished work.
