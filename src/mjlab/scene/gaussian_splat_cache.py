"""Versioned, optional disk caching for normalized single-asset Gaussian arrays."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

CACHE_SCHEMA_VERSION = 1
ARRAY_NAMES = ("position", "rotation", "scale", "rgba", "adr", "group_id")


def _source_hash(path: Path) -> str:
  digest = hashlib.sha256()
  try:
    with path.open("rb") as source:
      for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
  except OSError as exc:
    raise ValueError(f"Unable to read 3DGS PLY '{path}': {exc}") from exc
  return digest.hexdigest()


def _array_hash(array: np.ndarray) -> str:
  return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _source_stamp(path: Path) -> tuple[int, int, int, int, int]:
  try:
    stat = path.stat()
  except OSError as exc:
    raise ValueError(f"Unable to read 3DGS PLY '{path}': {exc}") from exc
  return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _validate_arrays(arrays: dict[str, np.ndarray]) -> None:
  if set(arrays) != set(ARRAY_NAMES):
    raise ValueError("Cache contains unexpected or missing array names.")
  count = len(arrays["position"])
  if count <= 0 or count > np.iinfo(np.int32).max:
    raise ValueError("Invalid cached splat count.")
  for name, width in (("position", 3), ("rotation", 4), ("scale", 3), ("rgba", 4)):
    array = arrays[name]
    if (
      array.dtype != np.float32
      or array.shape != (count, width)
      or not np.isfinite(array).all()
    ):
      raise ValueError(f"Invalid cached {name} array.")
  if np.any(arrays["scale"] <= 0) or np.any(
    (arrays["rgba"] < 0) | (arrays["rgba"] > 1)
  ):
    raise ValueError("Cached scales/colors/opacities are out of range.")
  if not np.allclose(
    np.linalg.norm(arrays["rotation"], axis=1), 1, rtol=1e-5, atol=1e-5
  ):
    raise ValueError("Cached rotations are not normalized.")
  for name, expected in (("adr", [0, count]), ("group_id", [0])):
    if arrays[name].dtype != np.int32 or not np.array_equal(arrays[name], expected):
      raise ValueError(f"Invalid cached single-asset {name} array.")


def _read_cache(path: Path, identity: str) -> dict[str, np.ndarray] | None:
  try:
    with np.load(path, allow_pickle=False) as archive:
      if set(archive.files) != {*ARRAY_NAMES, "metadata"}:
        raise ValueError("Unexpected cached fields.")
      raw_metadata = archive["metadata"]
      if raw_metadata.dtype != np.uint8 or raw_metadata.ndim != 1:
        raise ValueError("Invalid cache metadata encoding.")
      metadata = json.loads(raw_metadata.tobytes().decode("utf-8"))
      if metadata["identity"] != identity:
        raise ValueError("Cached source/config/schema identity does not match.")
      arrays = {name: archive[name] for name in ARRAY_NAMES}
    _validate_arrays(arrays)
    if metadata["array_hashes"] != {
      name: _array_hash(array) for name, array in arrays.items()
    }:
      raise ValueError("Cached array checksums do not match.")
    return arrays
  except FileNotFoundError:
    return None
  except (
    OSError,
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
    EOFError,
    zipfile.BadZipFile,
  ) as exc:
    warnings.warn(
      f"Ignoring invalid 3DGS cache '{path}': {exc}", RuntimeWarning, stacklevel=2
    )
    return None


def _write_cache(path: Path, identity: str, arrays: dict[str, np.ndarray]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  metadata = json.dumps(
    {
      "identity": identity,
      "array_hashes": {name: _array_hash(array) for name, array in arrays.items()},
    },
    sort_keys=True,
  ).encode("utf-8")
  temporary: Path | None = None
  try:
    with tempfile.NamedTemporaryFile(
      mode="wb", dir=path.parent, prefix=".splats-", suffix=".tmp", delete=False
    ) as file:
      temporary = Path(file.name)
      np.savez(
        file,
        metadata=np.frombuffer(metadata, dtype=np.uint8),
        position=arrays["position"],
        rotation=arrays["rotation"],
        scale=arrays["scale"],
        rgba=arrays["rgba"],
        adr=arrays["adr"],
        group_id=arrays["group_id"],
      )
      file.flush()
      os.fsync(file.fileno())
    os.replace(temporary, path)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def load_cached_splat_arrays(
  source: Path,
  cache_dir: Path,
  parameters: dict[str, object],
  build: Callable[[], dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
  """Reuse validated content/config-keyed arrays; never cache a changing source."""
  source = source.expanduser().resolve()
  cache_dir = cache_dir.expanduser()
  for _attempt in range(2):
    snapshot = _source_stamp(source)
    source_digest = _source_hash(source)
    if _source_stamp(source) != snapshot:
      continue
    identity = json.dumps(
      {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_path": str(source),
        "source_sha256": source_digest,
        "parameters": parameters,
      },
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
    )
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    destination = cache_dir / f"v{CACHE_SCHEMA_VERSION}-{key}.splats.npz"
    arrays = _read_cache(destination, identity)
    if arrays is not None:
      if _source_stamp(source) == snapshot:
        return arrays
      continue
    arrays = build()
    _validate_arrays(arrays)
    if _source_stamp(source) != snapshot or _source_hash(source) != source_digest:
      continue
    try:
      _write_cache(destination, identity, arrays)
    except OSError as exc:
      warnings.warn(
        f"Unable to write 3DGS cache '{destination}'; using uncached data: {exc}",
        RuntimeWarning,
        stacklevel=2,
      )
    return arrays
  raise ValueError(
    f"3DGS PLY '{source}' changed while preparing its cache; retry with a stable source."
  )
