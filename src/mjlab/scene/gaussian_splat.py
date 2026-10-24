"""Load static 3D Gaussian Splatting assets for MuJoCo Warp rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

_SH_C0 = 0.28209479177387814

_PLY_DTYPES = {
  "char": "i1",
  "int8": "i1",
  "uchar": "u1",
  "uint8": "u1",
  "short": "i2",
  "int16": "i2",
  "ushort": "u2",
  "uint16": "u2",
  "int": "i4",
  "int32": "i4",
  "uint": "u4",
  "uint32": "u4",
  "float": "f4",
  "float32": "f4",
  "double": "f8",
  "float64": "f8",
}


@dataclass(frozen=True, kw_only=True)
class GaussianSplatAssetCfg:
  """Configuration for one Graphdeco-style 3DGS PLY asset."""

  path: str
  """Path to the PLY file."""

  pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Translation from asset coordinates to MuJoCo world coordinates, in meters."""

  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  """Asset-to-world rotation quaternion in ``(w, x, y, z)`` order."""

  scale: float = 1.0
  """Uniform asset-to-world scale."""

  opacity_threshold: float = 0.0
  """Discard splats whose activated opacity is below this value."""

  bounds_min: tuple[float, float, float] | None = None
  """Optional inclusive lower crop bound, evaluated after the world transform."""

  bounds_max: tuple[float, float, float] | None = None
  """Optional inclusive upper crop bound, evaluated after the world transform."""

  max_splats: int | None = None
  """Keep at most this many highest-opacity splats."""

  scale_encoding: Literal["log", "linear"] = "log"
  """Encoding of ``scale_0`` through ``scale_2`` in the PLY."""

  opacity_encoding: Literal["logit", "linear"] = "logit"
  """Encoding of the ``opacity`` property in the PLY."""

  color_encoding: Literal["sh_dc", "rgb"] = "sh_dc"
  """Color fields: ``f_dc_0..2`` or ``red/green/blue``."""

  def __post_init__(self) -> None:
    if self.scale <= 0.0 or not np.isfinite(self.scale):
      raise ValueError(f"scale must be finite and positive, got {self.scale}.")
    if not 0.0 <= self.opacity_threshold <= 1.0:
      raise ValueError(
        f"opacity_threshold must be between 0 and 1, got {self.opacity_threshold}."
      )
    if (self.bounds_min is None) != (self.bounds_max is None):
      raise ValueError("bounds_min and bounds_max must be specified together.")
    if self.bounds_min is not None and self.bounds_max is not None:
      bounds_min = np.asarray(self.bounds_min, dtype=np.float64)
      bounds_max = np.asarray(self.bounds_max, dtype=np.float64)
      if (
        bounds_min.shape != (3,)
        or bounds_max.shape != (3,)
        or not np.all(np.isfinite(bounds_min))
        or not np.all(np.isfinite(bounds_max))
      ):
        raise ValueError("bounds_min and bounds_max must contain three finite values.")
      if np.any(bounds_min > bounds_max):
        raise ValueError("bounds_min must not exceed bounds_max.")
    if self.max_splats is not None and self.max_splats <= 0:
      raise ValueError(f"max_splats must be positive, got {self.max_splats}.")


@dataclass(frozen=True, kw_only=True)
class GaussianSplatSceneCfg:
  """Static 3DGS visual assets and their assignment to simulation worlds."""

  assets: tuple[GaussianSplatAssetCfg, ...]
  """Assets concatenated into MuJoCo Warp splat groups."""

  world_asset_ids: tuple[int, ...] | None = None
  """Asset index for every world. Omit when there is exactly one asset."""

  def __post_init__(self) -> None:
    if not self.assets:
      raise ValueError("GaussianSplatSceneCfg.assets must not be empty.")


@dataclass(frozen=True)
class GaussianSplatData:
  """Normalized arrays passed directly to MuJoCo Warp."""

  position: np.ndarray
  rotation: np.ndarray
  scale: np.ndarray
  rgba: np.ndarray
  adr: np.ndarray
  group_id: np.ndarray


def gaussian_splat_covariances(data: GaussianSplatData) -> np.ndarray:
  """Convert quaternion and axis scales to covariance matrices for Viser."""
  quat = data.rotation
  w, x, y, z = quat.T
  rotation = np.empty((len(quat), 3, 3), dtype=np.float32)
  rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
  rotation[:, 0, 1] = 2.0 * (x * y - z * w)
  rotation[:, 0, 2] = 2.0 * (x * z + y * w)
  rotation[:, 1, 0] = 2.0 * (x * y + z * w)
  rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
  rotation[:, 1, 2] = 2.0 * (y * z - x * w)
  rotation[:, 2, 0] = 2.0 * (x * z - y * w)
  rotation[:, 2, 1] = 2.0 * (y * z + x * w)
  rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

  scaled_rotation = rotation * data.scale[:, None, :]
  covariance = scaled_rotation @ scaled_rotation.transpose(0, 2, 1)
  return np.ascontiguousarray(covariance, dtype=np.float32)


def _read_ply_vertices(path: Path) -> dict[str, np.ndarray]:
  """Read scalar vertex properties from an ASCII or little-endian PLY file."""
  try:
    file = path.open("rb")
  except OSError as exc:
    raise ValueError(f"Unable to open 3DGS PLY '{path}': {exc}") from exc

  with file:
    first = file.readline()
    if first.rstrip() != b"ply":
      raise ValueError(f"3DGS asset '{path}' is not a PLY file.")

    ply_format: str | None = None
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    current_element: str | None = None
    preceding_element_count = 0

    while True:
      raw_line = file.readline()
      if not raw_line:
        raise ValueError(f"PLY header in '{path}' has no end_header marker.")
      try:
        line = raw_line.decode("ascii").strip()
      except UnicodeDecodeError as exc:
        raise ValueError(f"PLY header in '{path}' is not ASCII.") from exc
      if not line or line.startswith("comment ") or line.startswith("obj_info "):
        continue
      tokens = line.split()
      if tokens[0] == "format":
        if len(tokens) < 2:
          raise ValueError(f"Malformed PLY format declaration in '{path}'.")
        ply_format = tokens[1]
      elif tokens[0] == "element":
        if len(tokens) != 3:
          raise ValueError(f"Malformed PLY element declaration in '{path}'.")
        current_element = tokens[1]
        try:
          count = int(tokens[2])
        except ValueError as exc:
          raise ValueError(f"Invalid PLY element count in '{path}'.") from exc
        if count < 0:
          raise ValueError(f"Negative PLY element count in '{path}'.")
        if current_element == "vertex":
          vertex_count = count
        elif vertex_count is None:
          preceding_element_count += count
      elif tokens[0] == "property" and current_element == "vertex":
        if len(tokens) != 3 or tokens[1] == "list":
          raise ValueError(
            f"Vertex list properties are not supported in 3DGS PLY '{path}'."
          )
        type_name, property_name = tokens[1], tokens[2]
        if type_name not in _PLY_DTYPES:
          raise ValueError(f"Unsupported PLY property type '{type_name}' in '{path}'.")
        if any(name == property_name for name, _ in vertex_properties):
          raise ValueError(f"Duplicate PLY vertex property '{property_name}'.")
        vertex_properties.append((property_name, type_name))
      elif tokens[0] == "end_header":
        break

    if ply_format not in {"ascii", "binary_little_endian"}:
      raise ValueError(
        f"Unsupported PLY format '{ply_format}' in '{path}'; expected ASCII or "
        "binary_little_endian."
      )
    if vertex_count is None:
      raise ValueError(f"PLY '{path}' has no vertex element.")
    if vertex_count == 0:
      raise ValueError(f"PLY '{path}' has no vertices.")
    if not vertex_properties:
      raise ValueError(f"PLY '{path}' has no scalar vertex properties.")
    if preceding_element_count:
      raise ValueError(
        f"PLY '{path}' stores data before its vertex element; this layout is not "
        "supported for 3DGS assets."
      )

    if ply_format == "binary_little_endian":
      dtype = np.dtype(
        [(name, "<" + _PLY_DTYPES[type_name]) for name, type_name in vertex_properties]
      )
      records = np.fromfile(file, dtype=dtype, count=vertex_count)
      if len(records) != vertex_count:
        raise ValueError(
          f"PLY '{path}' ended after {len(records)} of {vertex_count} vertices."
        )
      return {name: np.asarray(records[name]) for name, _type_name in vertex_properties}

    rows: list[list[str]] = []
    property_count = len(vertex_properties)
    for vertex_idx in range(vertex_count):
      raw_line = file.readline()
      if not raw_line:
        raise ValueError(
          f"PLY '{path}' ended after {vertex_idx} of {vertex_count} vertices."
        )
      tokens = raw_line.split()
      if len(tokens) != property_count:
        raise ValueError(
          f"PLY '{path}' vertex {vertex_idx} has {len(tokens)} values; expected "
          f"{property_count}."
        )
      rows.append([token.decode("ascii") for token in tokens])

    columns = np.asarray(rows, dtype=np.float64)
    return {
      name: columns[:, idx] for idx, (name, _type_name) in enumerate(vertex_properties)
    }


def _require_fields(
  fields: dict[str, np.ndarray], names: tuple[str, ...], path: Path
) -> np.ndarray:
  missing = [name for name in names if name not in fields]
  if missing:
    raise ValueError(f"3DGS PLY '{path}' is missing vertex fields: {missing}.")
  return np.column_stack([fields[name] for name in names]).astype(np.float64)


def _sigmoid(values: np.ndarray) -> np.ndarray:
  """Numerically stable sigmoid."""
  result = np.empty_like(values, dtype=np.float64)
  positive = values >= 0.0
  result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
  exp_values = np.exp(values[~positive])
  result[~positive] = exp_values / (1.0 + exp_values)
  return result


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
  """Multiply ``(w, x, y, z)`` quaternions with broadcasted left operand."""
  lw, lx, ly, lz = left
  rw, rx, ry, rz = right.T
  return np.column_stack(
    (
      lw * rw - lx * rx - ly * ry - lz * rz,
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
    )
  )


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
  """Convert one normalized ``(w, x, y, z)`` quaternion to a matrix."""
  w, x, y, z = quat
  return np.asarray(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ],
    dtype=np.float64,
  )


def load_gaussian_splat_asset(cfg: GaussianSplatAssetCfg) -> GaussianSplatData:
  """Load and normalize one 3DGS asset as a single splat group."""
  path = Path(cfg.path).expanduser()
  fields = _read_ply_vertices(path)

  position = _require_fields(fields, ("x", "y", "z"), path)
  rotation = _require_fields(fields, ("rot_0", "rot_1", "rot_2", "rot_3"), path)
  scale = _require_fields(fields, ("scale_0", "scale_1", "scale_2"), path)
  opacity = _require_fields(fields, ("opacity",), path)[:, 0]

  if cfg.color_encoding == "sh_dc":
    sh_dc = _require_fields(fields, ("f_dc_0", "f_dc_1", "f_dc_2"), path)
    color = np.clip(0.5 + _SH_C0 * sh_dc, 0.0, 1.0)
  else:
    color = _require_fields(fields, ("red", "green", "blue"), path)
    if np.max(color) > 1.0:
      color = color / 255.0
    color = np.clip(color, 0.0, 1.0)

  if cfg.scale_encoding == "log":
    scale = np.exp(scale)
  if cfg.opacity_encoding == "logit":
    opacity = _sigmoid(opacity)

  if not all(
    np.all(np.isfinite(array)) for array in (position, rotation, scale, color, opacity)
  ):
    raise ValueError(f"3DGS PLY '{path}' contains NaN or infinite values.")
  if np.any(scale <= 0.0):
    raise ValueError(f"3DGS PLY '{path}' contains non-positive activated scales.")
  if np.any((opacity < 0.0) | (opacity > 1.0)):
    raise ValueError(f"3DGS PLY '{path}' contains opacity outside [0, 1].")

  rotation_norm = np.linalg.norm(rotation, axis=1)
  if np.any(rotation_norm <= 1.0e-12):
    raise ValueError(f"3DGS PLY '{path}' contains a zero-length rotation.")
  rotation = rotation / rotation_norm[:, None]

  transform_quat = np.asarray(cfg.quat, dtype=np.float64)
  if transform_quat.shape != (4,) or not np.all(np.isfinite(transform_quat)):
    raise ValueError("quat must contain four finite values in (w, x, y, z) order.")
  transform_norm = np.linalg.norm(transform_quat)
  if transform_norm <= 1.0e-12:
    raise ValueError("quat must not be zero length.")
  transform_quat = transform_quat / transform_norm
  transform_rotation = _quat_to_matrix(transform_quat)
  translation = np.asarray(cfg.pos, dtype=np.float64)
  if translation.shape != (3,) or not np.all(np.isfinite(translation)):
    raise ValueError("pos must contain three finite values.")

  position = (position * cfg.scale) @ transform_rotation.T + translation
  rotation = _quat_multiply(transform_quat, rotation)
  scale = scale * cfg.scale

  keep = opacity >= cfg.opacity_threshold
  if cfg.bounds_min is not None and cfg.bounds_max is not None:
    bounds_min = np.asarray(cfg.bounds_min, dtype=np.float64)
    bounds_max = np.asarray(cfg.bounds_max, dtype=np.float64)
    keep &= np.all(position >= bounds_min, axis=1)
    keep &= np.all(position <= bounds_max, axis=1)

  indices = np.flatnonzero(keep)
  if cfg.max_splats is not None and len(indices) > cfg.max_splats:
    ranked = np.argsort(-opacity[indices], kind="stable")[: cfg.max_splats]
    indices = np.sort(indices[ranked])
  if len(indices) == 0:
    raise ValueError(
      f"3DGS asset '{path}' has no splats after filtering; adjust crop or opacity."
    )

  rgba = np.column_stack((color, opacity))
  count = len(indices)
  return GaussianSplatData(
    position=np.ascontiguousarray(position[indices], dtype=np.float32),
    rotation=np.ascontiguousarray(rotation[indices], dtype=np.float32),
    scale=np.ascontiguousarray(scale[indices], dtype=np.float32),
    rgba=np.ascontiguousarray(rgba[indices], dtype=np.float32),
    adr=np.asarray([0, count], dtype=np.int32),
    group_id=np.zeros(1, dtype=np.int32),
  )


def load_gaussian_splat_scene(
  cfg: GaussianSplatSceneCfg, nworld: int
) -> GaussianSplatData:
  """Load all configured assets and construct MuJoCo Warp group arrays."""
  if nworld <= 0:
    raise ValueError(f"nworld must be positive, got {nworld}.")
  if len(cfg.assets) > 1 and cfg.world_asset_ids is None:
    raise ValueError(
      "world_asset_ids is required when more than one Gaussian splat asset is "
      "configured."
    )

  loaded = [load_gaussian_splat_asset(asset) for asset in cfg.assets]
  if cfg.world_asset_ids is None:
    group_id = np.zeros(nworld, dtype=np.int32)
  else:
    if len(cfg.world_asset_ids) != nworld:
      raise ValueError(
        f"world_asset_ids has length {len(cfg.world_asset_ids)}; expected {nworld}."
      )
    group_id = np.asarray(cfg.world_asset_ids, dtype=np.int32)
    if np.any((group_id < 0) | (group_id >= len(loaded))):
      raise ValueError(
        f"world_asset_ids must be between 0 and {len(loaded) - 1}, got "
        f"{cfg.world_asset_ids}."
      )

  counts = np.asarray([len(asset.position) for asset in loaded], dtype=np.int64)
  adr64 = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(counts)))
  if adr64[-1] > np.iinfo(np.int32).max:
    raise ValueError("Total splat count exceeds MuJoCo Warp's int32 group offsets.")

  return GaussianSplatData(
    position=np.concatenate([asset.position for asset in loaded]),
    rotation=np.concatenate([asset.rotation for asset in loaded]),
    scale=np.concatenate([asset.scale for asset in loaded]),
    rgba=np.concatenate([asset.rgba for asset in loaded]),
    adr=adr64.astype(np.int32),
    group_id=group_id,
  )
