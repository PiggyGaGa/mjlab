"""Versioned manifest for paired 3DGS visual and MuJoCo collision assets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal, cast
from xml.etree import ElementTree

import numpy as np

from mjlab.scene.gaussian_splat import (
  GaussianSplatAssetCfg,
  GaussianSplatSceneCfg,
)
from mjlab.terrains import TerrainEntityCfg

if TYPE_CHECKING:
  from mjlab.scene.scene import SceneCfg

_SCHEMA_VERSION = 1
_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class SceneManifestFile:
  """One resolved and hash-verified file in a scene bundle."""

  path: Path
  sha256: str


@dataclass(frozen=True)
class SceneManifestAsset(SceneManifestFile):
  """One primary asset referenced by a scene manifest."""

  type: Literal["3dgs_ply", "mjcf"]
  geom_group: int | None = None
  dependencies: tuple[SceneManifestFile, ...] = ()


@dataclass(frozen=True)
class SceneSpawnPoint:
  """Named robot spawn pose in scene-local coordinates."""

  name: str
  position: tuple[float, float, float]
  quaternion_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class SceneManifest:
  """Resolved and hash-verified visual/physical scene bundle."""

  source_path: Path
  schema_version: int
  visual: SceneManifestAsset
  collision: SceneManifestAsset
  world_position: tuple[float, float, float]
  world_quaternion_wxyz: tuple[float, float, float, float]
  spawn_points: tuple[SceneSpawnPoint, ...]


def _require_object(
  value: Any,
  label: str,
  *,
  required: set[str],
  optional: set[str] | None = None,
) -> dict[str, Any]:
  if not isinstance(value, dict):
    raise ValueError(f"{label} must be a JSON object")
  value = cast(dict[str, Any], value)
  missing = required - value.keys()
  if missing:
    raise ValueError(f"{label} is missing fields: {', '.join(sorted(missing))}")
  allowed = required | (optional or set())
  unexpected = value.keys() - allowed
  if unexpected:
    raise ValueError(
      f"{label} contains unsupported fields: {', '.join(sorted(unexpected))}"
    )
  return value


def _finite_tuple(value: Any, length: int, label: str) -> tuple[float, ...]:
  if not isinstance(value, list) or len(value) != length:
    raise ValueError(f"{label} must contain exactly {length} numbers")
  try:
    result = tuple(float(component) for component in value)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{label} must contain only numbers") from exc
  if not np.all(np.isfinite(result)):
    raise ValueError(f"{label} must contain only finite numbers")
  return result


def _unit_quaternion(value: Any, label: str) -> tuple[float, float, float, float]:
  quat = cast(tuple[float, float, float, float], _finite_tuple(value, 4, label))
  if not np.isclose(np.linalg.norm(quat), 1.0, rtol=1.0e-6, atol=1.0e-6):
    raise ValueError(f"{label} must be a unit quaternion in (w, x, y, z) order")
  return quat


def _resolve_relative_asset(manifest_dir: Path, value: Any, label: str) -> Path:
  if not isinstance(value, str) or not value:
    raise ValueError(f"{label} must be a non-empty relative path")
  posix_path = PurePosixPath(value)
  windows_path = PureWindowsPath(value)
  if (
    posix_path.is_absolute()
    or windows_path.is_absolute()
    or ".." in posix_path.parts
    or ".." in windows_path.parts
  ):
    raise ValueError(f"{label} must stay inside the manifest directory: {value}")
  resolved = (manifest_dir / value).resolve()
  if not resolved.is_relative_to(manifest_dir.resolve()):
    raise ValueError(f"{label} resolves outside the manifest directory: {value}")
  if not resolved.is_file():
    raise FileNotFoundError(f"{label} not found: {resolved}")
  return resolved


def _parse_sha256(value: Any, label: str) -> str:
  if not isinstance(value, str):
    raise ValueError(f"{label} must be a SHA-256 string")
  digest = value.lower()
  if len(digest) != 64 or any(char not in _HEX_DIGITS for char in digest):
    raise ValueError(f"{label} must contain 64 hexadecimal characters")
  return digest


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _resolve_mjcf_reference(
  mjcf_path: Path,
  manifest_dir: Path,
  raw_path: str,
  *,
  directory: str,
  label: str,
) -> Path:
  relative_path = PurePosixPath(directory) / PurePosixPath(raw_path)
  windows_path = PureWindowsPath(directory) / PureWindowsPath(raw_path)
  if (
    relative_path.is_absolute()
    or windows_path.is_absolute()
    or ".." in relative_path.parts
    or ".." in windows_path.parts
  ):
    raise ValueError(f"{label} must stay inside the manifest directory: {raw_path}")
  resolved = (mjcf_path.parent / Path(relative_path)).resolve()
  if not resolved.is_relative_to(manifest_dir.resolve()):
    raise ValueError(f"{label} resolves outside the manifest directory: {raw_path}")
  return resolved


def _discover_mjcf_dependencies(mjcf_path: Path, manifest_dir: Path) -> set[Path]:
  """Return all external asset files referenced by a collision MJCF."""
  try:
    root = ElementTree.parse(mjcf_path).getroot()
  except ElementTree.ParseError as exc:
    raise ValueError(f"Invalid collision MJCF XML '{mjcf_path}': {exc}") from exc

  compiler = root.find("compiler")
  compiler_attrs = compiler.attrib if compiler is not None else {}
  asset_dir = compiler_attrs.get("assetdir", "")
  mesh_dir = asset_dir or compiler_attrs.get("meshdir", "")
  texture_dir = asset_dir or compiler_attrs.get("texturedir", "")

  dependencies: set[Path] = set()
  for element in root.iter():
    raw_path = element.attrib.get("file")
    if raw_path is None:
      continue
    if element.tag == "include":
      directory = ""
    elif element.tag == "mesh":
      directory = mesh_dir
    elif element.tag == "texture":
      directory = texture_dir
    else:
      directory = asset_dir
    dependencies.add(
      _resolve_mjcf_reference(
        mjcf_path,
        manifest_dir,
        raw_path,
        directory=directory,
        label=f"collision MJCF <{element.tag}> file",
      )
    )
  return dependencies


def _validate_mjcf_dependencies(
  collision: SceneManifestAsset, manifest_dir: Path
) -> None:
  referenced = _discover_mjcf_dependencies(collision.path, manifest_dir)
  declared = {dependency.path for dependency in collision.dependencies}
  missing = referenced - declared
  if missing:
    paths = ", ".join(sorted(str(path.relative_to(manifest_dir)) for path in missing))
    raise ValueError(
      "collision.dependencies is missing files referenced by the MJCF: " + paths
    )
  unused = declared - referenced
  if unused:
    paths = ", ".join(sorted(str(path.relative_to(manifest_dir)) for path in unused))
    raise ValueError(
      "collision.dependencies contains files not referenced by the MJCF: " + paths
    )


def _quat_multiply(
  left: tuple[float, float, float, float],
  right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
  lw, lx, ly, lz = left
  rw, rx, ry, rz = right
  return (
    lw * rw - lx * rx - ly * ry - lz * rz,
    lw * rx + lx * rw + ly * rz - lz * ry,
    lw * ry - lx * rz + ly * rw + lz * rx,
    lw * rz + lx * ry - ly * rx + lz * rw,
  )


def _transform_spawn_point(
  manifest: SceneManifest, spawn: SceneSpawnPoint
) -> SceneSpawnPoint:
  w, x, y, z = manifest.world_quaternion_wxyz
  rotation = np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ],
    dtype=np.float64,
  )
  position = np.asarray(manifest.world_position) + rotation @ np.asarray(spawn.position)
  quaternion = _quat_multiply(manifest.world_quaternion_wxyz, spawn.quaternion_wxyz)
  return SceneSpawnPoint(
    name=spawn.name,
    position=cast(tuple[float, float, float], tuple(float(v) for v in position)),
    quaternion_wxyz=quaternion,
  )


def _parse_asset(
  value: Any,
  *,
  manifest_dir: Path,
  label: str,
  expected_type: Literal["3dgs_ply", "mjcf"],
  verify_hashes: bool,
) -> SceneManifestAsset:
  optional = {"geom_group", "dependencies"} if expected_type == "mjcf" else set()
  obj = _require_object(
    value,
    label,
    required={"type", "path", "sha256"},
    optional=optional,
  )
  if obj["type"] != expected_type:
    raise ValueError(f"{label}.type must be '{expected_type}'")
  path = _resolve_relative_asset(manifest_dir, obj["path"], f"{label}.path")
  expected_hash = _parse_sha256(obj["sha256"], f"{label}.sha256")
  if verify_hashes:
    actual_hash = _file_sha256(path)
    if actual_hash != expected_hash:
      raise ValueError(
        f"{label} SHA-256 mismatch for '{path}': expected {expected_hash}, "
        f"got {actual_hash}"
      )
  geom_group = None
  dependencies: list[SceneManifestFile] = []
  if expected_type == "mjcf":
    geom_group = obj.get("geom_group", 3)
    if not isinstance(geom_group, int) or isinstance(geom_group, bool):
      raise ValueError(f"{label}.geom_group must be an integer from 0 to 5")
    if not 0 <= geom_group <= 5:
      raise ValueError(f"{label}.geom_group must be between 0 and 5")
    raw_dependencies = obj.get("dependencies", [])
    if not isinstance(raw_dependencies, list):
      raise ValueError(f"{label}.dependencies must be a JSON array")
    dependency_paths: set[Path] = set()
    for index, raw_dependency in enumerate(raw_dependencies):
      dependency_label = f"{label}.dependencies[{index}]"
      dependency = _require_object(
        raw_dependency,
        dependency_label,
        required={"path", "sha256"},
      )
      dependency_path = _resolve_relative_asset(
        manifest_dir, dependency["path"], f"{dependency_label}.path"
      )
      if dependency_path in dependency_paths:
        raise ValueError(f"Duplicate collision dependency: {dependency_path}")
      dependency_paths.add(dependency_path)
      dependency_hash = _parse_sha256(
        dependency["sha256"], f"{dependency_label}.sha256"
      )
      if verify_hashes:
        actual_hash = _file_sha256(dependency_path)
        if actual_hash != dependency_hash:
          raise ValueError(
            f"{dependency_label} SHA-256 mismatch for '{dependency_path}': "
            f"expected {dependency_hash}, got {actual_hash}"
          )
      dependencies.append(SceneManifestFile(dependency_path, dependency_hash))
  return SceneManifestAsset(
    path=path,
    sha256=expected_hash,
    type=expected_type,
    geom_group=geom_group,
    dependencies=tuple(dependencies),
  )


def load_scene_manifest(
  path: str | Path, *, verify_hashes: bool = True
) -> SceneManifest:
  """Load, validate, resolve, and optionally hash-check a scene manifest."""
  manifest_path = Path(path).expanduser().resolve()
  if not manifest_path.is_file():
    raise FileNotFoundError(f"Scene manifest not found: {manifest_path}")
  try:
    raw = json.loads(manifest_path.read_text())
  except json.JSONDecodeError as exc:
    raise ValueError(f"Invalid scene manifest JSON '{manifest_path}': {exc}") from exc

  root = _require_object(
    raw,
    "manifest",
    required={
      "schema_version",
      "coordinate_system",
      "visual",
      "collision",
      "world_from_scene",
      "spawn_points",
    },
    optional={"$schema"},
  )
  if root["schema_version"] != _SCHEMA_VERSION:
    raise ValueError(
      f"Unsupported scene manifest schema_version: {root['schema_version']}; "
      f"expected {_SCHEMA_VERSION}"
    )

  coordinates = _require_object(
    root["coordinate_system"],
    "coordinate_system",
    required={"handedness", "up_axis", "length_unit"},
  )
  expected_coordinates = {
    "handedness": "right",
    "up_axis": "z",
    "length_unit": "meter",
  }
  if coordinates != expected_coordinates:
    raise ValueError(
      "coordinate_system must be right-handed, Z-up, and measured in meters"
    )

  manifest_dir = manifest_path.parent
  visual = _parse_asset(
    root["visual"],
    manifest_dir=manifest_dir,
    label="visual",
    expected_type="3dgs_ply",
    verify_hashes=verify_hashes,
  )
  collision = _parse_asset(
    root["collision"],
    manifest_dir=manifest_dir,
    label="collision",
    expected_type="mjcf",
    verify_hashes=verify_hashes,
  )
  _validate_mjcf_dependencies(collision, manifest_dir)

  transform = _require_object(
    root["world_from_scene"],
    "world_from_scene",
    required={"position", "quaternion_wxyz"},
  )
  world_position = cast(
    tuple[float, float, float],
    _finite_tuple(transform["position"], 3, "world_from_scene.position"),
  )
  world_quaternion = _unit_quaternion(
    transform["quaternion_wxyz"], "world_from_scene.quaternion_wxyz"
  )

  if not isinstance(root["spawn_points"], list) or not root["spawn_points"]:
    raise ValueError("spawn_points must be a non-empty JSON array")
  spawn_points: list[SceneSpawnPoint] = []
  names: set[str] = set()
  for index, value in enumerate(root["spawn_points"]):
    label = f"spawn_points[{index}]"
    spawn = _require_object(
      value,
      label,
      required={"name", "position", "quaternion_wxyz"},
    )
    name = spawn["name"]
    if not isinstance(name, str) or not name:
      raise ValueError(f"{label}.name must be a non-empty string")
    if name in names:
      raise ValueError(f"Duplicate spawn point name: {name}")
    names.add(name)
    position = cast(
      tuple[float, float, float],
      _finite_tuple(spawn["position"], 3, f"{label}.position"),
    )
    quaternion = _unit_quaternion(spawn["quaternion_wxyz"], f"{label}.quaternion_wxyz")
    spawn_points.append(SceneSpawnPoint(name, position, quaternion))

  return SceneManifest(
    source_path=manifest_path,
    schema_version=_SCHEMA_VERSION,
    visual=visual,
    collision=collision,
    world_position=world_position,
    world_quaternion_wxyz=world_quaternion,
    spawn_points=tuple(spawn_points),
  )


def apply_scene_manifest(
  scene_cfg: SceneCfg,
  path: str | Path,
  *,
  verify_hashes: bool = True,
  spawn_point: str | None = None,
) -> SceneManifest:
  """Apply one verified manifest and selected spawn to a scene configuration."""
  manifest = load_scene_manifest(path, verify_hashes=verify_hashes)

  requested_spawn = spawn_point
  if requested_spawn is None:
    selected_spawn = next(
      (spawn for spawn in manifest.spawn_points if spawn.name == "default"), None
    )
    if selected_spawn is None and len(manifest.spawn_points) == 1:
      selected_spawn = manifest.spawn_points[0]
    if selected_spawn is None:
      available = ", ".join(spawn.name for spawn in manifest.spawn_points)
      raise ValueError(
        "Scene manifest has multiple spawn points but no 'default'; select one "
        f"explicitly. Available: {available}"
      )
  else:
    selected_spawn = next(
      (spawn for spawn in manifest.spawn_points if spawn.name == requested_spawn),
      None,
    )
  if selected_spawn is None:
    available = ", ".join(spawn.name for spawn in manifest.spawn_points)
    raise ValueError(
      f"Unknown scene spawn point '{requested_spawn}'; available: {available}"
    )
  world_spawn = _transform_spawn_point(manifest, selected_spawn)
  scene_cfg.spawn_position = world_spawn.position
  scene_cfg.spawn_quaternion_wxyz = world_spawn.quaternion_wxyz

  splat_cfg = scene_cfg.gaussian_splats
  if splat_cfg is None:
    asset = GaussianSplatAssetCfg(
      path=str(manifest.visual.path),
      pos=manifest.world_position,
      quat=manifest.world_quaternion_wxyz,
    )
    scene_cfg.gaussian_splats = GaussianSplatSceneCfg(assets=(asset,))
  else:
    if len(splat_cfg.assets) != 1:
      raise ValueError("Scene manifests currently require exactly one 3DGS asset")
    asset = replace(
      splat_cfg.assets[0],
      path=str(manifest.visual.path),
      pos=manifest.world_position,
      quat=manifest.world_quaternion_wxyz,
      scale=1.0,
    )
    scene_cfg.gaussian_splats = replace(splat_cfg, assets=(asset,))

  terrain_cfg = scene_cfg.terrain
  if terrain_cfg is None:
    scene_cfg.terrain = TerrainEntityCfg(
      terrain_type="mjcf",
      terrain_generator=None,
      mjcf_path=str(manifest.collision.path),
      mjcf_keep_plane=False,
      mjcf_geom_group=manifest.collision.geom_group,
      mjcf_pos=manifest.world_position,
      mjcf_quat=manifest.world_quaternion_wxyz,
    )
  else:
    scene_cfg.terrain = replace(
      terrain_cfg,
      terrain_type="mjcf",
      terrain_generator=None,
      mjcf_path=str(manifest.collision.path),
      mjcf_keep_plane=False,
      mjcf_geom_group=manifest.collision.geom_group,
      mjcf_pos=manifest.world_position,
      mjcf_quat=manifest.world_quaternion_wxyz,
    )
  return manifest
