"""Fingerprint installed renderer files without initializing a GPU environment."""

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path


def fingerprint_tree(root: Path) -> dict:
  files = {
    str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.rglob("*"))
    if path.is_file()
    and "__pycache__" not in path.parts
    and (path.suffix in {".py", ".glsl"} or ".so" in path.name)
  }
  if not files:
    raise ValueError(f"No renderer source/native files found in {root}")
  encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
  return {
    "root": str(root),
    "files": files,
    "tree_sha256": hashlib.sha256(encoded).hexdigest(),
  }


def installed_build_identity() -> dict:
  identity = {}
  for module_name, distribution_name in (
    ("mujoco", "mujoco"),
    ("mujoco_warp", "mujoco-warp"),
  ):
    module = importlib.import_module(module_name)
    assert module.__file__ is not None
    distribution = importlib.metadata.distribution(distribution_name)
    package = fingerprint_tree(Path(module.__file__).resolve().parent)
    package["version"] = distribution.version
    direct_url = distribution.read_text("direct_url.json")
    package["direct_url"] = json.loads(direct_url) if direct_url else None
    identity[module_name] = package
  maps = Path("/proc/self/maps")
  paths = (
    {
      Path(line.split()[-1])
      for line in maps.read_text().splitlines()
      if "/libmujoco.so" in line
    }
    if maps.exists()
    else set()
  )
  identity["loaded_mujoco_libraries"] = {
    str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)
  }
  identity["scope"] = (
    "Installed files and mapped libraries, not proof of a source checkout/build revision"
  )
  return identity
