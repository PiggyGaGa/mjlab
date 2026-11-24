"""Benchmark primitive collision-only MJCF proxies with identical falling probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _benchmark_xml(
  path: Path,
  *,
  probe_grid: int,
  probe_spacing: float,
  probe_z: float,
  probe_radius: float,
  allow_file_assets: bool,
) -> tuple[str, list[dict[str, str]]]:
  root = ET.parse(path).getroot()
  if root.find(".//include") is not None:
    raise ValueError(
      "The benchmark accepts only self-contained MJCF without <include> elements."
    )
  asset = root.find("asset")
  dependencies: list[dict[str, str]] = []
  if asset is not None:
    compiler = root.find("compiler")
    mesh_dir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
    bundle_root = path.resolve().parent
    for child in asset:
      file_name = child.get("file")
      if child.tag == "hfield" and file_name is None:
        continue
      if child.tag != "mesh" or file_name is None or not allow_file_assets:
        raise ValueError(
          "The benchmark accepts primitive geoms and inline heightfields by "
          "default; pass allow_file_assets for bundle-local mesh files."
        )
      dependency = (bundle_root / mesh_dir / file_name).resolve()
      try:
        dependency.relative_to(bundle_root)
      except ValueError as exc:
        raise ValueError(
          f"Mesh asset must stay inside the MJCF bundle: {file_name}"
        ) from exc
      if not dependency.is_file():
        raise FileNotFoundError(f"Mesh asset not found: {dependency}")
      child.set("file", str(dependency))
      dependencies.append({"path": str(dependency), "sha256": _sha256(dependency)})
  worldbody = root.find("worldbody")
  if worldbody is None:
    raise ValueError(f"Collision MJCF '{path}' has no <worldbody>.")
  offset = (probe_grid - 1) * probe_spacing / 2
  probe_index = 0
  for x_index in range(probe_grid):
    for y_index in range(probe_grid):
      position = (
        x_index * probe_spacing - offset,
        y_index * probe_spacing - offset,
        probe_z,
      )
      body = ET.SubElement(
        worldbody,
        "body",
        name=f"benchmark_probe_{probe_index:04d}",
        pos=" ".join(f"{value:.8f}" for value in position),
      )
      ET.SubElement(body, "freejoint")
      ET.SubElement(
        body,
        "geom",
        type="sphere",
        size=f"{probe_radius:.8f}",
        mass="1",
        friction="1 0.005 0.0001",
        condim="3",
      )
      probe_index += 1
  return ET.tostring(root, encoding="unicode"), dependencies


def benchmark_collision_mjcf(
  path: Path,
  *,
  warmup_steps: int,
  measured_steps: int,
  probe_grid: int,
  probe_spacing: float,
  probe_z: float,
  probe_radius: float,
  allow_file_assets: bool = False,
) -> dict[str, object]:
  """Compile and benchmark one collision proxy with deterministic probes."""
  if warmup_steps < 0 or measured_steps <= 0:
    raise ValueError("warmup_steps must be non-negative and measured_steps positive.")
  if probe_grid <= 0:
    raise ValueError("probe_grid must be positive.")
  for name, value in (
    ("probe_spacing", probe_spacing),
    ("probe_z", probe_z),
    ("probe_radius", probe_radius),
  ):
    if not np.isfinite(value) or value <= 0:
      raise ValueError(f"{name} must be finite and positive.")
  xml, dependencies = _benchmark_xml(
    path,
    probe_grid=probe_grid,
    probe_spacing=probe_spacing,
    probe_z=probe_z,
    probe_radius=probe_radius,
    allow_file_assets=allow_file_assets,
  )
  compile_start = time.perf_counter()
  model = mujoco.MjModel.from_xml_string(xml)
  compile_seconds = time.perf_counter() - compile_start
  data = mujoco.MjData(model)
  for _ in range(warmup_steps):
    mujoco.mj_step(model, data)

  contact_counts = np.empty(measured_steps, dtype=np.int32)
  constraint_counts = np.empty(measured_steps, dtype=np.int32)
  solver_iterations = np.empty(measured_steps, dtype=np.int32)
  step_start = time.perf_counter()
  for step in range(measured_steps):
    mujoco.mj_step(model, data)
    contact_counts[step] = data.ncon
    constraint_counts[step] = data.nefc
    solver_iterations[step] = int(np.asarray(data.solver_niter).sum())
  step_seconds = time.perf_counter() - step_start
  return {
    "path": str(path),
    "sha256": _sha256(path),
    "dependencies": dependencies,
    "compile_seconds": compile_seconds,
    "measured_seconds": step_seconds,
    "physics_steps_per_second": measured_steps / step_seconds,
    "model": {
      "nbody": model.nbody,
      "ngeom": model.ngeom,
      "nv": model.nv,
      "timestep": model.opt.timestep,
    },
    "contacts": {
      "mean": float(contact_counts.mean()),
      "max": int(contact_counts.max()),
    },
    "constraints": {
      "mean": float(constraint_counts.mean()),
      "max": int(constraint_counts.max()),
    },
    "solver_iterations": {
      "mean": float(solver_iterations.mean()),
      "max": int(solver_iterations.max()),
    },
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("collision_mjcf", type=Path, nargs="+")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--warmup-steps", type=int, default=1_000)
  parser.add_argument("--measured-steps", type=int, default=5_000)
  parser.add_argument("--probe-grid", type=int, default=3)
  parser.add_argument("--probe-spacing", type=float, default=0.4)
  parser.add_argument("--probe-z", type=float, default=1.0)
  parser.add_argument("--probe-radius", type=float, default=0.08)
  parser.add_argument(
    "--allow-file-assets",
    action="store_true",
    help="Allow only bundle-local file-backed mesh assets.",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  report = {
    "schema_version": 1,
    "settings": {
      "warmup_steps": args.warmup_steps,
      "measured_steps": args.measured_steps,
      "probe_grid": args.probe_grid,
      "probe_spacing": args.probe_spacing,
      "probe_z": args.probe_z,
      "probe_radius": args.probe_radius,
    },
    "results": [
      benchmark_collision_mjcf(
        path,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        probe_grid=args.probe_grid,
        probe_spacing=args.probe_spacing,
        probe_z=args.probe_z,
        probe_radius=args.probe_radius,
        allow_file_assets=args.allow_file_assets,
      )
      for path in args.collision_mjcf
    ],
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"Wrote {len(args.collision_mjcf)} collision benchmark(s) to {args.output}")


if __name__ == "__main__":
  main()
