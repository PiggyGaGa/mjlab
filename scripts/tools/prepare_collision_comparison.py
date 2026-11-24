"""Freeze primitive collision variants with identical explicit common obstacles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_comparison(
  reference: Path,
  automatic: Path,
  visual: Path,
  output: Path,
  common_geoms: list[str],
  automatic_report: Path | None = None,
) -> Path:
  if len(set(common_geoms)) != len(common_geoms):
    raise ValueError("Common geom names must be unique.")
  sources = [path.resolve() for path in (reference, automatic, visual)]
  if any(not path.is_file() for path in sources):
    raise FileNotFoundError("All visual and collision inputs must exist.")
  if any(path.is_relative_to(output.resolve()) for path in sources):
    raise ValueError("Inputs must be outside the output directory.")
  if automatic_report is not None:
    if automatic_report.resolve().is_relative_to(output.resolve()):
      raise ValueError("The source report must be outside the output directory.")
    provenance = json.loads(automatic_report.read_text())
    if provenance["input"]["sha256"] != _sha256(visual) or provenance["outputs"][
      "mjcf"
    ]["sha256"] != _sha256(automatic):
      raise ValueError(
        "Automatic report hashes do not match the visual/collision inputs."
      )
  roots = [ET.parse(path).getroot() for path in (reference, automatic)]
  worlds = []
  for root in roots:
    world = root.find("worldbody")
    if (
      world is None
      or root.find(".//include") is not None
      or root.find("asset") is not None
      or any(child.tag != "geom" for child in world)
      or any(
        geom.get("type", "sphere")
        not in {"box", "sphere", "cylinder", "capsule", "plane"}
        for geom in world
      )
    ):
      raise ValueError(
        "Comparison preparation requires self-contained worldbody primitive geoms."
      )
    worlds.append(world)
  reference_model = mujoco.MjModel.from_xml_path(str(reference.resolve()))
  automatic_names = {geom.get("name") for geom in worlds[1]}
  sizes = {"box": 3, "sphere": 1, "cylinder": 2, "capsule": 2, "plane": 3}
  resolved_common = []
  for name in common_geoms:
    if name in automatic_names:
      raise ValueError(f"Automatic input already contains common geom: {name}")
    original = next((geom for geom in worlds[0] if geom.get("name") == name), None)
    if original is None:
      raise ValueError(f"Common geom not found in reference worldbody: {name}")
    geom_id = mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_GEOM, name)
    type_names = {
      int(getattr(mujoco.mjtGeom, f"mjGEOM_{kind.upper()}")): kind for kind in sizes
    }
    geom_type = type_names[int(reference_model.geom_type[geom_id])]
    attributes = {"name": name, "type": geom_type}
    arrays = {
      "pos": reference_model.geom_pos[geom_id],
      "quat": reference_model.geom_quat[geom_id],
      "size": reference_model.geom_size[geom_id, : sizes[geom_type]],
      "friction": reference_model.geom_friction[geom_id],
      "solref": reference_model.geom_solref[geom_id],
      "solimp": reference_model.geom_solimp[geom_id],
      "rgba": reference_model.geom_rgba[geom_id],
      "solmix": [reference_model.geom_solmix[geom_id]],
      "margin": [reference_model.geom_margin[geom_id]],
      "gap": [reference_model.geom_gap[geom_id]],
    }
    for key, values in arrays.items():
      attributes[key] = " ".join(format(float(value), ".17g") for value in values)
    for key in ("group", "contype", "conaffinity", "condim", "priority"):
      attributes[key] = str(int(getattr(reference_model, f"geom_{key}")[geom_id]))
    common = ET.Element("geom", attributes)
    position = list(worlds[0]).index(original)
    worlds[0].remove(original)
    worlds[0].insert(position, common)
    worlds[1].append(copy.deepcopy(common))
    resolved_common.append(attributes)
  for root in roots:
    mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
  output.mkdir(parents=True, exist_ok=True)
  variants = []
  for name, root in zip(("reference", "automatic"), roots, strict=True):
    ET.indent(root, space="  ")
    destination = output / f"{name}.xml"
    destination.write_text(ET.tostring(root, encoding="unicode") + "\n")
    variants.append(
      {
        "name": name,
        "collision_mjcf": destination.name,
        "geometry": {
          "common_geom_names": common_geoms,
          "source_role": "reference"
          if name == "reference"
          else "automatic proxy plus common obstacles",
        },
      }
    )
  manifest = output / "variants.json"
  manifest.write_text(
    json.dumps({"schema_version": 1, "variants": variants}, indent=2) + "\n"
  )
  report = {
    "schema_version": 1,
    "visual": {"path": str(visual.resolve()), "sha256": _sha256(visual)},
    "inputs": {
      "reference": {"path": str(reference.resolve()), "sha256": _sha256(reference)},
      "automatic": {"path": str(automatic.resolve()), "sha256": _sha256(automatic)},
    },
    "common_geoms": resolved_common,
    "automatic_report": (
      {"path": str(automatic_report.resolve()), "sha256": _sha256(automatic_report)}
      if automatic_report is not None
      else None
    ),
    "outputs": {
      name: _sha256(output / name)
      for name in ("reference.xml", "automatic.xml", "variants.json")
    },
    "limitations": [
      "Matching explicit common obstacles does not make other scene geometry equivalent.",
      "The automatic variant is an experimental composition, not the unchanged generator output.",
      "This preparation does not measure geometry quality, task success or rendering throughput.",
    ],
  }
  (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
  return manifest


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("reference", type=Path)
  parser.add_argument("automatic", type=Path)
  parser.add_argument("visual", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--common-geoms", nargs="*", default=[])
  parser.add_argument("--automatic-report", type=Path)
  args = parser.parse_args()
  print(
    prepare_comparison(
      args.reference,
      args.automatic,
      args.visual,
      args.output,
      args.common_geoms,
      args.automatic_report,
    )
  )


if __name__ == "__main__":
  main()
