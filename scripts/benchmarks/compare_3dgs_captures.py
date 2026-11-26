"""Compare matched raw captures; differences are not a task-quality verdict."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare_reports(reference_path: Path, candidate_path: Path) -> dict:
  reference = json.loads(reference_path.read_text())
  candidate = json.loads(candidate_path.read_text())
  if reference.get("status") != "passed" or candidate.get("status") != "passed":
    raise ValueError("Both capture reports must have status passed")
  for key in (
    "camera_capture",
    "manifest_sha256",
    "source_sha256",
    "collision_sha256",
    "scene_assets",
    "loaded_splats_shared_across_worlds",
  ):
    if key not in reference or reference[key] != candidate.get(key):
      raise ValueError(f"Capture inputs differ or are missing: {key}")
  arrays = []
  capture_hashes = []
  for report in (reference, candidate):
    path = Path(report["reset_capture"]["directory"]) / "reset.npz"
    with np.load(path, allow_pickle=False) as data:
      arrays.append({key: data[key] for key in ("rgb", "depth", "qpos")})
    capture_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
  ref, actual = arrays
  for key in ref:
    if ref[key].shape != actual[key].shape:
      raise ValueError(f"Capture shapes differ: {key}")
  if not np.isfinite(ref["qpos"]).all() or not np.array_equal(
    ref["qpos"], actual["qpos"]
  ):
    raise ValueError("Actual qpos differs or is non-finite; images are not comparable")
  rgb_ref, rgb_actual = ref["rgb"], actual["rgb"]
  depth_ref, depth_actual = ref["depth"], actual["depth"]
  if rgb_ref.dtype != np.uint8 or rgb_actual.dtype != np.uint8:
    raise ValueError("Expected uint8 raw RGB")
  valid_ref = np.isfinite(depth_ref) & (depth_ref > 0)
  valid_actual = np.isfinite(depth_actual) & (depth_actual > 0)
  common = valid_ref & valid_actual
  depth_error = np.abs(depth_ref[common] - depth_actual[common])
  rgb_error = np.abs(rgb_ref.astype(np.float64) - rgb_actual.astype(np.float64))
  return {
    "status": "compared",
    "scope": "Matched static frames against an archive, not ground truth or fix attribution",
    "reference_report": str(reference_path),
    "candidate_report": str(candidate_path),
    "report_sha256": [
      hashlib.sha256(p.read_bytes()).hexdigest()
      for p in (reference_path, candidate_path)
    ],
    "capture_sha256": capture_hashes,
    "qpos_equal": True,
    "rgb_exact": bool(np.array_equal(rgb_ref, rgb_actual)),
    "depth_exact": bool(np.array_equal(depth_ref, depth_actual, equal_nan=True)),
    "rgb_mae_0_255": float(rgb_error.mean()),
    "rgb_max_abs_error_0_255": float(rgb_error.max()),
    "reference_valid_depth_fraction": float(valid_ref.mean()),
    "candidate_valid_depth_fraction": float(valid_actual.mean()),
    "common_valid_depth_fraction": float(common.mean()),
    "reference_valid_depth_lost_fraction": (
      float((valid_ref & ~valid_actual).sum() / valid_ref.sum())
      if valid_ref.any()
      else None
    ),
    "common_valid_depth_mae_m": float(depth_error.mean()) if depth_error.size else None,
    "common_valid_depth_max_abs_error_m": float(depth_error.max())
    if depth_error.size
    else None,
    "common_valid_error_over_0_1m_fraction": float((depth_error > 0.1).mean())
    if depth_error.size
    else None,
    "reference_build_identity": reference.get("build_identity"),
    "candidate_build_identity": candidate.get("build_identity"),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--reference", required=True, type=Path)
  parser.add_argument("--candidate", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    parser.error("Output already exists")
  try:
    report = compare_reports(args.reference, args.candidate)
  except (ValueError, KeyError, OSError) as error:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
      json.dumps({"status": "not_comparable", "error": str(error)}, indent=2) + "\n"
    )
    raise
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  print(
    f"Compared: RGB exact={report['rgb_exact']}, depth exact={report['depth_exact']}"
  )


if __name__ == "__main__":
  main()
