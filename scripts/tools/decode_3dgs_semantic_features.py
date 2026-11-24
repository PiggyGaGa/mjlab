"""Decode aligned SceneSplat-style features into a collision semantic sidecar."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import cast

import numpy as np
import torch

from mjlab.asset_zoo.gaussian_semantics import decode_semantic_features


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
  with zipfile.ZipFile(path, "w") as archive:
    for name, array in arrays.items():
      buffer = io.BytesIO()
      np.save(buffer, array, allow_pickle=False)
      info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
      info.compress_type = zipfile.ZIP_DEFLATED
      info.external_attr = 0o600 << 16
      archive.writestr(info, buffer.getvalue(), compresslevel=9)


def _load_matrix(path: Path) -> np.ndarray:
  if path.suffix == ".npy":
    return np.load(path, allow_pickle=False)
  if path.suffix in {".pt", ".pth"}:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
      tensor_values = [
        item for item in value.values() if isinstance(item, torch.Tensor)
      ]
      if len(tensor_values) != 1:
        raise ValueError(
          f"{path} dictionary must contain exactly one tensor; found "
          f"{len(tensor_values)}."
        )
      value = tensor_values[0]
    if not isinstance(value, torch.Tensor):
      raise ValueError(f"{path} must contain a tensor.")
    return cast(np.ndarray, value.detach().cpu().numpy())
  raise ValueError(f"Unsupported feature file '{path}'; expected .npy, .pt, or .pth.")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("features", type=Path)
  parser.add_argument("class_embeddings", type=Path)
  parser.add_argument("output_labels", type=Path)
  parser.add_argument("--report", type=Path, required=True)
  parser.add_argument("--confidence-output", type=Path)
  parser.add_argument("--class-ids", type=Path)
  parser.add_argument("--class-names", type=Path)
  parser.add_argument("--minimum-similarity", type=float, default=0.0)
  parser.add_argument("--minimum-margin", type=float, default=0.0)
  parser.add_argument("--unknown-label", type=int, default=-1)
  parser.add_argument("--block-size", type=int, default=65_536)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  features = _load_matrix(args.features)
  class_embeddings = _load_matrix(args.class_embeddings)
  class_ids = (
    np.load(args.class_ids, allow_pickle=False) if args.class_ids is not None else None
  )
  class_names: list[str] | None = None
  if args.class_names is not None:
    raw_names = json.loads(args.class_names.read_text())
    if not isinstance(raw_names, list) or not all(
      isinstance(name, str) for name in raw_names
    ):
      raise ValueError("--class-names must contain a JSON string list.")
    if len(raw_names) != len(class_embeddings):
      raise ValueError("--class-names length must match class embeddings.")
    class_names = raw_names
  decoded = decode_semantic_features(
    features,
    class_embeddings,
    class_ids=class_ids,
    minimum_similarity=args.minimum_similarity,
    minimum_margin=args.minimum_margin,
    unknown_label=args.unknown_label,
    block_size=args.block_size,
  )
  confidence_path = args.confidence_output or args.report.with_suffix(".confidence.npz")
  for path in (args.output_labels, args.report, confidence_path):
    path.parent.mkdir(parents=True, exist_ok=True)
  np.save(args.output_labels, decoded.labels, allow_pickle=False)
  _write_npz(
    confidence_path,
    {
      "cosine_similarity": decoded.similarities,
      "cosine_margin": decoded.margins,
    },
  )
  unique_labels, counts = np.unique(decoded.labels, return_counts=True)
  ids = (
    np.arange(len(class_embeddings), dtype=np.int64)
    if class_ids is None
    else np.asarray(class_ids, dtype=np.int64)
  )
  report = {
    "schema_version": 1,
    "decoder": "cosine_nearest_class_embedding",
    "alignment_contract": (
      "feature row i and output label i refer to original PLY vertex i"
    ),
    "config": {
      "minimum_similarity": args.minimum_similarity,
      "minimum_margin": args.minimum_margin,
      "unknown_label": args.unknown_label,
      "block_size": args.block_size,
    },
    "input": {
      "features": {"path": str(args.features), "sha256": _sha256(args.features)},
      "class_embeddings": {
        "path": str(args.class_embeddings),
        "sha256": _sha256(args.class_embeddings),
      },
      "class_ids": (
        {"path": str(args.class_ids), "sha256": _sha256(args.class_ids)}
        if args.class_ids is not None
        else None
      ),
      "class_names": (
        {"path": str(args.class_names), "sha256": _sha256(args.class_names)}
        if args.class_names is not None
        else None
      ),
    },
    "statistics": {
      "point_count": len(features),
      "feature_dimension": features.shape[1],
      "class_count": len(class_embeddings),
      "label_counts": {
        str(int(label)): int(count)
        for label, count in zip(unique_labels, counts, strict=True)
      },
      "unknown_fraction": float(np.mean(decoded.labels == args.unknown_label)),
      "similarity_mean": float(decoded.similarities.mean()),
      "similarity_p05": float(np.quantile(decoded.similarities, 0.05)),
      "margin_mean": float(decoded.margins.mean()),
      "margin_p05": float(np.quantile(decoded.margins, 0.05)),
    },
    "classes": [
      {
        "id": int(class_id),
        "name": class_names[index] if class_names is not None else None,
      }
      for index, class_id in enumerate(ids)
    ],
    "outputs": {
      "labels": {
        "path": str(args.output_labels),
        "sha256": _sha256(args.output_labels),
      },
      "confidence": {
        "path": str(confidence_path),
        "sha256": _sha256(confidence_path),
      },
    },
  }
  args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"Decoded {len(features)} feature rows to {args.output_labels}")


if __name__ == "__main__":
  main()
