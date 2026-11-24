"""Score independently annotated Gaussian rows without treating abstention as truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def semantic_metrics(
  truth: np.ndarray, prediction: np.ndarray, class_count: int
) -> dict[str, object]:
  if class_count <= 0 or truth.ndim != 1 or prediction.shape != truth.shape:
    raise ValueError("Expected aligned vectors and a positive class count.")
  if truth.dtype.kind not in "iu" or prediction.dtype.kind not in "iu":
    raise ValueError("Labels must be integers.")
  if np.any((truth < 0) | (truth >= class_count)):
    raise ValueError("Human labels must be valid classes, not unknown predictions.")
  if np.any((prediction < -1) | (prediction >= class_count)):
    raise ValueError("Predictions must be valid classes or -1.")
  confusion = np.zeros((class_count, class_count + 1), dtype=np.int64)
  columns = np.where(prediction == -1, class_count, prediction)
  np.add.at(confusion, (truth, columns), 1)
  accepted = prediction != -1
  correct = truth == prediction
  per_class = []
  for class_id in range(class_count):
    true_positive = int(confusion[class_id, class_id])
    support = int(confusion[class_id].sum())
    predicted_count = int(confusion[:, class_id].sum())
    union = support + predicted_count - true_positive
    per_class.append(
      {
        "id": class_id,
        "support": support,
        "predicted_count": predicted_count,
        "precision": true_positive / predicted_count if predicted_count else None,
        "recall": true_positive / support if support else None,
        "iou": true_positive / union if union else None,
      }
    )
  return {
    "annotated_count": len(truth),
    "accepted_count": int(accepted.sum()),
    "coverage": float(accepted.mean()) if len(truth) else None,
    "accuracy_including_abstentions": float(correct.mean()) if len(truth) else None,
    "accuracy_on_accepted": float(correct[accepted].mean()) if accepted.any() else None,
    "per_class": per_class,
    "confusion_truth_rows_prediction_columns_unknown_last": confusion.tolist(),
  }


def load_annotations(
  path: Path, coordinates: np.ndarray, class_count: int
) -> tuple[np.ndarray, np.ndarray, int]:
  if coordinates.ndim != 2 or coordinates.shape[1] != 3:
    raise ValueError("Coordinates must have shape (N, 3).")
  source_ids: list[int] = []
  truth: list[int] = []
  seen_ids: set[int] = set()
  with path.open(newline="") as file:
    reader = csv.DictReader(file)
    required = {"source_id", "x", "y", "z", "human_class_id"}
    if not required.issubset(reader.fieldnames or []):
      raise ValueError("Annotation CSV is missing required fields.")
    for row in reader:
      source_id = int(row["source_id"])
      if source_id in seen_ids or not 0 <= source_id < len(coordinates):
        raise ValueError("Annotation source IDs must be unique and in range.")
      seen_ids.add(source_id)
      position = np.array([float(row[axis]) for axis in ("x", "y", "z")])
      if not np.isfinite(position).all() or not np.allclose(
        position, coordinates[source_id], rtol=0, atol=1e-6
      ):
        raise ValueError("Annotation coordinates do not match the source row.")
      label = row["human_class_id"].strip()
      if not label:
        continue
      class_id = int(label)
      if not 0 <= class_id < class_count:
        raise ValueError("Human labels must be valid class IDs; leave ambiguity blank.")
      source_ids.append(source_id)
      truth.append(class_id)
  return (
    np.array(source_ids, dtype=np.int64),
    np.array(truth, dtype=np.int64),
    len(seen_ids),
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("coordinates", type=Path)
  parser.add_argument("raw_labels", type=Path)
  parser.add_argument("confidence", type=Path)
  parser.add_argument("annotations", type=Path)
  parser.add_argument("class_names", type=Path)
  parser.add_argument("report", type=Path)
  parser.add_argument(
    "--margins", type=float, nargs="+", default=[0, 0.002, 0.005, 0.01, 0.02]
  )
  args = parser.parse_args()
  coordinates = np.load(args.coordinates, allow_pickle=False)
  labels = np.load(args.raw_labels, allow_pickle=False)
  names = json.loads(args.class_names.read_text())
  if (
    not isinstance(names, list)
    or not names
    or not all(isinstance(name, str) for name in names)
  ):
    raise ValueError("Expected a non-empty class-name list.")
  if labels.shape != (len(coordinates),) or labels.dtype.kind not in "iu":
    raise ValueError("Raw labels must align with coordinates.")
  if np.any((labels < 0) | (labels >= len(names))):
    raise ValueError("Raw labels must be ungated class IDs.")
  with np.load(args.confidence, allow_pickle=False) as confidence:
    margins = confidence["cosine_margin"]
    similarities = confidence["cosine_similarity"]
  if any(
    values.shape != labels.shape or not np.isfinite(values).all()
    for values in (margins, similarities)
  ):
    raise ValueError("Finite confidence arrays must align with labels.")
  if np.any((margins < 0) | (margins > 2)) or np.any(np.abs(similarities) > 1):
    raise ValueError("Cosine confidence is outside its valid range.")
  if any(not np.isfinite(value) or not 0 <= value <= 2 for value in args.margins):
    raise ValueError("Margin thresholds must be finite and in [0, 2].")
  source_ids, truth, total = load_annotations(args.annotations, coordinates, len(names))
  variants: dict[str, object] = {
    "raw": semantic_metrics(truth, labels[source_ids], len(names)),
  }
  for threshold in args.margins:
    prediction = np.where((margins >= threshold) & (similarities >= 0), labels, -1)
    variants[f"margin_{threshold:g}_similarity_0"] = {
      "all_input_coverage": float(np.mean(prediction != -1)),
      **semantic_metrics(truth, prediction[source_ids], len(names)),
    }
  report = {
    "schema_version": 1,
    "status": "scored_annotations" if len(truth) else "awaiting_annotations",
    "annotation_rows": total,
    "annotated_rows": len(truth),
    "unlabeled_rows": total - len(truth),
    "classes": names,
    "limitations": [
      "Human labels must be independently supplied; their provenance is not verified by this tool.",
      "Abstentions count as errors in full accuracy and as false negatives in class recall/IoU.",
      "Threshold sweep is diagnostic, not independent threshold selection and test evaluation.",
      "Metrics apply only to annotated Gaussian rows, not uniform surface area or the whole scene.",
    ],
    "inputs": {
      str(path): hashlib.sha256(path.read_bytes()).hexdigest()
      for path in (
        args.coordinates,
        args.raw_labels,
        args.confidence,
        args.annotations,
        args.class_names,
      )
    },
    "variants": variants,
  }
  if args.report.resolve() in {
    path.resolve()
    for path in (
      args.coordinates,
      args.raw_labels,
      args.confidence,
      args.annotations,
      args.class_names,
    )
  }:
    raise ValueError("The output report must not overwrite an input.")
  args.report.parent.mkdir(parents=True, exist_ok=True)
  args.report.write_text(json.dumps(report, indent=2) + "\n")
  print(f"{report['status']}: {len(truth)}/{total} rows annotated; {args.report}")


if __name__ == "__main__":
  main()
