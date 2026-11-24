"""Utilities for decoding per-Gaussian language features into semantic labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SemanticDecodeResult:
  """Decoded labels and confidence diagnostics in original point order."""

  labels: np.ndarray
  similarities: np.ndarray
  margins: np.ndarray


def decode_semantic_features(
  features: np.ndarray,
  class_embeddings: np.ndarray,
  *,
  class_ids: np.ndarray | None = None,
  minimum_similarity: float = -1.0,
  minimum_margin: float = 0.0,
  unknown_label: int = -1,
  block_size: int = 65_536,
) -> SemanticDecodeResult:
  """Decode normalized language features by deterministic cosine similarity."""
  features = np.asarray(features, dtype=np.float32)
  class_embeddings = np.asarray(class_embeddings, dtype=np.float32)
  if (
    features.ndim != 2
    or class_embeddings.ndim != 2
    or not len(features)
    or not len(class_embeddings)
    or features.shape[1] != class_embeddings.shape[1]
    or not np.isfinite(features).all()
    or not np.isfinite(class_embeddings).all()
  ):
    raise ValueError(
      "features and class_embeddings must be finite non-empty matrices with the "
      "same feature dimension."
    )
  if class_ids is None:
    class_ids = np.arange(len(class_embeddings), dtype=np.int64)
  else:
    class_ids = np.asarray(class_ids)
    if (
      class_ids.ndim != 1
      or len(class_ids) != len(class_embeddings)
      or class_ids.dtype.kind not in "iu"
      or len(np.unique(class_ids)) != len(class_ids)
    ):
      raise ValueError("class_ids must contain one unique integer per embedding.")
    class_ids = class_ids.astype(np.int64, copy=False)
  if unknown_label in class_ids:
    raise ValueError("unknown_label must not also be a class ID.")
  if not -1 <= minimum_similarity <= 1:
    raise ValueError("minimum_similarity must be in [-1, 1].")
  if not 0 <= minimum_margin <= 2:
    raise ValueError("minimum_margin must be in [0, 2].")
  if block_size <= 0:
    raise ValueError("block_size must be positive.")

  feature_norms = np.linalg.norm(features, axis=1)
  embedding_norms = np.linalg.norm(class_embeddings, axis=1)
  if np.any(feature_norms <= 1e-12) or np.any(embedding_norms <= 1e-12):
    raise ValueError("features and class_embeddings must not contain zero vectors.")
  normalized_embeddings = class_embeddings / embedding_norms[:, None]
  labels = np.full(len(features), unknown_label, dtype=np.int64)
  similarities = np.empty(len(features), dtype=np.float32)
  margins = np.empty(len(features), dtype=np.float32)
  for start in range(0, len(features), block_size):
    end = min(start + block_size, len(features))
    normalized_features = features[start:end] / feature_norms[start:end, None]
    scores = normalized_features @ normalized_embeddings.T
    best_indices = np.argmax(scores, axis=1)
    row_indices = np.arange(end - start)
    best_scores = scores[row_indices, best_indices]
    if len(class_embeddings) > 1:
      runner_up = np.partition(scores, -2, axis=1)[:, -2]
      score_margins = best_scores - runner_up
    else:
      score_margins = np.full(len(best_scores), 2.0, dtype=np.float32)
    accepted = (best_scores >= minimum_similarity) & (score_margins >= minimum_margin)
    labels[start:end][accepted] = class_ids[best_indices[accepted]]
    similarities[start:end] = best_scores
    margins[start:end] = score_margins
  return SemanticDecodeResult(labels, similarities, margins)
