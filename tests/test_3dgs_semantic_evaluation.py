"""Regression tests for abstention-aware, independent semantic scoring."""

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

SCRIPT = (
  Path(__file__).resolve().parents[1] / "scripts/tools/evaluate_3dgs_semantics.py"
)


def test_semantic_scoring_counts_unknown_as_missed_support() -> None:
  metrics = cast(
    Callable[..., dict[str, Any]], runpy.run_path(str(SCRIPT))["semantic_metrics"]
  )
  result = metrics(np.array([0, 0, 1, 1]), np.array([0, -1, 0, 1]), 2)
  assert result["coverage"] == 0.75
  assert result["accuracy_including_abstentions"] == 0.5
  assert result["accuracy_on_accepted"] == pytest.approx(2 / 3)
  assert result["per_class"][0]["precision"] == 0.5
  assert result["per_class"][0]["recall"] == 0.5
  assert result["per_class"][0]["iou"] == pytest.approx(1 / 3)
  assert result["per_class"][1]["recall"] == 0.5
  assert result["confusion_truth_rows_prediction_columns_unknown_last"] == [
    [1, 0, 1],
    [1, 1, 0],
  ]


def test_semantic_scoring_never_turns_missing_truth_into_perfect_quality() -> None:
  metrics = cast(
    Callable[..., dict[str, Any]], runpy.run_path(str(SCRIPT))["semantic_metrics"]
  )
  empty = np.array([], dtype=np.int64)
  result = metrics(empty, empty, 2)
  assert result["accuracy_including_abstentions"] is None
  assert result["coverage"] is None
  assert result["accuracy_on_accepted"] is None
  abstained = metrics(np.array([0]), np.array([-1]), 2)
  assert abstained["accuracy_including_abstentions"] == 0
  assert abstained["accuracy_on_accepted"] is None
  with pytest.raises(ValueError, match="Human labels"):
    metrics(np.array([-1]), np.array([0]), 2)
  with pytest.raises(ValueError, match="Predictions"):
    metrics(np.array([0]), np.array([2]), 2)


def test_annotation_loading_checks_alignment_and_preserves_blanks(
  tmp_path: Path,
) -> None:
  loader = cast(Callable[..., Any], runpy.run_path(str(SCRIPT))["load_annotations"])
  path = tmp_path / "manual.csv"
  header = "source_id,x,y,z,human_class_id\n"
  coordinates = np.array([[0, 0, 0], [1, 1, 1]])
  path.write_text(header + "0,0,0,0,\n1,1,1,1,1\n")
  source_ids, truth, total = loader(path, coordinates, 2)
  np.testing.assert_array_equal(source_ids, [1])
  np.testing.assert_array_equal(truth, [1])
  assert total == 2
  path.write_text(header + "0,0,0,0,0\n0,0,0,0,1\n")
  with pytest.raises(ValueError, match="unique"):
    loader(path, coordinates, 2)
  path.write_text(header + "0,1,0,0,0\n")
  with pytest.raises(ValueError, match="coordinates"):
    loader(path, coordinates, 2)
  path.write_text(header + "0,0,0,0,-1\n")
  with pytest.raises(ValueError, match="valid class IDs"):
    loader(path, coordinates, 2)
