"""Tests for prediction-blind semantic inspection exports."""

import csv
import json
import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tools/inspect_3dgs_semantics.py"


def test_region_sampling_is_reproducible_and_preserves_source_ids() -> None:
  sample_region = cast(
    Callable[[np.ndarray, np.ndarray, int, int], np.ndarray],
    runpy.run_path(str(SCRIPT))["sample_region"],
  )
  coordinates = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [np.nan, 0, 0]])
  bounds = np.array([-0.5, -1, -1, 1.5, 1, 1])
  np.testing.assert_array_equal(sample_region(coordinates, bounds, 10, 42), [0, 1])
  np.testing.assert_array_equal(
    sample_region(coordinates, bounds, 1, 42),
    sample_region(coordinates, bounds, 1, 42),
  )
  with pytest.raises(ValueError, match="No finite coordinates"):
    sample_region(coordinates, np.array([3, 3, 3, 4, 4, 4]), 1, 42)
  with pytest.raises(ValueError, match="sample count"):
    sample_region(coordinates, bounds, 0, 42)


def test_inspection_export_is_blind_and_does_not_overwrite_annotations(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  np.save(tmp_path / "coord.npy", np.array([[0, 0, 0], [1, 1, 1]]))
  np.save(tmp_path / "color.npy", np.array([[100, 0, 0], [0, 100, 0]]))
  np.save(tmp_path / "raw.npy", np.array([0, 1]))
  np.save(tmp_path / "gated.npy", np.array([0, -1]))
  (tmp_path / "names.json").write_text('["floor", "wall"]')
  output = tmp_path / "inspection"
  monkeypatch.setattr(
    sys,
    "argv",
    [
      str(SCRIPT),
      str(tmp_path),
      str(tmp_path / "raw.npy"),
      str(tmp_path / "gated.npy"),
      str(tmp_path / "names.json"),
      str(output),
      "--bounds",
      "-1",
      "-1",
      "-1",
      "2",
      "2",
      "2",
    ],
  )
  main = cast(Callable[[], None], runpy.run_path(str(SCRIPT))["main"])
  main()
  annotation_path = output / "manual_labels.csv"
  original_bytes = annotation_path.read_bytes()
  with annotation_path.open() as file:
    rows = list(csv.DictReader(file))
  assert [row["source_id"] for row in rows] == ["0", "1"]
  assert all(row["human_class_id"] == "" for row in rows)
  assert all("prediction" not in key for key in rows[0])
  assert (output / "blind_samples.png").is_file()
  assert (output / "semantic_projections.png").is_file()
  metadata = json.loads((output / "inspection.json").read_text())
  assert metadata["sample_count"] == 2
  assert len(metadata["inputs"]) == 5
  with pytest.raises(FileExistsError, match="Refusing to overwrite"):
    main()
  assert annotation_path.read_bytes() == original_bytes
