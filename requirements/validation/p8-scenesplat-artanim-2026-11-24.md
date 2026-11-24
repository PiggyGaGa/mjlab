# P8 SceneSplat Artanim feature and semantic diagnostics

Date: 2026-11-24. This is diagnostic evidence, not semantic-quality acceptance.

## Inputs and recovery

SceneSplat commit: `07e2c946203467e7dfc402832fad2b8b1e49b810`.
The isolated checkout and environment live in `artifacts/p8-scenesplat/`.
The checkout registers only the inference model and PT-v3, loads checkpoint
weights strictly after removing `module.` prefixes, and omits the unsupported
`return_backbone` keyword. Chunk size is 50,000; chunking reduces global context
and is not claimed equivalent to unchunked inference.

Checkpoint remains at
`/home/liana/SSD/Ubuntu/robot/models/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth`.
The user reran `bash artifacts/p8-scenesplat/run-inference.sh` on RTX 3060.
The log reports 393 strictly loaded keys and feature saving approximately
81 seconds after checkpoint loading. This is not a kernel-only benchmark.

Features: `artifacts/p8-scenesplat/features/artanim_feat.pt`, float16,
shape `(659010, 768)`, all finite.
SHA-256: `f727a15414e5d9812720bc692c6bb8a476e7f756f6c6b5089890170b4832bb2b`.
Preprocessed coordinates exactly equal original PLY XYZ in original row order.
The inference implementation restores sampled features using the inverse map;
coordinate equality and row counts do not independently prove semantic accuracy.

## Decoding

Use official InteriorGS 72-class SigLIP2 embeddings, not unrelated text features.
Embedding SHA-256:
`231d4fd81216ddc2d4d9ad48e99575ceb7a5d474567ab16f20dd966a4588d6e0`.
Class names were verified exactly against the official label file.

The existing decoder ran on CPU in blocks of 8,192 rows:

- Raw: minimum similarity -1, minimum margin 0, no rejected labels.
- Diagnostic gate: minimum similarity 0, minimum margin 0.01, unknown ID -1.
  These are uncalibrated experimental thresholds, not validated confidence cutoffs.
- Outputs, confidence arrays, and hashed reports are in
  `artifacts/p8-scenesplat/features/` with `raw` and `margin001` names.

Raw leading class assignments:

| Class | Gaussian count |
|---|---:|
| floor | 201,603 |
| ceiling | 131,896 |
| column | 102,947 |
| door | 57,262 |
| wall | 41,953 |
| drawer | 35,812 |

Mean cosine similarity is 0.118075 (p05 0.099123).
Mean top-two cosine margin is 0.007233 (median 0.004961, p05 0.000446).
The margin gate retains 135,237 labels and rejects 523,773 (79.4788%) as unknown.
In particular, only 8,386 of 201,603 raw floor assignments remain after gating.
Scores and class counts are not precision, recall, IoU, or calibrated probabilities.
No model category is automatically removed from collision based on these labels.

## Fixed-budget diagnostic

Ran the existing `compare_3dgs_collision_semantics.py` on the full original PLY
and `artanim_labels_margin001.npy`, using `uniform_priorities.json` containing
`{}` (all labels default to weight 1). This isolates label partitioning, not a
claimed affordance-weighted priority design. Both variants share the default
0.1 m voxel/fit settings and exactly 32 final primitives.

Outputs are in `artifacts/p8-scenesplat/semantic-comparison/`:
`geometry_only.xml`, `semantic_priority.xml`, `semantic_ablation_report.json`.

| Variant | Candidates before budget | Final geoms | CPU steps/s | Mean contacts |
|---|---:|---:|---:|---:|
| geometry-only | 257 | 32 | 49,700.25 | 3.9695 |
| semantic partition | 180 | 32 | 62,360.26 | 3.7440 |

Both compiled with four identical falling sphere probes, 100 warmup and 2,000
measured steps. This is one short run: timings are not statistically stable
speedups, and fewer contacts can indicate missing support rather than an improvement.
The 32-geom budget is diagnostic, not recommended as a complete room collider.
There are no independent surface/free-space evaluation points in this run;
quality distances and false-occupied fractions are null. A reported count of zero
without free-space probes is not evidence of zero false occupancy.
No native-viewer, robot-policy, or semantic-quality benefit is inferred here.

## Spatial inspection and blind annotation preparation

The standalone `scripts/tools/inspect_3dgs_semantics.py` exports three orthogonal
RGB/raw/gated projections, a class legend, and 48 prediction-blind samples. It
uses only NumPy and Matplotlib, without altering model or simulation behavior.
Existing manual annotation CSV files are protected against overwrite.

The fixed center bounds `[-2,-2,-0.5,2,2,3]` contain 23,933 original Gaussian
rows. Raw predictions assign 19,901 rows to floor; after the experimental 0.01
margin gate, zero floor labels remain and 22,060 rows (92.18%) are unknown.
The side projections show a low approximately horizontal band in the original
geometry, also broadly assigned floor by raw predictions. This is a visual
diagnostic, not independent ground truth; it shows why the chosen margin gate
must not silently remove support geometry or be treated as calibrated confidence.

Reproduce from the repository root:

```bash
uv run scripts/tools/inspect_3dgs_semantics.py \
  artifacts/p8-scenesplat/artanim \
  artifacts/p8-scenesplat/features/artanim_labels_raw.npy \
  artifacts/p8-scenesplat/features/artanim_labels_margin001.npy \
  artifacts/p8-scenesplat/features/class_names.json \
  artifacts/p8-scenesplat/inspection-center \
  --bounds -2 -2 -0.5 2 2 3
```

The output directory contains `semantic_projections.png`, `class_legend.png`,
`blind_samples.png`, `manual_labels.csv`, and `inspection.json` (input hashes,
class mapping, seed and sampling limitations). Manual labels are intentionally
blank; no model prediction is copied into human ground truth. Consult only the
RGB sample views and an independent original-scene view while annotating; if a
sample is occluded or ambiguous, leave it unlabeled and explain in `notes`.
Class IDs follow `class_names.json` (floor 7, wall 2, ceiling 5, etc.).
Unknown prediction ID -1 is not an annotation truth category.
This center crop is floor-heavy and cannot establish all-class performance.
The sampling unit is a Gaussian row, not uniform surface area or object instances.
Pyright excludes the Git-ignored `artifacts/` directory, preserving its standard
hidden/cache/node_modules exclusions. This keeps the isolated third-party
SceneSplat checkout out of mjlab type checking; project source and tests remain
checked. No simulation runtime or play/demo files change.
Validation: 32 collision-generation/inspection tests pass; `make check` passes
with `UV_NO_SYNC=1` (Ruff, ty and Pyright: zero errors); `make docs` succeeds.
`git diff --check` passes; play/demo and their config tests have zero diff.

## Independent scoring and remaining desktop runs

`scripts/tools/evaluate_3dgs_semantics.py` reads the manual CSV, verifies unique
in-range source IDs and XYZ alignment, then scores only nonblank human labels.
Unknown prediction -1 is never accepted as human truth. Confusion matrices include
an unknown prediction column; abstentions reduce full accuracy and class recall/IoU.
Coverage and accepted-only accuracy are reported separately. With no annotations,
quality metrics are null and status is `awaiting_annotations`.

```bash
uv run scripts/tools/evaluate_3dgs_semantics.py \
  artifacts/p8-scenesplat/artanim/coord.npy \
  artifacts/p8-scenesplat/features/artanim_labels_raw.npy \
  artifacts/p8-scenesplat/features/semantic_raw_report.confidence.npz \
  artifacts/p8-scenesplat/inspection-center/manual_labels.csv \
  artifacts/p8-scenesplat/features/class_names.json \
  artifacts/p8-scenesplat/inspection-center/quality_report.json
```

The report includes raw predictions and diagnostic similarity>=0 sweeps at margins
0/0.002/0.005/0.01/0.02; this is not independent threshold tuning and test evaluation.
The first actual CSV scoring run has 0/48 human labels and remains pending.
The user subsequently declined manual classification and requested automatic
abstention instead. Manual annotation is therefore optional and is not a runtime
prerequisite. No missing truth is inferred or filled in, and independent label
accuracy remains unverified. The 0.01 margin is an explicit conservative
experiment setting, not a calibrated probability or a guarantee against mistakes.
Unknown labels still enter the standard geometry filtering/fitting path; they are
not directly removed or treated as free space simply because their category is
uncertain. `bash artifacts/p8-scenesplat/run-automatic-collision.sh` regenerates
the gated labels and the full unbudgeted proxy under `automatic-collision/`.
The 32-geom variants remain diagnostics, not a proposed complete room asset.

Official demo checkpoint/motion were restored to
`artifacts/p8-scenesplat/policy-assets/`, using the upstream downloader and its
SHA-256 validation without modifying demo/play. A persistent manifest compares
plane control and three 32-geom variants: geometry-only, raw semantic partitioning,
and margin-0.01 semantic partitioning. All semantic weights remain 1. An additional
raw-label generation/CPU falling-probe run uses the same settings as the gated run.

After the raw run has completed, desktop GPU/native measurements use:

```bash
bash artifacts/p8-scenesplat/run-policy-benchmarks.sh
```

The script runs the existing evaluator for seeds 42/43/44 and 500-step horizons,
then opens native 3DGS viewers for four variants and records 500 submissions each.
Policy uses terminations; native measurement disables terminations and is throughput
evidence, not task success. Results go to `artifacts/p8-scenesplat/policy-results/`.
The plane is a protocol control, not a budget-matched collision variant.
These actual-semantic GPU/native measurements subsequently completed; results and
the report audit are recorded below.
The raw-label 32-geom run completed with 114 candidates before budgeting,
98.52 seconds generation and 47,907.12 CPU steps/s. Its geometry-only control had
257 candidates, 11.39 seconds generation and 41,549.96 CPU steps/s. Mean contacts
were 3.744 and 3.9695 respectively. Cross-run timing variation is substantial;
this does not establish a speedup or contact-quality improvement.

## Final acceptance status

### Automatic unknown handling and fallback

Added the opt-in `scripts/tools/generate_3dgs_collision_guarded.py` wrapper. It
reuses the existing generator without changing its defaults or runtime APIs.
It always generates a geometry baseline. If more than 50% of input labels are
unknown, semantic grouping is bypassed. Otherwise a same-input semantic proxy is
generated and must retain source mapping coverage and keep source distance
mean/p95/max within 1.05 times the baseline. Both thresholds are explicit tunable
engineering settings, not independently calibrated semantic or safety guarantees.
The report records input hashes, selection and rejection reasons. Geometry selection
does not backfill semantic categories or remove unknown points as free space.

Actual Artanim run: 79.4788% unknown triggers geometry selection. The selected
unbudgeted proxy has 257 geoms (254 boxes, 3 cylinders), compared with the
unbudgeted semantic candidate's 180 geoms (178 boxes, 2 cylinders).
The latter maps 58.47% of filtered source rows, with source-distance
mean/p95/max 0.14757/0.61680/2.15601 m; it is not promoted as a better collider.
The selected output and audit files are under
`artifacts/p8-scenesplat/guarded-collision/`, including `collision.xml`,
`report.json`, `report.source_ids.npz`, and `guard_report.json`.

```bash
uv run scripts/tools/generate_3dgs_collision_guarded.py \
  src/mjlab/asset_zoo/3dgs/ArtanimMotionCaptureStudio/scene_transform_20p.ply \
  artifacts/p8-scenesplat/features/artanim_labels_margin001.npy \
  artifacts/p8-scenesplat/guarded-collision
```

Independent truth-based label accuracy remains optional and unmeasured under the
user's clarified automatic/unknown requirement. This does not establish downstream
semantic benefit; actual-semantic GPU/native results follow.

### Actual semantic policy comparison

The same four-variant manifest completed CPU and desktop CUDA runs, each with
seeds 42/43/44, the verified official checkpoint/motion, 500-step maximum horizons,
enabled task termination and identical stochastic settings. The three non-plane
variants each have 32 collision geoms.

| Variant | CPU horizon completion | CUDA horizon completion | CPU control SPS | CUDA control SPS | CUDA MPKPE (m) |
|---|---:|---:|---:|---:|---:|
| plane control | 3/3 | 3/3 | 27.36 | 73.44 | 0.09612 |
| geometry 32 | 3/3 | 3/3 | 26.81 | 73.41 | 0.15193 |
| raw semantic 32 | 3/3 | 2/3 | 26.03 | 75.93 | 0.15205 |
| margin 0.01 semantic 32 | 3/3 | 2/3 | 25.48 | 74.30 | 0.16091 |

Both semantic CUDA seed-44 runs terminated at step 11 with `ee_body_pos`.
Early failures have shorter timed trajectories, so their throughput means are not
evidence of a useful speedup. CPU/CUDA trajectories are not required to match.
CPU plane seed 44 and raw-semantic seed 43 report `LS_ITERATIONS` (bit 1024).
CUDA plane seed 42, geometry seeds 42/43 and both semantic seed-42 runs also report
this flag. These are solver line-search iteration-limit warnings, not proof of
contact-buffer overflow or universally stable contact. No warning is hidden or
disabled to force acceptance. The data do not support a semantic-task benefit;
retaining the guarded geometry fallback is the conservative engineering decision.

The full unbudgeted geometry fallback maps 93.34% of filtered source rows, with
source-distance mean/p95/max 0.07787/0.18774/1.45173 m. It compiled and completed
the same 2x2 falling-probe run as the semantic candidate. Source metrics remain
input-coverage diagnostics, not independent truth or a robot-safety guarantee.

### Native rendering and final audit

| Variant | Actual realtime | Submissions/s | Overflow bitmask |
|---|---:|---:|---:|
| plane control | 0.52611 | 26.30563 | 0 |
| geometry 32 | 0.48462 | 24.23106 | 0 |
| raw semantic 32 | 0.48874 | 24.43693 | 0 |
| margin 0.01 semantic 32 | 0.49000 | 24.50007 | 0 |

All four native runs report renderer ready and 500 measured submissions. Native
measurement disables termination and is not evidence of task success or display FPS.
The audit verifies identical collision/checkpoint/motion hashes across CPU, CUDA
and native reports, all four variant names, and the three fixed policy seeds.

Persistent report SHA-256 values:

- `policy-results/cpu-policy.json`:
  `33702673f785622242fc1462cbd678b213a29a575609edb7d91d448793708980`.
- `policy-results/cuda-policy.json`:
  `f5eef4c080b898ea58bf69d1a2f9be32fc1470dbfccb7934ae7f802c06cfded4`.
- `policy-results/native-3dgs.json`:
  `49c9103169956a43d441f0b8e646a4c5c1a7cca32d61a3e0d902322aab82a742`.

42 focused collision, semantic inspection/scoring/guard and policy-protocol tests
pass. `make check` and `make docs` pass with `UV_NO_SYNC=1`; `git diff --check`
passes. Play/demo and their config tests have zero diff.

P8 engineering acceptance is complete under the user's explicit automatic
classification/unknown/fallback scope, not the earlier manual-label prerequisite.
It includes real model inference and actual CPU/CUDA/native comparison rather than
just synthetic sidecars. Independent semantic accuracy, real-scene free-space and
support ground truth, downstream semantic benefits, and collision safety guarantees
are not claimed. Further independent annotation-based validation belongs to later
research; the existing optional scorer still correctly reports no supplied truth.
