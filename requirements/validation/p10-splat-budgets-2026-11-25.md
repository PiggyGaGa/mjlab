# P10 apartment Gaussian-count budget audit

Date: 2026-11-25. Status: eight-case submatrix complete; P10 overall in progress.

Accepted run: `artifacts/p10-vision-matrix/splat-budgets-qg3ZlRyj/`.
All eight reports passed, with three retained samples numbered 1/2/3, finite losses
and positive finite timings. Recomputed medians match the reports. Logs contain
no traceback. Actual shared normalized counts equal 50,000/100,000/200,000/374,410;
these are not counts multiplied by world count.

Audit checked current manifest/PLY/collision hashes and matched camera, input
asset provenance, PPO settings, GPU, package versions and rollout context against
the full reference at each world count. Configuration is fixed at 64x48 RGB/depth,
32 measured steps, 16 warmup steps, three retained repeats after one discarded
repetition. GPU/package versions remain RTX 3060, MuJoCo 3.16.0+3dgs, Warp 1.18.0,
Torch 2.14.0. Counts use existing opacity-ranked truncation, not uniform sampling.

Every capped `reset.npz` has qpos exactly equal to its full reference at the same
world count. One-world and four-world references have different reset views;
image comparisons are made only within matching world count and world index.

## Timing and reset-view availability

Sense milliseconds are per batch across all worlds. Full-env SPS sums world
transitions. Values are medians of short within-process repeats; they do not
establish independent-trial confidence intervals. Positive finite depth fractions
are averaged across pixels/worlds; they are not geometry accuracy or global
coverage metrics. RGB MAE uses 0–255 channels against the full-render reference,
which is not human ground truth.

| Worlds | Budget | Sense ms/batch | Full-env SPS | Positive finite depth | RGB MAE |
|---|---:|---:|---:|---:|---:|
| 1 | 50,000 | 0.903 | 101.57 | 38.02% | 76.04 |
| 1 | 100,000 | 0.992 | 106.29 | 42.97% | 66.43 |
| 1 | 200,000 | 4.605 | 73.36 | 92.02% | 2.62 |
| 1 | 374,410 | 14.076 | 45.86 | 92.94% | 0 |
| 4 | 50,000 | 2.540 | 381.95 | 32.42% | 82.13 |
| 4 | 100,000 | 2.751 | 311.79 | 38.22% | 70.31 |
| 4 | 200,000 | 11.045 | 191.18 | 87.83% | 3.34 |
| 4 | 374,410 | 30.614 | 78.53 | 89.48% | 0 |

Relative to the full reference in this run, 200,000 splats gives a 3.06x/2.77x
stationary-sense speedup and 1.60x/2.43x aggregate full-env throughput for 1/4
worlds. These are workload measurements, not pure Gaussian-kernel speedups or
training-quality benefits. Use this run's full references rather than older runs;
the baseline has visible run-to-run timing variation.

## Visual assessment and limits

Generated review artifacts, without modifying source captures:

- `rgb-comparison.png`: five reset views by four budgets, nearest-neighbor display.
- `depth-comparison.png`: identical layout, fixed 0–5 m scale, invalid pixels magenta.
- `visual-audit.json`: report/capture SHA256, pose checks, pixel metrics and timings.

Visual inspection confirms large black holes and missing wall/floor regions at
50,000 and 100,000. Those caps are unsuitable as replacements for the full visual
reference in these views despite their throughput. At 200,000, the main visible
wall/floor/obstacle layout resembles full, but depth remains speckled and RGB wall
appearance differs. No source asset or task default is changed.

At 200,000, 0.98%/1.84% of full-reference-valid pixels become invalid in the 1/4
world captures. On pixels valid in both, depth MAE is 0.0403/0.0459 m and the
fraction differing by more than 0.1 m is 7.22%/7.93%. Common-valid metrics omit
missing pixels, so read them alongside lost-depth fractions. Geometry-derived
depth can mask missing Gaussian content; these metrics do not certify surface
accuracy. The full reference itself also has invalid pixels.

200,000 is a candidate for further multi-view validation, not a general optimum,
accepted training replacement or default. The five reset views cover a small part
of one scene. Other PLYs, unseen viewpoints, motion and downstream task outcomes
remain untested. Next assess the candidate at multiple fixed poses against full
before changing defaults or claiming preserved visual task quality.

Report SHA256 in table order:

```text
193863827c9d87e6b79e753b8bef96e7acbd7b8aaeb5c076fe1600d8038dd519
71594b9ced8ae12a086bd7fc43bffde77fd7a4e688703e392b498d26818b7f92
f54771583c28b2b9069fbb0b8594b1200bda2c7caf408e39268dce5f7c5d0931
a1ea8623145d312d0069eec14044008797ec92621d499cb02ba6c07a2d528e65
c24c6644d7e0f3c17152f814a24e299453284a38319afc15b3421b022f98bb05
9ee628860bb10eb9263cf75d3de81796f350a947e9768ca42a2fa07ab520a720
531736bb8867fbf3c28822840efe7fd2bf08c4c1986e374cae79197126086ef2
5d03b1b3b1fa856f79719121fe9cf9cab3981b494aa5d7a592bf0286fcc3b30e
```

This audit changes evidence/status only; no GPU experiments were launched during
the audit. Existing local benchmark tests and type checks do not substitute for
the desktop evidence above. Full P10 budget/viewer/test coverage remains open.
