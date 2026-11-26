# P10 apartment 250k/300k multi-view refinement audit

Date: 2026-11-25. Status: 24 static captures audited; P10 remains in progress.

Accepted directory:
`artifacts/p10-vision-matrix/multiview-refinement-iOIrPPHa/`.
All reports passed in static capture mode, with no PPO updates or throughput
samples. Eight views use two existing spawns and four spawn-relative yaw offsets;
each compares 250,000/300,000/full 374,410 shared Gaussians at 64x48, one world.

Audit checked actual counts, current manifest/source/collision hashes, dimensions,
requested views, finite qpos and absence of traceback. Within each group the
actual qpos is exactly equal, as are camera, input provenance, configuration,
GPU and reported versions. No timing medians or samples were present.

Generated artifacts in the accepted directory:

- `comparison.png`: eight rows, 250k/300k/full RGB followed by 250k/300k/full depth.
  Depth uses a fixed 0–5 m scale; invalid pixels are magenta.
- `visual-audit.json`: per-view measurements and all 24 report/capture hashes.
  SHA256: `f74cfc7707a8b4188ea7e0ab2445c0a107b51390b38e46abcd062a6ea53c113c`.

Source captures were not modified; audit and plotting were CPU-only.

## Matched-view results

Lost depth is the fraction of full-reference-valid pixels absent in the capped
frame. Depth MAE uses only pixels valid in both, excluding holes. Reference is
the full render, not independent geometry truth. No predeclared quality acceptance
threshold exists; these values do not establish training equivalence.

| Spawn | Yaw offset | Lost depth 250k | Lost depth 300k | Common-valid MAE 250k (cm) | Common-valid MAE 300k (cm) |
|---|---:|---:|---:|---:|---:|
| default | 0 | 0.175% | 0.070% | 1.567 | 0.621 |
| default | 90 | 1.205% | 0.293% | 1.280 | 0.616 |
| default | 180 | 0.423% | 0.065% | 1.267 | 0.665 |
| default | 270 | 0.721% | 0.131% | 1.594 | 0.782 |
| p7_free | 0 | 0.752% | 0.098% | 1.321 | 0.655 |
| p7_free | 90 | 0.651% | 0.033% | 1.183 | 0.626 |
| p7_free | 180 | 0.749% | 0.065% | 1.164 | 0.625 |
| p7_free | 270 | 0.149% | 0.075% | 2.101 | 0.851 |

Worst-view RGB MAE on 0–255 channels is 2.715 for 250k and 1.455 for 300k.
The largest common-valid fraction with depth error >0.1 m is 3.175%/1.194%
for 250k/300k, both at p7_free+270. These are fractions, not maximum pixel error.

Visual inspection shows reduced wall speckling and fewer additional depth holes
at 300k than 250k, with both closer to full than the prior 200k candidate. Small
differences remain, especially walls and depth boundaries. No default budget is
changed, and no speedup is inferred from static captures or initialization time.
Single-scene, low-resolution static views do not cover motion or task outcomes.

## Archived frame consistency and updated experiment priority

For all eight full views, qpos, RGB and raw depth are exactly equal to the prior
`multiview-3LNvW1Qt` full captures. This is archived-frame consistency only: reports
identify package versions but do not fingerprint the installed native binaries or
the source revision of the user's performance fix. It is not a controlled
before/after verification of that fix.

The user reports a large sensing-performance improvement from a MuJoCo source
fix. Earlier throughput runs remain historical build baselines; their bottleneck
conclusions are not assumed to describe the fixed build. Priority shifts to
fingerprinting and benchmarking the fixed full-Gaussian workload, including output
consistency, rather than further truncation as a workaround for the old cost.
No new GPU experiment, dependency install or driver change was performed during
this audit. Runtime code is unchanged; evidence/TODO updates and diff checks only.
