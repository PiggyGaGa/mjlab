# P10 native-viewer workload retest

Date: 2026-11-26. Status: three desktop repeats audited; native workload subtask
complete, full P10 in progress.

The current installed-environment camera retest completed with exact archived
RGB/depth equality but timings similar to the historical workload; installed Warp
remains 8ffab7a. The native viewer uses a separate path. Recent neighboring MuJoCo
source changes target `simulate/gaussian`, so this experiment measures native
policy/viewer workload separately rather than extrapolating training-camera data.

## Reused experiment

```bash
bash artifacts/p10-vision-matrix/run-native-retest.sh
```

The launcher directly reuses the unchanged
`scripts/tools/benchmark_native_3dgs_collision_policy.py`, existing apartment
`variants.json` and its reference/automatic collision copies, original full PLY,
cached checkpoint and motion. It does not regenerate collision, train policies,
modify shared code/play/demo, change dependencies, clear caches or alter drivers.
P8 remains closed; collision variants are held constant as workload inputs, not
reopened for geometric/semantic quality acceptance.

Three separate processes each run reference then automatic: six native windows
open/close automatically. Each variant waits for the Gaussian renderer to become
ready, warms 20 policy steps, resets environment/policy and measures 500 policy
steps/submissions with terminations disabled. Let the script close each window;
closing it early aborts the benchmark. Avoid camera/resize interaction or other
GPU work during the measurement.

Outputs use a fresh `artifacts/p10-vision-matrix/native-retest-XXXXXXXX` directory.
Separate CPU snapshots before/after each process record installed MuJoCo/Warp
file identities, mapped MuJoCo library hashes, input file hashes and benchmark
source hash. These are explicitly external snapshots, not fingerprints embedded
in the child renderer process. The launcher checks unchanged snapshots within
and across runs, renderer ready, both variants, 500 submissions, zero overflow
and positive finite throughput, then writes three-run medians to `summary.json`.
Failures retain partial reports/logs and stop subsequent runs.

## Interpretation boundaries

Actual RT includes policy inference, physics, CPU synchronization and native
viewer submissions. Submissions/s is not display refresh FPS; neither quantity
isolates Gaussian sorting, shading or GPU frame completion. GPU/window settings
and asynchronous rendering can affect results; window dimensions are not logged
by the reused tool, so do not claim resolution-controlled kernel attribution.
Within each process the reference always runs first; three repetitions do not
form randomized independent crossover trials.

The previous `artifacts/p10-apartment-comparison/native-3dgs.json` lacks installed
binary fingerprints. Preserve it as a historical workload report; do not treat a
same-version package string as a proven old build or assert a causal fix speedup.
Native performance success also does not prove training-camera acceleration,
visual equivalence, collision quality or policy task success. Full P10 remains open.

## Preparation checks

All required cached input paths exist. `bash -n`, embedded Python AST compilation
and `git diff --check` passed. A mocked uv launcher run verified three native
invocations with 20 warmup/500 measured steps plus six external identity calls
and one summary call. The summary block was executed on temporary CPU fixtures
copied from an archived native report to verify field compatibility and medians;
those fixture values are not new native/GPU performance results. No runtime code
changed in this preparation, and no window or GPU simulation was launched.

## Audited desktop results

Accepted directory: `artifacts/p10-vision-matrix/native-retest-C6pmr45y/`.
All three processes completed both variants. All six results have renderer ready,
500 measured submissions and zero overflow. Logs contain no traceback. CPU audit
recomputed submissions/s and Actual RT (step dt 0.02 s) from elapsed times and
recomputed all summary medians.

The six separate before/after snapshots are identical. Input, checkpoint/motion,
PLY, variant XML and benchmark source hashes match current files. Native report
input hashes agree across repeats. Installed file identities remain MuJoCo tree
`d87ce2bc7067b970d9c1d3d442a1dfbb5f0fc118d00ced598de488db9391a566` and Warp tree
`f635c513b1c0da16928ea4b50f88fa4f9b70e243d4069f0f69f716703e11bfe1`, with Warp
Git revision 8ffab7a. These are external CPU snapshots as described above, not
newly embedded renderer-process provenance.

| Variant | Median Actual RT | Actual RT range | Median submissions/s | Median elapsed (s) |
|---|---:|---:|---:|---:|
| reference | 1.02978 | 1.02960–1.05582 | 51.48888 | 9.71083 |
| automatic | 1.02834 | 1.01794–1.02981 | 51.41680 | 9.72445 |

Each 500-step measurement represents 10 simulated seconds. This current workload
thus runs slightly faster than real time in all six measurements. The results are
effectively comparable between the two fixed collision variants in this small
test; this does not establish collision quality or policy success with termination
disabled. No resolution-controlled or causal renderer-kernel speedup is claimed.
Historical native reports lack matching binary fingerprints and logged window
dimensions, so they are not treated as a controlled pre-fix arm.

`audit-manifest.json` records SHA256 values of the three reports, six snapshots and
summary. Its SHA256 is
`f0b68fd3743a01e8095c2d52f47a13ff75938f3274f4b9099e0f920cbf88aaea`.
This audit did not launch another GUI/GPU run or change dependencies/drivers.

The separate current installed-Warp camera retest still has historical-scale
timings. Native policy/viewer realtime throughput here must not be generalized
to camera-observation rendering. Native retest evidence is complete for this
protocol; broader P10 matrices, newer Warp evaluation and full-suite acceptance
remain separate work.
