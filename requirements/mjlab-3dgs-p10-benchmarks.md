# P10: fixed-asset collision replacement benchmarks

Started: 2026-11-24, after the user's explicit P8 completion decision.
Desktop runs: 2026-11-24; report audit: 2026-11-25.

## Scope

Freeze the apartment visual PLY and compare the existing scene-specific collision
against the generic P8 automatic collision using the existing falling-probe,
G1 tracking policy and native 3DGS benchmark tools. Reuse the official checkpoint,
motion, seeds 42/43/44 and finite 500-step evaluation protocol. This is not new
training, a reopening of P7 safety-map acceptance, or a claim of visual avoidance.
P8 floor seams and semantic limitations remain archived; they do not trigger
further P8 tuning in this task.

## Fairness boundary

The existing apartment collision has two extra artificial, camera-visible pillars
that are not reconstructed from its PLY. Directly comparing it against the raw
automatic output would conflate collision generation with obstacle removal.
`scripts/tools/prepare_collision_comparison.py` creates independent self-contained
primitive experiment copies. Explicit `--common-geoms pillar_00 pillar_01` resolves
their shape, pose, contact and color attributes with MuJoCo, then places the same
resolved common obstacles in both copies. This is experiment configuration, not
PLY-specific generation code. No artificial floor plane is added to the automatic
variant, and no geometry is removed from the reference.

The tool rejects nonprimitive/externally dependent bundles, missing/duplicate/name-
conflicting common geoms, and output locations that contain input files. With
`--automatic-report`, both the source PLY hash and generated MJCF hash must match
the generation report. Prepared XMLs are compiled before writing. It emits local
`variants.json` and `preparation.json` with source/output hashes and resolved
common-geom attributes. Source assets and the original manifest remain unchanged.

This controls the explicit common obstacles, not geometric quality or contact
equivalence of the rest of the scene. The composed automatic experiment is not
misrepresented as the unmodified generator output. Visual 3DGS is held fixed;
native benchmark hides physical collision overlays.

## Reproduction

```bash
uv run scripts/tools/prepare_collision_comparison.py \
  src/mjlab/asset_zoo/3dgs/apartment/collision/collision.xml \
  artifacts/p8-apartment/generic/collision.xml \
  src/mjlab/asset_zoo/3dgs/apartment/scene_transform_p5.ply \
  artifacts/p10-apartment-comparison \
  --common-geoms pillar_00 pillar_01 \
  --automatic-report artifacts/p8-apartment/generic/report.json
```

Existing evaluators consume `artifacts/p10-apartment-comparison/variants.json`.
CPU physics uses identical 3x3 falling spheres, 100 warmup and 2,000 measured steps.
CPU/CUDA policy use enabled termination, observation corruption, seeds 42/43/44,
10 warmup and at most 500 control steps. Early termination and solver flags are
reported, not hidden. Native throughput disables termination and measures 500
submissions after 20 warmup steps; submissions are not display refresh FPS.
Do not compare early-failure SPS as if trajectories had equal lengths.

Desktop command (does not change the NVIDIA driver):

```bash
bash artifacts/p10-apartment-comparison/run-desktop.sh
```

## Status

- Prepared and MuJoCo-compiled both actual apartment variants with matching pillars.
- CPU falling-probe and three-seed policy runs completed; raw results are preserved.
- CUDA/native desktop runs completed and their inputs/protocol were audited on
  2026-11-25. This fixed-asset comparison subtask is complete, not all of P10.
- No simulation shared interface or play/demo changes are required.

NPZ caching, broader sensor/training throughput matrices, full-suite testing and
release/PR preparation remain separate P10 items, not completed by this experiment.

## Initial CPU falling-probe evidence

Both variants compile with the same nine sphere probes:

| Variant | Scene geoms (excluding probes) | Physics steps/s | Mean contacts | Mean constraints | Mean solver iterations |
|---|---:|---:|---:|---:|---:|
| reference | 39 | 32,156.54 | 8.4735 | 33.894 | 1.728 |
| automatic plus common pillars | 79 | 28,735.70 | 8.5050 | 34.020 | 1.898 |

This is one short CPU run, not a statistically established speed difference or
geometry-quality result. Raw data: `artifacts/p10-apartment-comparison/cpu-physics.json`.
The source hashes are reference
`5f873d4fdab71c8e4e377eeb5258d77549f985315e1aa54c6754ae13094441a1`
and original automatic
`f3440ac33feee310025ff5ec22c81e13973fd1ddf1c94205c48257bc3c383dda`.
They identify the unmodified source inputs, not the normalized experimental copies.

## Initial CPU policy evidence

| Variant | 500-step horizon completion | Mean control SPS | MPKPE (m) | R-MPKPE (m) | Mean contacts | Overflow runs |
|---|---:|---:|---:|---:|---:|---:|
| reference | 3/3 | 22.66855 | 0.117990 | 0.034173 | 14.32867 | 0/3 |
| automatic plus common pillars | 3/3 | 19.62839 | 0.132351 | 0.051921 | 14.48600 | 0/3 |

All six runs reach `time_out` at 500 steps without task termination. This is a
short pretrained tracking smoke experiment, not evidence of safe-velocity behavior
or a complete-room geometry guarantee. No CPU semantic or automatic-proxy benefit
is asserted. Host load is not controlled; this is initial benchmark evidence, not
a statistically established performance ranking. Actual CUDA/native measurements
are recorded below rather than inferred from CPU timings.

Hashes in `artifacts/p10-apartment-comparison/`:

- `preparation.json`: `bd14a53503ab8df54ca2bfd9156afc8cfb4773498c5b0736afcd11a615eafade`.
- `variants.json`: `f082b2ec3d776517fba87dbe5390c3225bc99d122c1921975fa2b13b2cfd88ff`.
- `cpu-physics.json`: `bd10105f9a8f37c087a1569a4267ad38e8203501945462f4ed48dc3b8918f828`.
- `cpu-policy.json`: `2718258358f0e1281d1f8aa273408d6beae15be2a205da2c92103cdd9124a522`.

Actual prepared common pillars were checked for equal compiled geometry, contact
and RGBA/group attributes. Nine focused preparation/policy-protocol tests pass,
including inherited defaults, immutable inputs, reproducibility and generation
hash rejection. `make check`, `make docs`, shell syntax and `git diff --check`
pass. The only new implementation is an additive tool and its tests; no shared
simulation interface changed. Play/demo and their config tests have zero diff.

## Desktop CUDA and native evidence

The user completed `run-desktop.sh`. Both logs end with the expected report-written
message. The audit verifies actual checkpoint/motion file hashes against all three
CPU/CUDA/native reports, collision-bundle hashes against both prepared XMLs, exactly
the two expected variant names, and seeds 42/43/44 with 500-step maximum horizons.
Native visual PLY identity matches `preparation.json` and the actual source file.
The preparation, variant manifest and CPU report hashes above remain unchanged.

| Variant | CUDA 500-step completion | Control SPS (mean +/- std) | MPKPE (m, mean +/- std) | R-MPKPE (m) | Mean contacts | Mean constraints |
|---|---:|---:|---:|---:|---:|---:|
| reference | 3/3 | 80.50956 +/- 1.03119 | 0.097571 +/- 0.009687 | 0.033713 | 14.29267 | 57.17067 |
| automatic plus common pillars | 3/3 | 80.58901 +/- 0.70715 | 0.158031 +/- 0.072008 | 0.053888 | 11.85267 | 47.44400 |

All six policy runs reach `time_out` at step 500 without task termination. Both
seed-42 runs report `LS_ITERATIONS`, bit 1024: 168/500 steps for reference and
231/500 for automatic. Seeds 43/44 have zero flags. These are solver line-search
iteration-limit warnings, not contact-buffer overflow; they are neither suppressed
nor described as a zero-warning result. Mean solver iterations are 2.46267 and
2.32600 respectively. Fewer contacts/constraints do not demonstrate better geometry.

| Variant | Actual realtime | Submissions/s | Measured submissions | Renderer | Overflow |
|---|---:|---:|---:|---|---:|
| reference | 0.769702 | 38.48508 | 500 | ready | 0 |
| automatic plus common pillars | 0.760479 | 38.02394 | 500 | ready | 0 |

Native measurement has termination disabled and is throughput evidence, not task
success or display refresh FPS. CUDA control throughput is effectively comparable
in this limited experiment; native automatic throughput is slightly lower. The
automatic proxy has higher average tracking error and large seed variability.
No statistical significance, semantic improvement, geometric superiority or safe-
velocity generalization is claimed. The existing reference asset is not replaced.

Additional raw report hashes:

- `cuda-policy.json`: `5fcff297141232bb36818cb803b04036c921e14755f9854bc71e8253fdafb6ce`.
- `native-3dgs.json`: `88cee3408699039a7bc375d94a48465f60a0c2a2f92f7087bba21c3e181538a2`.

Only requirements/status evidence changed in this audit. No model, asset, script,
simulation runtime, play/demo or driver changes were made. P8 remains closed per
the user's decision. Remaining P10 work includes versioned atomic PLY caches,
broader sensor/training throughput matrices and full-suite/release validation.
