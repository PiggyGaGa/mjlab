.. _gaussian_splats:

3D Gaussian Splatting
=====================

.. note::

   On Linux x86_64 with Python 3.13, the repository's uv configuration
   downloads the custom MuJoCo 3DGS wheel from
   `the fork's v1.0 Release <https://github.com/PiggyGaGa/mujoco/releases/tag/v1.0>`_.
   The wheel requires glibc 2.35 or newer. Run
   ``uv run tests/run_g1_3dgs_native_demo.py --interactive`` for native playback;
   neither a neighboring MuJoCo checkout nor ``--no-sync`` is required.
   Other platforms and Python versions use official MuJoCo without the custom
   native 3DGS viewer.

mjlab can add static 3D Gaussian Splatting (3DGS) assets to interactive Viser
playback and RGB or depth camera observations. Viser uses its native splat
renderer for interactive playback. Camera observations and video recording use
MuJoCo Warp's native renderer, where splats and MuJoCo geometry share the same
camera and depth test.

3DGS assets are visual only. They do not participate in physics, contact
detection, or ``RayCastSensor`` queries. A training environment that needs
physical interaction must provide separate MuJoCo collision geometry aligned
with the splat asset.

Native viewer acceptance
------------------------

Run ``uv run tests/run_g1_3dgs_acceptance.py --checkpoint /path/to/model.pt``
to exercise native controls and repeated window cleanup, compare CPU contacts
with collision proxies hidden/shown, and record a JSON runtime report.
The default output is ``/tmp/p3-acceptance.json``. The August 2026 G1 velocity
checkpoint uses a different actor observation layout and requires the explicit
``--legacy-phase-observations`` adapter; this does not change production tasks.
Add ``--interactive --viewer native`` or ``--interactive --viewer viser``
for manual camera and occlusion comparisons with that adapter.

The automated camera check changes parameters through the viewer API; physical
keyboard and mouse operation still needs manual verification. CPU contact replay
does not establish full MJWarp policy trajectory equivalence. Reported submission
frequency is not display FPS, and GPU memory measurements include other processes.

G1 depth training smoke check
-----------------------------

The task ``Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth`` reuses the repository's
YAM Vision CNN path. Run:

.. code-block:: bash

    uv run tests/run_g1_3dgs_vision_smoke.py --num-envs 2 \
        --scene-manifest src/mjlab/asset_zoo/3dgs/room1/manifest.json \
        --output /tmp/p6-vision-smoke

The script saves normalized depth PNG/NPY samples, performs two short PPO updates,
checks checkpoint roundtrip and a resumed update, and writes ``report.json``.
It checks camera tensor updates and same-asset partial reset isolation; it does
not establish depth accuracy, training convergence, or a benefit from vision.

The learning algorithm is PPO (Schulman et al., 2017,
`paper <https://arxiv.org/abs/1707.06347>`_) with GAE (Schulman et al., 2015,
`paper <https://arxiv.org/abs/1506.02438>`_). The spatial-softmax/expected-position
encoder follows the architecture idea of Levine et al., JMLR 2016,
`End-to-End Training of Deep Visuomotor Policies <https://jmlr.org/papers/v17/15-522.html>`_.
The optimizer here is PPO, not that paper's guided policy search. Static Gaussian
assets follow Kerbl et al., 2023,
`3D Gaussian Splatting <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>`_;
their parameters are not optimized by the policy training loop.

For equations, exact architecture, attribution boundaries, and BibTeX, see the
repository files ``requirements/mjlab-3dgs-p6-method.md`` and
``requirements/mjlab-3dgs-p6-references.bib``. Current experiments are pipeline
checks; visual-task gains require task-level baselines and camera ablations.

The separate task ``Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity`` uses
``(B, 2, 48, 64)`` camera observations, ordered as normalized depth followed by
``isfinite(raw_depth) & (raw_depth > 0)``. The original ``3DGS-Depth`` task remains
the single-channel baseline. Both use the same cached CameraSensor output;
validity does not trigger additional rendering. Positive depth beyond the
cutoff remains valid, distinguishing missing ``(1, 0)`` from distant ``(1, 1)``.
It denotes renderer availability, not confidence, geometry accuracy, or free space.

Add ``--validity-mask`` to the smoke script to select that task and save validity
PNGs alongside depth images. Experiment names differ, checkpoint infos record
the encoding and channel count, and the first convolution's input shape prevents
strict loading across the two encodings. No automatic weight conversion is
provided. A mask-only output sensitivity probe checks connectivity; it does not
establish successful use of validity during task learning.

The independent task
``Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity-History`` keeps four
chronological depth+validity frames as ``(B, 4, 2, 48, 64)``. It reuses the
observation manager's standard history and partial-reset backfill behavior. At
the CNN boundary, time and channel dimensions are stacked into eight input
channels; the spatial-softmax encoder, PPO settings, rewards, and action space
otherwise match the single-frame baseline. This task does not yet add frame
stride, simulated sensor delay, or noise.

For deployment-corruption experiments, use
``Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity-History-Robust``. It stores
seven consecutive observations and selects four at stride two, adds a fixed
one-control-step delay, and holds camera samples for two control steps. During
training, depth+validity are corrupted together with depth range noise, blur,
structured missing regions, and occasional full-frame dropout; missing pixels
remain encoded as depth 1 and validity 0. Play disables random corruption while
retaining sample timing and latency. Append ``-Asymmetric`` to use the actor-only
camera baseline with an MLP critic.

The smoke script accepts ``--temporal-history``, ``--robust-history``, or
``--asymmetric-critic``. These modes also report actor inference throughput,
rollout environment-steps per second, PPO update time, and peak Torch allocation.
The latter excludes renderer/Warp allocations and is not total GPU memory.

For fixed-pose depth diagnostics, run
``uv run tests/run_g1_depth_diagnostics.py --output /tmp/p6-depth-diagnostics``.
It separates robot-only, Gaussian-only, and combined rendering, saves raw NPY
and depth PNG files, checks fixed-pose repeatability, and translates the camera
by 0.5 m to check image response. The splat-only case uses an offscreen,
non-colliding placeholder to avoid the installed backend's empty-BVH crash.
Add ``--pose-sweep`` to compare seven controlled root/arm poses, including
single-arm occlusion. This checks configurations, not a physical rollout or
all possible robot poses.

Add ``--depth-metrics`` to the vision smoke script to save per-world, per-step
raw depth fractions in ``rollout_depth.json`` and summary statistics in the main
report. Missing zeros, positive near depth, positive depth beyond cutoff, and
in-range depth are counted separately. These include post-reset images; they do
not attribute pixels to robot occlusion. CPU diagnostic sampling is included in
the reported runtime and should be disabled for performance benchmarks.

The installed renderer outputs axial camera Z-depth in meters. Gaussian depth
is assigned at the splat where accumulated transmittance falls below
``splat_depth_transmittance_threshold``; it is not expected depth or the nearest
Gaussian center. The locked MuJoCo Warp fork exposes this setting through
``create_render_context`` and defaults it to ``0.1``, corresponding to 90% accumulated
opacity. The value must be strictly between zero and one. A lower value requires
more accumulated opacity before recording depth and can therefore produce more
zero-depth holes; a higher value records an earlier surface along the ray.

If no mesh is hit and the threshold is not reached, the result is zero. G1 enables
``camera_depth(..., zero_depth_as_far=True)`` to map those missing values to the far
cutoff before normalization. The default for other tasks is unchanged. Missing
depth and valid depth beyond the cutoff are then indistinguishable; the encoding
does not prove free space or eliminate holes in the reconstruction. Changing the
threshold changes the policy observation distribution, so existing visual
checkpoints must be revalidated before results are compared.

Supported PLY format
--------------------

The initial importer accepts ASCII and binary little-endian Graphdeco-style
PLY files. The vertex element must be the first data element and contain scalar
properties with these names:

- ``x``, ``y``, ``z``;
- ``scale_0``, ``scale_1``, ``scale_2``;
- ``rot_0``, ``rot_1``, ``rot_2``, ``rot_3``;
- ``opacity``;
- ``f_dc_0``, ``f_dc_1``, ``f_dc_2`` for spherical-harmonic DC color, or
  ``red``, ``green``, ``blue`` when ``color_encoding="rgb"``.

The defaults follow the original 3DGS representation: log scale, logit
opacity, spherical-harmonic DC color, and ``(w, x, y, z)`` rotations. Higher
order spherical harmonics are not evaluated; rendering uses the view-independent
color reconstructed from the DC terms.


Scene configuration
-------------------

Configure splats on ``SceneCfg``. Add a ``CameraSensorCfg`` only when MuJoCo Warp
camera observations or recording are required:

.. code-block:: python

    from mjlab.scene import (
        GaussianSplatAssetCfg,
        GaussianSplatSceneCfg,
        SceneCfg,
    )
    from mjlab.sensor import CameraSensorCfg

    splats = GaussianSplatSceneCfg(
        assets=(
            GaussianSplatAssetCfg(
                path="assets/room/point_cloud.ply",
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
                scale=1.0,
                opacity_threshold=0.01,
            ),
        ),
    )

    camera = CameraSensorCfg(
        name="training_camera",
        pos=(0.0, -3.0, 1.0),
        quat=(0.707, 0.707, 0.0, 0.0),
        width=320,
        height=240,
        data_types=("rgb", "depth"),
    )

    scene = SceneCfg(
        entities={"robot": robot_cfg},
        sensors=(camera,),
        gaussian_splats=splats,
    )

``pos``, ``quat``, and ``scale`` define one asset-to-world transform. Uniform
scale affects both Gaussian centers and standard deviations. Non-uniform scale
is intentionally unsupported because it requires transforming each Gaussian's
full covariance.

``opacity_threshold`` removes low-opacity splats. ``bounds_min`` and
``bounds_max`` crop in transformed world coordinates. ``max_splats`` keeps the
highest-opacity splats, which is useful for memory and performance experiments.


Multiple assets and worlds
--------------------------

MuJoCo Warp stores each asset as a splat group. Assign one group to each world
with ``world_asset_ids``:

.. code-block:: python

    splats = GaussianSplatSceneCfg(
        assets=(room_a, room_b),
        world_asset_ids=(0, 0, 1, 1),
    )

The mapping length must equal ``SceneCfg.num_envs``. When only one asset is
configured, omit ``world_asset_ids`` and every world uses that asset.


Loading and training cost
-------------------------

Assets configured on ``SceneCfg`` are parsed during scene initialization so a
frontend such as Viser can consume them without creating a camera sensor. A
``CameraSensorCfg`` additionally creates the MuJoCo Warp render context and renders
on every sensing update.

The ordinary G1 and G1 3DGS playback tasks do not create a training camera, so
proprioceptive training does not parse a PLY or pay a rendering cost. Normal Viser
playback loads the PLY once but does not create the MuJoCo Warp camera. The camera
is enabled on demand only for explicit sensor-based video recording.

``Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth`` is the explicit visual-training
variant. It loads the Gaussian scene during training and renders a ``64 x 48``
forward/downward depth image every control step. Keep the non-visual task ID when
camera observations are not needed.

The current importer reads the source PLY once during environment initialization;
simulation and rendering then use the in-memory/GPU representation. Very large
assets should be filtered with bounds, opacity, or ``max_splats``. Optional
versioned preprocessing caches reduce repeated CPU preprocessing; they do not
cache the GPU render context or BVH and do not improve per-frame rendering by
themselves.

Optional preprocessing cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``GaussianSplatAssetCfg.cache_dir`` explicitly to enable caching:

.. code-block:: python

    asset = GaussianSplatAssetCfg(
        path="/path/to/splats.ply",
        cache_dir="artifacts/splat-cache",
    )

The default ``None`` follows the original loader without source hashing or cache
I/O. No play/demo or manifest-schema changes are needed; configure the asset in
the scene/task configuration. Manifest loading still leaves caching off unless
you explicitly replace/configure the resulting asset configuration.

Each single-asset cache is an uncompressed ``v<schema>-<key>.splats.npz`` file
containing normalized positions, quaternion rotations, scales, RGBA and single-
asset group arrays. Its key includes the resolved source path, full source SHA-256,
schema/normalization version and all import/filter/transform parameters. World
assignment is rebuilt separately and is not frozen in the asset cache. Source
content changes invalidate the key even when file size and mtime are unchanged.
The source PLY must still exist and is hashed on each cached load.

Cache reads disable pickle and verify names, shapes, dtypes, finite values, scale/
color/quaternion validity, group offsets and array checksums. Invalid entries warn
and rebuild from PLY. Writes use a unique temporary file in the destination
directory, flush/fsync, then atomic replacement. Write failures warn and return the
normal uncached result. Concurrent writers may duplicate preprocessing but never
publish partial entries. A source that changes during preprocessing is retried once
and then rejected rather than cached inconsistently. Obsolete content/config keys
are not deleted automatically.

Measure uncached, first-build and verified cache-hit CPU loading independently:

.. code-block:: bash

    uv run scripts/tools/benchmark_gaussian_splat_loading.py \
        /path/to/splats.ply artifacts/splat-load-report.json \
        --cache-dir artifacts/splat-cache --repeats 3

This tool retains a fresh cache subdirectory and checks exact array equality across
all phases. Timings exclude Python startup and do not measure GPU allocation, BVH,
Native Viewer FPS or training throughput. Cache creation may be slower than a
normal first load; OS file caching affects repeated measurements.


Repeated camera and training measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a prepared CUDA environment, benchmark a manifest without editing the generic
trainer or play/demo scripts:

.. code-block:: bash

    uv run --no-sync scripts/benchmarks/measure_3dgs_throughput.py \
        --scene-manifest /path/to/manifest.json \
        --device cuda:0 --num-envs 1 --width 64 --height 48 \
        --steps 32 --warmup-steps 16 --repeats 3 \
        --save-capture --output /tmp/gaussian-throughput.json

Use a fresh output path for each run. ``--no-sync`` preserves the installed build
being measured; prepare dependencies first. The benchmark reuses the G1 depth
task and its PPO settings, captures RGB and depth, discards the first complete
repetition and retains raw samples plus medians. Physics, full environment steps,
stationary sensing, fixed-observation actor inference and PPO rollout/update are
independent workloads, not additive components or pure renderer-kernel timing.
Snapshot and file I/O run outside measured loops. Torch peak allocation excludes
some renderer allocations and must not be labeled total device VRAM.

Reports include installed MuJoCo/MuJoCo-Warp runtime-file hashes, package source
metadata, mapped MuJoCo library hashes, source asset/collision hashes and benchmark
script hashes. Installed-file identity does not establish which neighboring source
checkout produced the binary, or identify the GPU driver/JIT cache. Same package
version does not guarantee the same build. Short repeats measure pipeline cost,
not policy convergence or visual task benefit.

``--without-gaussians`` retains the manifest collision and RGB/depth camera while
disabling only Gaussian assets. ``--max-splats N`` instead applies the loader's
existing per-asset opacity-ranked cap; these options are mutually exclusive.
Opacity ranking can leave spatial holes and does not guarantee uniform coverage.
The default remains uncapped. Compare images as well as timing before choosing a
budget; historical timing results do not describe every future renderer build.

Static capture and archive comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``--capture-only`` to save one fixed reset view per world without constructing
a PPO runner or performing updates. ``--spawn-point`` selects an existing manifest
spawn; ``--view-yaw-degrees`` is a finite fixed offset relative to spawn orientation:

.. code-block:: bash

    uv run --no-sync scripts/benchmarks/measure_3dgs_throughput.py \
        --scene-manifest /path/to/manifest.json --device cuda:0 \
        --capture-only --spawn-point default --view-yaw-degrees 90 \
        --output /tmp/gaussian-view.json

``--capture-only`` implies capture output; timed runs save it only when
``--save-capture`` is selected. The sibling ``<report-stem>.capture`` directory
contains raw ``reset.npz`` RGB/depth/qpos arrays and per-world RGB/depth PNGs.
Depth previews use a fixed 5 m cutoff; invalid pixels are represented at the far
cutoff in these previews, so inspect raw depth/validity statistics for missing data.
Capture completion does not certify safe placement or visual quality.

Compare two captures using passed reports with matching camera, input assets,
collision, Gaussian count and actual qpos:

.. code-block:: bash

    uv run --no-sync scripts/benchmarks/compare_3dgs_captures.py \
        --reference /path/to/previous-view.json \
        --candidate /tmp/gaussian-view.json \
        --output /tmp/gaussian-view-comparison.json

The comparator disables pickle, checks matching inputs/shapes and finite identical
qpos, and reports exact RGB/depth equality, raw errors and lost valid-depth pixels.
Nonmatching inputs are recorded as ``not_comparable`` and the command fails.
Different images are recorded as ``compared`` with errors, without inventing a
quality threshold. Common-valid depth errors omit holes; empty valid intersections
produce null error metrics. Full rendering is an archive reference, not independent
geometry truth; a few static views cannot certify whole-scene or training quality.

Integration test assets
~~~~~~~~~~~~~~~~~~~~~~~

Some slow tests instantiate the visual task with a real Graphdeco-compatible PLY.
Large reconstructed assets are not distributed with a clean checkout. Configure
the existing task and real-asset test overrides explicitly before the full suite:

.. code-block:: bash

    MJLAB_G1_3DGS_ASSET=/path/to/splats.ply \
    MJLAB_3DGS_TEST_ASSET=/path/to/splats.ply make test

The first variable selects the default G1 visual-task asset; the second selects
the real-asset import/render fixture. This tests the pipeline, not alignment or
collision quality for an arbitrary replacement scene. To run the full CPU suite,
also set ``FORCE_CPU=1 CUDA_VISIBLE_DEVICES=-1``; CUDA-gated skips remain separate
from GPU acceptance. Offline W&B tests need local IPC even with ``WANDB_MODE=offline``.
Default pytest discovery is restricted to ``tests/`` so downloaded experiments in
``artifacts/`` are not mistaken for repository tests.


External collision scene
------------------------

The visual PLY and collision MJCF are separate representations of one logical
scene. They must already use the same right-handed, Z-up coordinate frame, meter
units, and origin. The collision entry point can contain static primitive geoms,
multiple convex mesh assets, and heightfields. It must not contain joints, mocap
bodies, actuators, or sensors.

For a reproducible center-only baseline, voxelize a Graphdeco PLY and merge the
occupied cells into axis-aligned boxes:

.. code-block:: bash

    uv run scripts/tools/generate_3dgs_collision.py \
        /path/to/splats.ply /path/to/collision.xml \
        --report /path/to/collision_report.json \
        --floor-z 0 \
        --primitive-mode auto

The report records the input and output hashes, all filtering and merging
parameters, bounds, and generation statistics. A deterministic ``.source_ids.npz``
sidecar uses CSR offsets to map every retained primitive to zero-based source PLY
vertex IDs. ``--floor-z`` is opt-in because a Gaussian cloud alone does not prove
that its lowest horizontal samples form a safe support plane.

``--primitive-mode auto`` classifies each merged voxel candidate from its covariance
eigenvalues, then verifies the candidate's normalized surface-fit error. Approximately
isotropic regions may become spheres, elongated regions with a near-circular cross
section may become cylinders or capsules, and locally planar regions may become
oriented boxes. A planar box is rejected when projected support is too sparse or a
gap exceeds ``--max-surface-expansion``; every rejected fit falls back to the bounded
axis-aligned box baseline. Sphere, cylinder, and capsule fits must reduce normalized
surface error by 30% relative to the candidate's oriented-box fit. Configure this
gate with ``--min-round-error-improvement``. The report records both errors, their
relative improvement, every primitive's type, pose, size, source candidate, and
source-Gaussian count.

For Native Viewer inspection, add ``--debug-color-by-type``. It writes translucent
green boxes, magenta spheres, orange cylinders, blue capsules, and a translucent gray
ground slab. The option changes only per-geom ``rgba``; collision type, pose, size,
contact parameters, source mapping, and filtering remain unchanged. It requires
``--primitive-mode auto`` so that a color always identifies a fitted primitive type.

The default ``--primitive-mode boxes`` retains the original output and provenance
format. Use it as the geometry-only control when evaluating ``auto``; changing mode
is explicit and does not alter existing manifests or collision assets.

Bent surfaces such as stair profiles can satisfy a coarse fill test and otherwise
merge into a single ramp-like box. Add ``--max-planar-merge-residual 0.04`` to reject
an approximately planar merged candidate when any occupied voxel center is more than
4 cm from its best-fit plane. ``--planar-merge-eigenvalue-ratio`` controls which
candidates are considered planar; non-planar candidates continue through the normal
merge and primitive-fit path. The residual is an absolute distance in metres, so tune
it with the voxel size and expected feature scale. The guard is disabled by default
to preserve existing byte-stable output. It preserves synthetic doorway free space
and separates a three-step surface into the expected support heights, but does not
infer free-side thickness or merge coplanar candidates across disconnected regions.

If a trusted source provides negative occupancy evidence, store finite world-space
XYZ samples as an ``(N, 3)`` ``.npy`` array and add:

.. code-block:: bash

    --known-free-points /path/to/free_points.npy

The generator rejects occupied/free voxel conflicts, prevents merged boxes from
covering an in-bounds known-free sample, and prefers cuts that isolate constrained
voids with fewer boxes. In ``auto`` mode it also rejects an oriented box, sphere,
cylinder, or capsule fit that contains a known-free point and uses the safe bounded
box fallback. The report hashes the sidecar and records its point and in-bounds voxel
counts. Samples outside the generated quantile grid do not constrain the result.

This input must come from camera-ray free-space carving, synthetic ground truth,
manual annotation, or another explicit source. Gaussian centers provide positive
surface evidence only; the tool deliberately does not label every unoccupied or
unobserved voxel as free. The constraint therefore preserves demonstrated table
undersides, pits, and gaps between levels, but it does not yet infer a wall's solid
side or anchor slab thickness to that side.

With ``--primitive-mode auto``, add ``--planar-slab-thickness 0.1`` to request a
10 cm collision slab. Nearby known-free samples whose planar projections lie inside
the candidate footprint vote for the free side. The plane is used as that slab face
and thickness extends in the opposite direction. Configure the evidence window and
required sign agreement with ``--planar-free-side-max-distance`` and
``--planar-free-side-min-agreement``. If no eligible samples exist or the votes are
ambiguous, the tool keeps the existing symmetric oriented box rather than guessing.
Primitive diagnostics record whether anchoring occurred and the selected local-normal
sign. This separates collision thickness from contact-surface position, but depends
on explicit negative evidence and is not automatic wall-interior inference.

For the covariance-aware comparison, add ``--use-gaussian-covariance``. The tool
decodes ``scale_0..2`` and ``rot_0..3``, normalizes each quaternion, constructs
``R diag(scale^2) R^T``, and adds the mean splat covariance to the center covariance
used for candidate PCA. This changes shape classification axes, not the source
centers or occupancy grid. Use ``--max-gaussian-scale 0.2`` to reject a splat when
any activated axis exceeds 20 cm before voxelization. ``--scale-encoding`` selects
Graphdeco logarithmic or linear scales. Invalid or degenerate shape fields fail with
context. These options are disabled by default and are an experimental baseline;
larger covariance support does not by itself prove a solid collision boundary.

A replaceable semantic frontend can write one integer label per original PLY vertex
as an ``(N,)`` ``.npy`` array and pass ``--semantic-labels``. Each occupied voxel
uses deterministic majority voting with the smaller label winning exact ties. Boxes
for one label treat other-label occupied voxels as forbidden boundaries, so merging
cannot erase a labeled part boundary. ``--unknown-semantic-label`` identifies the
unknown value for reporting; unknown samples remain a geometry-only group and are
never silently discarded. Auto fits that expand into another label's source points
fall back to their bounded candidate box. The source-map NPZ records the label of
every output primitive, and the JSON report contains per-label output counts and a
hash of the label sidecar. This is an input contract, not a bundled segmentation
model; SceneSplat, multiview fusion, or manual labels must be evaluated separately.

When aligned language features and class embeddings are available, decode them with:

.. code-block:: bash

    uv run scripts/tools/decode_3dgs_semantic_features.py \
        /path/to/feat.pt /path/to/class_embeddings.pt /path/to/labels.npy \
        --report /path/to/semantic_decode.json \
        --minimum-similarity 0.2 --minimum-margin 0.05

The decoder accepts NumPy arrays or tensor-only PyTorch files, normalizes rows, and
uses blocked cosine nearest-class matching. Rows below either the best-score or
top-two-margin threshold remain ``unknown``. A confidence NPZ and hashed report keep
the label decision auditable and in original row order. The class embeddings must
come from the same language feature space as the scene features. This utility does
not download or run SceneSplat, encode text prompts, or establish label quality.

For a fixed output budget, add ``--max-primitives N`` in auto mode. The default
ranking retains candidates with more mapped source Gaussians. A JSON
``--semantic-priorities`` mapping label IDs to positive weights and an optional
non-negative ``--point-importance`` value per original PLY vertex can make the
ranking task-aware. Priority inputs never create geometry or labels; they only
select among fitted candidates, and the source-ID CSR is rebuilt for the retained
subset. The report records the pre-budget count, budget, priority files, and hashes.

Use ``compare_3dgs_collision_semantics.py`` to enforce a matched-budget experiment:

.. code-block:: bash

    uv run scripts/tools/compare_3dgs_collision_semantics.py \
        /path/to/splats.ply /path/to/labels.npy /path/to/priorities.json \
        /tmp/p8-semantic-ablation --max-primitives 32 \
        --evaluation-points /path/to/evaluation.npz

It produces geometry-only and semantic-priority MJCFs, requires both variants to
reach exactly the requested budget, and benchmarks them with identical falling
probes. Its report combines generation time, surface/free-space error, physics SPS,
contacts, constraints, and solver iterations. Visual throughput remains explicitly
unset because this rendering-independent CPU measurement cannot establish it.

Run a task-level collision replacement comparison from a versioned variant manifest:

.. code-block:: bash

    uv run scripts/tools/benchmark_g1_collision_policy.py \
        requirements/validation/p8-g1-policy-variants-2026-11-21.json \
        --checkpoint /tmp/mjlab_cache/demo_ckpt.pt \
        --motion /tmp/mjlab_cache/lafan1_dance1_subject1_demo_motion.npz \
        --output /tmp/p8-g1-collision-policy.json \
        --device cpu --seeds 42 43 44

The local evaluator uses one fixed motion start, removes push events, preserves
seeded startup randomization and actor corruption, and disables auto-reset so the
terminal state remains measurable. Geometry evidence supplied by the manifest is
stored beside MPKPE, root-relative MPKPE, end-effector and joint-velocity errors,
termination terms, contacts, constraints, solver iterations, overflow bit flags,
and policy-plus-environment throughput. It excludes rendering.

On a CUDA-visible desktop with the custom MuJoCo wheel, add native 3DGS submission
cost without changing the production play/demo scripts:

.. code-block:: bash

    uv run --no-sync scripts/tools/benchmark_native_3dgs_collision_policy.py \
        requirements/validation/p8-g1-policy-variants-2026-11-21.json \
        --gaussian-splat-file \
        src/mjlab/asset_zoo/3dgs/ArtanimMotionCaptureStudio/scene_transform_20p.ply \
        --checkpoint /tmp/mjlab_cache/demo_ckpt.pt \
        --motion /tmp/mjlab_cache/lafan1_dance1_subject1_demo_motion.npz \
        --output /tmp/p8-g1-native-3dgs.json --device cuda:0

This opens one native window per variant. ``actual_realtime`` includes policy
inference, MuJoCo Warp, CPU viewer synchronization, and native 3DGS submission;
``viewer_submissions_per_second`` is not a monitor refresh or rendered-frame FPS.
The finite throughput pass disables task terminations to avoid reset cost becoming
an uncontrolled representation-dependent part of the timing; use the local joint
evaluator above for success and termination comparisons.

For an evaluation set that was not used as a generation constraint, create an
``.npz`` containing ``surface_points`` and/or ``free_points`` arrays in world-space
XYZ and pass ``--evaluation-points``. The report records mean, 95th-percentile, and
maximum nearest-proxy surface distance plus the number and fraction of free probes
strictly inside any output primitive. Unknown array names and malformed coordinates
are rejected. Keep this archive independent from ``--known-free-points``; evaluating
the same samples that forced a split only measures constraint satisfaction. Reports
also state grid shape, retained connected components, discarded boxes, source
mapping coverage, primitive-per-voxel ratio, representation type, and zero mesh
vertex/face counts for the primitive-only path.

Add ``--evaluate-source-surface`` for a diagnostic distance from every retained
high-confidence Gaussian centre to the nearest generated primitive surface. This
uses generation input rather than independent ground truth, so report it as a
coverage diagnostic rather than an accuracy result.

Measure primitive-only CPU collision cost independently from rendering with the
same falling-sphere probes:

.. code-block:: bash

    uv run scripts/tools/benchmark_collision_mjcf.py \
        /path/to/boxes.xml /path/to/auto.xml \
        --output /tmp/collision_benchmark.json \
        --warmup-steps 1000 --measured-steps 10000 --probe-grid 3

The JSON records source hashes, compile and measured time, physics steps per second,
model sizes, contacts, active constraints, and solver iterations. It accepts
self-contained primitive and inline-heightfield MJCF files. File assets are rejected
by default; ``--allow-file-assets`` permits only mesh files that resolve inside the
MJCF bundle and records every dependency hash. Includes and external paths remain
rejected. Run several repetitions on an otherwise idle machine before drawing
performance conclusions; this tool deliberately does not include 3DGS rendering or
policy inference.

Compare reconstruction representations offline with:

.. code-block:: bash

    uv run scripts/tools/compare_3dgs_collision_representations.py \
        /path/to/splats.ply /tmp/p8-representations \
        --bounds -2 -2 -0.5 2 2 3 --voxel-size 0.1

The command exports exact exposed occupancy faces, a TSDF zero-isosurface extracted
with marching tetrahedra, a coarsened manifold TSDF mesh, a lowest-run heightfield,
and a heightfield-plus-residual-primitive MJCF. The report includes topology repair,
normal consistency, boundary/non-manifold edges, watertightness, surface distance,
generation time, and MuJoCo compile counts. A heightfield is single-valued and reports
columns containing multiple occupied runs; it cannot preserve ceilings or overhangs.
The OBJ meshes are reconstruction baselines, not ready-to-load concave room collision
geoms: MuJoCo uses the convex hull of a user mesh for collision, so a room requires
convex decomposition or multiple locally convex pieces.

Convert an external collision GLB/OBJ into an explicit MuJoCo convex bundle with:

.. code-block:: bash

    uv run scripts/tools/convert_collision_mesh_to_mjcf.py \
        /path/to/collision.glb /tmp/mesh-collision \
        --decomposition vhacd --max-convex-hulls 64 \
        --free-points /path/to/independent_free_points.npy

``component-hulls`` creates one convex geom per connected component and is useful as
a coarse diagnostic. ``vhacd`` uses the installed trimesh V-HACD backend and fixes
asynchronous decomposition off for byte-stable output. The report records hull budget,
volume inflation, source-vertex distance, output hashes, and false occupancy of the
optional free points. Neither method proves free-space correctness: inspect the probes
and collision overlay before using the bundle for a task. The resulting MJCF can be
passed to the benchmark with ``--allow-file-assets``.

This baseline uses centers and opacity only. It does not apply a scene transform,
infer units or axes, recover semantic parts, or prove that filled scan gaps are
occupied. Primitive classification is a geometric heuristic, not object recognition.
Review the collision overlay and contact behavior; use the report to compare parameter
sweeps rather than treating the generated MJCF as ground truth. The command rejects
voxel grids above 50 million cells by default; increase ``--voxel-size`` or tighten
the quantile bounds before raising that limit.

Use this asset layout so that the MJCF can resolve every dependency relative to
its own directory:

.. code-block:: text

    scene_name/
      visual/
        splats.ply
      collision/
        collision.xml
        meshes/
          ground_000.obj
          stair_000.obj
        heightfields/
          terrain.png

For example, ``collision.xml`` can define both a mesh and a primitive:

.. code-block:: xml

    <mujoco model="scene_collision">
      <compiler angle="radian" meshdir="meshes"/>
      <default>
        <geom group="3" contype="1" conaffinity="1" condim="3"
              friction="1.0 0.005 0.0001" rgba="0.2 0.8 0.2 0.25"/>
      </default>
      <asset>
        <mesh name="stair_000" file="stair_000.obj"/>
      </asset>
      <worldbody>
        <geom name="stair_000" type="mesh" mesh="stair_000"/>
        <geom name="ground" type="box" pos="0 0 -0.05" size="5 5 0.05"/>
      </worldbody>
    </mujoco>

Absolute paths, ``..`` path traversal, and MJCF includes are rejected. A mesh
geom follows MuJoCo's convex collision semantics, so a concave environment must
be exported as multiple convex mesh assets instead of one arbitrary scene mesh.
Contact parameters such as friction, ``condim``, ``priority``, ``contype``, and
``conaffinity`` remain authored in ``collision.xml``.

MuJoCo allows collision geoms directly under ``<worldbody>``. mjlab accepts this
form and automatically adds an internal fixed root body required by terrain
indexing; users do not need to add an artificial dynamic or mocap body.

Pass the external physics scene to training through the terrain configuration:

.. code-block:: bash

    uv run train Mjlab-Velocity-Flat-Unitree-G1-3DGS \
        --env.scene.terrain.terrain-type mjcf \
        --env.scene.terrain.mjcf-path /path/to/scene/collision/collision.xml \
        --env.scene.num-envs 4096

This replaces the default plane; all parallel worlds share the same static
collision model. Add ``--env.scene.terrain.mjcf-keep-plane True`` only for an
explicit fallback experiment. Use the ``-3DGS-Depth`` task ID when the policy
should consume rendered depth; the ordinary ``-3DGS`` task remains a
playback-oriented proprioceptive task.

The imported scene uses the conventional fixed ``terrain`` root body, so existing
contact sensors continue to recognize it. ``TerrainHeightSensorCfg`` instances are
automatically redirected to ``mjcf_geom_group``; this keeps foot-height observations
on the external collision surface instead of the removed plane.

Fixed-asset collision comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fixed-asset collision comparisons can use
``scripts/tools/prepare_collision_comparison.py`` to freeze reference and automatic
primitive MJCF copies with identical explicitly named common obstacles. This avoids
confounding a collision replacement with removing artificially added obstacles.
The original assets and manifest are not modified. Pass ``--automatic-report`` to
check that the automatic MJCF and visual PLY match the generator's provenance.
The resulting ``variants.json`` is accepted by the existing independent policy and
native-viewer benchmark tools; preparation alone does not establish collision
quality or task improvement.

Viser transparent collision overlays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Viser model geometry uses the same direct mesh API as reference-motion ghosts:
vertices, triangle indices, flat RGB colours, and explicit opacity. Fixed and
batched dynamic geoms and sites do not serialize or parse GLB. Faces with different
RGBA colours are submitted separately; texture maps and PBR appearance are replaced
by flat colours. Geometry and transforms remain intact. To composite transparent
meshes over Gaussian splats with the currently supported Viser client, rebuild its
browser bundle once per environment:

.. code-block:: bash

    uv run scripts/tools/build_viser_transparency.py

This requires Node.js 24 or later, npm, and network access to install frontend
build dependencies. Restart the viewer and reload the browser afterwards. Re-run
after reinstalling Viser. The tool fails if the installed client source no longer
matches the supported render-order patch.

The resulting order is opaque meshes, Gaussian splats, then transparent meshes.
This supports collision debugging overlays; it does not jointly sort Gaussians
and transparent mesh fragments. Transparent geometry behind a Gaussian wall may
remain visible. Physical collisions are unaffected by visualization alpha.

For ``play`` or ``demo``, prefer the versioned manifest:

.. code-block:: bash

    uv run demo \
        --use-3dgs True \
        --scene-manifest /path/to/scene/manifest.json

``manifest.json`` binds the PLY, collision MJCF, optional collision dependency
files, right-handed Z-up metre convention, one ``world_from_scene`` transform, and
named spawn-point metadata. Asset paths must be relative to the manifest directory
and cannot escape it. The loader parses the MJCF and requires
``collision.dependencies`` to match its external file references exactly; an
omitted mesh/heightfield/texture or an unused declaration fails before scene
compilation. The
formal schema is ``src/mjlab/scene/schemas/scene_manifest.schema.json``.

.. important::

    The PLY data itself must use MuJoCo's right-handed, Z-up coordinate convention
    with lengths in metres. Before loading, align the PLY and collision MJCF to
    the same scene-local origin, axis directions, and scale. Otherwise, the visual
    scene and physical collision geometry will be offset, rotated, or scaled
    differently after loading.

    ``coordinate_system`` declares the required convention; it does not convert
    PLY coordinates or detect geometric misalignment. ``world_from_scene`` applies
    the same rigid transform to both assets, so it cannot repair a mismatch
    between them. Convert and align source assets during preprocessing, including
    Gaussian orientations and scales when transforming the PLY.

The same manifest can be supplied to training:

.. code-block:: bash

    uv run train Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth \
        --scene-manifest /path/to/scene/manifest.json \
        --env.scene.num-envs 4096

The loader applies exactly the same translation and quaternion to the Gaussian
asset and the collision root. It transforms the manifest's ``default`` spawn from
scene coordinates into world coordinates and applies it to robot reset position and
orientation. A manifest with one differently named spawn uses that entry; with
multiple entries and no ``default``, select one explicitly:

.. code-block:: bash

    uv run play Mjlab-Velocity-Flat-Unitree-G1-3DGS \
        --scene-manifest /path/to/scene/manifest.json \
        --scene-spawn-point doorway

For parallel environments, the transformed spawn pose is composed with the existing
per-environment grid origins, so selecting one spawn does not stack all robots.

G1 depth visual training
~~~~~~~~~~~~~~~~~~~~~~~~

The depth task reuses the YAM Vision path: the camera observation has shape
``(B, 1, 48, 64)``, is clipped to ``0.05--5.0 m``, normalized to ``[0, 1]``, and
encoded by ``SpatialSoftmaxCNNModel`` before fusion with proprioception. Both actor
and critic initially consume the camera group. The action space and velocity command
interface are unchanged.

Run a one-environment configuration smoke test with the bundled room1 manifest:

.. code-block:: bash

    uv run train Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth \
        --scene-manifest src/mjlab/asset_zoo/3dgs/room1/manifest.json \
        --scene-spawn-point default \
        --env.scene.num-envs 1 \
        --agent.max-iterations 1

Camera and terrain-height ray sensors currently share one MuJoCo Warp BVH. To keep
the hidden physical plane out of the rendered 3DGS depth, this first visual task
removes ``foot_height_scan`` and the two rewards that depend on it
(``foot_clearance`` and ``foot_swing_height``). Contact, velocity tracking, posture,
smoothness, and the remaining locomotion signals are preserved. The planned P7
target-conditioned avoidance task will add deceleration, stopping-distance,
minimum-clearance, collision, and goal-reaching metrics. The bundled room1 smoke
asset exposes its synthetic group-1 pillars to the policy camera while keeping its
group-3 physical ground hidden. This validates the visual-control loop but exposes
exact obstacle geometry; final 3DGS perception experiments should instead render
obstacles as splats and keep their aligned collision proxies hidden from the actor.

Play a local velocity checkpoint in a manifest scene
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``play`` defaults to a trained agent, so a local run must provide
``--checkpoint-file`` (or alternatively a W&B run). The following command loads a
locally trained G1 velocity policy, room1's visual and collision assets, and the
manifest's ``default`` spawn point:

.. code-block:: bash

    uv run play Mjlab-Velocity-Flat-Unitree-G1-3DGS \
        --scene-manifest src/mjlab/asset_zoo/3dgs/room1/manifest.json \
        --scene-spawn-point default \
        --checkpoint-file \
          logs/rsl_rl/g1_velocity/<run>/model_<iteration>.pt \
        --num-envs 1 \
        --viewer viser \
        --collision-visible True

Replace ``<run>`` and ``<iteration>`` with the local training output. For example,
a completed run may contain ``model_29999.pt``. In Viser, enable the ``Twist``
controls and press ``Zero`` to command static standing. The Collision control can
be toggled at runtime under ``Visualization``.

For scene-only debugging, add ``--agent zero`` and omit ``--checkpoint-file``.
That mode sends zero joint actions and is not a closed-loop standing controller, so
the robot may fall.

For quick experiments without a manifest, the separate interfaces remain:

.. code-block:: bash

    uv run demo \
        --use-3dgs True \
        --gaussian-splat-file /path/to/scene/visual/splats.ply \
        --collision-mjcf-file /path/to/scene/collision/collision.xml

Do not combine either separate asset option (or ``--collision-keep-plane``) with
``--scene-manifest``. ``--collision-visible`` is a viewer option and works with
both input styles.

Imported collision geoms are assigned to viewer group 3 and hidden by default in
the bundled 3DGS tasks. Add ``--collision-visible True`` to inspect them, and use
semi-transparent ``rgba`` values in the MJCF. ``--collision-keep-plane True``
retains the plane for debugging; it is off by default to prevent double contacts.
While Viser is running, ``Visualization > Collision > Enabled`` toggles group 3
immediately. The equivalent ``Groups > Geoms > G3`` control remains available;
the named Collision checkbox follows changes made through either control.

The bundled room1 manual smoke test keeps all files belonging to the scene under
one asset directory:

.. code-block:: text

    src/mjlab/asset_zoo/3dgs/room1/
      manifest.json
      point_cloud_2_transform.ply
      collision/
        collision.xml

``collision.xml`` contains a ground box whose top is at ``z=0`` and six fixed
pseudo-random cylinders. The fixed layout makes contact failures reproducible.
It deliberately remains a separate file from the PLY because the PLY is visual
data and the MJCF is physical data. Run the standard pretrained demo policy with
both assets using:

.. code-block:: bash

    uv run tests/run_g1_3dgs_collision_demo.py

The script loads ``manifest.json``, downloads or reuses the same checkpoint and
motion as ``uv run demo``, selects ``Mjlab-Tracking-Flat-Unitree-G1-3DGS``, uses
one robot, and enables the semi-transparent collision proxy for visual alignment
inspection.

P4 collision validation
-----------------------

The room1 collision regression drops a 0.2 m box for two simulated seconds. It
requires a final center height of 95--105 mm, less than 0.5 mm steady-state height
standard deviation, no more than 10 mm contact penetration, less than 0.01 m/s
final vertical speed, and zero MuJoCo Warp overflow bits. It also initializes and
steps the complete G1 velocity environment, including its foot contact and terrain
height sensors.

Holding only G1's default joint-position targets is not a balance controller: in a
separate diagnostic the robot fell within approximately three seconds. Static
standing acceptance therefore requires a closed-loop standing policy or checkpoint;
it must not be claimed from an open-loop pose test. The pretrained tracking demo's
continuous room1 run remains the current closed-loop robot smoke test.


G1 flat-terrain video demo
--------------------------

``Mjlab-Velocity-Flat-Unitree-G1-3DGS`` uses the normal flat G1 configuration
for training. Its play configuration loads a 3DGS scene for native Viser display.
Without ``--collision-mjcf-file``, the physical ground remains the MuJoCo plane and
is hidden from Viser. When video recording is explicitly requested, an on-demand
camera wraps the robot's tracking camera and composes the robot's visual meshes
with the splats.

Set the asset path when it differs from the local development default. The command
line option takes precedence over ``MJLAB_G1_3DGS_ASSET``:

.. code-block:: bash

    export MJLAB_G1_3DGS_ASSET=/path/to/point_cloud.ply

Train without paying any 3DGS rendering cost:

.. code-block:: bash

    uv run train Mjlab-Velocity-Flat-Unitree-G1-3DGS \
        --env.scene.num-envs 4096

Then record the native MuJoCo Warp camera after selecting a checkpoint:

.. code-block:: bash

    uv run play Mjlab-Velocity-Flat-Unitree-G1-3DGS \
        --checkpoint-file /path/to/model.pt \
        --gaussian-splat-file /path/to/point_cloud.ply \
        --num-envs 1 \
        --video True \
        --video-sensor g1_3dgs_video \
        --video-length 300 \
        --viewer viser

The MP4 is written under ``<checkpoint-directory>/videos/play/``. If no collision
MJCF is supplied, the Gaussian scene remains visual-only and the robot walks on
the flat MuJoCo plane.

Viser loads the same normalized Gaussian data into its native interactive 3DGS
renderer. The splats are rendered directly in the 3D viewport together with the
robot meshes, so browser camera orbit, zoom, and tracking remain available without
streaming a low-resolution camera image. Normal Viser playback does not create a
MuJoCo Warp camera or pay its per-step rendering cost. Passing both ``--video True``
and ``--video-sensor g1_3dgs_video`` enables that camera on demand for MP4 recording.

Pretrained tracking demo
------------------------

The bundled tracking checkpoint is compatible with the 3DGS tracking play task.
Set the asset path and enable the demo option:

.. code-block:: bash

    uv run demo \
        --use-3dgs True \
        --gaussian-splat-file /path/to/point_cloud.ply

This selects ``Mjlab-Tracking-Flat-Unitree-G1-3DGS`` and defaults to one environment
while retaining the normal demo's cached checkpoint and motion. The ordinary
``uv run demo`` remains unchanged and defaults to eight environments without 3DGS.
When ``--use-3dgs False``, a supplied ``--gaussian-splat-file`` is ignored and the
PLY is neither validated nor loaded.

Multiple robots
---------------

The 3DGS demo defaults to one robot for performance, but ``--num-envs`` can show
multiple simulated robots in the same loaded splat scene:

.. code-block:: bash

    uv run demo \
        --use-3dgs True \
        --gaussian-splat-file /path/to/point_cloud.ply \
        --num-envs 4

In Viser, leave ``Scene > Environment > Hide others`` unchecked to display all
robots. Enable it and use ``Select`` to inspect one environment. The splat arrays
are loaded once rather than duplicated for every robot, although physics and policy
evaluation still run for every environment.

The equivalent explicit ``play`` command is:

.. code-block:: bash

    uv run play Mjlab-Tracking-Flat-Unitree-G1-3DGS \
        --checkpoint-file /tmp/mjlab_cache/demo_ckpt.pt \
        --motion-file /tmp/mjlab_cache/lafan1_dance1_subject1_demo_motion.npz \
        --gaussian-splat-file /path/to/point_cloud.ply \
        --num-envs 1 \
        --viewer viser


Performance notes
-----------------

Use Viser's ``Target RT`` and ``Actual RT`` values to distinguish requested speed
from achieved simulation speed. Startup includes PLY parsing, transfer to the
browser, and GPU initialization, so compare steady-state values after initialization.
The bundled G1 configuration does not run the MuJoCo Warp camera during ordinary
Viser playback. Selecting ``g1_3dgs_video`` for recording enables per-step camera
rendering and is substantially more expensive. For browser rendering bottlenecks,
use ``opacity_threshold``, crop bounds, or ``max_splats`` to reduce the asset.
