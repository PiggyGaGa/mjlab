.. _gaussian_splats:

3D Gaussian Splatting
=====================

mjlab can add static 3D Gaussian Splatting (3DGS) assets to interactive Viser
playback and RGB or depth camera observations. Viser uses its native splat
renderer for interactive playback. Camera observations and video recording use
MuJoCo Warp's native renderer, where splats and MuJoCo geometry share the same
camera and depth test.

3DGS assets are visual only. They do not participate in physics, contact
detection, or ``RayCastSensor`` queries. A training environment that needs
physical interaction must provide separate MuJoCo collision geometry aligned
with the splat asset.


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

The bundled G1 3DGS tasks attach ``GaussianSplatSceneCfg`` only to their play
configuration. Their training configuration therefore does not parse the PLY,
allocate splat arrays, create a splat BVH, or pay any rendering cost. Normal Viser
playback loads the PLY once but does not create the MuJoCo Warp camera. The camera
is enabled on demand only for explicit sensor-based video recording.

The current importer reads the source PLY during environment initialization.
Very large assets should be filtered with bounds, opacity, or ``max_splats``.
A versioned preprocessing cache and automatic collision-proxy generation are
planned separately.


G1 flat-terrain video demo
--------------------------

``Mjlab-Velocity-Flat-Unitree-G1-3DGS`` uses the normal flat G1 configuration
for training. Its play configuration loads a 3DGS scene for native Viser display.
The physical ground remains the MuJoCo plane and is hidden from Viser. When video
recording is explicitly requested, an on-demand camera wraps the robot's tracking
camera and composes the robot's visual meshes with the splats.

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

The MP4 is written under ``<checkpoint-directory>/videos/play/``. This demo is
for visual validation only: the Gaussian scene has no collision geometry, and
the robot still walks on the flat MuJoCo plane.

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
