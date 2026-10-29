.. _gaussian_splats:

3D Gaussian Splatting
=====================

.. note::

   On Linux x86_64 with Python 3.12 or 3.13, the repository's uv configuration
   downloads the custom MuJoCo 3DGS wheel from
   `the fork's v1.0 Release <https://github.com/PiggyGaGa/mujoco/releases/tag/v1.0>`_.
   The wheels require glibc 2.35 or newer. Run
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
assets should be filtered with bounds, opacity, or ``max_splats``. A versioned
preprocessing cache is an optional future startup-time optimization, while automatic
collision-proxy generation is planned separately.


External collision scene
------------------------

The visual PLY and collision MJCF are separate representations of one logical
scene. They must already use the same right-handed, Z-up coordinate frame, meter
units, and origin. The collision entry point can contain static primitive geoms,
multiple convex mesh assets, and heightfields. It must not contain joints, mocap
bodies, actuators, or sensors.

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

For ``play`` or ``demo``, prefer the versioned manifest:

.. code-block:: bash

    uv run demo \
        --use-3dgs True \
        --scene-manifest /path/to/scene/manifest.json

``manifest.json`` binds the PLY, collision MJCF, optional collision dependency
files, SHA-256 fingerprints, right-handed Z-up metre convention, one
``world_from_scene`` transform, and named spawn-point metadata. Asset paths must be
relative to the manifest directory and cannot escape it. Hashes are checked before
the scene is built, so mixing a stale PLY or collision file fails immediately. The
loader also parses the MJCF and requires ``collision.dependencies`` to match its
external file references exactly; an omitted mesh/heightfield/texture or an unused
declaration fails before scene compilation. The
formal schema is ``src/mjlab/scene/schemas/scene_manifest.schema.json``.

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
smoothness, and the remaining locomotion signals are preserved. P7 will add
task-specific step-clearance metrics without merging hidden collision geometry into
the visual camera context.

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
