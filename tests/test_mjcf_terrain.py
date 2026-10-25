"""Tests for static collision-only MJCF terrain input."""

from pathlib import Path

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import ContactSensorCfg, TerrainHeightSensorCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.terrains import TerrainEntityCfg

_ROOM1_COLLISION_PATH = (
  Path(__file__).parents[1] / "src/mjlab/asset_zoo/3dgs/room1/collision/collision.xml"
)


@pytest.fixture
def device() -> str:
  """Select the normal project test device."""
  return get_test_device()


def _write_mjcf(tmp_path: Path, xml: str) -> Path:
  path = tmp_path / "collision.xml"
  path.write_text(xml)
  return path


def _build_scene(path: Path, *, keep_plane: bool = False) -> Scene:
  return Scene(
    SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="mjcf",
        mjcf_path=str(path),
        mjcf_keep_plane=keep_plane,
      )
    ),
    device="cpu",
  )


def test_mjcf_terrain_loads_static_primitives(tmp_path: Path) -> None:
  path = _write_mjcf(
    tmp_path,
    """
    <mujoco>
      <worldbody>
        <body name="collision_scene">
          <geom name="ground" type="box" pos="0 0 -0.1" size="2 2 0.1"/>
          <geom name="step" type="box" pos="0.5 0 0.1" size="0.5 1 0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """,
  )

  scene = _build_scene(path)
  model = scene.compile()

  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground") >= 0
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "step") >= 0
  terrain = scene.terrain
  assert terrain is not None
  assert all(geom.group == 3 for geom in terrain.spec.geoms)
  assert not any(geom.name == "terrain" for geom in terrain.spec.geoms)


def test_mjcf_terrain_can_explicitly_keep_plane(tmp_path: Path) -> None:
  path = _write_mjcf(
    tmp_path,
    "<mujoco><worldbody><geom name='step' type='box' size='.5 .5 .5'/>"
    "</worldbody></mujoco>",
  )

  scene = _build_scene(path, keep_plane=True)

  terrain = scene.terrain
  assert terrain is not None
  assert any(geom.name == "terrain" for geom in terrain.spec.geoms)


def test_plane_terrain_remains_the_fallback() -> None:
  scene = Scene(SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane")), device="cpu")

  terrain = scene.terrain
  assert terrain is not None
  assert [geom.name for geom in terrain.spec.geoms] == ["terrain"]


def test_mjcf_terrain_is_shared_across_worlds(tmp_path: Path) -> None:
  path = _write_mjcf(
    tmp_path,
    "<mujoco><worldbody><geom name='ground' type='box' size='2 2 .1'/>"
    "</worldbody></mujoco>",
  )
  scene = Scene(
    SceneCfg(
      num_envs=4,
      terrain=TerrainEntityCfg(terrain_type="mjcf", mjcf_path=str(path)),
    ),
    device="cpu",
  )

  model = scene.compile()
  assert model.ngeom == 1
  assert scene.env_origins.shape == (4, 3)


def test_mjcf_terrain_applies_scene_to_world_transform(tmp_path: Path) -> None:
  path = _write_mjcf(
    tmp_path,
    "<mujoco><worldbody><geom name='ground' type='box' size='2 2 .1'/>"
    "</worldbody></mujoco>",
  )
  scene = Scene(
    SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="mjcf",
        mjcf_path=str(path),
        mjcf_pos=(1.0, 2.0, 3.0),
        mjcf_quat=(0.0, 0.0, 0.0, 1.0),
      )
    ),
    device="cpu",
  )

  model = scene.compile()
  body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")

  assert body_id >= 0
  assert model.body_pos[body_id].tolist() == [1.0, 2.0, 3.0]
  assert model.body_quat[body_id].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_mjcf_terrain_loads_relative_convex_meshes(tmp_path: Path) -> None:
  mesh_dir = tmp_path / "meshes"
  mesh_dir.mkdir()
  tetrahedron = (
    "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
  )
  (mesh_dir / "left.obj").write_text(tetrahedron)
  (mesh_dir / "right.obj").write_text(tetrahedron)
  path = _write_mjcf(
    tmp_path,
    """
    <mujoco>
      <compiler meshdir="meshes"/>
      <asset>
        <mesh name="left" file="left.obj"/>
        <mesh name="right" file="right.obj"/>
      </asset>
      <worldbody>
        <geom name="left_collision" type="mesh" mesh="left"/>
        <geom name="right_collision" type="mesh" mesh="right" pos="2 0 0"/>
      </worldbody>
    </mujoco>
    """,
  )

  scene = _build_scene(path)
  model = scene.compile()

  assert model.nmesh == 2


def test_mjcf_terrain_loads_heightfield(tmp_path: Path) -> None:
  path = _write_mjcf(
    tmp_path,
    """
    <mujoco>
      <asset>
        <hfield name="terrain_hfield" size="1 1 .2 .1" nrow="2" ncol="2"
                elevation="0 0 .2 .2"/>
      </asset>
      <worldbody>
        <geom name="heightfield" type="hfield" hfield="terrain_hfield"/>
      </worldbody>
    </mujoco>
    """,
  )

  assert _build_scene(path).compile().nhfield == 1


def test_room1_collision_asset_compiles() -> None:
  scene = _build_scene(_ROOM1_COLLISION_PATH)
  model = scene.compile()

  assert model.ngeom == 7
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground") >= 0
  for pillar_id in range(6):
    assert (
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"pillar_{pillar_id:02d}") >= 0
    )


def test_room1_worldbody_geoms_initialize_as_terrain_entity() -> None:
  scene = _build_scene(_ROOM1_COLLISION_PATH)
  model = scene.compile()
  sim = Simulation(
    num_envs=1,
    cfg=SimulationCfg(),
    model=model,
    device="cpu",
  )

  scene.initialize(sim.mj_model, sim.model, sim.data)

  terrain = scene.terrain
  assert terrain is not None
  assert terrain.root_body.name == "terrain"
  assert terrain.data.default_root_state.shape == (1, 7)


def test_g1_foot_contact_sensor_resolves_external_terrain(device: str) -> None:
  task_cfg = unitree_g1_flat_env_cfg(play=True)
  feet_sensor = next(
    sensor
    for sensor in task_cfg.scene.sensors or ()
    if sensor.name == "feet_ground_contact"
  )
  assert isinstance(feet_sensor, ContactSensorCfg)
  scene = Scene(
    SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="mjcf", mjcf_path=str(_ROOM1_COLLISION_PATH)
      ),
      entities={"robot": get_g1_robot_cfg()},
      sensors=(feet_sensor,),
    ),
    device=device,
  )
  model = scene.compile()
  sim = Simulation(
    num_envs=1,
    cfg=SimulationCfg(nconmax=70, njmax=300),
    model=model,
    device=device,
  )

  scene.initialize(sim.mj_model, sim.model, sim.data)

  assert tuple(scene["feet_ground_contact"].primary_names) == (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )


def test_external_mjcf_retargets_g1_terrain_height_sensor(device: str) -> None:
  task_cfg = unitree_g1_flat_env_cfg(play=True)
  height_sensor = next(
    sensor
    for sensor in task_cfg.scene.sensors or ()
    if sensor.name == "foot_height_scan"
  )
  assert isinstance(height_sensor, TerrainHeightSensorCfg)
  Scene(
    SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="mjcf",
        mjcf_path=str(_ROOM1_COLLISION_PATH),
        mjcf_geom_group=3,
      ),
      entities={"robot": get_g1_robot_cfg()},
      sensors=(height_sensor,),
    ),
    device=device,
  )

  assert height_sensor.include_geom_groups == (3,)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:overflow encountered in cast:RuntimeWarning")
def test_g1_velocity_env_steps_with_room1_external_collision(device: str) -> None:
  cfg = unitree_g1_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "mjcf"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.terrain.mjcf_path = str(_ROOM1_COLLISION_PATH)

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    actions = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
    observations, rewards, _, _, _ = env.step(actions)

    assert torch.isfinite(rewards).all()
    actor_observations = observations["actor"]
    assert isinstance(actor_observations, torch.Tensor)
    assert torch.isfinite(actor_observations).all()
    height_sensor = next(
      sensor for sensor in cfg.scene.sensors or () if sensor.name == "foot_height_scan"
    )
    assert isinstance(height_sensor, TerrainHeightSensorCfg)
    assert height_sensor.include_geom_groups == (3,)
  finally:
    env.close()


def test_room1_collision_stably_catches_falling_box(device: str) -> None:
  box_xml = """
    <mujoco>
      <worldbody>
        <body name="drop_box">
          <freejoint/>
          <geom name="drop_box_geom" type="box" size="0.1 0.1 0.1"
                density="500"/>
        </body>
      </worldbody>
    </mujoco>
  """
  scene = Scene(
    SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="mjcf", mjcf_path=str(_ROOM1_COLLISION_PATH)
      ),
      entities={
        "box": EntityCfg(
          spec_fn=lambda: mujoco.MjSpec.from_string(box_xml),
          init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )
      },
    ),
    device=device,
  )
  model = scene.compile()
  sim = Simulation(
    num_envs=1,
    cfg=SimulationCfg(
      nconmax=16,
      njmax=32,
      mujoco=MujocoCfg(timestep=0.002, iterations=20, ls_iterations=40),
    ),
    model=model,
    device=device,
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  box = scene["box"]
  z_samples: list[float] = []

  for step in range(1_000):
    sim.step()
    if step >= 800:
      z_index = int(box.indexing.free_joint_q_adr[2])
      z_samples.append(float(sim.data.qpos[0, z_index].item()))

  sim.forward()
  ncon = int(sim.data.nacon[0].item())
  assert ncon > 0
  min_contact_distance = float(sim.data.contact.dist[:ncon].min().item())
  final_vertical_speed = abs(
    float(sim.data.qvel[0, int(box.indexing.free_joint_v_adr[2])].item())
  )

  assert 0.095 <= z_samples[-1] <= 0.105
  assert torch.tensor(z_samples).std().item() < 5.0e-4
  assert min_contact_distance > -0.01
  assert final_vertical_speed < 0.01
  assert int(sim.data.overflow[0].item()) == 0


@pytest.mark.parametrize(
  ("xml", "error"),
  [
    (
      "<mujoco><worldbody><body><freejoint/><geom type='box' size='1 1 1'/>"
      "</body></worldbody></mujoco>",
      "cannot contain joints",
    ),
    (
      "<mujoco><worldbody><body mocap='true'><geom type='box' size='1 1 1'/>"
      "</body></worldbody></mujoco>",
      "mocap bodies",
    ),
    (
      "<mujoco><worldbody><body name='scene'><geom type='box' size='1 1 1'/>"
      "<site name='push'/></body></worldbody><actuator><general name='force' "
      "site='push'/></actuator></mujoco>",
      "cannot contain actuators",
    ),
    (
      "<mujoco><worldbody><body name='scene'><geom type='box' size='1 1 1'/>"
      "</body></worldbody><sensor><framepos name='pose' objtype='body' "
      "objname='scene'/></sensor></mujoco>",
      "cannot contain sensors",
    ),
    (
      "<mujoco><worldbody><geom name='visual' type='box' size='1 1 1' "
      "contype='0' conaffinity='0'/></worldbody></mujoco>",
      "no collision-enabled geoms",
    ),
  ],
)
def test_mjcf_terrain_rejects_non_static_or_non_collision_models(
  tmp_path: Path, xml: str, error: str
) -> None:
  path = _write_mjcf(tmp_path, xml)

  with pytest.raises(ValueError, match=error):
    _build_scene(path)


@pytest.mark.parametrize("asset_path", ["../outside.obj", "/tmp/outside.obj"])
def test_mjcf_terrain_rejects_escaping_asset_paths(
  tmp_path: Path, asset_path: str
) -> None:
  path = _write_mjcf(
    tmp_path,
    f"""
    <mujoco>
      <asset><mesh name="mesh" file="{asset_path}"/></asset>
      <worldbody><geom type="mesh" mesh="mesh"/></worldbody>
    </mujoco>
    """,
  )

  with pytest.raises(ValueError, match="must stay relative"):
    _build_scene(path)


def test_mjcf_terrain_requires_existing_file() -> None:
  with pytest.raises(FileNotFoundError, match="Collision MJCF not found"):
    _build_scene(Path("/does/not/exist.xml"))


def test_mjcf_terrain_reports_missing_mesh_asset(tmp_path: Path) -> None:
  path = _write_mjcf(
    tmp_path,
    """
    <mujoco>
      <asset><mesh name="missing" file="missing.obj"/></asset>
      <worldbody><geom type="mesh" mesh="missing"/></worldbody>
    </mujoco>
    """,
  )

  with pytest.raises(ValueError, match="failed to compile.*missing.obj"):
    _build_scene(path)


def test_mjcf_path_is_rejected_for_plane_terrain(tmp_path: Path) -> None:
  path = _write_mjcf(tmp_path, "<mujoco/>")

  with pytest.raises(ValueError, match="only valid"):
    Scene(
      SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane", mjcf_path=str(path))),
      device="cpu",
    )
