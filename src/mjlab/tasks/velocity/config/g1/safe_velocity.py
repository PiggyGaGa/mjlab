"""Safe perceptive velocity tracking configuration for Unitree G1."""

from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import apply_scene_manifest, load_scene_manifest
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.tasks.g1_3dgs import G1_3DGS_DEPTH_SENSOR
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from .env_cfgs import (
  unitree_g1_flat_3dgs_depth_validity_history_robust_env_cfg,
)
from .rl_cfg import (
  unitree_g1_3dgs_depth_validity_history_robust_asymmetric_ppo_runner_cfg,
  unitree_g1_ppo_runner_cfg,
)
from .safe_velocity_baseline import (
  SafeVelocityDepthAdapterCfg,
  SafeVelocityLocalGeometryTrackingReward,
  SafeVelocityPrivilegedSafetyCommand,
  SafeVelocityPrivilegedSafetyLabels,
  SafeVelocityPrivilegedStopCommand,
  SafeVelocitySafetyTeacherCfg,
  SafeVelocitySafetyTeacherTrackingReward,
  SafeVelocityTrackingReward,
  adapted_velocity_command,
  safe_velocity_base_obstacle_collision_cost,
  safe_velocity_local_geometry,
)
from .safe_velocity_scenarios import (
  SAFE_VELOCITY_SCENARIO_CATALOG,
  SafeVelocityScenarioCatalog,
)
from .safe_velocity_sensors import CollisionTruthRaySensorCfg

SAFE_VELOCITY_TASK_ID = "Mjlab-Velocity-Safe-Unitree-G1-3DGS"
SAFE_VELOCITY_ADAPTER_TASK_ID = "Mjlab-Velocity-Safe-Adapter-Unitree-G1-3DGS"
SAFE_VELOCITY_DEPTH_STOP_TASK_ID = "Mjlab-Velocity-Safe-Depth-Stop-Unitree-G1-3DGS"
SAFE_VELOCITY_BLIND_TASK_ID = "Mjlab-Velocity-Safe-Blind-Unitree-G1-3DGS"
SAFE_VELOCITY_PRIVILEGED_TASK_ID = "Mjlab-Velocity-Safe-Privileged-Unitree-G1-3DGS"
SAFE_VELOCITY_PRIVILEGED_STOP_GATE_TASK_ID = (
  "Mjlab-Velocity-Safe-Privileged-Stop-Gate-Unitree-G1-3DGS"
)
SAFE_VELOCITY_SAFETY_TEACHER_TASK_ID = (
  "Mjlab-Velocity-Safe-Privileged-Safety-Teacher-Unitree-G1-3DGS"
)
SAFE_VELOCITY_VISION_DATA_TASK_ID = "Mjlab-Velocity-Safe-Vision-Data-Unitree-G1-3DGS"
SAFE_VELOCITY_COLLISION_SENSOR_PREFIX = "safe_velocity_base_obstacle_contact_"
SAFE_VELOCITY_NONFOOT_GROUND_SENSOR = "safe_velocity_nonfoot_ground_contact"
SAFE_VELOCITY_LOCAL_GEOMETRY_SENSOR_PREFIX = "safe_velocity_local_geometry_ray_"
SAFE_VELOCITY_ALLOWED_CONTACT_GEOMS = frozenset({"ground"})
SAFE_VELOCITY_MIN_PELVIS_HEIGHT_M = 0.5
_LOCAL_GEOMETRY_RAY_ANGLES_DEG = (
  80.0,
  60.0,
  40.0,
  20.0,
  0.0,
  -20.0,
  -40.0,
  -60.0,
  -80.0,
)


def _safe_velocity_adapter_cfg(
  catalog: SafeVelocityScenarioCatalog,
) -> SafeVelocityDepthAdapterCfg:
  """Build one adapter threshold set shared by reward and baseline tasks."""
  return SafeVelocityDepthAdapterCfg(
    collision_risk_distance_m=catalog.thresholds.collision_risk_distance_m,
    stop_clearance_m=catalog.thresholds.stop_clearance_m,
    search_trigger_distance_m=catalog.thresholds.search_trigger_distance_m,
    search_time_budget_s=catalog.thresholds.search_time_budget_s,
  )


def _safe_velocity_depth_stop_cfg() -> SafeVelocityDepthAdapterCfg:
  """Build the shape-independent depth threshold safety layer."""
  return SafeVelocityDepthAdapterCfg(
    collision_risk_distance_m=1.4,
    stop_clearance_m=0.4,
    search_trigger_distance_m=0.8,
    clearance_quantile=0.2,
    vertical_roi=(0.2, 0.85),
    stop_only=True,
    clearance_sector_indices=(0, 1, 2),
    horizontal_window_fraction=0.25,
    temporal_median_window=7,
    maximum_forward_speed_m_s=0.5,
  )


def _set_safe_velocity_play_command(
  cfg: ManagerBasedRlEnvCfg,
  scenario_id: str,
) -> None:
  """Use the selected scenario command for deterministic safety playback."""
  scenario = SAFE_VELOCITY_SCENARIO_CATALOG.scenario(scenario_id)
  _set_safe_velocity_play_twist(cfg, scenario.nominal_command)


def _set_safe_velocity_play_twist(
  cfg: ManagerBasedRlEnvCfg,
  command: tuple[float, float, float],
) -> None:
  """Freeze one deterministic body-frame twist for playback."""
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  vx, vy, wz = command
  twist.ranges.lin_vel_x = (vx, vx)
  twist.ranges.lin_vel_y = (vy, vy)
  twist.ranges.ang_vel_z = (wz, wz)
  twist.rel_forward_envs = 0.0


def _safe_velocity_obstacle_geom_names(
  scene_manifest: str | Path,
) -> tuple[str, ...]:
  """Return collision geoms that must not contact the robot base link."""
  manifest = load_scene_manifest(scene_manifest)
  try:
    root = ElementTree.parse(manifest.collision.path).getroot()
  except ElementTree.ParseError as exc:
    raise ValueError(
      f"Invalid safe-velocity collision MJCF XML '{manifest.collision.path}': {exc}"
    ) from exc
  geom_names = tuple(
    geom_name
    for geom in root.findall(".//worldbody//geom")
    if (geom_name := geom.attrib.get("name"))
    and geom_name not in SAFE_VELOCITY_ALLOWED_CONTACT_GEOMS
  )
  if not geom_names:
    raise ValueError("Safe-velocity collision MJCF has no obstacle geoms")
  return geom_names


def unitree_g1_safe_velocity_3dgs_env_cfg(
  play: bool = False,
  scenario_id: str = "both_open",
  scene_manifest: str | Path | None = None,
  scene_spawn_point: str | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the isolated safe-velocity task for one deterministic scenario."""
  cfg = unitree_g1_flat_3dgs_depth_validity_history_robust_env_cfg(play=play)
  catalog = SAFE_VELOCITY_SCENARIO_CATALOG
  scenario = catalog.scenario(scenario_id)
  manifest_path = catalog.scene_manifest if scene_manifest is None else scene_manifest
  spawn_point = scene_spawn_point
  if spawn_point is None and scene_manifest is None:
    spawn_point = scenario.spawn_point
  apply_scene_manifest(
    cfg.scene,
    manifest_path,
    spawn_point=spawn_point,
  )

  # MuJoCo Warp already isolates vectorized environments as separate worlds that
  # share this collision model and splat asset. Keep every robot in scene-local
  # coordinates so collision, camera, and raycast geometry stay aligned per world.
  cfg.scene.env_spacing = 0.0
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 256)
  cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch, 256)
  obstacle_geom_names = _safe_velocity_obstacle_geom_names(manifest_path)
  obstacle_contact_sensors = tuple(
    ContactSensorCfg(
      name=f"{SAFE_VELOCITY_COLLISION_SENSOR_PREFIX}{index}",
      primary=ContactMatch(mode="body", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="geom", pattern=geom_name),
      fields=("found", "force", "dist"),
      reduce="maxforce",
      history_length=cfg.decimation,
    )
    for index, geom_name in enumerate(obstacle_geom_names)
  )
  nonfoot_ground_sensor = ContactSensorCfg(
    name=SAFE_VELOCITY_NONFOOT_GROUND_SENSOR,
    primary=ContactMatch(
      mode="body",
      pattern=".*",
      entity="robot",
      exclude=("left_ankle_roll_link", "right_ankle_roll_link"),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    history_length=cfg.decimation,
  )
  cfg.scene.sensors = (
    (cfg.scene.sensors or ()) + obstacle_contact_sensors + (nonfoot_ground_sensor,)
  )
  cfg.terminations["pelvis_too_low"] = TerminationTermCfg(
    func=envs_mdp.root_height_below_minimum,
    params={"minimum_height": SAFE_VELOCITY_MIN_PELVIS_HEIGHT_M},
  )
  cfg.terminations["nonfoot_ground_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={
      "sensor_name": nonfoot_ground_sensor.name,
      "force_threshold": 10.0,
    },
  )

  # Replace unconditional nominal tracking with the same reward shape evaluated
  # against a depth-conditioned safe twist. This preserves free-space locomotion
  # incentives while allowing braking, search, and blocked standing near obstacles.
  cfg.rewards["track_linear_velocity"].weight = 0.0
  cfg.rewards["track_angular_velocity"].weight = 0.0
  adapter_cfg = _safe_velocity_adapter_cfg(catalog)
  cfg.rewards["safe_velocity_tracking"] = RewardTermCfg(
    func=SafeVelocityTrackingReward,
    weight=2.0,
    params={
      "command_name": "twist",
      "sensor_name": G1_3DGS_DEPTH_SENSOR,
      "adapter_cfg": adapter_cfg,
      "linear_std": math.sqrt(0.25),
      "angular_std": math.sqrt(0.5),
    },
  )
  cfg.rewards["base_obstacle_collision"] = RewardTermCfg(
    func=safe_velocity_base_obstacle_collision_cost,
    weight=-5.0,
    params={
      "sensor_names": tuple(sensor.name for sensor in obstacle_contact_sensors),
      "force_threshold": 10.0,
    },
  )

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_heading_envs = 0.0
  twist.rel_world_envs = 0.0
  twist.rel_standing_envs = 0.0
  twist.rel_forward_envs = 0.8
  twist.resampling_time_range = catalog.command.resampling_time_s
  twist.ranges.lin_vel_x = catalog.command.lin_vel_x
  twist.ranges.lin_vel_y = catalog.command.lin_vel_y
  twist.ranges.ang_vel_z = catalog.command.ang_vel_z
  # The generic G1 curriculum starts at reverse-capable (-1, 1) commands and
  # would overwrite the frozen P7 forward-only protocol on the first reset.
  cfg.curriculum.pop("command_vel", None)

  # Use deterministic scenario poses. Randomized spawn curricula are added
  # only after the catalog has passed visual/collision review.
  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.02, 0.02),
    "yaw": (0.0, 0.0),
  }
  cfg.events.pop("push_robot", None)

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  return cfg


def unitree_g1_safe_velocity_3dgs_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reuse the robust actor-camera/asymmetric-critic runner."""
  cfg = unitree_g1_3dgs_depth_validity_history_robust_asymmetric_ppo_runner_cfg()
  cfg.experiment_name = "g1_3dgs_safe_velocity"
  return cfg


def unitree_g1_safe_velocity_adapter_3dgs_env_cfg(
  play: bool = False,
  scenario_id: str = "both_open",
) -> ManagerBasedRlEnvCfg:
  """Create the evaluation task for a blind policy behind the depth adapter."""
  cfg = unitree_g1_safe_velocity_3dgs_env_cfg(
    play=play,
    scenario_id=scenario_id,
  )
  catalog = SAFE_VELOCITY_SCENARIO_CATALOG
  cfg.observations["actor"].terms["command"] = ObservationTermCfg(
    func=adapted_velocity_command,
    params={
      "command_name": "twist",
      "sensor_name": G1_3DGS_DEPTH_SENSOR,
      "adapter_cfg": _safe_velocity_adapter_cfg(catalog),
    },
  )
  # The camera remains in the scene for the adapter, but the blind policy and its
  # MLP runner do not consume an image observation group.
  cfg.observations.pop("camera")
  return cfg


def unitree_g1_safe_velocity_adapter_3dgs_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Keep the adapter baseline actor compatible with the standard G1 MLP."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_3dgs_safe_velocity_adapter"
  return cfg


def unitree_g1_safe_velocity_depth_stop_3dgs_env_cfg(
  play: bool = False,
  scenario_id: str = "both_open",
  scene_manifest: str | Path | None = None,
  scene_spawn_point: str | None = None,
  nominal_command: tuple[float, float, float] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create a manifest-driven blind locomotion task behind a depth stop layer."""
  cfg = unitree_g1_safe_velocity_3dgs_env_cfg(
    play=play,
    scenario_id=scenario_id,
    scene_manifest=scene_manifest,
    scene_spawn_point=scene_spawn_point,
  )
  if play:
    if nominal_command is None and scene_manifest is None:
      _set_safe_velocity_play_command(cfg, scenario_id)
    else:
      _set_safe_velocity_play_twist(
        cfg,
        nominal_command or (0.5, 0.0, 0.0),
      )
  cfg.observations["actor"].terms["command"] = ObservationTermCfg(
    func=adapted_velocity_command,
    params={
      "command_name": "twist",
      "sensor_name": G1_3DGS_DEPTH_SENSOR,
      "adapter_cfg": _safe_velocity_depth_stop_cfg(),
      "debug_print": play,
    },
  )
  cfg.rewards["safe_velocity_tracking"].params["adapter_cfg"] = (
    _safe_velocity_depth_stop_cfg()
  )
  cfg.observations.pop("camera")
  return cfg


def unitree_g1_safe_velocity_depth_stop_3dgs_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reuse the frozen-compatible blind G1 actor for depth-only stopping."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_3dgs_safe_velocity_depth_stop"
  return cfg


def unitree_g1_safe_velocity_blind_3dgs_env_cfg(
  play: bool = False,
  scenario_id: str = "both_open",
) -> ManagerBasedRlEnvCfg:
  """Create the renderer-free blind baseline in the frozen collision scene."""
  cfg = unitree_g1_safe_velocity_3dgs_env_cfg(
    play=play,
    scenario_id=scenario_id,
  )
  cfg.scene.sensors = tuple(
    sensor for sensor in cfg.scene.sensors if sensor.name != G1_3DGS_DEPTH_SENSOR
  )
  cfg.scene.gaussian_splats = None
  cfg.observations.pop("camera")
  cfg.rewards.pop("safe_velocity_tracking")
  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["track_angular_velocity"].weight = 2.0
  return cfg


def unitree_g1_safe_velocity_blind_3dgs_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Fine-tune the standard G1 MLP conservatively for the blind baseline."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_3dgs_safe_velocity_blind"
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg = {
    **cfg.actor.distribution_cfg,
    "init_std": 0.15,
    "std_range": (0.15, 0.15),
    "learn_std": False,
  }
  cfg.algorithm.learning_rate = 1.0e-4
  cfg.algorithm.entropy_coef = 0.0
  cfg.algorithm.num_learning_epochs = 2
  return cfg


def _safe_velocity_local_geometry_sensors(
  play: bool,
  isolate_from_camera: bool = False,
) -> tuple[RayCastSensorCfg, ...]:
  """Build a pelvis-frame collision-truth ray fan ordered left to right."""
  sensors = []
  sensor_cfg_type = (
    CollisionTruthRaySensorCfg if isolate_from_camera else RayCastSensorCfg
  )
  for index, angle_deg in enumerate(_LOCAL_GEOMETRY_RAY_ANGLES_DEG):
    angle_rad = math.radians(angle_deg)
    sensors.append(
      sensor_cfg_type(
        name=f"{SAFE_VELOCITY_LOCAL_GEOMETRY_SENSOR_PREFIX}{index:02d}",
        frame=ObjRef(type="body", name="pelvis", entity="robot"),
        pattern=GridPatternCfg(
          size=(0.0, 0.0),
          resolution=1.0,
          direction=(math.cos(angle_rad), math.sin(angle_rad), 0.0),
        ),
        ray_alignment="base",
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(1, 3),
        debug_vis=play,
      )
    )
  return tuple(sensors)


def unitree_g1_safe_velocity_privileged_3dgs_env_cfg(
  play: bool = False,
  scenario_id: str = "both_open",
) -> ManagerBasedRlEnvCfg:
  """Create the non-deployable local-geometry stop-policy baseline."""
  cfg = unitree_g1_safe_velocity_blind_3dgs_env_cfg(
    play=play,
    scenario_id=scenario_id,
  )
  ray_sensors = _safe_velocity_local_geometry_sensors(play=play)
  sensor_names = tuple(sensor.name for sensor in ray_sensors)
  cfg.scene.sensors += ray_sensors
  observation = ObservationTermCfg(
    func=safe_velocity_local_geometry,
    params={
      "sensor_names": sensor_names,
      "cutoff_distance_m": 5.0,
    },
  )
  cfg.observations["actor"].terms["local_geometry"] = observation
  cfg.observations["critic"].terms["local_geometry"] = observation

  cfg.rewards["track_linear_velocity"].weight = 0.0
  cfg.rewards["track_angular_velocity"].weight = 0.0
  cfg.rewards["safe_velocity_tracking"] = RewardTermCfg(
    func=SafeVelocityLocalGeometryTrackingReward,
    weight=2.0,
    params={
      "command_name": "twist",
      "sensor_names": sensor_names,
      "adapter_cfg": _safe_velocity_adapter_cfg(SAFE_VELOCITY_SCENARIO_CATALOG),
      "linear_std": math.sqrt(0.25),
      "angular_std": math.sqrt(0.5),
    },
  )
  return cfg


def unitree_g1_safe_velocity_privileged_3dgs_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reuse conservative locomotion fine-tuning for privileged local geometry."""
  cfg = unitree_g1_safe_velocity_blind_3dgs_ppo_runner_cfg()
  cfg.experiment_name = "g1_3dgs_safe_velocity_privileged"
  return cfg


def unitree_g1_safe_velocity_privileged_stop_gate_3dgs_env_cfg(
  play: bool = False,
  scenario_id: str = "both_open",
) -> ManagerBasedRlEnvCfg:
  """Gate a stable locomotion actor with privileged stop/clear geometry."""
  cfg = unitree_g1_safe_velocity_privileged_3dgs_env_cfg(
    play=play,
    scenario_id=scenario_id,
  )
  sensor_names = tuple(
    sensor.name
    for sensor in cfg.scene.sensors
    if sensor.name.startswith(SAFE_VELOCITY_LOCAL_GEOMETRY_SENSOR_PREFIX)
  )
  params = {
    "command_name": "twist",
    "sensor_names": sensor_names,
    "adapter_cfg": _safe_velocity_adapter_cfg(SAFE_VELOCITY_SCENARIO_CATALOG),
  }
  cfg.observations["actor"].terms["command"] = ObservationTermCfg(
    func=SafeVelocityPrivilegedStopCommand,
    params=params,
  )
  cfg.observations["critic"].terms["command"] = ObservationTermCfg(
    func=SafeVelocityPrivilegedStopCommand,
    params=params,
  )
  return cfg


def unitree_g1_safe_velocity_privileged_stop_gate_3dgs_ppo_runner_cfg() -> (
  RslRlOnPolicyRunnerCfg
):
  """Load the privileged warm-start actor for stop-gate evaluation only."""
  cfg = unitree_g1_safe_velocity_privileged_3dgs_ppo_runner_cfg()
  cfg.experiment_name = "g1_3dgs_safe_velocity_privileged_stop_gate"
  return cfg


def unitree_g1_safe_velocity_safety_teacher_3dgs_env_cfg(
  play: bool = False,
  scenario_id: str = "both_open",
) -> ManagerBasedRlEnvCfg:
  """Expose continuous collision-truth labels for the future visual student."""
  cfg = unitree_g1_safe_velocity_privileged_3dgs_env_cfg(
    play=play,
    scenario_id=scenario_id,
  )
  if play:
    _set_safe_velocity_play_command(cfg, scenario_id)
  sensor_names = tuple(
    sensor.name
    for sensor in cfg.scene.sensors
    if sensor.name.startswith(SAFE_VELOCITY_LOCAL_GEOMETRY_SENSOR_PREFIX)
  )
  teacher_cfg = SafeVelocitySafetyTeacherCfg()
  params = {
    "command_name": "twist",
    "sensor_names": sensor_names,
    "teacher_cfg": teacher_cfg,
  }
  cfg.observations["actor"].terms["command"] = ObservationTermCfg(
    func=SafeVelocityPrivilegedSafetyCommand,
    params={**params, "debug_print": play},
  )
  cfg.observations["critic"].terms["command"] = ObservationTermCfg(
    func=SafeVelocityPrivilegedSafetyCommand,
    params=params,
  )
  cfg.observations["safety_teacher"] = ObservationGroupCfg(
    terms={
      "labels": ObservationTermCfg(
        func=SafeVelocityPrivilegedSafetyLabels,
        params=params,
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.rewards["safe_velocity_tracking"] = RewardTermCfg(
    func=SafeVelocitySafetyTeacherTrackingReward,
    weight=2.0,
    params={
      **params,
      "linear_std": math.sqrt(0.25),
      "angular_std": math.sqrt(0.5),
    },
  )
  return cfg


def unitree_g1_safe_velocity_safety_teacher_3dgs_ppo_runner_cfg() -> (
  RslRlOnPolicyRunnerCfg
):
  """Keep the teacher task checkpoint-compatible with the privileged warm-start."""
  cfg = unitree_g1_safe_velocity_privileged_3dgs_ppo_runner_cfg()
  cfg.experiment_name = "g1_3dgs_safe_velocity_safety_teacher"
  return cfg


def unitree_g1_safe_velocity_vision_data_3dgs_env_cfg(
  play: bool = False,
  scenario_id: str = "both_open",
  *,
  isolate_teacher_rays: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Pair temporal 3DGS input with collision-truth safety supervision."""
  cfg = unitree_g1_safe_velocity_3dgs_env_cfg(
    play=play,
    scenario_id=scenario_id,
  )
  if play:
    _set_safe_velocity_play_command(cfg, scenario_id)
  ray_sensors = _safe_velocity_local_geometry_sensors(
    play=play, isolate_from_camera=isolate_teacher_rays
  )
  cfg.scene.sensors += ray_sensors
  params = {
    "command_name": "twist",
    "sensor_names": tuple(sensor.name for sensor in ray_sensors),
    "teacher_cfg": SafeVelocitySafetyTeacherCfg(),
  }
  cfg.observations["actor"].terms["command"] = ObservationTermCfg(
    func=SafeVelocityPrivilegedSafetyCommand,
    params={**params, "debug_print": play},
  )
  cfg.observations["critic"].terms["command"] = ObservationTermCfg(
    func=SafeVelocityPrivilegedSafetyCommand,
    params=params,
  )
  cfg.observations["safety_teacher"] = ObservationGroupCfg(
    terms={
      "labels": ObservationTermCfg(
        func=SafeVelocityPrivilegedSafetyLabels,
        params=params,
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.rewards["safe_velocity_tracking"] = RewardTermCfg(
    func=SafeVelocitySafetyTeacherTrackingReward,
    weight=2.0,
    params={
      **params,
      "linear_std": math.sqrt(0.25),
      "angular_std": math.sqrt(0.5),
    },
  )
  return cfg


def unitree_g1_safe_velocity_vision_data_3dgs_ppo_runner_cfg() -> (
  RslRlOnPolicyRunnerCfg
):
  """Keep temporal images and labels outside the frozen locomotion MLP."""
  cfg = unitree_g1_safe_velocity_blind_3dgs_ppo_runner_cfg()
  cfg.experiment_name = "g1_3dgs_safe_velocity_vision_data"
  return cfg


__all__ = [
  "SAFE_VELOCITY_ADAPTER_TASK_ID",
  "SAFE_VELOCITY_ALLOWED_CONTACT_GEOMS",
  "SAFE_VELOCITY_BLIND_TASK_ID",
  "SAFE_VELOCITY_COLLISION_SENSOR_PREFIX",
  "SAFE_VELOCITY_DEPTH_STOP_TASK_ID",
  "SAFE_VELOCITY_LOCAL_GEOMETRY_SENSOR_PREFIX",
  "SAFE_VELOCITY_MIN_PELVIS_HEIGHT_M",
  "SAFE_VELOCITY_NONFOOT_GROUND_SENSOR",
  "SAFE_VELOCITY_PRIVILEGED_TASK_ID",
  "SAFE_VELOCITY_PRIVILEGED_STOP_GATE_TASK_ID",
  "SAFE_VELOCITY_SAFETY_TEACHER_TASK_ID",
  "SAFE_VELOCITY_TASK_ID",
  "SAFE_VELOCITY_VISION_DATA_TASK_ID",
  "unitree_g1_safe_velocity_blind_3dgs_env_cfg",
  "unitree_g1_safe_velocity_blind_3dgs_ppo_runner_cfg",
  "unitree_g1_safe_velocity_3dgs_env_cfg",
  "unitree_g1_safe_velocity_3dgs_ppo_runner_cfg",
  "unitree_g1_safe_velocity_adapter_3dgs_env_cfg",
  "unitree_g1_safe_velocity_adapter_3dgs_ppo_runner_cfg",
  "unitree_g1_safe_velocity_depth_stop_3dgs_env_cfg",
  "unitree_g1_safe_velocity_depth_stop_3dgs_ppo_runner_cfg",
  "unitree_g1_safe_velocity_privileged_3dgs_env_cfg",
  "unitree_g1_safe_velocity_privileged_3dgs_ppo_runner_cfg",
  "unitree_g1_safe_velocity_privileged_stop_gate_3dgs_env_cfg",
  "unitree_g1_safe_velocity_privileged_stop_gate_3dgs_ppo_runner_cfg",
  "unitree_g1_safe_velocity_safety_teacher_3dgs_env_cfg",
  "unitree_g1_safe_velocity_safety_teacher_3dgs_ppo_runner_cfg",
  "unitree_g1_safe_velocity_vision_data_3dgs_env_cfg",
  "unitree_g1_safe_velocity_vision_data_3dgs_ppo_runner_cfg",
]
