"""Acceptance telemetry must detect contact independently of camera rendering."""

from unittest.mock import MagicMock

import pytest
import torch
from run_g1_3dgs_safe_velocity_adapter import _collision_metrics

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.tasks.velocity.config.g1.safe_velocity import (
  SAFE_VELOCITY_COLLISION_SENSOR_PREFIX,
)


@pytest.mark.parametrize(
  ("found", "force", "expected_contact"),
  [(1, 0.0, True), (0, 6.0, True), (0, 0.0, False)],
)
def test_contact_metric_keeps_found_and_short_force_history(
  found, force, expected_contact
):
  sensor = MagicMock(spec=ContactSensor)
  sensor.cfg = ContactSensorCfg(
    name=SAFE_VELOCITY_COLLISION_SENSOR_PREFIX + "0",
    primary=ContactMatch(mode="body", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="hidden_wall"),
  )
  sensor.data.found = torch.tensor([[found]])
  sensor.data.force_history = torch.zeros(1, 1, 4, 3)
  sensor.data.force_history[0, 0, 2, 0] = force
  env = MagicMock(spec=ManagerBasedRlEnv)
  env.scene = MagicMock()
  env.scene.sensors = {sensor.cfg.name: sensor}
  env.physics_dt = 0.005
  events = []

  contact, peak, impulse = _collision_metrics(env, events)

  assert contact is expected_contact
  assert peak == force
  assert impulse == pytest.approx(force * 0.005)
  assert len(events) == int(expected_contact)
  if expected_contact:
    assert events[0]["obstacle_geom"] == "hidden_wall"
    assert events[0]["current_found"] == found
