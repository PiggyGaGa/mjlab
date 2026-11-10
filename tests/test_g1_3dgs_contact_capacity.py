"""Keep contact headroom local to G1 3DGS play configurations."""

import pytest

from mjlab.tasks.g1_3dgs import apply_g1_3dgs_play_cfg
from mjlab.tasks.registry import load_env_cfg


@pytest.mark.parametrize(
  "task,plain_play_capacity", [("Tracking", 35), ("Velocity", None)]
)
def test_3dgs_play_has_contact_headroom_without_changing_training(
  task: str, plain_play_capacity: int | None
) -> None:
  task_id = f"Mjlab-{task}-Flat-Unitree-G1"
  baseline = load_env_cfg(task_id, play=True)
  training = load_env_cfg(task_id + "-3DGS", play=False)
  play = load_env_cfg(task_id + "-3DGS", play=True)

  assert baseline.sim.nconmax == plain_play_capacity
  assert training.sim.nconmax == load_env_cfg(task_id, play=False).sim.nconmax
  assert play.sim.nconmax == 256


def test_3dgs_play_preserves_larger_contact_allocation() -> None:
  cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=True)
  cfg.sim.nconmax = 512

  apply_g1_3dgs_play_cfg(cfg)

  assert cfg.sim.nconmax == 512
