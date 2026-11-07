import torch
from run_g1_3dgs_vision_smoke import check_depth


def test_check_depth_allows_full_frame_dropout() -> None:
  observation = torch.ones((2, 7, 2, 48, 64), dtype=torch.float32)
  observation[:, :, 1] = 0.0

  check_depth(observation, num_envs=2, channels=2, history_length=7)
