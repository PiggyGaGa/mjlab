"""Prepare a locomotion-actor warm-start checkpoint for safe-velocity training.

The target critic and optimizer remain freshly initialized. For the privileged
baseline, the standard 99 actor inputs are copied and the 18 appended local-geometry
input columns start at zero.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.g1.safe_velocity import (
  SAFE_VELOCITY_BLIND_TASK_ID,
  SAFE_VELOCITY_PRIVILEGED_TASK_ID,
)

_BASELINE_TASKS = {
  "blind": SAFE_VELOCITY_BLIND_TASK_ID,
  "privileged": SAFE_VELOCITY_PRIVILEGED_TASK_ID,
}


def _expanded_actor_state(
  source: dict[str, torch.Tensor],
  target: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
  """Copy an actor state, expanding only its appended observation dimension."""
  expanded = {key: value.clone() for key, value in target.items()}
  for key, source_value in source.items():
    if key not in expanded:
      raise KeyError(f"Source actor key is absent from target: {key}")
    target_value = expanded[key]
    if source_value.shape == target_value.shape:
      expanded[key] = source_value.clone()
      continue
    if key == "mlp.0.weight":
      if source_value.shape[0] != target_value.shape[0]:
        raise ValueError("Actor first-layer output width changed")
      if source_value.shape[1] > target_value.shape[1]:
        raise ValueError("Target actor has fewer observation inputs than source")
      target_value.zero_()
      target_value[:, : source_value.shape[1]] = source_value
      continue
    if key in {
      "obs_normalizer._mean",
      "obs_normalizer._var",
      "obs_normalizer._std",
    }:
      if source_value.shape[0] != 1 or target_value.shape[0] != 1:
        raise ValueError(f"Unexpected normalizer shape for {key}")
      if source_value.shape[1] > target_value.shape[1]:
        raise ValueError("Target normalizer has fewer inputs than source")
      fill = 0.0 if key.endswith("_mean") else 1.0
      target_value.fill_(fill)
      target_value[:, : source_value.shape[1]] = source_value
      continue
    raise ValueError(
      f"Unsupported actor shape change for {key}: "
      f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
    )
  return expanded


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--baseline", choices=tuple(_BASELINE_TASKS), required=True)
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  source_path = args.source.expanduser().resolve()
  if not source_path.is_file():
    raise FileNotFoundError(f"Source checkpoint not found: {source_path}")
  source = torch.load(source_path, map_location=args.device, weights_only=False)
  source_actor = source.get("actor_state_dict")
  if not isinstance(source_actor, dict):
    raise ValueError("Source checkpoint has no actor_state_dict")

  task_id = _BASELINE_TASKS[args.baseline]
  env_cfg = load_env_cfg(task_id)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(
      wrapped,
      asdict(agent_cfg),
      log_dir=None,
      device=args.device,
    )
    checkpoint = runner.alg.save()
    target_actor = checkpoint["actor_state_dict"]
    checkpoint["actor_state_dict"] = _expanded_actor_state(
      source_actor,
      target_actor,
    )
    checkpoint["iter"] = 0
    checkpoint["infos"] = {
      "env_state": {"common_step_counter": 0},
      "warmstart": {
        "source": str(source_path),
        "baseline": args.baseline,
        "copied_actor_inputs": source_actor["mlp.0.weight"].shape[1],
        "target_actor_inputs": target_actor["mlp.0.weight"].shape[1],
        "critic": "fresh",
        "optimizer": "fresh",
      },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Wrote {args.baseline} warm-start checkpoint: {output}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
