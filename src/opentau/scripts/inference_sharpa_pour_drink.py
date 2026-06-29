#!/usr/bin/env python
# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run inference with a fine-tuned OpenTau pi05 policy on the Sharpa pour-drink task.

This script loads a trained checkpoint and predicts an action chunk from either:

1. A dummy observation (default) -- useful for a quick smoke test of a checkpoint.
2. A real frame from the LeRobot dataset -- useful for sanity-checking predictions
   against the ground-truth action.

Example (dummy observation):
    python src/opentau/scripts/inference_sharpa_pour_drink.py \
        --checkpoint outputs/train/pi05_sharpa_pour_drink/20260101_120000/checkpoint-10000

Example (dataset frame):
    python src/opentau/scripts/inference_sharpa_pour_drink.py \
        --checkpoint outputs/train/pi05_sharpa_pour_drink/20260101_120000/checkpoint-10000 \
        --mode dataset \
        --dataset-root /mnt/nas/guchenyang/Data/Sharpa/lerobot_v30_pour_drink_0624 \
        --frame-index 100

The checkpoint directory must contain config.json and model.safetensors.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

import draccus

from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset_mixture
from opentau.policies.factory import get_policy_class
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    attempt_torch_compile,
    auto_torch_device,
    create_dummy_observation,
    init_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a trained checkpoint directory (contains config.json + model.safetensors).",
    )
    p.add_argument(
        "--mode",
        choices=["dummy", "dataset"],
        default="dummy",
        help="Observation source: 'dummy' zeros or a real 'dataset' frame.",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/nas/guchenyang/Data/Sharpa/lerobot_v30_pour_drink_0624"),
        help="Root of the LeRobot v3.0 dataset (only used when --mode dataset).",
    )
    p.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Global frame index in the dataset to use as the observation (only used when --mode dataset).",
    )
    p.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of inference passes to run for timing (default: 10).",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup passes before timing (default: 2).",
    )
    p.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Inference dtype (default: bfloat16).",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the policy's sample_actions method (CUDA only).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Random seed (default: 1000).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to write the predicted action chunk to.",
    )
    return p.parse_args()


def load_config_from_checkpoint(checkpoint_dir: Path) -> TrainPipelineConfig:
    """Load the training config that was saved alongside the checkpoint."""
    train_config_path = checkpoint_dir / "train_config.json"
    if not train_config_path.exists():
        raise FileNotFoundError(f"train_config.json not found in {checkpoint_dir}")

    with open(train_config_path) as f:
        cfg_dict = json.load(f)

    # Point the policy config at the local checkpoint so weights are loaded from disk.
    cfg_dict.setdefault("policy", {})
    cfg_dict["policy"]["pretrained_path"] = str(checkpoint_dir)

    return draccus.decode(TrainPipelineConfig, cfg_dict)


def build_dummy_observation(cfg: TrainPipelineConfig, device: torch.device, dtype: torch.dtype) -> dict:
    """Create a batch-1 dummy observation for a quick smoke test."""
    obs = create_dummy_observation(cfg, device, dtype=dtype)
    # Use a task description relevant to the pour-drink task.
    obs["prompt"] = ["Pour the drink from the bottle into the cup."]
    return obs


def build_dataset_observation(
    cfg: TrainPipelineConfig,
    dataset_root: Path,
    frame_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict, np.ndarray | None]:
    """Load a real frame from the dataset and convert it to policy observation format."""
    # Ensure the dataset mixture points at the requested local root.
    cfg.dataset_mixture.datasets[0].root = str(dataset_root)

    dataset = make_dataset_mixture(cfg)
    if frame_index < 0 or frame_index >= len(dataset):
        raise IndexError(
            f"frame_index {frame_index} is out of range for dataset of length {len(dataset)}"
        )

    sample = dataset[frame_index]
    ground_truth_action = sample.get("actions")
    if ground_truth_action is not None:
        ground_truth_action = ground_truth_action.cpu().to(torch.float32).numpy()

    observation = {}
    for i in range(cfg.num_cams):
        key = f"camera{i}"
        if key in sample:
            observation[key] = sample[key].unsqueeze(0).to(device=device, dtype=dtype)
        else:
            observation[key] = torch.zeros((1, 3, *cfg.resolution), dtype=dtype, device=device)

    observation["state"] = sample["state"].unsqueeze(0).to(device=device, dtype=dtype)
    observation["prompt"] = [sample["task"]] if isinstance(sample["task"], str) else sample["task"]
    observation["img_is_pad"] = sample.get(
        "img_is_pad", torch.zeros((1, cfg.num_cams), dtype=torch.bool)
    ).to(device=device)
    observation["action_is_pad"] = sample.get(
        "action_is_pad", torch.zeros((1, cfg.action_chunk), dtype=torch.bool)
    ).to(device=device)
    observation["dataset_index"] = torch.zeros((1,), dtype=torch.long, device=device)

    return observation, ground_truth_action


def main():
    args = parse_args()
    init_logging()

    if args.seed is not None:
        set_seed(args.seed)

    checkpoint_dir = args.checkpoint
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not (checkpoint_dir / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in checkpoint directory: {checkpoint_dir}")
    if not (checkpoint_dir / "model.safetensors").exists():
        raise FileNotFoundError(
            f"model.safetensors not found in checkpoint directory: {checkpoint_dir}"
        )

    device = auto_torch_device()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    logging.info(f"Loading config from {checkpoint_dir}")
    cfg = load_config_from_checkpoint(checkpoint_dir)
    logging.info(f"Policy type: {cfg.policy.type}")
    logging.info(f"Resolution: {cfg.resolution}, num_cams: {cfg.num_cams}")
    logging.info(f"max_state_dim: {cfg.max_state_dim}, max_action_dim: {cfg.max_action_dim}")
    logging.info(f"action_chunk: {cfg.action_chunk}")

    logging.info("Loading policy")
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(str(checkpoint_dir), config=cfg.policy)
    policy.to(device=device, dtype=dtype)
    policy.eval()

    if args.compile:
        policy.model.sample_actions = attempt_torch_compile(
            policy.model.sample_actions, device_hint=device
        )

    # Build observation.
    if args.mode == "dummy":
        observation = build_dummy_observation(cfg, device, dtype)
        ground_truth_action = None
    else:
        observation, ground_truth_action = build_dataset_observation(
            cfg,
            args.dataset_root,
            args.frame_index,
            device,
            dtype,
        )

    # Print observation keys/shapes.
    print("Observation keys:")
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype} on {value.device})")
        else:
            print(f"  {key}: {value}")

    policy.reset()

    with torch.inference_mode():
        # Warmup passes.
        for _ in range(args.warmup):
            _ = policy.sample_actions(observation)

        # Timed passes.
        times_ms = []
        for _ in range(args.n_runs):
            t0 = time.perf_counter()
            actions = policy.sample_actions(observation)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    actions_np = actions.to("cpu", torch.float32).numpy()
    print(f"\nPredicted action chunk shape: {actions_np.shape}")
    print(f"Predicted action chunk (first 3 dims, first 5 steps):")
    print(actions_np[0, :5, :3])

    if ground_truth_action is not None:
        print(f"\nGround-truth action chunk shape: {ground_truth_action.shape}")
        print(f"Ground-truth action chunk (first 3 dims, first 5 steps):")
        print(ground_truth_action[:5, :3])
        mse = np.mean((actions_np[0] - ground_truth_action) ** 2)
        print(f"MSE vs ground truth: {mse:.6f}")

    times_tensor = torch.tensor(times_ms)
    print(
        f"\nInference time (ms) over {args.n_runs} runs: "
        f"min={times_tensor.min().item():.2f}, "
        f"max={times_tensor.max().item():.2f}, "
        f"avg={times_tensor.mean().item():.2f}, "
        f"std={times_tensor.std().item():.2f}"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(checkpoint_dir),
            "mode": args.mode,
            "frame_index": args.frame_index if args.mode == "dataset" else None,
            "action_chunk": actions_np.tolist(),
            "shape": list(actions_np.shape),
            "timing_ms": {
                "min": float(times_tensor.min().item()),
                "max": float(times_tensor.max().item()),
                "mean": float(times_tensor.mean().item()),
                "std": float(times_tensor.std().item()),
            },
        }
        if ground_truth_action is not None:
            payload["ground_truth_action_chunk"] = ground_truth_action.tolist()
            payload["mse_vs_ground_truth"] = float(mse)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
