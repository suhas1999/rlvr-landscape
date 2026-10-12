#!/usr/bin/env python3
"""
Compute and save a raw SGD direction d = -grad of the GRPO loss at a checkpoint.

1. vLLM samples group_size completions per prompt and computes group advantages.
2. An HF copy of the checkpoint backpropagates the GRPO surrogate loss.
3. -gradient is saved (not normalized) as
   <direction-dir>/<model-name>_step<N>_sgd_raw_direction_<i>_gs<G>.pt

Feed the file to landscape/along_direction.py --direction-type grpo-grad --direction-path ...
Launcher: scripts/compute_sgd_direction.sh
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from rlvr_landscape.directions import load_hf_model, save_direction
from rlvr_landscape.grpo import compute_group_advantages, compute_grpo_gradient, vllm_grpo_rollout
from rlvr_landscape.tasks import load_eval_data
from rlvr_landscape.vllm_utils import cleanup_cuda, create_vllm


# env CUDA_VISIBLE_DEVICES=0 nohup python sgd_direction.py > sgd_gs32_stp110_train1.log 2>&1 &


# ============================================================
# Save direction
# ============================================================


 

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Model
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--model-name", default="qwen3_0.6b")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="Only used for the output file name.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    # Data
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--task", default="math500", choices=["math500"])
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--pg-num-prompts", type=int, default=1)
    # vLLM
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--max-model-len", type=int, default=16384)
    # GRPO rollout
    parser.add_argument("--num-directions", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    # Output
    parser.add_argument("--direction-dir", default="./sgd_directions")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.tensor_parallel_size != 1:
        raise ValueError("This script requires tensor_parallel_size=1.")

    if args.num_directions < 1:
        raise ValueError("num_directions must be >= 1.")

    if args.group_size < 2:
        raise ValueError("group_size must be >= 2.")

    if args.num_samples < 1 or args.pg_num_prompts < 1:
        raise ValueError("num_samples and pg_num_prompts must be >= 1.")

    print("=" * 80)
    print("Configuration")
    print("=" * 80)

    for name, value in vars(args).items():
        print(f"{name}: {value}")

    # ========================================================
    # Tokenizer 和数据
    # ========================================================
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_ckpt,
        trust_remote_code=True,
    )

    data = load_eval_data(
        args.eval_json,
        args.num_samples,
        args.task,
    )

    if len(data) == 0:
        raise RuntimeError("No examples loaded.")

    print(f"\nLoaded {len(data)} examples.")

    direction_dir = Path(args.direction_dir)
    direction_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 1. 使用原始 checkpoint 生成 rollout 并计算 advantage
    # ========================================================
    grpo_rollouts = []
    llm = create_vllm(args)

    try:
        for direction_idx in range(args.num_directions):
            rollout = vllm_grpo_rollout(
                llm,
                tokenizer,
                data,
                args,
                direction_idx,
            )

            compute_group_advantages(
                rollout,
                args.group_size,
            )

            active_groups = {
                item["group_id"]
                for item in rollout
                if item["advantage"] != 0.0
                and len(item["completion_token_ids"]) > 0
            }

            print(
                f"Direction {direction_idx}: "
                f"{len(active_groups)} groups with usable "
                "nonzero advantages."
            )

            if not active_groups:
                raise RuntimeError(
                    f"Direction {direction_idx} has no usable "
                    "nonzero advantages. Increase pg_num_prompts "
                    "or group_size and try again."
                )

            grpo_rollouts.append(rollout)

    finally:
        print("\nDestroying vLLM after rollout...")
        del llm
        cleanup_cuda()

    # ========================================================
    # 2. 计算并保存原始 SGD 方向：direction = -gradient
    #    不归一化，不更新 checkpoint 参数
    # ========================================================
    hf_model = load_hf_model(args)
    direction_paths = []

    try:
        # 关闭 dropout，但仍然允许计算梯度
        hf_model.eval()

        step_name = (
            f"step{args.checkpoint_step}"
            if args.checkpoint_step is not None
            else "checkpoint"
        )

        for direction_idx, rollout in enumerate(grpo_rollouts):
            print()
            print("=" * 80)
            print(
                f"Computing raw SGD direction "
                f"{direction_idx + 1}/{args.num_directions}"
            )
            print("=" * 80)

            # 使用前面提供的无归一化版本
            direction = compute_grpo_gradient(
                hf_model,
                rollout,
                args,
            )

            direction_path = direction_dir / (
                f"{args.model_name}_"
                f"{step_name}_"
                f"sgd_raw_direction_{direction_idx}_gs{args.group_size}.pt"
            )

            try:
                save_direction(
                    direction,
                    direction_path,
                )
            finally:
                del direction

            direction_paths.append(direction_path.resolve())
            cleanup_cuda()

    finally:
        del hf_model
        cleanup_cuda()

    print()
    print("=" * 80)
    print("Saved raw SGD direction files:")

    for path in direction_paths:
        print(f"  {path}")

    print("=" * 80)


if __name__ == "__main__":
    main()