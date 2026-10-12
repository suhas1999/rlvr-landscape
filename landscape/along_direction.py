#!/usr/bin/env python3
"""
MATH500 reward landscape along a single parameter-space direction, evaluated with vLLM.

    theta(alpha) = theta_ckpt + alpha * scale * d

--direction-type picks d:
  grpo-grad     SGD direction -grad of the GRPO loss. Pass --direction-path to use
                a file from directions/compute_sgd_direction.py; otherwise it is
                computed here from fresh rollouts and rescaled to ||theta||.
  displacement  theta_j - theta_i from directions/compute_displacement.py
                (--direction-path required). alpha = 1 is added to the grid so the
                sweep hits checkpoint j exactly when scale = 1.
  random        Gaussian direction rescaled to ||theta||, saved to --direction-dir.

The direction is loaded inside the vLLM worker and every alpha is rebuilt from an
fp32 copy of the checkpoint (see rlvr_landscape/perturb.py).

Launchers: scripts/landscape_sgd.sh, scripts/landscape_displacement.sh
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from rlvr_landscape.directions import (
    get_target_state,
    global_norm,
    load_hf_model,
    random_direction_like,
    save_direction,
)
from rlvr_landscape.grpo import compute_group_advantages, compute_grpo_gradient, vllm_grpo_rollout
from rlvr_landscape.perturb import install_direction_on_vllm, set_vllm_coefficient
from rlvr_landscape.tasks import load_eval_data
from rlvr_landscape.vllm_utils import cleanup_cuda, create_vllm, vllm_eval_reward


# ============================================================
# vLLM
# ============================================================


@torch.inference_mode()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # ========================================================
    # Model
    # ========================================================

    parser.add_argument(
        "--model-ckpt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dir-step",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--pg-num-prompts",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--task",
        choices=("math500",),
        default="math500",
    )

    parser.add_argument(
        "--direction-path",
        type=str,
        default=None,
        help=(
            "Existing SGD direction .pt file. "
            "If provided, skip GRPO gradient computation and load it directly."
        ),
    )

    parser.add_argument(
        "--eval-json",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen3_0.6b",
    )

    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
    )

    # ========================================================
    # vLLM
    # ========================================================

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=[
            "bfloat16",
            "float16",
        ],
    )

    # ========================================================
    # Reward landscape
    # ========================================================

    parser.add_argument(
        "--num-samples",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "Kept for compatibility. "
            "vLLM automatically batches prompts."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--alpha-range",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--alpha-left",
        type=float,
        default=10.0,
        help="Absolute alpha range on the left side. "
            "The actual left endpoint is -alpha-left.",
    )

    parser.add_argument(
        "--alpha-right",
        type=float,
        default=10.0,
        help="Absolute alpha range on the right side. "
            "The actual right endpoint is +alpha-right.",
    )

    parser.add_argument(
        "--num-points",
        type=int,
        default=15,
    )

    # ========================================================
    # IMPORTANT:
    #
    # One SGD rollout -> one SGD direction.
    # ========================================================

    parser.add_argument(
        "--num-directions",
        type=int,
        default=1,
    )

    # ========================================================
    # Direction
    # ========================================================

    parser.add_argument(
        "--direction-type",
        type=str,
        choices=[
            "random",
            "grpo-grad",
            "displacement",
        ],
        default="grpo-grad",
    )

    # ========================================================
    # GRPO
    # ========================================================

    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    # ========================================================
    # Output
    # ========================================================

    parser.add_argument(
        "--output-dir",
        type=str,
        default="figs_qwen_math500_sgd",
    )

    parser.add_argument(
        "--direction-dir",
        type=str,
        default=None,
        help=(
            "Directory for saved SGD direction files. "
            "Defaults to <output-dir>/directions."
        ),
    )

    args = parser.parse_args()

    if args.direction_type == "displacement" and args.direction_path is None:
        parser.error("--direction-type displacement requires --direction-path")

    # ========================================================
    # Environment
    # ========================================================

    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "false",
    )

    print("=" * 80)
    print("Configuration")
    print("=" * 80)

    for k, v in vars(args).items():

        print(
            f"{k}: {v}"
        )

    # ========================================================
    # Seed
    # ========================================================

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    # ========================================================
    # Check single GPU
    # ========================================================

    if args.tensor_parallel_size != 1:

        raise ValueError(
            "This version is configured "
            "for SINGLE-GPU vLLM. "
            "Use --tensor-parallel-size 1."
        )

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model_ckpt,
            trust_remote_code=True,
        )
    )

    # ========================================================
    # Data
    # ========================================================

    task = (
        args.task
        .lower()
        .strip()
    )

    data = load_eval_data(
        args.eval_json,
        args.num_samples,
        task,
    )

    print()
    print(
        f"Loaded {len(data)} "
        f"evaluation examples."
    )

    # ========================================================
    # Output directories
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.direction_dir is None:

        direction_dir = (
            output_dir
            / "directions"
        )

    else:

        direction_dir = Path(
            args.direction_dir
        )

    direction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # PHASE 1
    #
    # vLLM SINGLE GPU
    #
    # Generate GRPO rollout
    # ========================================================

    grpo_rollouts = []

    if (
        args.direction_type == "grpo-grad"
        and args.direction_path is None
    ):

        llm = create_vllm(
            args
        )

        try:

            for direction_idx in range(
                args.num_directions
            ):

                rollout = (
                    vllm_grpo_rollout(
                        llm,
                        tokenizer,
                        data,
                        args,
                        direction_idx,
                    )
                )

                compute_group_advantages(
                    rollout,
                    args.group_size,
                )

                grpo_rollouts.append(
                    rollout
                )

        finally:

            print(
                "Destroying vLLM "
                "after rollout..."
            )

            del llm

            cleanup_cuda()

    # ========================================================
    # PHASE 2
    #
    # HF model
    #
    # Compute SGD / GRPO direction
    # ========================================================

    directions = []
    direction_paths = []

    if args.direction_type in ("grpo-grad", "displacement"):

        # ========================================================
        # Existing direction: directly use it
        # ========================================================

        if args.direction_path is not None:

            direction_path = Path(
                args.direction_path
            )

            if not direction_path.exists():
                raise FileNotFoundError(
                    f"SGD or displacement direction file does not exist: "
                    f"{direction_path}"
                )

            print()
            print("=" * 80)
            print("Using existing SGD or displacement direction")
            print("=" * 80)
            print(f"Direction path: {direction_path}")

            direction_paths.append(
                str(direction_path.resolve())
            )

        # ========================================================
        # No existing direction: compute from GRPO gradient
        # ========================================================

        else:

            hf_model = load_hf_model(args)

            theta_state = get_target_state(
                hf_model
            )

            theta_norm = global_norm(
                theta_state
            )

            print()
            print(
                "Checkpoint parameter norm:",
                theta_norm,
            )

            for direction_idx, rollout in enumerate(
                grpo_rollouts
            ):

                print()
                print("=" * 80)

                print(
                    f"Computing SGD direction "
                    f"{direction_idx + 1}/"
                    f"{args.num_directions}"
                )

                print("=" * 80)

                direction = compute_grpo_gradient(
                    hf_model,
                    rollout,
                    args,
                    normalize_to_theta_norm=True,
                )

                step_name = (
                    f"step{args.checkpoint_step}"
                    if args.checkpoint_step is not None
                    else "checkpoint"
                )

                direction_path = (
                    direction_dir
                    / (
                        f"{args.model_name}_"
                        f"{step_name}_"
                        f"grpo_direction_"
                        f"{direction_idx}.pt"
                    )
                )

                save_direction(
                    direction,
                    direction_path,
                )

                direction_paths.append(
                    str(direction_path.resolve())
                )

                del direction

                cleanup_cuda()

            del theta_state
            del hf_model

            cleanup_cuda()


    # ========================================================
    # Random direction
    # ========================================================

    else:

        # Random direction requires checkpoint parameter shapes.
        hf_model = load_hf_model(args)

        theta_state = get_target_state(
            hf_model
        )

        for direction_idx in range(
            args.num_directions
        ):

            direction = random_direction_like(
                theta_state,
                args.seed + direction_idx,
            )

            step_name = (
                f"step{args.checkpoint_step}"
                if args.checkpoint_step is not None
                else "checkpoint"
            )

            direction_path = (
                direction_dir
                / (
                    f"{args.model_name}_"
                    f"{step_name}_"
                    f"random_direction_"
                    f"{direction_idx}.pt"
                )
            )

            save_direction(
                direction,
                direction_path,
            )

            direction_paths.append(
                str(direction_path.resolve())
            )

            del direction

            cleanup_cuda()

        del theta_state
        del hf_model

        cleanup_cuda()

    # ========================================================
    # IMPORTANT:
    #
    # We do NOT keep the direction tensors in memory.
    #
    # directions = []
    #
    # Everything will be loaded by vLLM worker from disk.
    # ========================================================

    directions = []

    print()
    print(
        "Saved direction files:"
    )

    for path in direction_paths:

        print(
            "  ",
            path
        )

    # ========================================================
    # PHASE 3
    #
    # Reload SAME checkpoint
    #
    # SINGLE GPU vLLM
    # ========================================================

    llm = create_vllm(
        args
    )

    # ========================================================
    # Alpha grid
    # ========================================================

    # alpha_values = np.linspace(
    #     -args.alpha_range,
    #     args.alpha_range,
    #     args.num_points,
    # )

    alpha_values = np.linspace(
        -args.alpha_left,
        args.alpha_right,
        args.num_points,
    )

    # Always evaluate the checkpoint itself; for a displacement also evaluate
    # alpha = 1, which is checkpoint j when scale = 1.
    extra_alphas = [0.0, 1.0] if args.direction_type == "displacement" else [0.0]
    alpha_values = np.unique(np.append(alpha_values, extra_alphas))


    # ========================================================
    # Results
    # ========================================================

    all_results = []

    # ========================================================
    # PHASE 4
    #
    # Direction file -> worker
    #
    # Parameter perturbation
    # + vLLM evaluation
    # ========================================================

    try:

        for direction_idx, direction_path in enumerate(
            direction_paths
        ):

            print()
            print("=" * 80)

            print(
                f"DIRECTION "
                f"{direction_idx + 1}/"
                f"{len(direction_paths)}"
            )

            print("=" * 80)

            print(
                "Direction file:",
                direction_path,
            )

            # ------------------------------------------------
            # Worker reads direction from disk.
            # ------------------------------------------------

            install_direction_on_vllm(
                llm,
                direction_path,
            )

            direction_results = []

            # ------------------------------------------------
            # Sweep alpha
            # ------------------------------------------------
            all_generation_records = []
            for alpha in alpha_values:

                coefficient = (
                    float(alpha)
                    * args.scale
                )

                print()
                print(
                    "-" * 70
                )

                print(
                    f"direction="
                    f"{direction_idx}"
                )

                print(
                    f"alpha="
                    f"{alpha:.6f}"
                )

                print(
                    f"coefficient="
                    f"{coefficient:.8f}"
                )

                # ------------------------------------------------
                # theta =
                #
                # theta_ckpt
                # +
                # coefficient * direction
                # ------------------------------------------------

                set_vllm_coefficient(
                    llm,
                    coefficient,
                )

                # ------------------------------------------------
                # Evaluate
                # ------------------------------------------------
                

                (
                    mean_reward,
                    prompt_mean_std,
                    prompt_mean_se,
                    generation_mean_std,
                    std_error_generation_mean,
                    prompt_within_group_rms_std,
                    reward_std,
                    reward_se,
                    generation_records,
                ) = vllm_eval_reward(llm, tokenizer, data, args)

                row = {
                    "direction": direction_idx,
                    "direction_path": direction_path,
                    "alpha": float(alpha),
                    "coefficient": float(coefficient),
                    "mean_reward": mean_reward,
                    "prompt_mean_std": prompt_mean_std,
                    "prompt_mean_se": prompt_mean_se,
                    "generation_mean_std": generation_mean_std,
                    "std_error_generation_mean": std_error_generation_mean,
                    "prompt_within_group_rms_std": prompt_within_group_rms_std,
                    "reward_std": reward_std,
                    "reward_se": reward_se,
                }
                direction_results.append(row)
                all_results.append(row)

                for record in generation_records:
                    all_generation_records.append(
                        {
                            "direction": direction_idx,
                            "direction_path": direction_path,
                            "alpha": float(alpha),
                            "coefficient": float(coefficient),
                            **record,
                        }
                    )

                print(f"Mean reward = {mean_reward:.6f}")
                # print(f"Prompt-mean std = {prompt_mean_std:.6f}")
                # print(f"Prompt-mean SE = {prompt_mean_se:.6f}")
                print(f"Generation-mean std = {generation_mean_std:.6f}")
                print(f"Generation-mean SE = {std_error_generation_mean:.6f}")
                # print(
                #     f"Prompt within-group RMS std = "
                #     f"{prompt_within_group_rms_std:.6f}"
                # )
                # print(f"Reward std = {reward_std:.6f}")
                # print(f"Reward SE = {reward_se:.6f}", flush=True)

            # ------------------------------------------------
            # Restore checkpoint
            # ------------------------------------------------

            print()
            print(
                "Restoring checkpoint "
                "parameters by setting "
                "coefficient to 0..."
            )

            set_vllm_coefficient(
                llm,
                0.0,
            )

            print(
                "Direction finished."
            )

    finally:

        print(
            "Destroying final "
            "vLLM instance..."
        )

        del llm

        cleanup_cuda()

    # ========================================================
    # Save CSV
    # ========================================================

    df = pd.DataFrame(all_results)
    generation_df = pd.DataFrame(all_generation_records)

    step_name = (
        f"stp{args.checkpoint_step}"
        if args.checkpoint_step is not None
        else "checkpoint"
    )
    stem = f"{args.model_name}_{step_name}_to{args.dir_step}_{args.direction_type}_grp{args.group_size}_tmp{args.temperature}_scale{args.scale:.5f}_alpha{args.alpha_left}_{args.alpha_right}_npts{args.num_points}_seed{args.seed}"

    csv_path = output_dir / f"{stem}.csv"
    generation_csv_path = output_dir / f"completion_lengths_{stem}.csv"
    df.to_csv(csv_path, index=False)

    if not generation_df.empty:
        generation_df = generation_df.sort_values(
            ["direction", "alpha", "prompt_index", "group_index"]
        )
    generation_df.to_csv(generation_csv_path, index=False)

    def plot_landscape(relative: bool, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))

        for direction_idx, subset in df.groupby("direction"):
            subset = subset.sort_values("coefficient")

            x = subset["coefficient"].to_numpy(dtype=float)
            y = subset["mean_reward"].to_numpy(dtype=float)
            se = subset["std_error_generation_mean"].to_numpy(dtype=float)

            if relative:
                # 使用该 direction 在 alpha=0 时实际测得的 reward。
                zero_rows = subset.loc[subset["alpha"] == 0.0]

                if len(zero_rows) != 1:
                    raise ValueError(
                        f"Direction {direction_idx}: 需要唯一的 alpha=0 结果"
                    )

                baseline = float(zero_rows["mean_reward"].iloc[0])

                if not np.isfinite(baseline) or baseline == 0.0:
                    raise ValueError(
                        f"Direction {direction_idx}: "
                        f"0 点 reward={baseline}，无法归一化"
                    )

                y = y / baseline
                se = se / abs(baseline)

            line = ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=f"Direction {direction_idx}",
            )[0]

            lower = y - se
            upper = y + se

            # 原始 reward 保留 [0, 1] 裁剪；
            # relative reward 可以超过 1，不能按 [0, 1] 裁剪。
            if not relative:
                lower = np.clip(lower, 0.0, 1.0)
                upper = np.clip(upper, 0.0, 1.0)

            ax.fill_between(
                x,
                lower,
                upper,
                alpha=0.2,
                color=line.get_color(),
            )

        ax.axvline(0.0, linestyle="--", linewidth=1, color="black")

        if relative:
            ax.axhline(1.0, linestyle=":", linewidth=1, color="gray")
            ax.set_ylabel("Relative Reward: R(coefficient) / R(0)")
            ax.set_title(f"Math500 Relative Reward Landscape ({args.direction_type}) step {args.checkpoint_step} gs {args.group_size}")
        else:
            ax.set_ylabel("Mean Math500 Reward")
            ax.set_title(f"Math500 Absolute Reward Landscape ({args.direction_type}) step {args.checkpoint_step} gs {args.group_size}")

        ax.set_ylim(0.0, 1.5)
        ax.set_xlabel("Perturbation coefficient")
        ax.minorticks_on()
        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.8)
        ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)

    plot_landscape(
        relative=False,
        filename=f"{stem}_absolute_reward_generation_mean_se.png",
    )

    plot_landscape(
        relative=True,
        filename=f"{stem}_relative_reward_generation_mean_se.png",
    )

    # plot_landscape(
    #     "prompt_mean_se",
    #     f"{stem}_prompt_mean_se.png",
    #     "prompt-level mean SE",
    # )

    # # Prompt-level 标准差。
    # plot_landscape(
    #     "prompt_mean_std",
    #     f"{stem}_prompt_mean_std.png",
    #     "prompt-level mean std",
    # )


    # # 不同 generation slot 的平均 reward 波动。
    # plot_landscape(
    #     "generation_mean_std",
    #     f"{stem}_generation_mean_std.png",
    #     "generation-mean std",
    # )

    # # 基于 generation slot 的标准误。
    # plot_landscape(
    #     "std_error_generation_mean",
    #     f"{stem}_generation_mean_se.png",
    #     "generation-mean SE",
    # )

    # # 同一道题内部，不同生成之间的典型波动。
    # plot_landscape(
    #     "prompt_within_group_rms_std",
    #     f"{stem}_prompt_rms.png",
    #     "within-prompt generation RMS std",
    # )

    # # 所有 generation reward 的总体标准差。
    # plot_landscape(
    #     "reward_std",
    #     f"{stem}_reward_std.png",
    #     "reward std",
    # )

    # # 假设所有 generation 独立时的 naive SE。
    # plot_landscape(
    #     "reward_se",
    #     f"{stem}_reward_se.png",
    #     "naive reward SE",
    # )

    print(f"Saved results to: {csv_path}")
    print(f"Saved generation records to: {generation_csv_path}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":

    main()