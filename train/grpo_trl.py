#!/usr/bin/env python3
"""
GRPO training on MATH500 with TRL. Launch through train/grpo_trl.sh, which places vLLM
rollouts and training on separate GPUs (or shares one GPU) and starts this script with
accelerate.

Defaults follow the ms-swift runs of train/grpo_math500.sh: loss_type grpo, beta 0.04,
lr 5e-6 constant with 5% warmup, T=0.7 / top_p 0.8 / top_k 20, 2048 completion tokens,
one prompt x GROUP_SIZE completions per optimizer step, non-thinking chat template.

Reward and prompt are the ones the landscape tools use (rlvr_landscape.tasks.math500), so
training reward and landscape reward are the same function. Every rollout is saved to
    <output-dir>/rollouts/<split>_step<N>_rank<R>.jsonl.gz
where N is the number of optimizer steps already taken: train_step00010 holds the
completions sampled from the weights saved as checkpoint-10.

Options this script does not define are GRPOConfig fields and override the defaults below,
e.g. --beta 0 --loss_type dapo --save_steps 1 --report_to none.
"""

import argparse
import gzip
import json
import netrc
import os
from pathlib import Path

from datasets import Dataset
from transformers import HfArgumentParser
from trl import GRPOConfig, GRPOTrainer

from rlvr_landscape.tasks import load_eval_data
from rlvr_landscape.tasks.math500 import add_answer_instruction, extract_math500_answer, math500_reward_func


def build_dataset(path, limit, split):
    examples = load_eval_data(path, limit, "math500")
    return Dataset.from_list([
        {
            # The dataset's user message as-is (it already asks for \boxed{}), as ms-swift used it.
            # TRL applies the chat template itself.
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "answer": ex["answer"],
            "problem_index": i,
            "split": split,
        }
        for i, ex in enumerate(examples)
    ])


class Math500Reward:
    """math500_reward_func per completion; also appends every rollout to a gzipped JSONL."""

    __name__ = "math500_acc"

    def __init__(self, rollout_dir):
        self.rollout_dir = rollout_dir
        self.rank = int(os.environ.get("RANK", "0"))
        if rollout_dir is not None:
            rollout_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, prompts, completions, answer, problem_index, split, trainer_state, **_):
        texts = [completion[0]["content"] for completion in completions]
        rewards = [math500_reward_func(text, gold) for text, gold in zip(texts, answer)]

        if self.rollout_dir is not None:
            step = trainer_state.global_step
            path = self.rollout_dir / f"{split[0]}_step{step:05d}_rank{self.rank}.jsonl.gz"
            # Append mode: eval calls the reward once per eval batch at the same step.
            with gzip.open(path, "at", encoding="utf-8") as f:
                for text, gold, reward, index in zip(texts, answer, rewards, problem_index):
                    f.write(json.dumps({
                        "step": step,
                        "problem_index": index,
                        "reward": reward,
                        "prediction": extract_math500_answer(text)[0],
                        "answer": gold,
                        "completion": text,
                    }) + "\n")

        return rewards


def wandb_available():
    if os.environ.get("WANDB_MODE") == "disabled":
        return False
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return netrc.netrc().authenticators("api.wandb.ai") is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        return False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--train-limit", type=int, default=10**9)
    parser.add_argument("--val-data", default=None, help="Eval set; omit to skip evaluation.")
    parser.add_argument("--val-limit", type=int, default=10**9)
    parser.add_argument("--group-size", type=int, default=4, help="Completions per prompt (num_generations).")
    parser.add_argument("--prompts-per-step", type=int, default=1,
                        help="Prompts per optimizer step; the batch is prompts_per_step * group_size completions.")
    parser.add_argument("--micro-batch-size", type=int, default=None,
                        help="Completions per GPU per forward/backward. Default: the whole per-GPU share, up to 8.")
    parser.add_argument("--no-save-rollouts", action="store_true")
    args, rest = parser.parse_known_args()

    # Split the batch over the training processes; accumulate the rest.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    completions_per_step = args.prompts_per_step * args.group_size
    if completions_per_step % world_size:
        raise ValueError(
            f"prompts_per_step * group_size = {completions_per_step} completions per step "
            f"does not split over {world_size} training GPUs."
        )
    per_gpu = completions_per_step // world_size
    micro = args.micro_batch_size or max(d for d in range(1, min(per_gpu, 8) + 1) if per_gpu % d == 0)
    if per_gpu % micro:
        raise ValueError(f"micro_batch_size {micro} does not divide the {per_gpu} completions per GPU.")

    config_parser = HfArgumentParser(GRPOConfig)
    config_parser.set_defaults(
        output_dir="output_grpo_trl",
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=per_gpu // micro,
        num_generations=args.group_size,
        num_generations_eval=2,
        per_device_eval_batch_size=8,
        max_steps=80,
        learning_rate=5e-6,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=0.05,  # < 1: fraction of max_steps
        beta=0.04,
        loss_type="grpo",
        scale_rewards="group",
        num_iterations=1,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_completion_length=2048,
        chat_template_kwargs={"enable_thinking": False},
        use_vllm=True,
        vllm_mode="server",
        vllm_server_timeout=900.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=1,
        eval_strategy="steps" if args.val_data else "no",
        eval_steps=10,
        # Checkpoints stay fp32: lr 5e-6 updates are below bf16 resolution for most weights,
        # and compute_displacement needs the exact difference between checkpoints.
        save_strategy="steps",
        save_steps=10,
        save_only_model=True,
        report_to="wandb" if wandb_available() else "none",
        seed=42,
    )
    (config,) = config_parser.parse_args_into_dataclasses(args=rest)
    return args, config


def main():
    args, config = parse_args()

    train_dataset = build_dataset(args.train_data, args.train_limit, "train")
    eval_dataset = build_dataset(args.val_data, args.val_limit, "val") if args.val_data else None
    rollout_dir = None if args.no_save_rollouts else Path(config.output_dir) / "rollouts"

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=Math500Reward(rollout_dir),
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if trainer.accelerator.is_main_process:
        print(f"train prompts: {len(train_dataset)} | "
              f"completions per step: {args.prompts_per_step * args.group_size} | "
              f"per GPU: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} accum | "
              f"vLLM: {config.vllm_mode if config.use_vllm else 'off'}")

    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
