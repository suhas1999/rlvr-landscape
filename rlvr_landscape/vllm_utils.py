"""vLLM engine creation, CUDA cleanup and MATH500 reward evaluation."""

import gc
import os

import numpy as np
import torch
from vllm import LLM, SamplingParams

from rlvr_landscape.tasks.math500 import (
    build_prompt_text,
    extract_math500_answer,
    extract_math500_gold,
    math500_reward_func,
)


def create_vllm(args):

    print()
    print("=" * 80)
    print("Initializing vLLM")
    print("=" * 80)

    print(
        f"Model:              {args.model_ckpt}"
    )

    print(
        f"Tensor parallel:    {args.tensor_parallel_size}"
    )

    print(
        f"GPU memory util:    {args.gpu_memory_utilization}"
    )

    print(
        f"Max model length:   {args.max_model_len}"
    )

    llm = LLM(
        model=args.model_ckpt,

        # ====================================================
        # SINGLE GPU
        # ====================================================

        tensor_parallel_size=1,

        trust_remote_code=True,

        dtype=args.dtype,

        gpu_memory_utilization=(
            args.gpu_memory_utilization
        ),

        max_model_len=args.max_model_len,

        # Weights are updated in place (param.copy_), so CUDA graphs stay valid.
        # Eager stays the default until cloud/smoke_test.py has confirmed that
        # on the installed vLLM; RLVR_ENFORCE_EAGER=0 enables graphs (faster).
        enforce_eager=os.environ.get("RLVR_ENFORCE_EAGER", "1") != "0",

        enable_prefix_caching=False,

        seed=args.seed,
    )

    return llm


def apply_to_model(llm, fn):
    """Run fn(model) inside every vLLM worker; returns one result per worker.

    Use this instead of LLM.apply_model, which is broken in some V1 releases
    (vLLM 0.10.2: 'LLMEngine' object has no attribute 'model_executor').
    collective_rpc works across V1 releases. Needs
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 to send the closure to the engine process.
    """

    def call_on_worker(worker):
        return fn(worker.get_model())

    return llm.collective_rpc(call_on_worker)


def cleanup_cuda():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


def vllm_eval_reward(
    llm,
    tokenizer,
    data,
    args,
):
    """Evaluate vLLM and return reward statistics plus per-generation rows."""
    selected = data
    prompts = [
        build_prompt_text(tokenizer, ex)
        for ex in selected
    ]

    # sampling_params = SamplingParams(
    #     n=args.group_size,
    #     temperature=(
    #         args.temperature
    #         # if args.group_size > 1
    #         # else 0.0
    #     ),
    #     top_p=args.top_p,
    #     top_k=args.top_k,
    #     max_tokens=args.max_new_tokens,
    #     seed=args.seed,
    #     skip_special_tokens=True,
    # )

    sampling_params = SamplingParams(
        n=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        stop=[
            "<|im_end|>",
            "<|endoftext|>",
        ],
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )

    if len(outputs) != len(selected):
        raise RuntimeError(
            f"Expected {len(selected)} outputs, got {len(outputs)}"
        )

    reward_matrix = np.zeros(
        (len(selected), args.group_size),
        dtype=np.float64,
    )
    generation_records = []

    for index, (example, prompt, request_output) in enumerate(
        zip(selected, prompts, outputs)
    ):
        if len(request_output.outputs) != args.group_size:
            raise RuntimeError(
                f"Example {index}: expected {args.group_size} completions, "
                f"got {len(request_output.outputs)}"
            )

        sample_rewards = []

        for sample_idx, completion_output in enumerate(
            request_output.outputs
        ):
            completion = completion_output.text
            token_ids = list(completion_output.token_ids)
            finish_reason = completion_output.finish_reason

            reward = float(
                math500_reward_func(
                    completion,
                    example["answer"],
                )
            )

            reward_matrix[index, sample_idx] = reward
            sample_rewards.append(reward)

            generation_records.append(
                {
                    "prompt_index": index,
                    "group_index": sample_idx,
                    "group_sample": sample_idx + 1,
                    "completion_tokens": len(token_ids),
                    "raw_completion_tokens": len(token_ids),
                    "completion_characters": len(completion),
                    "ended_with_eos": finish_reason == "stop",
                    "likely_truncated": finish_reason == "length",
                    "finish_reason": finish_reason,
                    "reward": reward,
                }
            )

            if index < 3:
                prediction_text, extraction_method = (
                    extract_math500_answer(completion)
                )
                target_text = extract_math500_gold(
                    example["answer"]
                )

                print("\n" + "#" * 100)
                print(
                    f"[DEBUG EVAL EXAMPLE {index} / "
                    f"SAMPLE {sample_idx + 1}/{args.group_size}]"
                )
                print("#" * 100)

                print("\nPROMPT:\n" + "-" * 80)
                print(prompt)
                print("-" * 80)

                print(
                    "\nPROMPT TOKEN COUNT:",
                    len(
                        tokenizer.encode(
                            prompt,
                            add_special_tokens=True,
                        )
                    ),
                )

                print("\nGENERATED COMPLETION:\n" + "-" * 80)
                print(completion)
                print("-" * 80)

                print("\nCOMPLETION TOKEN COUNT:", len(token_ids))
                print("FINISH REASON:", finish_reason)
                print("EXTRACTION METHOD:", extraction_method)
                print("PREDICTION:", repr(prediction_text))
                print("GOLD:", repr(target_text))
                print("REWARD:", reward)
                print("#" * 100)

        if index < 3:
            print(f"[EXAMPLE {index} SUMMARY]")
            print(f"Rewards:       {sample_rewards}")
            print(f"Mean reward:   {np.mean(sample_rewards):.6f}")
            print(
                f"Pass@{args.group_size}:       "
                f"{int(any(x > 0.5 for x in sample_rewards))}"
            )

    flat_rewards = reward_matrix.reshape(-1)
    if flat_rewards.size == 0:
        raise RuntimeError("No rewards were computed.")

    # y value: mean reward over all M * N generations.
    mean_reward = float(reward_matrix.mean())

    # Per-prompt mean reward. Its mean is exactly equal to mean_reward
    # because every prompt has the same number of generations.
    prompt_mean_rewards = reward_matrix.mean(axis=1)

    if len(prompt_mean_rewards) >= 2:
        prompt_mean_std = float(
            prompt_mean_rewards.std(ddof=1)
        )
        prompt_mean_se = float(
            prompt_mean_std / np.sqrt(len(prompt_mean_rewards))
        )
    else:
        prompt_mean_std = 0.0
        prompt_mean_se = 0.0

    # Diagnostics based on generation slots.
    generation_means = reward_matrix.mean(axis=0)

    if args.group_size >= 2:
        generation_mean_std = float(
            generation_means.std(ddof=1)
        )
        std_error_generation_mean = float(
            generation_mean_std / np.sqrt(args.group_size)
        )

        prompt_variances = reward_matrix.var(
            axis=1,
            ddof=1,
        )
        prompt_within_group_rms_std = float(
            np.sqrt(prompt_variances.mean())
        )
    else:
        generation_mean_std = 0.0
        std_error_generation_mean = 0.0
        prompt_within_group_rms_std = 0.0

    # Naive generation-level dispersion and SE.
    if flat_rewards.size >= 2:
        reward_std = float(
            flat_rewards.std(ddof=1)
        )
        reward_se = float(
            reward_std / np.sqrt(flat_rewards.size)
        )
    else:
        reward_std = 0.0
        reward_se = 0.0

    generation_correct_mask = flat_rewards > 0.5
    num_correct_generations = int(
        generation_correct_mask.sum()
    )
    generation_accuracy = float(
        generation_correct_mask.mean()
    )

    pass_at_n = float(
        (reward_matrix > 0.5).any(axis=1).mean()
    )

    print("\n" + "=" * 100)
    print("EVALUATION RESULT")
    print("=" * 100)
    print(f"Examples:              {len(selected)}")
    print(f"Generations/example:   {args.group_size}")
    print(f"Total generations:     {flat_rewards.size}")
    print(f"Correct generations:   {num_correct_generations}")
    print(f"Generation accuracy:   {generation_accuracy:.6f}")
    print(f"Pass@{args.group_size}:                {pass_at_n:.6f}")
    print(f"Mean reward:           {mean_reward:.6f}")
    # print(f"Prompt-mean std:       {prompt_mean_std:.6f}")
    # print(f"Prompt-mean SE:        {prompt_mean_se:.6f}")
    print(f"Generation-mean std:   {generation_mean_std:.6f}")
    print(
        f"Generation-mean SE:    "
        f"{std_error_generation_mean:.6f}"
    )
    # print(
    #     f"Prompt within-group RMS std: "
    #     f"{prompt_within_group_rms_std:.6f}"
    # )
    # print(f"Reward std:            {reward_std:.6f}")
    # print(f"Reward SE:             {reward_se:.6f}")
    print("=" * 100)

    return (
        mean_reward,
        prompt_mean_std,
        prompt_mean_se,
        generation_mean_std,
        std_error_generation_mean,
        prompt_within_group_rms_std,
        reward_std,
        reward_se,
        generation_records,
    )
