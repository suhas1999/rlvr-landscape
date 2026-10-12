"""GRPO rollouts with vLLM and the policy-gradient direction computed with HF."""

import math
import random

import numpy as np
import torch
from tqdm import tqdm
from vllm import SamplingParams

from rlvr_landscape.directions import get_target_state, global_norm, scale_to_state_norm
from rlvr_landscape.tasks.math500 import build_prompt_text, math500_reward_func


def vllm_grpo_rollout(
    llm,
    tokenizer,
    data,
    args,
    direction_index,
):

    rng = random.Random(
        args.seed
        + 1000
        + direction_index
    )

    num_prompts = min(
        args.pg_num_prompts,
        len(data),
    )

    prompt_indices = rng.sample(
        range(len(data)),
        num_prompts,
    )

    selected = [
        data[i]
        for i in prompt_indices
    ]

    prompt_texts = [
        build_prompt_text(
            tokenizer,
            ex,
        )
        for ex in selected
    ]

    sampling_params = SamplingParams(
        n=args.group_size,

        temperature=args.temperature,

        top_p=args.top_p,

        top_k=args.top_k,

        max_tokens=args.max_new_tokens,

        seed=(
            args.seed
            + 1000
            + direction_index
        ),

        skip_special_tokens=True,
    )

    print()
    print("=" * 80)

    print(
        f"GRPO rollout for direction "
        f"{direction_index}"
    )

    print(
        f"Prompts: {num_prompts}"
    )

    print(
        f"Group size: "
        f"{args.group_size}"
    )

    print(
        f"Total completions: "
        f"{num_prompts * args.group_size}"
    )

    print("=" * 80)

    outputs = llm.generate(
        prompt_texts,
        sampling_params,
        use_tqdm=True,
    )

    rollout = []

    for group_id, (
        example,
        prompt_text,
        request_output,
    ) in enumerate(
        zip(
            selected,
            prompt_texts,
            outputs,
        )
    ):

        for completion_output in (
            request_output.outputs
        ):

            completion_text = (
                completion_output.text
            )

            completion_token_ids = list(
                completion_output.token_ids
            )

            reward = math500_reward_func(
                completion_text,
                example["answer"],
            )

            # Already rendered chat template.
            #
            # We use the HF tokenization for
            # the backward pass.
            prompt_token_ids = (
                tokenizer.encode(
                    prompt_text,
                    add_special_tokens=True,
                )
            )

            rollout.append(
                {
                    "group_id":
                        group_id,

                    "prompt":
                        prompt_text,

                    "prompt_token_ids":
                        prompt_token_ids,

                    "completion_token_ids":
                        completion_token_ids,

                    "completion_text":
                        completion_text,

                    "reward":
                        float(reward),
                }
            )

    rewards = [
        x["reward"]
        for x in rollout
    ]

    print(
        "Rollout mean reward:",
        np.mean(rewards),
    )

    return rollout


def compute_group_advantages(
    rollout,
    group_size,
):

    rewards_by_group = {}

    for item in rollout:

        gid = item["group_id"]

        if gid not in rewards_by_group:
            rewards_by_group[gid] = []

        rewards_by_group[gid].append(
            float(item["reward"])
        )

    for gid, rewards in (
        rewards_by_group.items()
    ):

        if len(rewards) != group_size:

            raise RuntimeError(
                f"Group {gid} has "
                f"{len(rewards)} rewards, "
                f"expected {group_size}"
            )

        rewards_tensor = torch.tensor(
            rewards,
            dtype=torch.float32,
        )

        mean = rewards_tensor.mean()

        std = rewards_tensor.std(
            unbiased=False
        )

        for item in rollout:

            if item["group_id"] != gid:
                continue

            reward = torch.tensor(
                float(item["reward"]),
                dtype=torch.float32,
            )

            advantage = (
                reward - mean
            ) / (
                std + 1e-4
            )

            item["advantage"] = float(
                advantage.item()
            )


def compute_response_log_probs(
    model,
    prompt_token_ids,
    completion_token_ids,
):

    input_ids = (
        prompt_token_ids
        +
        completion_token_ids
    )

    device = next(
        model.parameters()
    ).device

    input_tensor = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    attention_mask = torch.ones_like(
        input_tensor
    )

    outputs = model(
        input_ids=input_tensor,
        attention_mask=attention_mask,
    )

    logits = outputs.logits

    shift_logits = logits[:, :-1, :]

    shift_labels = input_tensor[:, 1:]

    log_probs = torch.log_softmax(
        shift_logits,
        dim=-1,
    )

    token_log_probs = (
        log_probs
        .gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    prompt_len = len(
        prompt_token_ids
    )

    completion_len = len(
        completion_token_ids
    )

    if completion_len == 0:

        return token_log_probs.new_empty(
            (0,)
        )

    start = prompt_len - 1

    end = (
        start
        + completion_len
    )

    response_log_probs = (
        token_log_probs[
            0,
            start:end
        ]
    )

    return response_log_probs


def compute_grpo_gradient(
    model,
    rollout,
    args,
    normalize_to_theta_norm=False,
):
    """
    Return the SGD direction d = -grad of the GRPO surrogate loss.

    normalize_to_theta_norm=False: raw -gradient (what compute_sgd_direction saves).
    normalize_to_theta_norm=True:  -gradient rescaled so ||d|| == ||theta||.
    """
    model.zero_grad(set_to_none=True)

    print()
    print("=" * 80)
    print("Computing GRPO gradient")
    print("=" * 80)

    # Count total completion tokens.
    total_tokens = sum(
        len(item["completion_token_ids"])
        for item in rollout
    )

    if total_tokens == 0:
        raise RuntimeError("No valid completion tokens.")

    print("Total completion tokens:", total_tokens)

    total_loss_value = 0.0

    for item in tqdm(
        rollout,
        desc="GRPO backward",
    ):
        if len(item["completion_token_ids"]) == 0:
            continue

        advantage = float(item["advantage"])

        if not math.isfinite(advantage):
            raise RuntimeError(
                f"Non-finite advantage: {advantage}"
            )

        log_probs = compute_response_log_probs(
            model,
            item["prompt_token_ids"],
            item["completion_token_ids"],
        )

        if log_probs.numel() == 0:
            continue

        # GRPO policy-gradient surrogate:
        # average over tokens within each completion,
        # then average over all completions.

        # loss = -sum(advantage * token_log_prob) / total_tokens

        loss_i = -(advantage * log_probs).sum() / total_tokens
        # loss_i = -advantage * log_probs.mean() / num_completions

        if not torch.isfinite(loss_i).item():
            raise RuntimeError(
                "Non-finite GRPO loss encountered."
            )

        loss_i.backward()

        total_loss_value += loss_i.detach().float().item()

        del log_probs
        del loss_i

    print("GRPO loss:", total_loss_value)

    # Raw SGD direction: d = -gradient.
    # No normalization and no scaling to parameter norm.
    direction = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.grad is None:
            direction[name] = torch.zeros_like(
                param.detach(),
                dtype=torch.float32,
                device="cpu",
            )
        else:
            direction[name] = (
                param.grad
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .neg()
            )

    # Diagnostic only; this does not modify the direction.
    grad_norm = global_norm(direction)

    print("Raw SGD direction norm:", grad_norm)

    model.zero_grad(set_to_none=True)

    if not math.isfinite(grad_norm):
        raise RuntimeError(
            f"GRPO gradient norm is non-finite: {grad_norm}"
        )

    if grad_norm == 0.0:
        raise RuntimeError("GRPO gradient is zero.")

    if normalize_to_theta_norm:
        theta_norm = global_norm(get_target_state(model))
        print("Theta norm:", theta_norm)
        direction = scale_to_state_norm(direction, theta_norm)
        print("Direction norm:", global_norm(direction))

    return direction
