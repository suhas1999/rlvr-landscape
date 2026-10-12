"""Parameter-space direction utilities (norms, scaling, saving, random directions)."""

import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def get_target_state(
    model,
):

    state = {}

    for name, param in (
        model.named_parameters()
    ):

        if param.requires_grad:

            state[name] = (
                param.detach()
                .cpu()
                .float()
                .clone()
            )

    return state


def global_norm(
    state,
):

    total = 0.0

    for tensor in state.values():

        total += (
            tensor.float()
            .pow(2)
            .sum()
            .item()
        )

    return math.sqrt(total)


def scale_to_state_norm(
    direction,
    target_norm,
):

    current_norm = global_norm(
        direction
    )

    if current_norm == 0:

        raise RuntimeError(
            "Direction has zero norm."
        )

    scale = (
        target_norm
        / current_norm
    )

    return {
        name:
            tensor * scale
        for name, tensor
        in direction.items()
    }


def save_direction(
    direction,
    path,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("Saving direction")
    print("=" * 80)

    print(
        "Path:",
        path,
    )

    print(
        "Direction norm:",
        global_norm(direction),
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Store CPU float32.
    #
    # The worker will read this file itself.
    #
    # We DO NOT pass this dictionary through apply_model().
    # --------------------------------------------------------

    cpu_direction = {}

    for name, tensor in direction.items():

        cpu_direction[name] = (
            tensor.detach()
            .cpu()
            .float()
            .contiguous()
        )

    torch.save(
        cpu_direction,
        path,
    )

    file_size_gb = (
        path.stat().st_size
        / (1024 ** 3)
    )

    print(
        f"Direction file size: "
        f"{file_size_gb:.3f} GB"
    )

    print(
        "Direction saved."
    )


def load_hf_model(
    args,
):

    print()
    print("=" * 80)
    print(
        "Loading HuggingFace model "
        "for GRPO backward"
    )
    print("=" * 80)

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_ckpt,

            torch_dtype=(
                torch.bfloat16
                if args.dtype == "bfloat16"
                else torch.float16
            ),

            trust_remote_code=True,

            # Single GPU
            device_map="cuda",
        )
    )

    # --------------------------------------------------------
    # Important for memory-efficient backward
    # --------------------------------------------------------

    model.gradient_checkpointing_enable()

    model.config.use_cache = False

    model.train()

    return model


def random_direction_like(
    theta_state,
    seed,
):

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed
    )

    direction = {}

    for name, tensor in (
        theta_state.items()
    ):

        direction[name] = torch.randn(
            tensor.shape,
            generator=generator,
            dtype=torch.float32,
        )

    norm = global_norm(
        direction
    )

    direction = {
        name:
            tensor / norm
        for name, tensor
        in direction.items()
    }

    direction = (
        scale_to_state_norm(
            direction,
            global_norm(theta_state),
        )
    )

    return direction
