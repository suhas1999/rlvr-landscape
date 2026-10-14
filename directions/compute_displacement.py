#!/usr/bin/env python3
"""
Compute the parameter displacement theta_j - theta_i between two checkpoints.

Not normalized: theta_i + displacement reproduces theta_j exactly (checked below).
Feed the file to landscape/along_direction.py --direction-type displacement.

Example:
    python -m directions.compute_displacement \
        --ckpt-i .../checkpoint-10 --ckpt-j .../checkpoint-50 \
        --output param_displace/displacement_10_to_50.pt
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def load_state_dict(checkpoint: str) -> dict[str, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        device_map="cpu",
    )

    # state_dict rather than named_parameters: it also includes buffers.
    # Only floating-point tensors take part in the displacement.
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point()
    }

    del model
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-i", required=True, help="Start checkpoint (theta_i).")
    parser.add_argument("--ckpt-j", required=True, help="End checkpoint (theta_j).")
    parser.add_argument("--output", required=True, help="Output .pt path.")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint i: {args.ckpt_i}")
    state_i = load_state_dict(args.ckpt_i)

    print(f"Loading checkpoint j: {args.ckpt_j}")
    state_j = load_state_dict(args.ckpt_j)

    if state_i.keys() != state_j.keys():
        raise ValueError(
            f"Checkpoint structures differ.\n"
            f"Only in checkpoint i: {state_i.keys() - state_j.keys()}\n"
            f"Only in checkpoint j: {state_j.keys() - state_i.keys()}"
        )

    displacement = {}

    for name in state_i:
        if state_i[name].shape != state_j[name].shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"{state_i[name].shape} vs {state_j[name].shape}"
            )

        displacement[name] = state_j[name] - state_i[name]

    # Check: checkpoint i + displacement == checkpoint j
    max_error = max(
        (state_i[name] + displacement[name] - state_j[name]).abs().max().item()
        for name in displacement
    )
    print(f"Maximum reconstruction error: {max_error:.8e}")

    torch.save(displacement, output)
    print(f"Saved displacement to {output}")


if __name__ == "__main__":
    main()
