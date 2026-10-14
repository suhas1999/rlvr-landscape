"""End-to-end GPU check for the landscape code on the installed vLLM.

Runs the real perturbation path (install_direction_on_vllm / set_vllm_coefficient)
on a real model and checks:

  1. vLLM generates and the MATH500 reward works.
  2. Setting alpha != 0 changes the model (prompt log-probs move).
  3. Setting alpha back to 0 restores the checkpoint exactly.
  4. CUDA-graph mode (RLVR_ENFORCE_EAGER=0) sees the in-place weight update
     the same way eager mode does, so graphs are safe to enable.

It also prints generation throughput so runs can be sized.

    python -m cloud.smoke_test --model /workspace/models/Qwen3-0.6B
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/models/Qwen3-0.6B")
    parser.add_argument(
        "--eval-json",
        default=str(REPO_ROOT / "train_data/math500/train_8.jsonl"),
    )
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--work-dir", default="/workspace/smoke_test")
    # Internal: run one engine mode in a child process.
    parser.add_argument("--mode", choices=["eager", "graph"], default=None)
    return parser.parse_args()


def make_direction(model_dir, path):
    """Random Gaussian direction scaled to ||theta||, HF parameter names."""
    from safetensors.torch import load_file

    theta = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        theta.update(load_file(str(shard)))

    generator = torch.Generator().manual_seed(0)
    direction = {
        name: torch.randn(t.shape, generator=generator, dtype=torch.float32)
        for name, t in theta.items()
    }
    theta_norm = torch.sqrt(sum((t.float() ** 2).sum() for t in theta.values()))
    dir_norm = torch.sqrt(sum((d ** 2).sum() for d in direction.values()))
    direction = {k: v * (theta_norm / dir_norm) for k, v in direction.items()}

    torch.save(direction, path)
    print(f"Direction: {len(direction)} tensors, ||d|| = ||theta|| = {theta_norm:.2f}")


def run_mode(args):
    """Child process: one vLLM engine, the full perturbation round trip."""
    os.environ["RLVR_ENFORCE_EAGER"] = "1" if args.mode == "eager" else "0"

    from vllm import SamplingParams

    from rlvr_landscape.perturb import (
        install_direction_on_vllm,
        set_vllm_coefficient,
    )
    from rlvr_landscape.tasks import load_eval_data
    from rlvr_landscape.tasks.math500 import build_prompt_text, math500_reward_func
    from rlvr_landscape.vllm_utils import create_vllm

    engine_args = argparse.Namespace(
        model_ckpt=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        dtype="bfloat16",
        seed=0,
    )

    t0 = time.time()
    llm = create_vllm(engine_args)
    startup_s = time.time() - t0

    tokenizer = llm.get_tokenizer()
    data = load_eval_data(args.eval_json, 8, "math500")
    prompts = [build_prompt_text(tokenizer, ex) for ex in data]

    # Fixed text scored with prompt_logprobs: a deterministic forward pass,
    # so it compares weights without sampling noise.
    scored = [p + "Let me solve this step by step." for p in prompts]
    score_params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)

    def prompt_logprobs():
        values = []
        for out in llm.generate(scored, score_params, use_tqdm=False):
            for token_id, entry in zip(out.prompt_token_ids[1:], out.prompt_logprobs[1:]):
                values.append(entry[token_id].logprob)
        return values

    greedy = SamplingParams(temperature=0, max_tokens=64)

    def greedy_texts():
        return [o.outputs[0].text for o in llm.generate(prompts, greedy, use_tqdm=False)]

    # 1. Sampling throughput + reward.
    sample_params = SamplingParams(
        n=args.group_size,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=args.max_new_tokens,
        seed=0,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sample_params, use_tqdm=False)
    gen_s = time.time() - t0
    gen_tokens = sum(len(c.token_ids) for o in outputs for c in o.outputs)
    rewards = [
        float(math500_reward_func(c.text, ex["answer"]))
        for o, ex in zip(outputs, data)
        for c in o.outputs
    ]

    # 2-3. Perturbation round trip.
    lp_base = prompt_logprobs()
    text_base = greedy_texts()

    install_direction_on_vllm(llm, args.direction_path)
    set_vllm_coefficient(llm, args.alpha)
    lp_alpha = prompt_logprobs()

    set_vllm_coefficient(llm, 0.0)
    lp_restored = prompt_logprobs()
    text_restored = greedy_texts()

    result = {
        "mode": args.mode,
        "startup_s": startup_s,
        "gen_s": gen_s,
        "gen_tokens": gen_tokens,
        "tokens_per_s": gen_tokens / gen_s,
        "mean_reward": sum(rewards) / len(rewards),
        "lp_base": lp_base,
        "lp_alpha": lp_alpha,
        "lp_restored": lp_restored,
        "greedy_restored_equal": text_base == text_restored,
    }
    Path(args.result_path).write_text(json.dumps(result))


def mean_abs_diff(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    args.direction_path = str(work_dir / "random_direction.pt")

    if args.mode is not None:
        args.result_path = str(work_dir / f"result_{args.mode}.json")
        run_mode(args)
        return

    import vllm

    print(f"vLLM {vllm.__version__}, torch {torch.__version__}, "
          f"GPU {torch.cuda.get_device_name(0)}")

    make_direction(args.model, args.direction_path)

    # Separate processes: a V1 engine does not reliably free GPU memory on del.
    results = {}
    for mode in ("eager", "graph"):
        print(f"\n===== {mode} =====", flush=True)
        subprocess.run(
            [sys.executable, "-m", "cloud.smoke_test", *sys.argv[1:], "--mode", mode],
            check=True,
            cwd=REPO_ROOT,
        )
        results[mode] = json.loads((work_dir / f"result_{mode}.json").read_text())

    eager, graph = results["eager"], results["graph"]
    effect = mean_abs_diff(eager["lp_base"], eager["lp_alpha"])
    d0 = mean_abs_diff(graph["lp_base"], eager["lp_base"])
    d1 = mean_abs_diff(graph["lp_alpha"], eager["lp_alpha"])
    checks = {
        "reward computed": 0.0 <= eager["mean_reward"] <= 1.0,
        "alpha changes the model (eager)": effect > 1e-3,
        "alpha=0 restores exactly (eager)": eager["lp_base"] == eager["lp_restored"]
        and eager["greedy_restored_equal"],
        "alpha=0 restores exactly (graph)": graph["lp_base"] == graph["lp_restored"]
        and graph["greedy_restored_equal"],
        # Compiled and eager kernels differ slightly even at alpha=0 (d0). If graphs
        # ignored the update, graph-vs-eager at alpha (d1) would be about `effect`.
        "graph sees the weight update": d1 < 0.2 * effect and d1 < 3 * d0 + 0.02 * effect,
    }

    print("\n" + "=" * 70)
    for mode in ("eager", "graph"):
        r = results[mode]
        print(f"{mode:6s} startup {r['startup_s']:6.1f}s   "
              f"{r['gen_tokens']} tokens in {r['gen_s']:.1f}s = {r['tokens_per_s']:,.0f} tok/s   "
              f"mean reward {r['mean_reward']:.3f}")
    print(f"perturbation effect |dlogp| = {effect:.4f}; graph vs eager: "
          f"alpha=0 {d0:.6f}, alpha={args.alpha} {d1:.6f}")
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 70)

    if not all(checks.values()):
        sys.exit(1)
    if checks["graph sees the weight update"]:
        print("CUDA graphs are safe: `export RLVR_ENFORCE_EAGER=0` (set in cloud/env.sh).")


if __name__ == "__main__":
    main()
