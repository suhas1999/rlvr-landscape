"""2-D reward landscape with filter-normalized directions (Li et al. 2018).

    R(alpha, beta) = mean reward of theta_canonical + alpha*d1 + beta*d2

theta is the fp32, canonicalized checkpoint (rlvr_landscape/filter_norm.py); d1 and d2 are
filter-normalized Gaussian directions with different seeds. Each grid point samples
group_size completions per prompt, num_repeats times (different sampling seeds), and
stores one row per repeat. Rows are appended to the CSV as they finish, so a killed run
resumes where it stopped.

    PYTHONPATH=. python landscape/2d_perturb.py --model-ckpt $BASE_MODEL --output-dir results/landscape2d
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from landscape.plot_2d import plot  # noqa: E402
from rlvr_landscape.filter_norm import (  # noqa: E402
    canonicalize,
    check_function_preserved,
    filter_normalized_direction,
    install_plane,
    load_fp32_params,
    set_plane_point,
)
from rlvr_landscape.tasks import load_eval_data  # noqa: E402
from rlvr_landscape.tasks.math500 import build_prompt_text, math500_reward_func  # noqa: E402
from rlvr_landscape.tracking import finish_run, init_run, log_point  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-ckpt", required=True)
    p.add_argument("--model-name", default="qwen3_0.6b")
    p.add_argument("--eval-json", default=str(REPO_ROOT / "train_data/math500/test.jsonl"))
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--prompt-seed", type=int, default=0)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--num-repeats", type=int, default=2)
    p.add_argument("--sample-seed", type=int, default=1234)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--grid-range", type=float, default=1.0)
    p.add_argument("--grid-points", type=int, default=11)
    p.add_argument("--dir-seeds", type=int, nargs=2, default=[0, 1])
    p.add_argument("--no-canonicalize", action="store_true")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--output-dir", default="results/landscape2d")
    p.add_argument("--work-dir", default="/workspace/landscape2d_tmp",
                   help="Large .pt files (base + directions); not results.")
    return p.parse_args()


def grid_order(values):
    """All (alpha, beta) pairs, centre first, then by distance from the centre."""
    pairs = [(a, b) for a in values for b in values]
    return sorted(pairs, key=lambda ab: (round(ab[0] ** 2 + ab[1] ** 2, 9), ab))


def prompt_logprobs(llm, texts):
    params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    values = []
    for out in llm.generate(texts, params, use_tqdm=False):
        for token_id, entry in zip(out.prompt_token_ids[1:], out.prompt_logprobs[1:]):
            values.append(entry[token_id].logprob)
    return np.array(values)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    canon_tag = "raw" if args.no_canonicalize else "canon"
    stem = (f"{args.model_name}_reward2d_{canon_tag}_{args.dtype}_seeds{args.dir_seeds[0]}-"
            f"{args.dir_seeds[1]}_r{args.grid_range}_n{args.grid_points}_p{args.num_prompts}"
            f"_g{args.group_size}x{args.num_repeats}_tok{args.max_new_tokens}")
    csv_path = out_dir / f"{stem}.csv"
    (out_dir / f"{stem}_config.json").write_text(json.dumps(vars(args), indent=2))

    # ---------------- data ----------------
    data = load_eval_data(args.eval_json, 10**9, "math500")
    data = random.Random(args.prompt_seed).sample(data, args.num_prompts)

    # ---------------- base + directions (fp32, CPU) ----------------
    base_path = work_dir / f"base_{canon_tag}.pt"
    dir_paths = [work_dir / f"dir_{canon_tag}_seed{s}.pt" for s in args.dir_seeds]
    if not (base_path.exists() and all(p.exists() for p in dir_paths)):
        params, config = load_fp32_params(args.model_ckpt)
        base = params if args.no_canonicalize else canonicalize(params, config)

        if not args.no_canonicalize:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.model_ckpt)
            token_lists = [tok(build_prompt_text(tok, ex))["input_ids"] for ex in data[:8]]
            before, after = check_function_preserved(args.model_ckpt, base, token_lists)
            print(f"[canonicalize] NLL before {before:.7f} after {after:.7f} "
                  f"|diff| {abs(before - after):.2e}", flush=True)
            if abs(before - after) > 1e-4:
                raise RuntimeError("Canonicalization changed the model output: bug.")

        torch.save(base, base_path)
        for seed, path in zip(args.dir_seeds, dir_paths):
            torch.save(filter_normalized_direction(base, seed), path)
        del params, base

    # ---------------- vLLM ----------------
    llm = LLM(
        model=args.model_ckpt,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_new_tokens + 1024,
        enable_prefix_caching=False,  # weights change in place
        seed=args.sample_seed,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [build_prompt_text(tokenizer, ex) for ex in data]

    # Mapping check: canonical base at (0, 0) must give the original model's outputs.
    lp_original = prompt_logprobs(llm, prompts[:8])
    print("Installed:", install_plane(llm, str(base_path), [str(p) for p in dir_paths]))
    set_plane_point(llm, [0.0, 0.0])
    drift = float(np.abs(prompt_logprobs(llm, prompts[:8]) - lp_original).max())
    print(f"[mapping] max |dlogp| original vs installed base: {drift:.2e}", flush=True)
    if drift > 1e-2:
        raise RuntimeError("Installed base does not reproduce the checkpoint: mapping bug.")

    # ---------------- grid ----------------
    values = np.round(np.linspace(-args.grid_range, args.grid_range, args.grid_points), 6)
    done = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        done = {(round(a, 6), round(b, 6)) for a, b in zip(prev["alpha"], prev["beta"])}
        print(f"Resuming: {len(done)} points already in {csv_path}")

    wandb_run = init_run(args, job_type="landscape-2d", name=stem)
    sampling = SamplingParams(
        n=args.group_size * args.num_repeats,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_tokens=args.max_new_tokens, seed=args.sample_seed,
    )

    todo = [ab for ab in grid_order(values) if (round(ab[0], 6), round(ab[1], 6)) not in done]
    t_start = time.time()
    for k, (alpha, beta) in enumerate(todo):
        t0 = time.time()
        set_plane_point(llm, [alpha, beta])
        outputs = llm.generate(prompts, sampling, use_tqdm=False)

        rows = []
        for r in range(args.num_repeats):
            rewards, lengths, truncated = [], [], 0
            for ex, out in zip(data, outputs):
                for c in out.outputs[r * args.group_size:(r + 1) * args.group_size]:
                    rewards.append(float(math500_reward_func(c.text, ex["answer"])))
                    lengths.append(len(c.token_ids))
                    truncated += c.finish_reason == "length"
            rows.append({
                "alpha": alpha, "beta": beta, "repeat": r,
                "reward": float(np.mean(rewards)),
                "mean_tokens": float(np.mean(lengths)),
                "truncated_frac": truncated / len(lengths),
                "n": len(rewards),
            })
        pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

        mean_reward = float(np.mean([row["reward"] for row in rows]))
        dt = time.time() - t0
        eta = (time.time() - t_start) / (k + 1) * (len(todo) - k - 1)
        print(f"[{k + 1}/{len(todo)}] alpha={alpha:+.2f} beta={beta:+.2f} "
              f"reward={mean_reward:.3f} tokens={rows[0]['mean_tokens']:.0f} "
              f"trunc={rows[0]['truncated_frac']:.2f} {dt:.0f}s  ETA {eta / 60:.0f} min",
              flush=True)
        log_point(wandb_run, {"alpha": alpha, "beta": beta, "reward": mean_reward}, "2d")

    plot(csv_path, title=f"{args.model_name} MATH500 reward landscape ({canon_tag})")
    finish_run(wandb_run, out_dir.glob(f"{stem}*"), results_df=pd.read_csv(csv_path))
    print("Done:", csv_path)


if __name__ == "__main__":
    main()
