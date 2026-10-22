"""2-D reward landscape along two random directions (Li et al. 2018), five variants.

    R(alpha, beta) = mean reward of theta + alpha*d1 + beta*d2

--variant picks how theta and the directions are built (rlvr_landscape/filter_norm.py):

    canon_qkn   canonicalized theta, filter-normalized, q_norm/k_norm perturbed (per RoPE pair)
    canon       canonicalized theta, filter-normalized, no norm perturbed
    raw_all     raw theta, filter-normalized, every norm perturbed (q/k per pair, rest per element)
    raw_nonqk   raw theta, filter-normalized, every norm except q_norm/k_norm perturbed
    raw_global  raw theta, one Gaussian over all params with ||d|| = ||theta|| (no normalization)
    raw_global_nonqk  raw_global with q_norm/k_norm entries zeroed (same draw otherwise)

--grid-range 0 (default) picks the range automatically: probe (alpha, 0) for growing alpha
until the reward falls below --probe-threshold x the centre reward, then use
--range-factor x that alpha. The choice is saved, so a resumed run reuses it.

Each grid point samples group_size completions per prompt, num_repeats times, and stores one
row per repeat. Rows are appended to the CSV as they finish, so a killed run resumes where it
stopped. Completions are saved per point to <stem>_rollouts/a<alpha>_b<beta>.jsonl.gz.

    PYTHONPATH=. python landscape/2d_perturb.py --model-ckpt $BASE_MODEL --variant canon_qkn
"""

import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Building directions uses torch CPU thread pools; a forked vLLM engine can then deadlock on
# their locks (seen with the raw variants, which never touch CUDA before vLLM starts).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

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
    global_direction,
    install_plane,
    load_fp32_params,
    set_plane_point,
)
from rlvr_landscape.tasks import load_eval_data  # noqa: E402
from rlvr_landscape.tasks.math500 import build_prompt_text, math500_reward_func  # noqa: E402
from rlvr_landscape.tracking import finish_run, init_run, log_point  # noqa: E402

# variant -> (canonicalize, direction, perturbed norms)
VARIANTS = {
    "canon_qkn": (True, "filter", "qk"),
    "canon": (True, "filter", "none"),
    "raw_all": (False, "filter", "all"),
    "raw_nonqk": (False, "filter", "nonqk"),
    "raw_global": (False, "global", "all"),
    "raw_global_nonqk": (False, "global", "nonqk"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-ckpt", required=True)
    p.add_argument("--model-name", default="qwen3_0.6b")
    p.add_argument("--variant", default="canon_qkn", choices=list(VARIANTS))
    p.add_argument("--eval-json", default=str(REPO_ROOT / "train_data/math500/test.jsonl"))
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--prompt-seed", type=int, default=0)
    p.add_argument("--resample-prompts", action="store_true",
                   help="Draw a fresh set of --num-prompts prompts from --eval-json at every grid "
                        "(and probe) point, seeded by (prompt-seed, alpha, beta). Default: one "
                        "fixed set for all points.")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--num-repeats", type=int, default=1)
    p.add_argument("--sample-seed", type=int, default=1234)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--grid-range", type=float, default=0.0, help="0 = automatic (probe).")
    p.add_argument("--grid-points", type=int, default=11)
    p.add_argument("--probe-start", type=float, default=0.002)
    p.add_argument("--probe-threshold", type=float, default=0.15)
    p.add_argument("--range-factor", type=float, default=1.7)
    p.add_argument("--probe-confirm", type=int, default=2,
                   help="Consecutive probe points below the threshold needed to stop.")
    p.add_argument("--stop-zero-rings", type=int, default=2,
                   help="Stop once this many consecutive complete rings (points at the same "
                        "distance from the centre) all have reward 0. 0 = run the full grid.")
    p.add_argument("--dir-seeds", type=int, nargs=2, default=[0, 1])
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--output-dir", default="results/landscape2d")
    p.add_argument("--rollouts-per-prompt", type=int, default=-1,
                   help="Completions saved per prompt per grid point (0 = none, -1 = all).")
    p.add_argument("--work-dir", default=os.path.join(os.environ.get("WORK", "/workspace"), "landscape2d_tmp"),
                   help="Large .pt files (base + directions); not results. Default: $WORK/landscape2d_tmp.")
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


class PromptPool:
    """The prompts scored at a grid point: one fixed sample, or (resample=True) a fresh sample
    per point seeded by (seed, alpha, beta), so a resumed run and another model evaluated at the
    same point see the same prompts."""

    def __init__(self, pool, tokenizer, num_prompts, seed, resample):
        self.pool, self.tokenizer = pool, tokenizer
        self.num_prompts, self.seed, self.resample = num_prompts, seed, resample
        self.fixed = random.Random(seed).sample(pool, num_prompts)

    def at(self, alpha, beta):
        data = self.fixed
        if self.resample:
            a, b = float(alpha) + 0.0, float(beta) + 0.0  # -0.0 -> 0.0
            rng = random.Random(f"{self.seed}:{a:+.8f}:{b:+.8f}")
            data = rng.sample(self.pool, self.num_prompts)
        return data, [build_prompt_text(self.tokenizer, ex) for ex in data]


def run_point(llm, pool, sampling, args, alpha, beta):
    """Set theta + alpha*d1 + beta*d2, sample, and score. Returns (outputs, rows, data)."""
    set_plane_point(llm, [alpha, beta])
    data, prompts = pool.at(alpha, beta)
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
    return outputs, rows, data


def mean_reward(rows):
    return float(np.mean([row["reward"] for row in rows]))


def auto_range(llm, pool, sampling, args, probe_path):
    """Walk (alpha, 0) up by sqrt(2) until reward < threshold x centre on probe_confirm
    consecutive points; range = factor x the first of them."""
    centre = mean_reward(run_point(llm, pool, sampling, args, 0.0, 0.0)[1])
    probe = [{"alpha": 0.0, "reward": centre}]
    print(f"[probe] alpha=0 reward={centre:.3f}", flush=True)
    alpha, below = args.probe_start, []
    while alpha < 100:
        reward = mean_reward(run_point(llm, pool, sampling, args, alpha, 0.0)[1])
        probe.append({"alpha": alpha, "reward": reward})
        pd.DataFrame(probe).to_csv(probe_path, index=False)
        print(f"[probe] alpha={alpha:.4g} reward={reward:.3f}", flush=True)
        below = below + [alpha] if reward <= args.probe_threshold * centre else []
        if len(below) >= args.probe_confirm:
            alpha = below[0]
            value = args.range_factor * alpha
            digits = 1 - int(math.floor(math.log10(value)))  # two significant figures
            return round(value, digits), alpha
        alpha *= math.sqrt(2)
    raise RuntimeError("Reward never dropped: check the direction.")


def zero_ring_streak(values, csv_path):
    """Consecutive complete all-zero rings at the outer end of the measured grid, walking
    rings outward from the centre (the grid is evaluated in this order)."""
    if not csv_path.exists():
        return 0
    ring_of = lambda a, b: round(float(np.hypot(a, b)), 6)  # noqa: E731
    sizes = {}
    for a in values:
        for b in values:
            sizes[ring_of(a, b)] = sizes.get(ring_of(a, b), 0) + 1
    df = pd.read_csv(csv_path).groupby(["alpha", "beta"])["reward"].mean().reset_index()
    seen = {}
    for a, b, r in zip(df["alpha"], df["beta"], df["reward"]):
        seen.setdefault(ring_of(a, b), []).append(r)
    streak = 0
    for radius in sorted(sizes):
        rewards = seen.get(radius, [])
        if len(rewards) < sizes[radius]:
            break  # ring not complete yet
        streak = streak + 1 if max(rewards) == 0 else 0
    return streak


def save_rollouts(path, data, outputs, alpha, beta, group_size, per_prompt):
    """Write the first `per_prompt` completions of each prompt (all if -1), gzipped JSONL."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, (ex, out) in enumerate(zip(data, outputs)):
            completions = out.outputs if per_prompt < 0 else out.outputs[:per_prompt]
            for j, c in enumerate(completions):
                f.write(json.dumps({
                    "alpha": alpha, "beta": beta, "prompt_idx": i,
                    "repeat": j // group_size, "sample": j % group_size,
                    "problem": ex["prompt"], "answer": ex["answer"],
                    "completion": c.text,
                    "reward": float(math500_reward_func(c.text, ex["answer"])),
                    "num_tokens": len(c.token_ids), "finish_reason": c.finish_reason,
                }, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    do_canon, dir_kind, norms = VARIANTS[args.variant]
    out_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    prefix = (f"{args.model_name}_reward2d_{args.variant}_{args.dtype}_seeds{args.dir_seeds[0]}-"
              f"{args.dir_seeds[1]}")
    suffix = (f"_n{args.grid_points}_p{args.num_prompts}_g{args.group_size}x{args.num_repeats}"
              f"_tok{args.max_new_tokens}{'_resample' if args.resample_prompts else ''}")

    # ---------------- data ----------------
    pool_data = load_eval_data(args.eval_json, 10**9, "math500")
    data = random.Random(args.prompt_seed).sample(pool_data, args.num_prompts)  # checks below

    # ---------------- base + directions (fp32, CPU) ----------------
    canon_tag = "canon" if do_canon else "raw"
    base_path = work_dir / f"base_{canon_tag}_v2.pt"  # v2: q_norm/k_norm balanced
    dir_paths = [work_dir / f"dir_{args.variant}_v3_seed{s}.pt" for s in args.dir_seeds]
    if not (base_path.exists() and all(p.exists() for p in dir_paths)):
        params, config = load_fp32_params(args.model_ckpt)
        base = canonicalize(params, config) if do_canon else params

        if do_canon:
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
            if dir_kind == "global":
                direction = global_direction(base, seed, perturb_norms=norms)
            else:
                direction = filter_normalized_direction(base, seed, perturb_norms=norms)
            torch.save(direction, path)
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
    pool = PromptPool(pool_data, tokenizer, args.num_prompts, args.prompt_seed,
                      args.resample_prompts)
    prompts = [build_prompt_text(tokenizer, ex) for ex in data]

    # Mapping check: the installed base at (0, 0) must give the original model's outputs.
    lp_original = prompt_logprobs(llm, prompts[:8])
    print("Installed:", install_plane(llm, str(base_path), [str(p) for p in dir_paths]))
    set_plane_point(llm, [0.0, 0.0])
    # Canonical weights round differently and vLLM kernels are not bit-exact, so allow
    # float noise; a mapping bug (e.g. wrong qkv split) scrambles the model (|dlogp| ~ 1-10).
    diff = np.abs(prompt_logprobs(llm, prompts[:8]) - lp_original)
    print(f"[mapping] |dlogp| original vs installed base: mean {diff.mean():.2e} "
          f"max {diff.max():.2e}", flush=True)
    if diff.mean() > 1e-2 or diff.max() > 0.5:
        raise RuntimeError("Installed base does not reproduce the checkpoint: mapping bug.")

    sampling = SamplingParams(
        n=args.group_size * args.num_repeats,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_tokens=args.max_new_tokens, seed=args.sample_seed,
    )

    # ---------------- grid range ----------------
    grid_range = args.grid_range
    range_path = out_dir / f"{prefix}{suffix}_range.json"
    if grid_range <= 0 and range_path.exists():
        grid_range = json.loads(range_path.read_text())["grid_range"]
        print(f"[probe] reusing grid range {grid_range} from {range_path}", flush=True)
    elif grid_range <= 0:
        grid_range, alpha_drop = auto_range(llm, pool, sampling, args,
                                            out_dir / f"{prefix}{suffix}_probe.csv")
        range_path.write_text(json.dumps({"grid_range": grid_range, "alpha_drop": alpha_drop}))
        print(f"[probe] reward dropped at alpha={alpha_drop:.4g}; grid range {grid_range}",
              flush=True)

    stem = f"{prefix}_r{grid_range:g}{suffix}"
    csv_path = out_dir / f"{stem}.csv"
    rollout_dir = out_dir / f"{stem}_rollouts"
    if args.rollouts_per_prompt:
        rollout_dir.mkdir(exist_ok=True)
    (out_dir / f"{stem}_config.json").write_text(
        json.dumps({**vars(args), "grid_range_used": grid_range}, indent=2))

    # ---------------- grid ----------------
    values = np.round(np.linspace(-grid_range, grid_range, args.grid_points), 8)
    done = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        done = {(round(a, 8), round(b, 8)) for a, b in zip(prev["alpha"], prev["beta"])}
        print(f"Resuming: {len(done)} points already in {csv_path}")

    wandb_run = init_run(args, job_type="landscape-2d", name=stem)
    todo = [ab for ab in grid_order(values) if (round(ab[0], 8), round(ab[1], 8)) not in done]
    if args.stop_zero_rings and zero_ring_streak(values, csv_path) >= args.stop_zero_rings:
        print("[stop] already stopped on zero rings; nothing left to run", flush=True)
        todo = []
    t_start = time.time()
    for k, (alpha, beta) in enumerate(todo):
        t0 = time.time()
        outputs, rows, point_data = run_point(llm, pool, sampling, args, alpha, beta)
        if args.rollouts_per_prompt:
            save_rollouts(rollout_dir / f"a{alpha:+.6g}_b{beta:+.6g}.jsonl.gz", point_data, outputs,
                          alpha, beta, args.group_size, args.rollouts_per_prompt)
        pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

        reward = mean_reward(rows)
        dt = time.time() - t0
        eta = (time.time() - t_start) / (k + 1) * (len(todo) - k - 1)
        print(f"[{k + 1}/{len(todo)}] alpha={alpha:+.4g} beta={beta:+.4g} "
              f"reward={reward:.3f} tokens={rows[0]['mean_tokens']:.0f} "
              f"trunc={rows[0]['truncated_frac']:.2f} {dt:.0f}s  ETA {eta / 60:.0f} min",
              flush=True)
        log_point(wandb_run, {"alpha": alpha, "beta": beta, "reward": reward}, "2d")
        if args.stop_zero_rings and zero_ring_streak(values, csv_path) >= args.stop_zero_rings:
            print(f"[stop] {args.stop_zero_rings} consecutive rings all at reward 0: "
                  f"stopping with {len(todo) - k - 1} points left", flush=True)
            break

    plot(csv_path, title=f"{args.model_name} MATH500 reward \u00b7 {args.variant}")
    finish_run(wandb_run, out_dir.glob(f"{stem}*"), results_df=pd.read_csv(csv_path))
    print("Done:", csv_path)


if __name__ == "__main__":
    main()
