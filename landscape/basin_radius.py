"""Basin radius and volume along random directions (Huang et al. 2019, arXiv:1906.03291, sec. 5).

The basin is the set of points around theta whose reward stays above a threshold. Along each
random direction d_k (filter-normalized on the canonicalized checkpoint, the --variant canon
directions of landscape/2d_perturb.py; or, with --direction unit, a plain Gaussian over every raw
parameter with ||d_k|| = 1, so alpha is weight-space distance), the reward R(alpha) = R(theta + alpha * d_k) is swept
outward in steps of --alpha-step until it falls below --stop-frac x R(0). The radius r_k(tau) is
the first crossing of the threshold tau (linear interpolation), computed later by
landscape/basin_volume.py for several thresholds from the same sweep.

Volume as in the paper: V = omega_n E_k[r_k^n] with n = number of perturbed parameters; for n ~ 6e8
log V is dominated by the largest radius, so basin_volume.py also reports mean / min radii.

Rows (direction seed, alpha, reward, ...) are appended to the CSV as they finish; a killed run
resumes. Every completion is saved under <output-dir>/rollouts/<stem>/ for pass@k.

    PYTHONPATH=. python landscape/basin_radius.py --model-ckpt .../checkpoint-100 --model-name step100
"""

import argparse
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import pandas as pd
import torch
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rlvr_landscape.filter_norm import (  # noqa: E402
    canonicalize,
    check_function_preserved,
    filter_normalized_direction,
    install_plane,
    load_fp32_params,
    set_plane_point,
    unit_direction,
)
from rlvr_landscape.tasks import load_eval_data  # noqa: E402
from rlvr_landscape.tasks.math500 import build_prompt_text, math500_reward_func  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-ckpt", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--eval-json", default=str(REPO_ROOT / "train_data/math500/train.jsonl"))
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--prompt-seed", type=int, default=0)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--sample-seed", type=int, default=1234)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--dir-seeds", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--direction", choices=["filter", "unit", "unit_norms", "files"], default="filter",
                   help="filter: filter-normalized on the canonicalized checkpoint. unit: plain "
                        "Gaussian over every raw parameter with ||d|| = 1 (alpha = weight-space distance). "
                        "unit_norms: the same, on the RMSNorm gains only. files: --direction-files.")
    p.add_argument("--direction-files", nargs="+", default=None,
                   help="With --direction files: unit directions saved as {HF param name: tensor} "
                        "(e.g. by reward_hessian.py); dir_seed in the CSV is the file index.")
    p.add_argument("--signed", action="store_true",
                   help="Sweep -alpha as well as +alpha (each sign stops on its own).")
    p.add_argument("--centre-repeats", type=int, default=2,
                   help="Independent samplings of R(0) (sample seeds differ).")
    p.add_argument("--alpha-step", type=float, default=0.02)
    p.add_argument("--alpha-max", type=float, default=0.24)
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="Explicit increasing alpha grid; overrides --alpha-step / --alpha-max.")
    p.add_argument("--stop-frac", type=float, default=0.2,
                   help="Stop a direction after --stop-confirm points below stop_frac x R(0).")
    p.add_argument("--stop-confirm", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                   help="vLLM weight dtype. bfloat16 rounds the perturbation: below alpha ~1 most of "
                        "it is lost (per-weight shift < bf16 spacing); fine for alpha >= ~5.")
    p.add_argument("--output-dir", default="results/basin")
    p.add_argument("--work-dir", default=os.path.join(os.environ.get("WORK", "/workspace"), "basin_tmp"))
    p.add_argument("--keep-work", action="store_true", help="Keep base/direction .pt files.")
    return p.parse_args()


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
    work_dir = Path(args.work_dir) / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model_name}_{Path(args.eval_json).stem}_p{args.num_prompts}_g{args.group_size}"
    if args.direction != "filter":
        stem += f"_{args.direction}"
    csv_path = out_dir / f"{stem}.csv"
    rollout_dir = out_dir / "rollouts" / stem
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}_config.json").write_text(json.dumps(vars(args), indent=2))

    pool = load_eval_data(args.eval_json, 10**9, "math500")
    data = random.Random(args.prompt_seed).sample(pool, args.num_prompts)

    if args.direction == "files":
        if not args.direction_files:
            raise SystemExit("--direction files needs --direction-files")
        args.dir_seeds = list(range(len(args.direction_files)))
        base_path = work_dir / "base_files.pt"
        if not base_path.exists():
            params, _ = load_fp32_params(args.model_ckpt)
            torch.save(params, base_path)
            n = sum(v.numel() for v in params.values())
            (work_dir / "dims.json").write_text(json.dumps({"n_params": n, "n_perturbed": n}))
            del params
        dir_paths = [Path(f) for f in args.direction_files]

    # ---------------- base + directions (fp32, CPU) ----------------
    tag = "canon" if args.direction == "filter" else args.direction
    if args.direction != "files":
        base_path = work_dir / f"base_{tag}.pt"
        dir_paths = [work_dir / f"dir_{tag}_seed{s}.pt" for s in args.dir_seeds]
    if args.direction != "files" and not (base_path.exists() and all(p.exists() for p in dir_paths)):
        params, config = load_fp32_params(args.model_ckpt)
        if args.direction == "filter":
            base = canonicalize(params, config)
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.model_ckpt)
            token_lists = [tok(build_prompt_text(tok, ex))["input_ids"] for ex in data[:8]]
            before, after = check_function_preserved(args.model_ckpt, base, token_lists)
            print(f"[canonicalize] NLL before {before:.7f} after {after:.7f}", flush=True)
            if abs(before - after) > 1e-4:
                raise RuntimeError("Canonicalization changed the model output: bug.")
        else:
            base = params  # raw checkpoint, no canonicalization
        n_params = sum(v.numel() for v in base.values())
        torch.save(base, base_path)
        for seed, path in zip(args.dir_seeds, dir_paths):
            if args.direction == "filter":
                direction = filter_normalized_direction(base, seed, perturb_norms="none")
            else:
                direction = unit_direction(base, seed, norms_only=args.direction == "unit_norms")
            n_perturbed = sum(int((v != 0).sum()) for v in direction.values())
            torch.save(direction, path)
        (work_dir / "dims.json").write_text(json.dumps({"n_params": n_params, "n_perturbed": n_perturbed}))
        del params, base, direction
    (out_dir / f"{stem}_dims.json").write_text((work_dir / "dims.json").read_text())

    # ---------------- vLLM ----------------
    llm = LLM(model=args.model_ckpt, dtype=args.dtype,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_new_tokens + 1024, enable_prefix_caching=False,
              seed=args.sample_seed)
    tokenizer = llm.get_tokenizer()
    prompts = [build_prompt_text(tokenizer, ex) for ex in data]
    lp_original = prompt_logprobs(llm, prompts[:8])
    print("Installed:", install_plane(llm, str(base_path), [str(p) for p in dir_paths]), flush=True)
    set_plane_point(llm, [0.0] * len(dir_paths))
    diff = np.abs(prompt_logprobs(llm, prompts[:8]) - lp_original)
    print(f"[mapping] |dlogp| original vs installed base: mean {diff.mean():.2e} "
          f"max {diff.max():.2e}", flush=True)
    if diff.mean() > 1e-2 or diff.max() > 0.5:
        raise RuntimeError("Installed base does not reproduce the checkpoint: mapping bug.")
    k = len(dir_paths)

    def evaluate(seed_index, alpha, repeat=0):
        coefficients = [0.0] * k
        if seed_index is not None:
            coefficients[seed_index] = alpha
        set_plane_point(llm, coefficients)
        sampling = SamplingParams(n=args.group_size, temperature=args.temperature,
                                  top_p=args.top_p, top_k=args.top_k,
                                  max_tokens=args.max_new_tokens,
                                  seed=args.sample_seed + 1000 * repeat)
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        rewards, lengths, truncated, records = [], [], 0, []
        for j, (ex, out) in enumerate(zip(data, outputs)):
            for comp in out.outputs:
                rewards.append(float(math500_reward_func(comp.text, ex["answer"])))
                lengths.append(len(comp.token_ids))
                truncated += comp.finish_reason == "length"
                records.append({"prompt_index": j, "answer": ex["answer"], "completion": comp.text,
                                "reward": rewards[-1], "tokens": lengths[-1],
                                "finish_reason": comp.finish_reason})
        dir_seed = -1 if seed_index is None else args.dir_seeds[seed_index]
        with gzip.open(rollout_dir / f"dir{dir_seed:+d}_a{alpha:.3f}_r{repeat}.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
        per_prompt = np.array(rewards).reshape(len(data), args.group_size).mean(axis=1)
        return {"dir_seed": dir_seed,
                "alpha": round(alpha, 6), "repeat": repeat, "reward": float(np.mean(rewards)),
                "se": float(per_prompt.std(ddof=1) / np.sqrt(len(data))),
                "mean_tokens": float(np.mean(lengths)),
                "truncated_frac": truncated / len(lengths), "n": len(rewards)}

    prev = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(
        columns=["dir_seed", "alpha", "repeat", "reward"])
    done = {(int(s), round(float(a), 6), int(r)) for s, a, r in
            zip(prev["dir_seed"], prev["alpha"], prev["repeat"])}

    def record(row):
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        print(f"dir {row['dir_seed']:>2} alpha {row['alpha']:.3f} reward {row['reward']:.3f}"
              f"±{row['se']:.3f} tokens {row['mean_tokens']:.0f} trunc {row['truncated_frac']:.2f}"
              f" {time.time() - t0:.0f}s", flush=True)
        return row

    t0 = time.time()
    centre = [r for r in prev.to_dict("records") if int(r["dir_seed"]) == -1]
    for rep in range(args.centre_repeats):
        if (-1, 0.0, rep) not in done:
            t0 = time.time()
            centre.append(record(evaluate(None, 0.0, rep)))
    r0 = float(np.mean([r["reward"] for r in centre]))
    print(f"R(0) = {r0:.3f}", flush=True)

    if args.alphas:
        alphas = np.round(np.array(sorted(args.alphas)), 6)
    else:
        alphas = np.round(np.arange(1, int(round(args.alpha_max / args.alpha_step)) + 1) * args.alpha_step, 6)
    for i, seed in enumerate(args.dir_seeds):
        for sign in ((1, -1) if args.signed else (1,)):
            mine = prev[(prev["dir_seed"] == seed) & (np.sign(prev["alpha"]) == sign)]
            rows = mine.assign(a=mine["alpha"].abs()).sort_values("a")["reward"].tolist()
            below = sum(1 for r in rows[-args.stop_confirm:] if r < args.stop_frac * r0)
            for alpha in alphas * sign:
                if below >= args.stop_confirm:
                    break
                if (seed, float(alpha), 0) in done:
                    continue
                t0 = time.time()
                row = record(evaluate(i, float(alpha)))
                below = below + 1 if row["reward"] < args.stop_frac * r0 else 0
    set_plane_point(llm, [0.0] * k)
    if not args.keep_work:
        # never delete caller-provided direction files
        for path in [base_path, *(dir_paths if args.direction != "files" else [])]:
            path.unlink(missing_ok=True)
    print("Done:", csv_path)


if __name__ == "__main__":
    main()
