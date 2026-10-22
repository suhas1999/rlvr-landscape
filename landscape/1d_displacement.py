"""1-D reward landscape along the training displacement, starting from the base checkpoint.

    R(c) = mean reward of theta_base + c * (theta_final - theta_base)

No canonicalization. --direction-norm raw (default): c = 0 is the base model, c = 1 is the final
checkpoint exactly, c < 0 moves away from it and c > 1 extrapolates past it. --direction-norm
unit: the direction is (theta_final - theta_base) / ||theta_final - theta_base|| over all
parameters, so c is a distance in weight space and the final checkpoint sits at c = ||delta||.
--direction random replaces the displacement by a plain Gaussian over every parameter (seed
--dir-seed), scaled to ||delta|| (raw) or to 1 (unit): a baseline of the same length. Prompts,
sampling and the CSV / rollout layout follow landscape/2d_perturb.py (seed-0 prompt subset,
group_size samples per prompt), so R(0) and R(1) match the 2-D centres.

Rows are appended to the CSV as points finish; a killed run resumes. --shard i/n evaluates
every n-th coefficient so several GPUs can split one sweep into the same folder
(each shard writes its own CSV; plot_1d merges them).

    PYTHONPATH=. python landscape/1d_displacement.py --base-ckpt $BASE_MODEL \
        --final-ckpt .../checkpoint-188 --eval-json train_data/math500/train.jsonl
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

from rlvr_landscape.filter_norm import install_plane, load_fp32_params, set_plane_point  # noqa: E402
from rlvr_landscape.tasks import load_eval_data  # noqa: E402
from rlvr_landscape.tasks.math500 import build_prompt_text, math500_reward_func  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--final-ckpt", required=True)
    p.add_argument("--model-name", default="qwen3_0.6b")
    p.add_argument("--eval-json", default=str(REPO_ROOT / "train_data/math500/test.jsonl"))
    p.add_argument("--split", default=None, help="Label for file names (default: eval-json stem).")
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--prompt-seed", type=int, default=0)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--sample-seed", type=int, default=1234)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--direction", default="displacement", choices=["displacement", "random"])
    p.add_argument("--dir-seed", type=int, default=0)
    p.add_argument("--direction-norm", default="raw", choices=["raw", "unit"])
    p.add_argument("--c-min", type=float, default=-3.0)
    p.add_argument("--c-max", type=float, default=4.0)
    p.add_argument("--c-step", type=float, default=0.1)
    p.add_argument("--shard", default="0/1", help="i/n: evaluate every n-th coefficient from i.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--output-dir", default="results/landscape1d")
    p.add_argument("--work-dir", default=os.path.join(os.environ.get("WORK", "/workspace"), "landscape1d_tmp"))
    return p.parse_args()


def coefficients(args):
    """The sweep, ordered by distance from the [0, 1] segment so the trained region runs first."""
    n = int(round((args.c_max - args.c_min) / args.c_step))
    values = np.round(args.c_min + args.c_step * np.arange(n + 1), 6)
    return sorted(values.tolist(), key=lambda c: (max(0.0, -c, c - 1.0), c))


def prompt_logprobs(llm, texts):
    params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    values = []
    for out in llm.generate(texts, params, use_tqdm=False):
        for token_id, entry in zip(out.prompt_token_ids[1:], out.prompt_logprobs[1:]):
            values.append(entry[token_id].logprob)
    return np.array(values)


def main():
    args = parse_args()
    split = args.split or Path(args.eval_json).stem
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    out_dir, work_dir = Path(args.output_dir), Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"{args.model_name}_{split}_reward1d_{args.direction if args.direction == 'displacement' else f'random{args.dir_seed}'}"
            f"{'_unit' if args.direction_norm == 'unit' else ''}_p{args.num_prompts}"
            f"_g{args.group_size}_tok{args.max_new_tokens}")
    csv_path = out_dir / f"{stem}_shard{shard_i}of{shard_n}.csv"
    rollout_dir = out_dir / f"{stem}_rollouts"
    rollout_dir.mkdir(exist_ok=True)
    (out_dir / f"{stem}_config.json").write_text(json.dumps(vars(args), indent=2))

    pool = load_eval_data(args.eval_json, 10**9, "math500")
    data = random.Random(args.prompt_seed).sample(pool, args.num_prompts)

    # ---------------- base + displacement (fp32, CPU), shared by all shards ----------------
    base_path, delta_path = work_dir / "base_raw.pt", work_dir / "delta_final_minus_base.pt"
    if not (base_path.exists() and delta_path.exists()):
        base, _ = load_fp32_params(args.base_ckpt)
        final, _ = load_fp32_params(args.final_ckpt)
        if base.keys() != final.keys():
            raise RuntimeError(f"Checkpoints differ in parameters: {base.keys() ^ final.keys()}")
        delta = {name: final[name] - base[name] for name in base}
        norms = {"base": float(torch.sqrt(sum((v.double() ** 2).sum() for v in base.values()))),
                 "delta": float(torch.sqrt(sum((v.double() ** 2).sum() for v in delta.values())))}
        print(f"||theta_base|| {norms['base']:.4f}  ||delta|| {norms['delta']:.6f}  "
              f"ratio {norms['delta'] / norms['base']:.3e}", flush=True)
        tmp = work_dir / f"tmp{os.getpid()}"
        torch.save(base, f"{tmp}_b.pt"), torch.save(delta, f"{tmp}_d.pt")
        os.replace(f"{tmp}_b.pt", base_path), os.replace(f"{tmp}_d.pt", delta_path)
        (work_dir / "norms.json").write_text(json.dumps(norms))
        del base, final, delta
    (out_dir / "norms.json").write_text((work_dir / "norms.json").read_text())
    if args.direction == "random":
        norms = json.loads((work_dir / "norms.json").read_text())
        rand_path = work_dir / f"random_seed{args.dir_seed}_{args.direction_norm}.pt"
        if not rand_path.exists():
            from rlvr_landscape.filter_norm import global_direction

            # global_direction has ||d|| = ||theta_base||; rescale to ||delta|| or 1.
            target = norms["delta"] if args.direction_norm == "raw" else 1.0
            base = torch.load(base_path, weights_only=True)
            rand = {k: v * (target / norms["base"]) for k, v in global_direction(base, args.dir_seed).items()}
            print(f"random direction norm {torch.sqrt(sum((v.double() ** 2).sum() for v in rand.values())):.6f}",
                  flush=True)
            tmp = work_dir / f"tmp{os.getpid()}_r.pt"
            torch.save(rand, tmp)
            os.replace(tmp, rand_path)
            del base, rand
        delta_path = rand_path
    elif args.direction_norm == "unit":
        unit_path = work_dir / "delta_unit.pt"
        if not unit_path.exists():
            scale = 1.0 / json.loads((work_dir / "norms.json").read_text())["delta"]
            delta = torch.load(delta_path, weights_only=True)
            tmp = work_dir / f"tmp{os.getpid()}_u.pt"
            torch.save({name: v * scale for name, v in delta.items()}, tmp)
            os.replace(tmp, unit_path)
            del delta
        delta_path = unit_path

    # ---------------- vLLM ----------------
    llm = LLM(model=args.base_ckpt, dtype="float32",
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_new_tokens + 1024, enable_prefix_caching=False,
              seed=args.sample_seed)
    tokenizer = llm.get_tokenizer()
    prompts = [build_prompt_text(tokenizer, ex) for ex in data]

    # c = 0 must reproduce the loaded base model.
    lp_original = prompt_logprobs(llm, prompts[:8])
    print("Installed:", install_plane(llm, str(base_path), [str(delta_path)]))
    set_plane_point(llm, [0.0])
    diff = np.abs(prompt_logprobs(llm, prompts[:8]) - lp_original)
    print(f"[mapping] |dlogp| base vs installed c=0: mean {diff.mean():.2e} max {diff.max():.2e}",
          flush=True)
    if diff.mean() > 1e-2 or diff.max() > 0.5:
        raise RuntimeError("Installed base does not reproduce the checkpoint: mapping bug.")

    sampling = SamplingParams(n=args.group_size, temperature=args.temperature, top_p=args.top_p,
                              top_k=args.top_k, max_tokens=args.max_new_tokens,
                              seed=args.sample_seed)

    done = set()
    if csv_path.exists():
        done = {round(c, 6) for c in pd.read_csv(csv_path)["c"]}
        print(f"Resuming: {len(done)} points already in {csv_path}")
    todo = [c for k, c in enumerate(coefficients(args)) if k % shard_n == shard_i and c not in done]

    t_start = time.time()
    for k, c in enumerate(todo):
        t0 = time.time()
        set_plane_point(llm, [c])
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        rewards, lengths, truncated, records = [], [], 0, []
        for i, (ex, out) in enumerate(zip(data, outputs)):
            for j, comp in enumerate(out.outputs):
                reward = float(math500_reward_func(comp.text, ex["answer"]))
                rewards.append(reward)
                lengths.append(len(comp.token_ids))
                truncated += comp.finish_reason == "length"
                records.append({"c": c, "prompt_idx": i, "sample": j, "problem": ex["prompt"],
                                "answer": ex["answer"], "completion": comp.text, "reward": reward,
                                "num_tokens": len(comp.token_ids),
                                "finish_reason": comp.finish_reason})
        per_prompt = np.array(rewards).reshape(len(data), args.group_size).mean(axis=1)
        row = {"c": c, "reward": float(np.mean(rewards)),
               "se": float(per_prompt.std(ddof=1) / np.sqrt(len(data))),
               "mean_tokens": float(np.mean(lengths)),
               "truncated_frac": truncated / len(lengths), "n": len(rewards)}
        with gzip.open(rollout_dir / f"c{c:+.2f}.jsonl.gz", "wt", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        eta = (time.time() - t_start) / (k + 1) * (len(todo) - k - 1)
        print(f"[{k + 1}/{len(todo)}] c={c:+.2f} reward={row['reward']:.3f}±{row['se']:.3f} "
              f"tokens={row['mean_tokens']:.0f} trunc={row['truncated_frac']:.2f} "
              f"{time.time() - t0:.0f}s  ETA {eta / 60:.0f} min", flush=True)
    print("Done:", csv_path)


if __name__ == "__main__":
    main()
