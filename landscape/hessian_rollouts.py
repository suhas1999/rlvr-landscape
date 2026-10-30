"""Rollouts for the reward Hessian (landscape/reward_hessian.py).

Samples from the plain temperature policy (no top-p / top-k), because the Hessian surrogate's
importance ratio pi_theta'(y) / pi_theta(y) needs pi_theta to be exactly the sampling distribution.
fp32 vLLM, so the sampler matches the fp32 HF model reward_hessian.py differentiates.

Writes one JSON line per completion: prompt index, prompt / completion token ids, reward.

    python landscape/hessian_rollouts.py --model-ckpt <ckpt> --out rollouts.jsonl
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rlvr_landscape.tasks import load_eval_data  # noqa: E402
from rlvr_landscape.tasks.math500 import build_prompt_text, math500_reward_func  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--eval-json", default=str(REPO_ROOT / "train_data/math500/train_32.jsonl"))
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--prompt-seed", type=int, default=0)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=4321)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = p.parse_args()

    pool = load_eval_data(args.eval_json, 10**9, "math500")
    data = random.Random(args.prompt_seed).sample(pool, args.num_prompts)

    llm = LLM(model=args.model_ckpt, dtype="float32", gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_new_tokens + 1024, enable_prefix_caching=False, seed=args.seed)
    tokenizer = llm.get_tokenizer()
    prompts = [build_prompt_text(tokenizer, ex) for ex in data]
    sampling = SamplingParams(n=args.group_size, temperature=args.temperature, top_p=1.0, top_k=0,
                              max_tokens=args.max_new_tokens, seed=args.seed)
    outputs = llm.generate(prompts, sampling, use_tqdm=False)

    rows = []
    for j, (ex, out) in enumerate(zip(data, outputs)):
        for comp in out.outputs:
            rows.append({"prompt_index": j, "prompt_token_ids": list(out.prompt_token_ids),
                         "completion_token_ids": list(comp.token_ids),
                         "reward": float(math500_reward_func(comp.text, ex["answer"])),
                         "finish_reason": comp.finish_reason})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    rewards = [r["reward"] for r in rows]
    print(f"{len(rows)} completions, mean reward {sum(rewards) / len(rewards):.3f}, "
          f"mean tokens {sum(len(r['completion_token_ids']) for r in rows) / len(rows):.0f}, "
          f"truncated {sum(r['finish_reason'] == 'length' for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
