# GRPO (TRL) run that did not learn: importance-sampling ratio ≈ 0.01

2026-10-15 · Qwen3-0.6B · MATH500 train300 · TRL 1.15 + vLLM 0.31 (server mode)

## TL;DR

The first full GRPO run ran without errors but learned nothing, and slowly got worse. The cause
was a sampling setting, not the data or the reward. We sampled rollouts with
`top_p=0.8, top_k=20`. vLLM reports logprobs **after** top-k/top-p renormalization, but the
trainer recomputes them with temperature only. TRL's default IS correction
(`vllm_importance_sampling_mode="sequence_mask"`) takes `exp(sum over tokens of the logp gap)`
per completion. With a ~0.03 per-token gap over ~700 tokens, every completion's loss was
multiplied by **~0.01 or less**.

**Fix:** `top_p=1.0, top_k=0, vllm_importance_sampling_mode="token_truncate"`, now the default in
`train/grpo_trl.py`. The IS ratio is 0.9998 at step 1 of the new run.

## Symptoms (run `8ukwdsd4`)

Setup: 300 train problems, 8 prompts × 4 rollouts per step, 188 steps (~5 epochs), lr 5e-6,
β 0.04, 2048 tokens, 3 vLLM GPUs + 1 training GPU.

| | epoch 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| train reward (same 300 prompts) | 0.475 | 0.450 | 0.430 | 0.423 | 0.437 |

- Per prompt, epoch 5 vs epoch 1: 69 went down, 50 went up, 158 unchanged (mean −0.034).
  RL on a fixed prompt set should push these **up**.
- Test eval (200 MATH500 × 2): ~49.5% at steps 20–60, then down to 43% at the end.
- Lengths and truncation were stable, so this was not a length or format collapse.
- The 2-D landscape on train prompts showed the final checkpoint **below** the base model
  (centre 0.28 vs 0.375). For example, the "three divisors → 121" prompt went from 3/4 to 0/4.
- Prompts, gold answers and grading were checked by hand and are correct.

## Diagnosis

TRL logs `sampling/importance_sampling_ratio/*` and `sampling/sampling_logp_difference/*`:

| epoch | per-token logp gap (vLLM vs trainer) | IS ratio mean | IS ratio max (over 32 completions) |
|---|---|---|---|
| 0.03 (step 1, **before any update**) | 0.034 | 0.011 | 0.13 |
| 1.1 | 0.032 | 0.0025 | 0.018 |
| 2.2 | 0.027 | 0.00058 | 0.0064 |
| 4.9 | 0.030 | 0.021 | 0.13 |

- The ratio was already tiny at step 1, so this was not weights drifting.
- The gap is flat over training, so the **vLLM weight sync is fine**. Stale weights would make it
  grow.
- No completion ever got a weight near 1, so every policy-gradient update was scaled down
  ~100× or more. Longer completions were scaled down exponentially more. The run was
  effectively driven by noise plus the KL term.
- loss and grad_norm looked plausible, so nothing in the usual curves flagged it.

### Why the gap exists

- `trl vllm-serve` starts vLLM with `--logprobs-mode processed_logprobs`, so the sampled token's
  logprob reflects temperature **and** top-k/top-p truncation with renormalization.
- The trainer (`_get_per_token_logps_and_entropies`) divides logits by the temperature only.
  It never applies top-k/top-p.
- Every sampled token therefore looks more likely to vLLM than to the trainer, so
  `old_logp − sampling_logp < 0` on almost every token.
- In `sequence_mask` mode the ratio is `exp(Σ_t gap_t)`: about −0.03 × 700 = −21, and
  e^−21 ≈ 1e-9. Masking only removes ratios above 3, so these tiny weights were kept.

Code: `.venv/.../trl/trainer/grpo_trainer.py:2586–2630` (IS ratio) and `:2786–2815` (metrics).

## A/B check (5 steps each, same setup)

| | broken | **A**: top_p 1, top_k 0, sequence IS | **B**: top-p 0.8 / top-k 20, token IS |
|---|---|---|---|
| per-token logp gap | 0.034 | 0.008–0.011 | 0.022–0.034 |
| IS ratio mean | 0.003–0.06 | 0.70–1.01 | 0.971–0.982 |
| IS ratio range | ≤ 0.33 | 0.07–2.9 (a few > 3 masked) | 0.40–1.35 |

- **A** removes the cause. Sampling and training use the same distribution. The leftover
  ~0.01 per token is most likely vLLM in bf16 vs the trainer in fp32. At sequence level that
  still spreads the ratios out.
- **B** keeps the mismatch, since rollouts still come from the truncated distribution. Token IS
  just stops it from compounding.
- `--vllm_importance_sampling_correction false` (ms-swift-like) removes the down-weighting, but
  leaves the bias uncorrected and also removes the `sampling/*` metrics that found this bug.

**Chosen:** A's sampling + B's token-level IS: `top_p=1.0, top_k=0, token_truncate`.

### Modes, for reference

| mode | ratio | out-of-range values |
|---|---|---|
| `sequence_mask` (TRL default) | `exp(Σ_t gap_t)` per completion | ratio > clip_max (3) → weight 0 |
| `sequence_truncate` | same | clamped to [clip_min, clip_max] |
| `token_mask` | `exp(gap_t)` per token | > clip_max → 0 |
| `token_truncate` (ours) | `exp(gap_t)` per token | clamped to ≤ clip_max |

## After the fix: does GRPO learn?

**train300, group 4, 8 prompts per step, lr 5e-6, IS fixed (wandb `ucg2nndz`, stopped at
step 131).** The IS ratio stayed ≈ 1.00 and KL rose to 0.03–0.05, so the policy moved. But on the
same prompts, the reward change from the first to the second visit was still −0.020 ± 0.020.
Each problem gets very few updates in this setup (≈ 4 rollouts per visit, one visit per epoch).

**train300, group 8, 32 prompts per step, lr 1e-6 (wandb `z7j24the`, stopped at step 13).** KL
stayed around 0.0005, so the model barely moved.

**8-prompt overfit test (wandb `b30t1r94`, stopped at step 30).** The 8 prompts were the
landscape's seed-0 train sample (`train_data/math500/train_8_landscape_seed0.jsonl`), with
group 8, all 8 prompts every step and lr 5e-6:

| problem | steps 0–5 | 24–29 |
|---|---|---|
| parallel segment (28) | 0.48 | 0.81 |
| probability (1/2) | 0.25 | 0.60 |
| real roots (3) | 0.10 | 0.54 |
| three divisors (121) | 0.42 | 0.85 |
| polynomial / min xyz | 0.00 / 0.12 | 0.00 / 0.06 (no signal) |
| **mean** | **0.39** | **0.59** |

So GRPO learns once the IS ratio is fixed. The earlier 300-problem runs looked flat mainly
because there were too few updates per problem.

**Big run (done, wandb `vk2bj85a`, 2 h 42 min):** train300, group 8, 32 prompts per step
(256 completions), lr 5e-6, 20 epochs = 188 steps, 2 vLLM + 2 training GPUs.

- Train reward per epoch: 0.467 → 0.699
  (0.47 0.50 0.54 0.54 0.55 0.58 0.58 0.63 0.63 0.59 0.65 0.65 0.65 0.65 0.68 0.69 0.63 0.70 0.68 0.70).
- Per prompt, last visit − first visit: **+0.235 ± 0.018** (174 up, 17 down). The broken run was
  −0.03 on this measure.
- Test eval (200 × 2, top_p 1), every 20 steps: 50.0 49.0 50.0 52.2 51.6 50.8 54.3 52.5 52.7 54.1.
- IS ratio ≈ 1.00 throughout; KL 0.02–0.04; clipped completions 2–11%.
`checkpoints/output_qwen3_0.6b_grpo_trl_train300_seed42_gs8_p32_lr5e-6_max2048_ep20_topp1_tokenIS/`.
After it finishes: canon landscapes for base vs final, on train and test prompts, written to
`results/landscape2d/2026-10-15_train300_gs8p32_ep20/fixed8_r0.22_n11/`.

**Landscapes, base vs big-run `checkpoint-188`** (canon, 8 prompts × 4, 2048 tokens, auto range
0.22, landscape sampling still top-p 0.8 / top-k 20). Results in
`results/landscape2d/2026-10-15_train300_gs8p32_ep20/fixed8_r0.22_n11/compare_*.png`. Mean reward per ring of
distance r = √(α²+β²) from the centre:

| run | r=0 | 0.044 | 0.062 | 0.088 | 0.098 | 0.124 |
|---|---|---|---|---|---|---|
| base · train prompts | 0.38 | 0.39 | 0.20 | 0.03 | 0.05 | 0.00 |
| final · train prompts | **0.66** | 0.48 | 0.41 | 0.18 | 0.07 | 0.00 |
| base · test prompts | 0.50 | 0.57 | 0.52 | 0.35 | 0.19 | 0.02 |
| final · test prompts | **0.66** | 0.66 | 0.55 | 0.34 | 0.28 | 0.03 |

- The peak is higher after RL on both prompt sets. The train prompts gain the most (+0.28 at the
  centre).
- The basin has the same width: all four collapse at r ≈ 0.12–0.13, and every 1-D probe drops
  between α = 0.09 and 0.128.
- On the train prompts the final peak is sharper. Most of the gain is gone by r ≈ 0.044, so the
  improvement sits in a narrow region around the trained weights. On the test prompts the final
  model keeps its plateau out to 0.044.
- Caveats: 32 samples per point (SE ≈ ±0.08), and 5×5 effective grid inside the basin. The base
  train grid stopped at 29 points this time vs 45 before (vLLM sampling is not bit-reproducible
  across runs; the probes matched exactly).

## Changes made (not committed)

- `train/grpo_trl.py`: defaults `top_p=1.0`, `top_k=0`, `vllm_importance_sampling_mode="token_truncate"`.
  The docstring explains why.
- `README.md`: training-defaults paragraph updated.
- `rlvr_landscape/tracking.py`: `finish_run` skips directories. The landscape runs had crashed at
  the wandb artifact upload, because `glob(f"{stem}*")` matched the `<stem>_rollouts/` folder.
  CSVs and PNGs had still been written locally.
- New file `train_data/math500/train_line215.jsonl`: the single-problem run, "two-digit primes
  with digit sum 8", pass rate ~62%.

## Other problems hit today

1. **sbatch from inside the interactive job.** `--export=ALL` leaked the outer job's
   `SLURM_STEP_GPUS`, so `cluster/env.sh` pinned GPU UUIDs the new job did not own, and vLLM died
   with `NVMLError_NotFound`. Fix: run interactively in the existing job, after
   `source cluster/env.sh` and `source /scratch/micah/vm2825/misc/slurm_gpu_pin.sh`.
2. **train_1 (p(8) polynomial problem) is never solved** by Qwen3-0.6B non-thinking: 0/40
   rollouts, mostly answering 8/63. Every group had zero variance, so there was nothing to learn.
   Hence the switch to line 215.
3. **Landscape grid too coarse.** Reward is flat out to α ≈ 0.064 and reaches 0 by ≈ 0.13. The
   auto range picked 0.22 (1.7 × 0.128), so with spacing 0.044 only ~9 points per run fall inside
   the basin. The contours are mostly interpolation. Next time use
   `--grid-range 0.12 --grid-points 13` and `--num-repeats 2` or more prompts. Each point is only
   8 × 4 = 32 samples (SE ≈ ±0.08).
4. **`2d_perturb.py --work-dir` caching** uses model-agnostic filenames (`base_canon_v2.pt`).
   Always give each model and split its own work dir, or a run silently reuses another model's
   base.

## Status / where things are

- **Fixed run (in progress):** wandb `ucg2nndz`,
  `checkpoints/output_qwen3_0.6b_grpo_trl_train300_seed42_gs4_p8_lr5e-6_max2048_ep5_topp1_tokenIS/`.
  Step 1: IS 0.9998, logp gap 0.011, grad_norm 0.71. Step 6: IS 0.9996, grad_norm 1.34 (clipped
  to 1.0).
- **Broken run (for comparison):** wandb `8ukwdsd4`,
  `checkpoints/output_qwen3_0.6b_grpo_trl_train300_seed42_gs4_p8_lr5e-6_max2048_ep5/`.
- **Landscapes of the broken run:** `results/landscape2d/2026-10-14_broken_train300_gs4p8/` (`compare_*.png`).
- **Landscapes for the big run:** `results/landscape2d/2026-10-15_train300_gs8p32_ep20/`.
  `fixed8_r0.22_n11/` uses the same 8 prompts at every point. `resample8_r0.12_n13x2/` uses a
  new 8 at each point (`--resample-prompts`), range 0.12, 13×13, 2 repeats. Folder index:
  `results/landscape2d/README.md`.

  Resampled run, mean reward by distance from the trained weights (stopped at 158–165/169 points):

  | run | r=0 | <0.03 | <0.06 | <0.09 | <0.12 | <0.15 |
  |---|---|---|---|---|---|---|
  | base · train | 0.45 | 0.46 | 0.41 | 0.26 | 0.08 | 0.02 |
  | final · train | 0.77 | 0.73 | 0.61 | 0.38 | 0.13 | 0.04 |
  | base · test | 0.34 | 0.50 | 0.39 | 0.30 | 0.10 | 0.02 |
  | final · test | 0.56 | 0.57 | 0.43 | 0.36 | 0.12 | 0.03 |

  Final is above base at every radius up to ≈ 0.1. The gain is larger on train prompts (≈ +0.3)
  than on test prompts (≈ +0.07), and it shrinks with distance. The basin width is unchanged.
  The contours are patchy because prompt difficulty varies point to point; `compare_radial.png`
  is the clearest view.
- **Results directory:** moved into the repo, `results/` (git-ignored). `RESULTS_DIR` in
  `cluster/env.sh` now points there, and `/scratch/micah/vm2825/results/rlvr-landscape` is a
  symlink to it.
- **Logs:** `/scratch/micah/vm2825/logs/` (`grpo-trl-train300-v2.out`, `is_check_{A,B}.out`,
  `landscape_canon_*.out`).

## Checklist for future TRL runs

- In the first steps, check `sampling/importance_sampling_ratio/mean` ≈ 1 and
  `sampling/sampling_logp_difference/mean` ≲ 0.01.
- Don't combine top-k/top-p rollouts with sequence-level IS.
- Check training reward on the training prompts across epochs. If it doesn't go up, something
  is wrong, even when the loss looks fine.
- The wandb API key was pasted in chat; rotate it, or use `wandb login` (`~/.netrc`).
