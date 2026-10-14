# RLVR Reward Landscape

Visualize the **reward landscape** of RLVR (GRPO)-trained language models: take a
checkpoint θ, pick a direction *d* in parameter space, and measure mean reward at

    θ(α) = θ_ckpt + α · scale · d

for a grid of α. Current focus: Qwen3-0.6B (non-thinking) on **MATH500**, trained
until it overfits on 1 or 8 problems, and compared across training stages.

Three kinds of direction:

| Direction | Meaning | Where it comes from |
|---|---|---|
| **random** | random Gaussian direction | generated on the fly inside vLLM |
| **SGD** (`grpo-grad`) | −∇ of the GRPO loss at θ | `directions/compute_sgd_direction.py` |
| **displacement** | θ_j − θ_i between two checkpoints | `directions/compute_displacement.py` |

## Layout

```
rlvr_landscape/            shared library (imported by everything below)
  tasks/                   dataset loading, prompts, answer extraction, rewards
    math500.py  gsm8k.py  chess_single_turn.py  common.py
  grading/                 PRM800K MATH answer checker (grade_answer)
  vllm_utils.py            create_vllm, vllm_eval_reward, cleanup_cuda
  perturb.py               install a direction file in vLLM, set the coefficient
  grpo.py                  GRPO rollouts + policy-gradient direction
  directions.py            norms, scaling, saving, random directions

train/grpo_math500.sh      GRPO training with ms-swift (group size / seed / data via env vars)

directions/                build direction files (.pt)
  compute_sgd_direction.py   -∇ GRPO loss at a checkpoint (raw, not normalized)
  compute_displacement.py    θ_j − θ_i

landscape/                 sweep α and plot reward
  random_direction.py        random directions (math500 | gsm8k | chess)
  along_direction.py         one direction from a file: grpo-grad | displacement | random

eval/eval_math500.py       plain MATH500 accuracy (base or trained model), no perturbation

cloud/                     single-GPU cloud setup (uv env, Vast.ai helper, GPU smoke test)

scripts/                   launchers with the experiment configs (edit these)
  landscape_random.sh  landscape_sgd.sh  landscape_displacement.sh  compute_sgd_direction.sh

legacy/                    older HF-backend scripts for chess/GSM8K (not maintained)
train_data/                datasets (see below)
chess_llm/                 small chess-LLM checkpoint used by the legacy chess runs
```

## Workflow

```bash
# 1. Train (needs ms-swift checkout at SWIFT_DIR)
export SWANLAB_API_KEY=...
GROUP_SIZE=4 SEED=42 TRAIN_NAME=train1 bash train/grpo_math500.sh

# 2a. Random-direction landscape
bash scripts/landscape_random.sh

# 2b. SGD-direction landscape
bash scripts/compute_sgd_direction.sh       # writes sgd_directions_train1/*.pt
bash scripts/landscape_sgd.sh               # set DIRECTION_PATH to that file

# 2c. Displacement landscape
python -m directions.compute_displacement --ckpt-i .../checkpoint-109 \
    --ckpt-j .../checkpoint-110 --output param_displace/disp_109_to_110.pt
bash scripts/landscape_displacement.sh      # set DIRECTION_PATH to that file

# Plain accuracy
python -m eval.eval_math500 --model-path .../checkpoint-50 --model-name step50 \
    --data-path train_data/math500/train.jsonl --temperature 0 --max-new-tokens 2048
```

Run Python entry points as modules from the repo root (`python -m landscape.along_direction ...`)
so `rlvr_landscape` is importable. The `scripts/*.sh` launchers do this for you.

Each landscape run writes a CSV of mean reward per α, a CSV of per-completion
records, and absolute and relative (R(α)/R(0)) plots to `--output-dir`.

## Running on a rented GPU (Vast.ai)

Everything runs on one GPU. The Python env is pinned by `pyproject.toml` / `uv.lock`
(vLLM 0.10.2, torch 2.8.0 / CUDA 12.8; Linux x86_64 only).

```bash
# laptop, once
uvx vastai set api-key <KEY>
uvx vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"

# laptop: pick and rent a box (setup_instance.sh runs automatically on boot)
bash cloud/vast.sh offers A100_SXM4            # or H100_SXM; add "bid" for interruptible
bash cloud/vast.sh create <OFFER_ID> [BID]     # BID only for interruptible
bash cloud/vast.sh log <INSTANCE_ID>           # wait for "Setup done"
bash cloud/vast.sh push <INSTANCE_ID> path/to/checkpoint-50   # -> /workspace/checkpoints/

# on the box
source cloud/env.sh
python -m cloud.smoke_test --model $BASE_MODEL  # vLLM + perturbation checks, prints tok/s
export RLVR_ENFORCE_EAGER=0                     # only if the smoke test says graphs are safe
MODEL_CKPT=$CKPT_ROOT/checkpoint-50 CHECKPOINT_STEP=50 bash scripts/landscape_random.sh

# laptop: results back, then stop paying
bash cloud/vast.sh pull <INSTANCE_ID>           # -> ./results/
bash cloud/vast.sh destroy <INSTANCE_ID>
```

`cloud/env.sh` keeps the venv, HF cache and vLLM compile cache under `/workspace`, so a
stop/start or an interruptible pause does not reinstall anything. On a dedicated GPU it
sets `GPU_MEMORY_UTILIZATION=0.85` (the launchers default to 0.25-0.3 for shared cluster
GPUs). The launchers take `MODEL_CKPT`, `EVAL_JSON`, `DIRECTION_PATH`, `CHECKPOINT_STEP`,
`GPU` and `GPU_MEMORY_UTILIZATION` from the environment.

`RLVR_ENFORCE_EAGER` controls the engines built by `create_vllm` (SGD / displacement
landscapes). Weights are updated with an in-place `copy_`, so CUDA graphs should stay
valid. `cloud/smoke_test.py` checks this directly by comparing eager and graph log-probs
at a perturbed α. `landscape/random_direction.py` already ran with graphs on.

## Data (`train_data/math500`)

- `test/test_500.jsonl`: all 500 problems (raw format; used by `eval/`)
- `train.jsonl`: 300-problem training subset; `test.jsonl`: 200-problem test subset
- `train_1.jsonl`, `train_8.jsonl`, `train_8_50_80.jsonl`, `train_id299.jsonl`, ...:
  small subsets for the overfitting experiments

## Where the old files went

| Old file | New location |
|---|---|
| `visualize_reward_landscape_vllm.py` | `landscape/random_direction.py` |
| `visualize_reward_landscape_sgd.py` | `landscape/along_direction.py --direction-type grpo-grad` |
| `visualize_reward_landscape_displace.py`, `..._displace_no_accm_error.py` | `landscape/along_direction.py --direction-type displacement` |
| `visualize_reward_landscape.py`, `_paral.py`, `_sto.py` | `legacy/` (chess / GSM8K, HF backend) |
| `sgd_direction.py` | `directions/compute_sgd_direction.py` (now takes CLI args) |
| `param_displace_direc.py` | `directions/compute_displacement.py` (now takes CLI args) |
| `eval_math500_base.py`, `eval_math500_displacement.py` | `eval/eval_math500.py` (`--base-model` is auto-detected) |
| `eval_math500.py` | `legacy/eval_math500_old.py` |
| `swift_scripts/train1_gs{4,8,16}.sh` | `train/grpo_math500.sh` (`GROUP_SIZE=4\|8\|16`) |
| `scripts/run_qwen_math500_{vllm,sgd,displacement}.sh` | `scripts/landscape_{random,sgd,displacement}.sh` |
| `scripts/run_chessllm.sh`, `run_qwen_{chessdata,gsm8k,math500}.sh` | `legacy/scripts/` |
| `grading/` | `rlvr_landscape/grading/` (import: `rlvr_landscape.grading.grader`) |

Functions shared between scripts now live once in `rlvr_landscape/` and were moved
verbatim. The task code was checked against the old scripts on the MATH500, GSM8K and
chess data and gives identical outputs.

## Fixes in the reorganization

- **Displacement landscapes.** The old displacement scripts matched HF parameter names
  directly to vLLM's, so vLLM's fused `qkv_proj` and `gate_up_proj` layers were **never
  perturbed**, and prefix caching stayed on while weights changed. Displacement now
  uses the same installer as the SGD direction (maps fused layers, rebuilds each α
  from an fp32 copy, prefix caching off). **Re-run displacement plots made before this
  change.**
- **Prefix caching in random-direction runs.** `visualize_reward_landscape_vllm.py` also
  changed weights in place with prefix caching left at vLLM's default (on in the V1
  engine). The prefix cache is keyed by prompt tokens only, so the same prompt at the
  next α reused prompt KV computed with the previous α's weights. Every vLLM instance
  whose weights are perturbed now sets `enable_prefix_caching=False`. Within a request
  the normal KV cache is still used, so decoding speed is unchanged. **Random-direction
  plots made with a vLLM version that enables prefix caching by default should be
  re-run.**
- **SGD direction computed inline.** `visualize_reward_landscape_sgd.py` crashed
  (`NameError`) without `--direction-path`; `along_direction.py` now works either way.
- **Tied `lm_head`.** Displacement files include `lm_head.weight`; it is now ignored when
  it duplicates the tied embedding instead of raising.
- **`eval_math500_base.py`** crashed on import (`MAX_NEW_TOKENS` used before definition).
- **SwanLab key.** No longer hardcoded; `export SWANLAB_API_KEY=...` before training.

## Notes

- All checkpoint paths in `scripts/` point at the cluster (`/mnt/swordfish-pool2/...`).
- Training code (the `math500_acc` reward and the `custom_save` / `sample_pass_rate` plugins)
  lives in the ms-swift checkout, not in this repo. ms-swift's own copy of the grader
  must be importable there: the old `swift_scripts` gs8 run failed with
  `No module named 'grading'`.
- Output filenames of displacement runs differ slightly from the old scripts
  (`scale` is printed with 5 decimals, as for SGD runs).
- `legacy/` is kept for reference and still imports `swift.grading`.
