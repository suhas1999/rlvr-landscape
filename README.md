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

## Data (`train_data/math500`)

- `test/test_500.jsonl`: all 500 problems (raw format; used by `eval/`)
- `train.jsonl`: 300-problem training subset; `test.jsonl`: 200-problem test subset
- `train_1.jsonl`, `train_8.jsonl`, `train_8_50_80.jsonl`, `train_id299.jsonl`, ...:
  small subsets for the overfitting experiments

## Notes

- All checkpoint paths in `scripts/` point at the cluster (`/mnt/swordfish-pool2/...`).
- Training code (the `math500_acc` reward and the `custom_save` / `sample_pass_rate` plugins)
  lives in the ms-swift checkout, not in this repo.
- `along_direction.py` loads a direction inside the vLLM worker, maps HF names onto
  vLLM's fused layers (`qkv_proj`, `gate_up_proj`) and rebuilds every α from an fp32
  copy of the checkpoint. Displacement results produced before this reorganization
  (`visualize_reward_landscape_displace*.py`) did **not** perturb the fused layers. See
  the commit message.
