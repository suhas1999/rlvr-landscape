#!/usr/bin/env bash
# GRPO training of Qwen3-0.6B (non-thinking) on a small MATH500 subset with ms-swift.
#
# Every setting can be overridden from the environment, e.g.
#   SWANLAB_API_KEY=... GROUP_SIZE=16 SEED=7 GPU=0 MASTER_PORT=29503 VLLM_MEM=0.4 bash train/grpo_math500.sh
#   TRAIN_NAME=train8 bash train/grpo_math500.sh          # uses train_data/math500/train_8.jsonl
#
# Seeds used so far: 42 50 7 10 99 100 200 300 400 500
# Group sizes used so far: 4 (VLLM_MEM=0.5), 8, 16 (VLLM_MEM=0.4)
#
# The training code lives in ms-swift (SWIFT_DIR), which must provide the
# math500_acc reward and the custom_save / sample_pass_rate plugins.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEED="${SEED:-99}"
GROUP_SIZE="${GROUP_SIZE:-4}"       # num_generations; also used as gradient_accumulation_steps
MAX_STEPS="${MAX_STEPS:-80}"
LR="${LR:-5e-6}"
MAX_LEN="${MAX_LEN:-2048}"
GPU="${GPU:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
VLLM_MEM="${VLLM_MEM:-0.5}"

TRAIN_NAME="${TRAIN_NAME:-train1}"                                   # train1 | train8 | ...
TRAIN_DATA="${TRAIN_DATA:-${REPO_ROOT}/train_data/math500/train_${TRAIN_NAME#train}.jsonl}"
VAL_DATA="${VAL_DATA:-${REPO_ROOT}/train_data/math500/test.jsonl}"

MODEL="${MODEL:-/tmp/erin/Qwen3-0.6B}"
SWIFT_DIR="${SWIFT_DIR:-/mnt/swordfish-pool2/erinxia/ms-swift}"
RUN_DIR="${RUN_DIR:-$(pwd)}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-math500-grpo-run30}"

RUN_NAME="${TRAIN_NAME}_seed${SEED}_gs${GROUP_SIZE}_lr${LR}_max${MAX_LEN}"
LOG_FILE="${RUN_DIR}/grpo_qwen3_0.6b_nonthink_math500_${RUN_NAME}_${MAX_STEPS}stp.log"
OUTPUT_DIR="${RUN_DIR}/output_qwen3_0.6b_nonthink_grpo_math500_${RUN_NAME}"

if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "ERROR: export SWANLAB_API_KEY first (it is no longer stored in this repo)." >&2
  exit 1
fi

cd "${SWIFT_DIR}"

nohup env \
CUDA_VISIBLE_DEVICES="${GPU}" \
SWANLAB_API_KEY="${SWANLAB_API_KEY}" \
SWANLAB_GROUP="qwen3_0.6b/${TRAIN_NAME}_gs${GROUP_SIZE}" \
torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" \
-m swift.cli.rlhf \
--seed "${SEED}" \
--callbacks custom_save \
--save_strategy steps \
--external_plugins "${SWIFT_DIR}/custom_save.py" "${SWIFT_DIR}/sample_pass_rate.py" \
--load_best_model_at_end false \
--rlhf_type grpo \
--loss_type grpo \
--deepspeed zero3 \
--model "${MODEL}" \
--remove_unused_columns false \
--num_iterations 1 \
--max_steps "${MAX_STEPS}" \
--log_completions true \
--log_entropy true \
--swanlab_project "${SWANLAB_PROJECT}" \
--swanlab_exp_name "${RUN_NAME}" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps "${GROUP_SIZE}" \
--temperature 0.7 \
--top_k 20 \
--top_p 0.8 \
--lr_scheduler_type constant_with_warmup \
--warmup_ratio 0.05 \
--learning_rate "${LR}" \
--eval_steps 10 \
--eval_strategy steps \
--save_only_model false \
--enable_thinking false \
--reward_func math500_acc \
--max_completion_length "${MAX_LEN}" \
--max_new_tokens "${MAX_LEN}" \
--logging_steps 1 \
--num_generations "${GROUP_SIZE}" \
--num_generations_eval 2 \
--tuner_type full \
--report_to swanlab \
--use_vllm true \
--vllm_mode colocate \
--vllm_gpu_memory_utilization "${VLLM_MEM}" \
--dataset "${TRAIN_DATA}" \
--val_dataset "${VAL_DATA}" \
--output_dir "${OUTPUT_DIR}" \
> "${LOG_FILE}" 2>&1 &

echo "Started GRPO training."
echo "PID: $!"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
