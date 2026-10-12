#!/usr/bin/env bash
# Compute a raw SGD (-GRPO gradient) direction at one checkpoint.
# bash scripts/compute_sgd_direction.sh
# Output feeds scripts/landscape_sgd.sh (DIRECTION_PATH).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

GPU="0"
CHECKPOINT_STEP=110
MODEL_CKPT="/mnt/swordfish-pool2/erinxia/ms-swift/output_qwen3_0.6b_nonthink_grpo_math500_train1_lr_5e-6_max2048/v0-20260916-223901/checkpoint-${CHECKPOINT_STEP}"
EVAL_JSON="${REPO_ROOT}/train_data/math500/train_id299.jsonl"
MODEL_NAME="qwen3_0.6b"
NUM_SAMPLES=1
PG_NUM_PROMPTS=1
GROUP_SIZE=32
NUM_DIRECTIONS=1
DIRECTION_DIR="${REPO_ROOT}/sgd_directions_train1"
LOG_DIR="logs_sgd_directions"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/sgd_${MODEL_NAME}_step${CHECKPOINT_STEP}_gs${GROUP_SIZE}.log"

nohup env CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u -m directions.compute_sgd_direction \
    --model-ckpt "${MODEL_CKPT}" \
    --model-name "${MODEL_NAME}" \
    --checkpoint-step "${CHECKPOINT_STEP}" \
    --eval-json "${EVAL_JSON}" \
    --num-samples "${NUM_SAMPLES}" \
    --pg-num-prompts "${PG_NUM_PROMPTS}" \
    --group-size "${GROUP_SIZE}" \
    --num-directions "${NUM_DIRECTIONS}" \
    --temperature 0.7 --top-p 0.8 --top-k 20 \
    --max-new-tokens 2048 \
    --gpu-memory-utilization 0.25 \
    --seed 42 \
    --direction-dir "${DIRECTION_DIR}" \
  > "${LOG_FILE}" 2>&1 &

echo "Started PID $!"
echo "Follow log with: tail -f ${LOG_FILE}"
