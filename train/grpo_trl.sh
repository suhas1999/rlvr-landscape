#!/usr/bin/env bash
# GRPO on MATH500 with TRL (train/grpo_trl.py). vLLM rollouts and training can run on
# separate GPUs; the trainer pushes new weights to the vLLM server after every step.
#
#   bash train/grpo_trl.sh [extra grpo_trl.py / GRPOConfig args]
#
# GPU split (everything via env vars):
#   GPUS              GPUs to use, comma-separated indices or UUIDs. Default: $GPU_UUIDS
#                     (cluster/env.sh: all Slurm-allocated GPUs), else $CUDA_VISIBLE_DEVICES,
#                     else every GPU nvidia-smi lists.
#   ROLLOUT_GPUS      how many of GPUS (the first ones) serve vLLM rollouts; the rest train.
#                     Default 1 if there are >= 2 GPUs, else 0. 0 = colocate: vLLM shares the
#                     training GPU(s), one engine per training process.
#   ROLLOUT_TP        tensor-parallel size of each vLLM replica (default 1); ROLLOUT_GPUS/ROLLOUT_TP
#                     replicas serve requests in data parallel.
#
# Examples:
#   GPUS=0 bash train/grpo_trl.sh                                # 1 GPU, colocated
#   GPUS=0,1 bash train/grpo_trl.sh                              # 1 rollout + 1 train
#   GPUS=0,1,2,3 ROLLOUT_GPUS=2 bash train/grpo_trl.sh           # 2 rollout replicas + 2 train (DDP)
#   GROUP_SIZE=16 PROMPTS_PER_STEP=4 TRAIN_NAME=train8 bash train/grpo_trl.sh --beta 0
#
# Run settings: MODEL, TRAIN_NAME (train1 -> train_data/math500/train_1.jsonl) or TRAIN_DATA,
# VAL_DATA ("" disables eval), GROUP_SIZE, PROMPTS_PER_STEP, MAX_STEPS, LR, MAX_LEN, SEED,
# SAVE_STEPS, EVAL_STEPS, VLLM_MEM, VLLM_MAX_MODEL_LEN, OUTPUT_ROOT, RUN_NAME.
# Full fp32 (training and rollouts): MIXED_PRECISION=no VLLM_DTYPE=float32 ... --bf16 false
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-${BASE_MODEL:-Qwen/Qwen3-0.6B}}"
TRAIN_NAME="${TRAIN_NAME:-train1}"
TRAIN_DATA="${TRAIN_DATA:-train_data/math500/train_${TRAIN_NAME#train}.jsonl}"
VAL_DATA="${VAL_DATA-train_data/math500/test.jsonl}"
GROUP_SIZE="${GROUP_SIZE:-4}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-1}"
MAX_STEPS="${MAX_STEPS:-80}"
LR="${LR:-5e-6}"
MAX_LEN="${MAX_LEN:-2048}"
SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-10}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((MAX_LEN + 1024))}"   # prompt + completion

RUN_NAME="${RUN_NAME:-${TRAIN_NAME}_seed${SEED}_gs${GROUP_SIZE}_p${PROMPTS_PER_STEP}_lr${LR}_max${MAX_LEN}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CKPT_ROOT:-${REPO_ROOT}}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/output_qwen3_0.6b_grpo_trl_${RUN_NAME}}"
mkdir -p "${OUTPUT_DIR}"

# ---- GPU split ----
if [[ -z "${GPUS:-}" ]]; then
    if [[ -n "${GPU_UUIDS:-}" ]]; then
        GPUS="${GPU_UUIDS}"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        GPUS="${CUDA_VISIBLE_DEVICES}"
    else
        GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
    fi
fi
IFS=, read -r -a ALL_GPUS <<< "${GPUS}"
NUM_GPUS=${#ALL_GPUS[@]}
ROLLOUT_GPUS="${ROLLOUT_GPUS:-$(( NUM_GPUS >= 2 ? 1 : 0 ))}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"

if (( NUM_GPUS == 0 )); then
    echo "ERROR: no GPUs found (set GPUS)." >&2; exit 1
fi
if (( ROLLOUT_GPUS >= NUM_GPUS )); then
    echo "ERROR: ROLLOUT_GPUS=${ROLLOUT_GPUS} leaves no training GPU out of ${NUM_GPUS} (${GPUS})." >&2; exit 1
fi
if (( ROLLOUT_GPUS > 0 && ROLLOUT_GPUS % ROLLOUT_TP )); then
    echo "ERROR: ROLLOUT_GPUS=${ROLLOUT_GPUS} is not a multiple of ROLLOUT_TP=${ROLLOUT_TP}." >&2; exit 1
fi

join() { local IFS=,; echo "$*"; }
TRAIN_GPU_LIST="$(join "${ALL_GPUS[@]:${ROLLOUT_GPUS}}")"
NUM_TRAIN=$(( NUM_GPUS - ROLLOUT_GPUS ))

free_port() { python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])'; }

VLLM_ARGS=()
SERVER_PID=""
# The server runs in its own process group (setsid) so its vLLM workers die with it.
cleanup() { [[ -n "${SERVER_PID}" ]] && kill -- "-${SERVER_PID}" 2>/dev/null; true; }
trap cleanup EXIT

if (( ROLLOUT_GPUS > 0 )); then
    ROLLOUT_GPU_LIST="$(join "${ALL_GPUS[@]:0:${ROLLOUT_GPUS}}")"
    VLLM_PORT="${VLLM_PORT:-$(free_port)}"
    VLLM_GROUP_PORT="${VLLM_GROUP_PORT:-$(free_port)}"
    EAGER_ARGS=()
    [[ "${RLVR_ENFORCE_EAGER:-0}" == 1 ]] && EAGER_ARGS=(--enforce_eager true)

    echo "vLLM server: GPUs ${ROLLOUT_GPU_LIST} (tp=${ROLLOUT_TP}, dp=$(( ROLLOUT_GPUS / ROLLOUT_TP ))), port ${VLLM_PORT}"
    CUDA_VISIBLE_DEVICES="${ROLLOUT_GPU_LIST}" setsid python -m trl.scripts.vllm_serve \
        --model "${MODEL}" \
        --tensor_parallel_size "${ROLLOUT_TP}" \
        --data_parallel_size $(( ROLLOUT_GPUS / ROLLOUT_TP )) \
        --host 127.0.0.1 \
        --port "${VLLM_PORT}" \
        --gpu_memory_utilization "${VLLM_MEM:-0.85}" \
        --max_model_len "${VLLM_MAX_MODEL_LEN}" \
        --dtype "${VLLM_DTYPE:-auto}" \
        ${EAGER_ARGS[@]+"${EAGER_ARGS[@]}"} \
        > "${OUTPUT_DIR}/vllm_server.log" 2>&1 &
    SERVER_PID=$!

    # First start compiles CUDA graphs; wait for /health instead of racing the trainer.
    for _ in $(seq 1 180); do
        curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null && break
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: vLLM server died; see ${OUTPUT_DIR}/vllm_server.log" >&2; exit 1
        fi
        sleep 5
    done
    curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null \
        || { echo "ERROR: vLLM server not up after 15 min; see ${OUTPUT_DIR}/vllm_server.log" >&2; exit 1; }

    VLLM_ARGS=(--vllm_mode server
               --vllm_server_base_url "http://127.0.0.1:${VLLM_PORT}"
               --vllm_group_port "${VLLM_GROUP_PORT}")
else
    echo "vLLM colocated with training on GPUs ${TRAIN_GPU_LIST}"
    VLLM_ARGS=(--vllm_mode colocate
               --vllm_gpu_memory_utilization "${VLLM_MEM:-0.3}"
               --vllm_max_model_length "${VLLM_MAX_MODEL_LEN}")
fi

echo "Training: GPUs ${TRAIN_GPU_LIST} (${NUM_TRAIN} processes) -> ${OUTPUT_DIR}"
MULTI_GPU=()
(( NUM_TRAIN > 1 )) && MULTI_GPU=(--multi_gpu)

CUDA_VISIBLE_DEVICES="${TRAIN_GPU_LIST}" accelerate launch \
    --num_processes "${NUM_TRAIN}" \
    --num_machines 1 \
    ${MULTI_GPU[@]+"${MULTI_GPU[@]}"} \
    --mixed_precision "${MIXED_PRECISION:-bf16}" \
    --dynamo_backend no \
    --main_process_port "$(free_port)" \
    train/grpo_trl.py \
    --model "${MODEL}" \
    --train-data "${TRAIN_DATA}" \
    ${VAL_DATA:+--val-data "${VAL_DATA}"} \
    --group-size "${GROUP_SIZE}" \
    --prompts-per-step "${PROMPTS_PER_STEP}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --max_steps "${MAX_STEPS}" \
    --learning_rate "${LR}" \
    --max_completion_length "${MAX_LEN}" \
    --seed "${SEED}" \
    --save_steps "${SAVE_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    "${VLLM_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
