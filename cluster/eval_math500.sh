#!/usr/bin/env bash
# MATH500 accuracy with vLLM on one GPU. Interactive (inside srun) or batch:
#   bash cluster/eval_math500.sh [MODEL_PATH] [extra eval_math500.py args]
#   sbatch cluster/eval_math500.sh /scratch/micah/$USER/models/Qwen3-0.6B-Base
#   GPU_INDEX=1 bash cluster/eval_math500.sh ...   # 2nd allocated GPU
# Default model: Qwen3-0.6B-Base (plain prompt, greedy, 8192 new tokens).
#SBATCH --job-name=math500-eval
#SBATCH --partition=micah
#SBATCH --account=micah
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/micah/%u/logs/%x-%j.out
set -euo pipefail

# sbatch runs a copy of this file from the spool dir, so locate the repo via the submit dir.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "${REPO}/cluster/env.sh" ] || REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO}"
source cluster/env.sh

MODEL_PATH="${1:-${PRETRAINED_MODEL}}"
shift || true

python -m eval.eval_math500 \
    --model-path "${MODEL_PATH}" \
    --output-dir "${RESULTS_DIR}/math500_eval" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    "$@"
