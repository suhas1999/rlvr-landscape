#!/usr/bin/env bash
# GRPO training (train/grpo_trl.sh) on the AICS cluster, interactive (inside srun) or batch:
#   bash cluster/train_grpo.sh [extra grpo_trl.py / GRPOConfig args]
#   sbatch cluster/train_grpo.sh                          # 2 GPUs: 1 rollout + 1 train
#   sbatch --gres=gpu:4 cluster/train_grpo.sh             # 1 rollout + 3 train
#   ROLLOUT_GPUS=2 sbatch --gres=gpu:4 cluster/train_grpo.sh
# All Slurm-allocated GPUs are used (pinned by UUID, see env.sh); env vars of grpo_trl.sh apply.
#SBATCH --job-name=grpo-trl
#SBATCH --partition=micah
#SBATCH --account=micah
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/micah/%u/logs/%x-%j.out
set -euo pipefail

# sbatch runs a copy of this file from the spool dir, so locate the repo via the submit dir.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "${REPO}/cluster/env.sh" ] || REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO}"
source cluster/env.sh

bash train/grpo_trl.sh "$@"
