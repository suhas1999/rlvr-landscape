# Source on the GPU box before running anything:  source cloud/env.sh
# Everything lives under $WORK so it survives a stop/start (and interruptible pauses).

export WORK="${WORK:-/workspace}"
export REPO_ROOT="${WORK}/rlvr-landscape"

export UV_CACHE_DIR="${WORK}/.cache/uv"
export HF_HOME="${WORK}/.cache/huggingface"
# torch.compile / CUDA-graph cache: later engine starts skip recompilation.
export VLLM_CACHE_ROOT="${WORK}/.cache/vllm"
# apply_model sends a closure to the V1 engine-core process.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# One dedicated GPU: use most of it (the cluster launchers used 0.25-0.3 on shared GPUs).
# Lower GPU_MEMORY_UTILIZATION to 0.3 for landscape_sgd.sh without DIRECTION_PATH,
# where vLLM and the HF backward share the GPU in one process.
export GPU=0
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# Set to 0 after cloud/smoke_test.py passes "graph sees the weight update".
export RLVR_ENFORCE_EAGER="${RLVR_ENFORCE_EAGER:-1}"

export BASE_MODEL="${WORK}/models/Qwen3-0.6B"            # Qwen/Qwen3-0.6B (post-trained, run non-thinking)
export PRETRAINED_MODEL="${WORK}/models/Qwen3-0.6B-Base"  # Qwen/Qwen3-0.6B-Base (pretrain only)
export CKPT_ROOT="${WORK}/checkpoints"

# Launcher defaults: the HF model instead of the cluster checkpoints.
export MODEL_CKPT="${MODEL_CKPT:-${BASE_MODEL}}"
export CHECKPOINT_STEP="${CHECKPOINT_STEP:-0}"

# Weights & Biases: run `wandb login` once on the box (key goes to ~/.netrc).
# Without a key the scripts still run and just skip logging.
export WANDB_PROJECT="${WANDB_PROJECT:-rlvr-landscape}"
export WANDB_DIR="${WORK}/wandb"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi
