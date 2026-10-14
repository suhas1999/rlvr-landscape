# Source on an AICS cluster node (interactive srun shell or inside an sbatch script):
#   source cluster/env.sh                 # uses the 1st GPU Slurm gave you
#   GPU_INDEX=2 source cluster/env.sh     # 3rd allocated GPU (run independent jobs side by side)
# Everything heavy (venv, uv/HF/vLLM caches, models) lives on /scratch, not the NFS $HOME.

export WORK="${WORK:-/scratch/micah/${USER}}"
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export UV_CACHE_DIR="${WORK}/.cache/uv"
export UV_PYTHON_INSTALL_DIR="${WORK}/.local/uv-python"
export HF_HOME="${WORK}/.cache/huggingface"
# torch.compile / CUDA-graph cache: later engine starts skip recompilation.
export VLLM_CACHE_ROOT="${WORK}/.cache/vllm"
export TRITON_CACHE_DIR="${WORK}/.cache/triton"
# FlashInfer JIT-compiles kernels for sm_120 (RTX PRO 6000) on first use. /usr/local/cuda on the
# nodes is a stub without nvcc, so point at the full toolkit; cache lands in $WORK/.cache/flashinfer.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export FLASHINFER_WORKSPACE_BASE="${WORK}"
# apply_to_model (collective_rpc) sends a closure to the V1 engine-core process.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Stream prints into sbatch / nohup logs instead of block-buffering them.
export PYTHONUNBUFFERED=1

# ---- GPU pinning ----
# On csrtx6000-2 one GPU (/dev/nvidia7) is missing from gres.conf, so Slurm's cgroup does not
# hide it: plain indices (CUDA_VISIBLE_DEVICES=1) can land on a GPU you were not allocated.
# Map Slurm's allocated device minors to UUIDs and pin by UUID.
_slurm_gpus="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
GPU_UUIDS=""
for _minor in ${_slurm_gpus//,/ }; do
    _info=$(grep -l -E "Device Minor:[[:space:]]+${_minor}\$" /proc/driver/nvidia/gpus/*/information 2>/dev/null | head -1)
    _uuid=$(awk -F: '/GPU UUID/ {gsub(/[[:space:]]/,"",$2); print $2}' "${_info}" 2>/dev/null)
    if [ -z "${_uuid}" ]; then
        echo "cluster/env.sh: cannot map /dev/nvidia${_minor} to a UUID" >&2
        GPU_UUIDS=""; break
    fi
    GPU_UUIDS="${GPU_UUIDS:+${GPU_UUIDS},}${_uuid}"
done
export GPU_UUIDS
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${GPU_UUIDS}" ]; then
    IFS=, read -r -a _uuids <<< "${GPU_UUIDS}"
    _idx="${GPU_INDEX:-0}"
    if [ "${_idx}" -ge "${#_uuids[@]}" ]; then
        echo "cluster/env.sh: GPU_INDEX=${_idx} but only ${#_uuids[@]} GPU(s) allocated" >&2
    else
        # Launchers do CUDA_VISIBLE_DEVICES="${GPU}", so GPU carries the UUID.
        export GPU="${_uuids[${_idx}]}"
        export CUDA_VISIBLE_DEVICES="${GPU}"
        echo "cluster/env.sh: job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}: GPU_INDEX=${_idx} -> ${GPU}" >&2
    fi
else
    echo "cluster/env.sh: no Slurm GPU allocation found; leaving CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
fi
unset _slurm_gpus _minor _info _uuid _uuids _idx

# One dedicated GPU (96GB RTX PRO 6000): use most of it.
# Lower to 0.3 for landscape_sgd.sh without DIRECTION_PATH (vLLM + HF backward in one process).
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
# Keep one mode across runs you compare (eager and graph kernels differ by ~0.04 logp).
export RLVR_ENFORCE_EAGER="${RLVR_ENFORCE_EAGER:-0}"

export MODELS_DIR="${WORK}/models"
export BASE_MODEL="${MODELS_DIR}/Qwen3-0.6B"            # Qwen/Qwen3-0.6B (post-trained, run non-thinking)
export PRETRAINED_MODEL="${MODELS_DIR}/Qwen3-0.6B-Base"  # Qwen/Qwen3-0.6B-Base (pretrain only)
export CKPT_ROOT="${WORK}/checkpoints"
export RESULTS_DIR="${WORK}/results/rlvr-landscape"

export MODEL_CKPT="${MODEL_CKPT:-${BASE_MODEL}}"
export CHECKPOINT_STEP="${CHECKPOINT_STEP:-0}"

export WANDB_PROJECT="${WANDB_PROJECT:-rlvr-landscape}"
export WANDB_DIR="${WORK}/wandb"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
