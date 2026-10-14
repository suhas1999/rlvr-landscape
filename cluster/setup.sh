#!/usr/bin/env bash
# One-time setup on the AICS cluster (run from a GPU node so the import check sees CUDA):
#   bash cluster/setup.sh
# Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh

command -v uv >/dev/null || { echo "uv not on PATH (expected ${WORK}/bin/uv)"; exit 1; }
mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${VLLM_CACHE_ROOT}" "${MODELS_DIR}" "${CKPT_ROOT}" \
    "${RESULTS_DIR}" "${WANDB_DIR}"

# Exact versions from uv.lock (torch 2.13 / CUDA 13, vLLM 0.31).
uv sync --frozen
source .venv/bin/activate
python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'| vllm', vllm.__version__, '| GPU ok:', torch.cuda.is_available())"

[ -f "${BASE_MODEL}/config.json" ] || hf download Qwen/Qwen3-0.6B --local-dir "${BASE_MODEL}"
[ -f "${PRETRAINED_MODEL}/config.json" ] || hf download Qwen/Qwen3-0.6B-Base --local-dir "${PRETRAINED_MODEL}"

echo
echo "Setup done. In each new shell:  source cluster/env.sh"
echo "  wandb login        # once (optional)"
echo "  bash cluster/eval_math500.sh \${PRETRAINED_MODEL}"
