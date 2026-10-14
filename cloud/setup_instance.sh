#!/usr/bin/env bash
# One-time setup on a fresh GPU box (Vast.ai or any Linux + NVIDIA machine):
#   curl -LsSf https://raw.githubusercontent.com/suhas1999/rlvr-landscape/refactor/organize/cloud/setup_instance.sh | bash
# or, with the repo already cloned:  bash cloud/setup_instance.sh
# Safe to re-run.
set -euo pipefail

WORK="${WORK:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/suhas1999/rlvr-landscape.git}"
BRANCH="${BRANCH:-refactor/organize}"
REPO_ROOT="${WORK}/rlvr-landscape"

mkdir -p "${WORK}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ---- system packages (gcc for Triton, tmux for detached runs) ----
need=()
for cmd in git gcc tmux rsync curl; do
    command -v "${cmd}" >/dev/null || need+=("${cmd}")
done
if [ ${#need[@]} -gt 0 ]; then
    pkgs="${need[*]}"
    pkgs="${pkgs/gcc/build-essential}"
    SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    ${SUDO} apt-get update -qq
    DEBIAN_FRONTEND=noninteractive ${SUDO} apt-get install -y -qq ${pkgs}
fi

# ---- uv ----
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

# ---- repo ----
if [ ! -d "${REPO_ROOT}/.git" ]; then
    git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO_ROOT}"
else
    git -C "${REPO_ROOT}" pull --ff-only || echo "git pull skipped (local changes?)"
fi
cd "${REPO_ROOT}"

source cloud/env.sh

# ---- Python env (exact versions from uv.lock) ----
uv sync --frozen
source .venv/bin/activate
python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'| vllm', vllm.__version__, '| GPU ok:', torch.cuda.is_available())"

# ---- HF models ----
[ -f "${BASE_MODEL}/config.json" ] || hf download Qwen/Qwen3-0.6B --local-dir "${BASE_MODEL}"
[ -f "${PRETRAINED_MODEL}/config.json" ] || hf download Qwen/Qwen3-0.6B-Base --local-dir "${PRETRAINED_MODEL}"
mkdir -p "${CKPT_ROOT}" "${WANDB_DIR}"

# ---- shell convenience ----
line="source ${REPO_ROOT}/cloud/env.sh && cd ${REPO_ROOT}"
grep -qxF "${line}" "${HOME}/.bashrc" 2>/dev/null || echo "${line}" >> "${HOME}/.bashrc"

echo
echo "Setup done. Next:"
echo "  wandb login        # once"
echo "  python -m cloud.smoke_test --model \${BASE_MODEL}"
