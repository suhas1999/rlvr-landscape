#!/usr/bin/env bash
# Local helper for Vast.ai (runs on your laptop). Needs a Vast API key stored once with
#   uvx vastai set api-key <KEY>        (saved to ~/.config/vastai/vast_api_key)
#
#   bash cloud/vast.sh offers [GPU_NAME] [on-demand|bid]   cheapest single-GPU offers
#   bash cloud/vast.sh create OFFER_ID [BID_PRICE]         rent it; setup runs automatically
#   bash cloud/vast.sh list                                your instances
#   bash cloud/vast.sh ssh ID                              shell on the box
#   bash cloud/vast.sh log ID                              follow the automatic setup log
#   bash cloud/vast.sh push ID LOCAL_PATH [REMOTE_DIR]     copy e.g. a checkpoint up
#   bash cloud/vast.sh pull ID                             copy figs_*/logs_* back to ./results/
#   bash cloud/vast.sh stop ID | start ID | destroy ID
#
# Stopped instances still bill for disk; `destroy` when finished (pull results first).
set -euo pipefail

VAST="uvx vastai"
IMAGE="${IMAGE:-vastai/base-image:cuda-12.8.1-auto}"
DISK_GB="${DISK_GB:-80}"
BRANCH="${BRANCH:-refactor/organize}"
SETUP_URL="https://raw.githubusercontent.com/suhas1999/rlvr-landscape/${BRANCH}/cloud/setup_instance.sh"

ssh_parts() {
    # ssh://root@HOST:PORT -> "HOST PORT"
    local url
    url="$(${VAST} ssh-url "$1")"
    url="${url#ssh://}"; url="${url#*@}"
    echo "${url%:*} ${url##*:}"
}

ssh_opts() {
    echo "-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
}

cmd="${1:-help}"; shift || true
case "${cmd}" in
    offers)
        gpu="${1:-H100_SXM}"; type="${2:-on-demand}"
        ${VAST} search offers \
            "gpu_name=${gpu} num_gpus=1 reliability>0.98 cuda_vers>=12.8 inet_down>200 disk_space>=${DISK_GB}" \
            -t "${type}" --storage "${DISK_GB}" -o dph_total --limit 10
        ;;
    create)
        offer="$1"; bid="${2:-}"
        extra=()
        [ -n "${bid}" ] && extra+=(--bid_price "${bid}")
        ${VAST} create instance "${offer}" \
            --image "${IMAGE}" --disk "${DISK_GB}" --ssh --direct \
            --label rlvr-landscape \
            --onstart-cmd "curl -LsSf ${SETUP_URL} | WORK=/workspace BRANCH=${BRANCH} bash > /workspace/setup.log 2>&1" \
            ${extra[@]+"${extra[@]}"}
        echo "Wait ~1-3 min, then: bash cloud/vast.sh log <INSTANCE_ID>"
        ;;
    list)
        ${VAST} show instances
        ;;
    ssh)
        read -r host port <<<"$(ssh_parts "$1")"
        exec ssh $(ssh_opts) -p "${port}" "root@${host}"
        ;;
    log)
        read -r host port <<<"$(ssh_parts "$1")"
        exec ssh $(ssh_opts) -p "${port}" "root@${host}" "tail -n 50 -f /workspace/setup.log"
        ;;
    push)
        read -r host port <<<"$(ssh_parts "$1")"
        src="$2"; dst="${3:-/workspace/checkpoints/}"
        rsync -avP -e "ssh $(ssh_opts) -p ${port}" "${src}" "root@${host}:${dst}"
        ;;
    pull)
        read -r host port <<<"$(ssh_parts "$1")"
        mkdir -p results
        rsync -avP -e "ssh $(ssh_opts) -p ${port}" \
            --include='figs_*/***' --include='logs_*/***' --exclude='*' \
            "root@${host}:/workspace/rlvr-landscape/" results/
        ;;
    stop)    ${VAST} stop instance "$1" ;;
    start)   ${VAST} start instance "$1" ;;
    destroy) ${VAST} destroy instance "$1" ;;
    *)
        sed -n '2,15p' "$0"
        ;;
esac
