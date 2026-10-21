#!/usr/bin/env bash
# Fetch the userspace NVML (libnvidia-ml) + nvidia-smi that match the *loaded* kernel driver.
#
# When a node's driver packages are upgraded without a reboot, the kernel module stays old and
# NVML fails with "Driver/library version mismatch": nvidia-smi breaks, vLLM cannot detect the
# GPU platform, and NCCL refuses to init (so GRPO weight sync to the vLLM server dies). CUDA
# itself keeps working. cluster/env.sh puts the matching copy on LD_LIBRARY_PATH/PATH.
#
#   bash cluster/fetch_nvml.sh     # -> $WORK/nvidia-<version>/{lib,bin}
#
# Packages are only unpacked (dpkg-deb -x); nothing is installed and no package scripts run.
set -euo pipefail

WORK="${WORK:-/scratch/micah/${USER}}"
VER="$(sed -nE 's/.*Kernel Module( for x86_64)?[[:space:]]+([0-9.]+).*/\2/p' /proc/driver/nvidia/version)"
[[ -n "${VER}" ]] || { echo "cannot read the driver version from /proc/driver/nvidia/version" >&2; exit 1; }
DEST="${WORK}/nvidia-${VER}"
REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64"
DEB="libnvidia-compute_${VER}-0ubuntu1_amd64.deb"   # has libnvidia-ml and nvidia-smi

mkdir -p "${DEST}/debs" "${DEST}/root" "${DEST}/lib" "${DEST}/bin"
curl -fsSL -o "${DEST}/debs/${DEB}" "${REPO}/${DEB}"
dpkg-deb -x "${DEST}/debs/${DEB}" "${DEST}/root"
cp -P "${DEST}"/root/usr/lib/x86_64-linux-gnu/libnvidia-ml.so* "${DEST}/lib/"
cp "${DEST}/root/usr/bin/nvidia-smi" "${DEST}/bin/"
echo "NVML ${VER} -> ${DEST}"
LD_LIBRARY_PATH="${DEST}/lib" "${DEST}/bin/nvidia-smi" -L
