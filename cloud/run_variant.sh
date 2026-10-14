#!/usr/bin/env bash
# On the box: run one 2d_perturb.py variant, restarting it on failure (it resumes from its
# CSV). Writes OUT_DIR/DONE or OUT_DIR/FAILED at the end.
#
#   nohup bash cloud/run_variant.sh VARIANT OUT_DIR [extra 2d_perturb args] \
#       >> /workspace/run2d.log 2>&1 < /dev/null &
set -u
VARIANT="$1"; OUT_DIR="$2"; shift 2
MAX_TRIES="${MAX_TRIES:-4}"
cd /workspace/rlvr-landscape && source cloud/env.sh
mkdir -p "${OUT_DIR}"; rm -f "${OUT_DIR}/DONE" "${OUT_DIR}/FAILED"

for attempt in $(seq 1 "${MAX_TRIES}"); do
    echo "=== ${VARIANT} attempt ${attempt} $(date '+%F %T')"
    if PYTHONPATH=. python landscape/2d_perturb.py --model-ckpt "${BASE_MODEL}" \
            --variant "${VARIANT}" --output-dir "${OUT_DIR}" "$@"; then
        touch "${OUT_DIR}/DONE"; echo "=== ${VARIANT} done $(date '+%F %T')"; exit 0
    fi
    echo "=== ${VARIANT} attempt ${attempt} failed $(date '+%F %T')"
    pkill -9 -f 'VLLM::EngineCore'; sleep 20
done
touch "${OUT_DIR}/FAILED"; exit 1
