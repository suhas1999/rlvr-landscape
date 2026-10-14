#!/usr/bin/env bash
# Run on the laptop. Waits for a remote 2d_perturb.py run to end, downloads the results,
# verifies them, then destroys the instance. Never destroys unverified data: if the
# download cannot be verified it STOPS the instance instead (disk kept, GPU billing ends).
#
#   caffeinate -ims bash cloud/finish_and_destroy.sh INSTANCE_ID HOST PORT REMOTE_DIR LOCAL_DIR
#
# Also ends the run early (keeping partial results) after MAX_HOURS, or when the CSV has
# not grown for STALL_MIN minutes.
set -uo pipefail

ID="$1"; HOST="$2"; PORT="$3"; REMOTE_DIR="$4"; LOCAL_DIR="$5"
MAX_HOURS="${MAX_HOURS:-4}"
STALL_MIN="${STALL_MIN:-40}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SSH=(ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -p "${PORT}" "root@${HOST}")
mkdir -p "${LOCAL_DIR}"
LOG="${LOCAL_DIR}/auto_finish.log"
log() { echo "$(date '+%F %T') $*" | tee -a "${LOG}"; }

remote() { "${SSH[@]}" "$@" 2>/dev/null | grep -v "Welcome to vast.ai\|Have fun"; }

start=$(date +%s); last_rows=-1; last_change=$(date +%s); ssh_fail=0; reason=""
log "watching instance ${ID} (${HOST}:${PORT}) ${REMOTE_DIR}"
while true; do
    out=$(remote "pgrep -f '[l]andscape/2d_perturb' >/dev/null && echo RUN || echo END; cat ${REMOTE_DIR}/*.csv 2>/dev/null | wc -l")
    if [ -z "${out}" ]; then
        ssh_fail=$((ssh_fail + 1)); log "ssh failed (${ssh_fail})"
        if [ ${ssh_fail} -ge 60 ]; then reason="ssh unreachable"; break; fi
        sleep 60; continue
    fi
    ssh_fail=0
    state=$(echo "${out}" | head -1); rows=$(echo "${out}" | sed -n 2p | tr -d ' ')
    now=$(date +%s)
    [ "${rows}" != "${last_rows}" ] && { last_rows=${rows}; last_change=${now}; log "state=${state} csv_lines=${rows}"; }
    if [ "${state}" = END ]; then reason="run finished"; break; fi
    if [ $((now - start)) -ge $((MAX_HOURS * 3600)) ]; then reason="hit ${MAX_HOURS}h cap"; break; fi
    if [ $((now - last_change)) -ge $((STALL_MIN * 60)) ]; then reason="stalled ${STALL_MIN} min"; break; fi
    sleep 60
done
log "ending: ${reason}"

if [ "${reason}" = "ssh unreachable" ]; then
    log "cannot reach the box; STOPPING it (data kept on disk)"
    uvx vastai stop instance "${ID}" 2>&1 | tee -a "${LOG}"
    exit 1
fi

if [ "${reason}" != "run finished" ]; then
    log "stopping the run early and plotting the partial grid"
    remote "pkill -f '[l]andscape/2d_perturb'; sleep 10; pkill -9 -f 'VLLM::EngineCore'; cd /workspace/rlvr-landscape && source cloud/env.sh && for f in ${REMOTE_DIR}/*.csv; do python -m landscape.plot_2d \$f; done" | tail -2 | tee -a "${LOG}"
fi

verified=0
for attempt in 1 2 3 4 5; do
    rsync -a -e "ssh -o ConnectTimeout=20 -p ${PORT}" "root@${HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}/" 2>>"${LOG}"
    rsync -a -e "ssh -o ConnectTimeout=20 -p ${PORT}" "root@${HOST}:/workspace/run2d.log" "${LOCAL_DIR}/" 2>>"${LOG}"
    remote_lines=$(remote "cat ${REMOTE_DIR}/*.csv | wc -l" | tr -d ' ')
    local_lines=$(cat "${LOCAL_DIR}"/*.csv 2>/dev/null | wc -l | tr -d ' ')
    log "attempt ${attempt}: remote csv lines=${remote_lines} local=${local_lines}"
    if [ -n "${remote_lines}" ] && [ "${remote_lines}" = "${local_lines}" ] && [ "${local_lines}" -gt 1 ]; then
        verified=1; break
    fi
    sleep 30
done

if [ ${verified} -ne 1 ]; then
    log "download NOT verified; STOPPING the instance instead of destroying (data kept)"
    uvx vastai stop instance "${ID}" 2>&1 | tee -a "${LOG}"
    exit 1
fi

# Make sure the plots exist locally (re-plot from the CSV if the remote plot step failed).
if ! ls "${LOCAL_DIR}"/*_surface3d.png >/dev/null 2>&1; then
    log "plots missing; plotting locally"
    for f in "${LOCAL_DIR}"/*.csv; do
        PYTHONPATH="${REPO}" uv run --no-project -q --with matplotlib --with pandas --with plotly --with scipy \
            python -m landscape.plot_2d "${f}" 2>&1 | tail -1 | tee -a "${LOG}"
    done
fi

log "verified download:"; ls -la "${LOCAL_DIR}" | tee -a "${LOG}"
log "destroying instance ${ID}"
uvx vastai destroy instance "${ID}" -y 2>&1 | tee -a "${LOG}"
uvx vastai show instances 2>&1 | tail -3 | tee -a "${LOG}"
log "done"
