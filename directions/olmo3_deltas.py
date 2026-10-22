#!/usr/bin/env python3
"""
Weight displacements along the OLMo 3 7B pipeline: pretraining stages 1-3, then SFT -> DPO -> RL
(Think and Instruct lines) and RL-Zero (RL straight from the base model).

Downloads the selected checkpoints in pipeline order, at most --max-on-disk at a time (each is
deleted once every pair using it is done). As soon as both ends of a pair are on disk it
computes ||theta_a||, ||theta_b||, ||theta_b - theta_a||, unchanged fractions, per-module shares
of ||Delta||^2 and a log10|Delta_i| histogram in a process pool. One JSON per pair under
--out-dir/pairs/ (existing ones are skipped), so reruns resume. Plot with directions/plot_olmo3_deltas.py.

Example:
    python -m directions.olmo3_deltas --ckpt-dir /scratch/micah/vm2825/models/olmo3 \
        --out-dir results/olmo3_deltas
"""

import argparse
import json
import multiprocessing
import os
import re
import shutil
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

BASE = "allenai/Olmo-3-1025-7B"
LOG_MIN, LOG_MAX, NBINS = -14.0, 2.0, 1600

# (stage name, repo, revisions in order). A stage starts from the previous stage's last checkpoint
# unless START says otherwise.
STAGES = [
    ("pretrain-s1", BASE, [f"stage1-step{s}" for s in
                           (0, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 400000, 700000,
                            1000000, 1413814)]),
    ("midtrain-s2", BASE, [f"stage2-step{s}" for s in (1000, 5000, 10000, 20000, 30000, 40000, 47684)]),
    ("longctx-s3", BASE, [f"stage3-step{s}" for s in (1000, 4000, 8000, 11921)] + ["main"]),
    ("think-sft", "allenai/Olmo-3-7B-Think-SFT", [f"step{s}" for s in (1000, 5000, 10000, 20000, 30000, 43000)]
     + ["main"]),
    ("think-dpo", "allenai/Olmo-3-7B-Think-DPO", ["main"]),
    ("think-rl", "allenai/Olmo-3-7B-Think", [f"step_{s:04d}" for s in (25, 100, 300, 600, 1000, 1375)] + ["main"]),
    ("instruct-sft", "allenai/Olmo-3-7B-Instruct-SFT", ["main"]),
    ("instruct-dpo", "allenai/Olmo-3-7B-Instruct-DPO", ["main"]),
    ("instruct-rl", "allenai/Olmo-3-7B-Instruct", [f"step_{s:03d}" for s in (50, 150, 250, 400)] + ["main"]),
    ("rlzero-math", "allenai/Olmo-3-7B-RL-Zero-Math", [f"step_{s}" for s in (100, 300, 600, 1000, 1500, 1900)]
     + ["main"]),
    ("rlzero-mix", "allenai/Olmo-3-7B-RL-Zero-Mix", [f"step_{s}" for s in (50, 250, 500, 950)] + ["main"]),
]
BASE_MAIN = (BASE, "main")
START = {
    "pretrain-s1": None,  # starts at its own first entry (random init, stage1-step0)
    "think-sft": BASE_MAIN,
    "instruct-sft": BASE_MAIN,
    "rlzero-math": BASE_MAIN,
    "rlzero-mix": BASE_MAIN,
}
# Extra pairs to check which checkpoint a stage really starts from.
# Stage-1 pretraining split at its midpoint (1413814 / 2 ~ step 707000), reported as stages of their own.
SPLITS = [
    ("pretrain-s1-1st50", (BASE, "stage1-step0"), (BASE, "stage1-step707000")),
    ("pretrain-s1-2nd50", (BASE, "stage1-step707000"), (BASE, "stage1-step1413814")),
]
EXTRA = [
    ((BASE, "stage3-step11921"), BASE_MAIN),
    (("allenai/Olmo-3-7B-Think-SFT", "main"), ("allenai/Olmo-3-7B-Instruct-SFT", "main")),
]


def ckpt_path(root: Path, ckpt) -> Path:
    repo, rev = ckpt
    return root / repo.split("/")[1] / rev


def pair_name(a, b) -> str:
    f = lambda c: f"{c[0].split('/')[1]}@{c[1]}"  # noqa: E731
    return f"{f(a)}__{f(b)}"


def build_pairs():
    """Returns [(kind, stage, a, b)] with kind in consecutive | from_start | stage_total | extra."""
    pairs, prev_end = [], None
    for stage, repo, revs in STAGES:
        ckpts = [(repo, r) for r in revs]
        start = START.get(stage, prev_end)
        chain = ckpts if start is None else [start] + ckpts
        for a, b in zip(chain, chain[1:]):
            pairs.append(("consecutive", stage, a, b))
        for b in chain[2:]:
            pairs.append(("from_start", stage, chain[0], b))
        pairs.append(("stage_total", stage, chain[0], chain[-1]))
        prev_end = chain[-1]
    pairs += [("stage_total", stage, a, b) for stage, a, b in SPLITS]
    pairs += [("extra", "extra", a, b) for a, b in EXTRA]
    seen, out = set(), []
    for p in pairs:
        key = (p[0], p[2], p[3])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def download(root: Path, ckpt) -> Path:
    from huggingface_hub import snapshot_download

    path = ckpt_path(root, ckpt)
    if (path / ".done").exists():
        return path
    for attempt in range(5):
        try:
            snapshot_download(ckpt[0], revision=ckpt[1], local_dir=path, max_workers=4,
                              allow_patterns=["*.safetensors", "*.json"])
            (path / ".done").touch()
            return path
        except Exception as e:  # transient network / rate-limit errors
            print(f"[download] {ckpt} attempt {attempt}: {e!r}", flush=True)
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"download failed: {ckpt}")


def module_kind(name: str) -> str:
    if "embed" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    if "norm" in name:
        return "norms"
    m = re.search(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.", name)
    return m.group(1) if m else "other"


def tensor_index(path: Path) -> dict[str, Path]:
    from safetensors import safe_open

    index = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                index[k] = f
    return index


_WORKER_DEVICE = "cpu"


def init_worker(counter, gpus: int) -> None:
    """Pin each pool process to one GPU for its lifetime (round-robin)."""
    global _WORKER_DEVICE
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    _WORKER_DEVICE = f"cuda:{i % gpus}" if gpus else "cpu"


def compute_pair(path_a: str, path_b: str, threads: int, device: str | None = None) -> dict:
    try:
        return _compute_pair(path_a, path_b, threads, device or _WORKER_DEVICE)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _compute_pair(path_a: str, path_b: str, threads: int, device: str) -> dict:
    from safetensors import safe_open

    torch.set_num_threads(threads)
    dev = torch.device(device)
    ia, ib = tensor_index(Path(path_a)), tensor_index(Path(path_b))
    keys = sorted(k for k in ia if k in ib and "rotary" not in k)
    missing = sorted(set(ia) ^ set(ib))
    sq_a = sq_b = sq_d = 0.0
    n = zero = le5 = 0
    dtypes = set()
    kn, ks, kz = defaultdict(int), defaultdict(float), defaultdict(int)
    counts = torch.zeros(NBINS, dtype=torch.float64, device=dev)
    sqh = torch.zeros(NBINS, dtype=torch.float64, device=dev)
    handles = {}

    def get(index, k):
        f = index[k]
        if f not in handles:
            handles[f] = safe_open(f, "pt")
        return handles[f].get_tensor(k)

    for k in keys:
        a, b = get(ia, k), get(ib, k)
        if not a.is_floating_point():
            continue
        a, b = a.to(dev, non_blocking=True), b.to(dev, non_blocking=True)
        dtypes.add(f"{a.dtype}/{b.dtype}")
        d = b.float() - a.float()
        sa, sb, sd = ((a.double() ** 2).sum().item(), (b.double() ** 2).sum().item(), (d.double() ** 2).sum().item())
        sq_a += sa
        sq_b += sb
        sq_d += sd
        n += d.numel()
        z = (d == 0).sum().item()
        zero += z
        le5 += (d.abs() <= 1e-5).sum().item()
        kind = module_kind(k)
        kn[kind] += d.numel()
        ks[kind] += sd
        kz[kind] += z
        ad = d.abs().flatten()
        ad = ad[ad > 0]
        idx = ((ad.log10() - LOG_MIN) / (LOG_MAX - LOG_MIN) * NBINS).long().clamp_(0, NBINS - 1)
        counts += torch.bincount(idx, minlength=NBINS).double()
        sqh += torch.bincount(idx, weights=ad.double() ** 2, minlength=NBINS)
    return {
        "numel": n, "norm_a": sq_a ** 0.5, "norm_b": sq_b ** 0.5, "norm_delta": sq_d ** 0.5,
        "frac_exact_zero": zero / n, "frac_le_1e-5": le5 / n, "dtypes": sorted(dtypes),
        "keys_not_in_both": missing[:20],
        "by_kind": {k: {"numel": kn[k], "sq_delta": ks[k], "frac_zero": kz[k] / kn[k]} for k in kn},
        "log_hist": {"log_min": LOG_MIN, "log_max": LOG_MAX, "counts": counts.cpu().tolist(), "sq": sqh.cpu().tolist()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-on-disk", type=int, default=8,
                        help="Checkpoints held on disk at once (~15 GB each); each is deleted once its pairs are done.")
    parser.add_argument("--compute-workers", type=int, default=6)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count(),
                        help="Spread workers over this many GPUs (0 = CPU).")
    parser.add_argument("--threads-per-worker", type=int, default=6)
    parser.add_argument("--keep", action="store_true", help="Do not delete checkpoints after use.")
    args = parser.parse_args()

    root, out = Path(args.ckpt_dir), Path(args.out_dir)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    pairs = build_pairs()
    (out / "pairs.json").write_text(json.dumps(
        [{"kind": k, "stage": s, "a": list(a), "b": list(b), "file": pair_name(a, b) + ".json"}
         for k, s, a, b in pairs], indent=1))
    pending = [p for p in pairs if not (out / "pairs" / (pair_name(p[2], p[3]) + ".json")).exists()]
    # Pipeline order: a stage's start checkpoint comes before its members, so it stays on disk while needed.
    queue = list(dict.fromkeys(c for p in pending for c in (p[2], p[3])))
    print(f"{len(pairs)} pairs, {len(pending)} to compute, {len(queue)} checkpoints, "
          f"at most {args.max_on_disk} on disk", flush=True)

    users = defaultdict(int)
    for p in pending:
        users[p[2]] += 1
        users[p[3]] += 1
    cond = threading.Condition()
    ready, downloading, running = {}, set(), set()
    ctx = multiprocessing.get_context("spawn")
    compute = ProcessPoolExecutor(args.compute_workers, mp_context=ctx, initializer=init_worker,
                                  initargs=(ctx.Value("i", 0), args.gpus))
    dl = ThreadPoolExecutor(args.max_on_disk)

    def release(c):  # called with cond held
        users[c] -= 1
        if users[c] == 0:
            ready.pop(c, None)
            if not args.keep:
                shutil.rmtree(ckpt_path(root, c), ignore_errors=True)
                print(f"[disk] deleted {c[0].split('/')[1]}@{c[1]}", flush=True)

    def finish(p, fut):
        try:
            res = fut.result()
            res.update(kind=p[0], stage=p[1], a=list(p[2]), b=list(p[3]))
            (out / "pairs" / (pair_name(p[2], p[3]) + ".json")).write_text(json.dumps(res))
            print(f"[compute] {p[0]:12s} {p[1]:13s} {p[2][1]:>20s} -> {p[3][1]:<20s} "
                  f"||D|| {res['norm_delta']:.4g}  ||a|| {res['norm_a']:.1f}  "
                  f"unchanged {res['frac_exact_zero']:.1%}", flush=True)
        except Exception as e:
            print(f"[compute] FAILED {pair_name(p[2], p[3])}: {e!r}", flush=True)
        with cond:
            running.discard(p)
            release(p[2])
            release(p[3])
            cond.notify_all()

    def fetch(c):
        try:
            path = download(root, c)
        except Exception as e:
            print(f"[download] FAILED {c}: {e!r}", flush=True)
            path = None
        with cond:
            downloading.discard(c)
            if path is None:  # drop every pair that needs it
                for p in [p for p in pending if c in (p[2], p[3])]:
                    pending.remove(p)
                    release(p[2] if p[3] == c else p[3])
            else:
                ready[c] = str(path)
                print(f"[download] done {c[0].split('/')[1]}@{c[1]}", flush=True)
            cond.notify_all()

    with cond:
        while queue or downloading or running or pending:
            for p in [p for p in pending if p[2] in ready and p[3] in ready]:
                pending.remove(p)
                running.add(p)
                f = compute.submit(compute_pair, ready[p[2]], ready[p[3]], args.threads_per_worker)
                f.add_done_callback(lambda fut, p=p: finish(p, fut))
            while queue and len(ready) + len(downloading) < args.max_on_disk:
                c = queue.pop(0)
                downloading.add(c)
                dl.submit(fetch, c)
            if not downloading and not running and pending and queue:
                raise RuntimeError("window full and nothing can run; raise --max-on-disk")
            if not queue and not downloading and not running and pending:
                print(f"{len(pending)} pairs could not run", flush=True)
                break
            cond.wait(timeout=30)
    compute.shutdown()
    dl.shutdown()
    print("done", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    main()
