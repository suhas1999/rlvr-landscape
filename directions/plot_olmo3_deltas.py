#!/usr/bin/env python3
"""
Plots for directions/olmo3_deltas.py output.

    python -m directions.plot_olmo3_deltas --out-dir results/olmo3_deltas

Writes stage_totals.png (per-stage ||Delta||, relative size, unchanged fraction, |Delta_i| histograms),
trajectories.png (||theta_t - theta_stage_start|| along each stage) and modules.png.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ORDER = ["pretrain-s1", "pretrain-s1-1st50", "pretrain-s1-2nd50", "midtrain-s2", "longctx-s3", "think-sft", "think-dpo", "think-rl",
         "instruct-sft", "instruct-dpo", "instruct-rl", "rlzero-math", "rlzero-mix"]
GROUP_COLOR = {"pretrain": "C7", "midtrain": "C7", "longctx": "C7", "think": "C0", "instruct": "C1", "rlzero": "C3"}
KINDS = ["embed", "lm_head", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "norms"]
# Reference RL runs from this project (relative size ||Delta|| / ||theta_start||).
REFS = {"Qwen3-0.6B GRPO (ours)": 1.136 / 1440.8, "DeepSeek-Math-7B instruct→RL": 5.077 / 6751.3}


def color(stage: str) -> str:
    return GROUP_COLOR[stage.split("-")[0]]


def step_of(rev: str) -> float | None:
    m = re.search(r"step_?(\d+)", rev)
    return float(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    # One file per (a, b); a pair can play several roles (e.g. a stage's last from_start is also its
    # stage_total), so roles come from pairs.json, not from the file.
    rows = []
    for p in json.loads((out / "pairs.json").read_text()):
        f = out / "pairs" / p["file"]
        if f.exists():
            rows.append(json.loads(f.read_text()) | {"kind": p["kind"], "stage": p["stage"]})
    totals = {r["stage"]: r for r in rows if r["kind"] == "stage_total"}
    stages = [s for s in ORDER if s in totals]

    # ---------------- stage totals ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    x = np.arange(len(stages))
    cols = [color(s) for s in stages]

    ax = axes[0, 0]
    v = [totals[s]["norm_delta"] for s in stages]
    ax.bar(x, v, color=cols)
    for i, val in enumerate(v):
        ax.text(i, val, f"{val:.3g}", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x, stages, rotation=35, ha="right")
    ax.set_ylabel("‖θ_end − θ_start‖")
    ax.set_title("(a) displacement over each stage")

    ax = axes[0, 1]
    v = [totals[s]["norm_delta"] / totals[s]["norm_a"] for s in stages]
    ax.bar(x, v, color=cols)
    for i, val in enumerate(v):
        ax.text(i, val, f"{val:.1e}", ha="center", va="bottom", fontsize=8)
    for (label, ref), ls in zip(REFS.items(), ["--", ":"]):
        ax.axhline(ref, color="k", ls=ls, lw=1, label=f"{label}: {ref:.1e}")
    ax.set_yscale("log")
    ax.set_xticks(x, stages, rotation=35, ha="right")
    ax.set_ylabel("‖Δ‖ / ‖θ_start‖")
    ax.set_title("(b) relative displacement")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    v = [totals[s]["frac_exact_zero"] for s in stages]
    ax.bar(x, v, color=cols)
    for i, val in enumerate(v):
        ax.text(i, val, f"{val:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xticks(x, stages, rotation=35, ha="right")
    ax.set_ylabel("fraction of weights exactly unchanged")
    ax.set_title("(c) 'update sparsity' (bf16 checkpoints, tolerance 0)")

    ax = axes[1, 1]
    for s in stages:
        h = totals[s]["log_hist"]
        counts = np.array(h["counts"]).reshape(160, -1).sum(1)
        edges = np.logspace(h["log_min"], h["log_max"], 161)
        ls = "-" if s.split("-")[1] in ("rl", "math", "mix", "s1", "s2", "s3") and not s.endswith("50") else "--"
        ax.stairs(counts / totals[s]["numel"], edges, color=color(s), ls=ls, lw=1.3, label=s)
    ax.set_xscale("log")
    ax.set_xlim(1e-8, 10)
    ax.set_xlabel("|Δ_i|  (exact zeros not shown)")
    ax.set_ylabel("fraction of weights")
    ax.set_title("(d) per-weight |Δ_i| over each stage")
    ax.legend(fontsize=7, ncol=2)

    n = totals[stages[0]]["numel"]
    fig.suptitle(f"OLMo 3 7B (N = {n / 1e9:.2f}B): weight displacement per training stage   "
                 "(grey = base pretraining, blue = Think, orange = Instruct, red = RL-Zero from base)")
    fig.tight_layout()
    fig.savefig(out / "stage_totals.png", dpi=150)

    # ---------------- trajectories ----------------
    traj = [s for s in stages if any(r["kind"] == "from_start" and r["stage"] == s for r in rows)]
    ncol = 3
    fig, axes = plt.subplots(int(np.ceil(len(traj) / ncol)), ncol, figsize=(15, 4 * np.ceil(len(traj) / ncol)),
                             squeeze=False)
    for ax, s in zip(axes.flat, traj):
        pts = [r for r in rows if r["stage"] == s and r["kind"] in ("from_start", "stage_total")]
        pts += [r for r in rows if r["stage"] == s and r["kind"] == "consecutive" and r["a"] == totals[s]["a"]]
        xy = {}
        main_val = None
        for r in pts:
            st = step_of(r["b"][1])
            if st is None:
                main_val = r["norm_delta"]
            else:
                xy[st] = r["norm_delta"]
        xs = sorted(xy)
        ax.plot(xs, [xy[k] for k in xs], "o-", color=color(s), ms=4)
        if main_val is not None:
            ax.axhline(main_val, color=color(s), ls=":", lw=1, label=f"released 'main': {main_val:.3g}")
            ax.legend(fontsize=8)
        if s == "pretrain-s1":
            ax.set_xscale("symlog", linthresh=1000)
        ax.set_xlabel("step")
        ax.set_ylabel("‖θ_t − θ_start‖")
        ax.set_title(f"{s}   (start: {totals[s]['a'][0].split('/')[1]}@{totals[s]['a'][1]})", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(traj):]:
        ax.axis("off")
    fig.suptitle("OLMo 3 7B: distance from the start of each stage")
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=150)

    # ---------------- per-module share ----------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    kinds = [k for k in KINDS if any(k in totals[s]["by_kind"] for s in stages)]
    m = np.array([[totals[s]["by_kind"].get(k, {"sq_delta": 0})["sq_delta"] for k in kinds] for s in stages])
    share = m / m.sum(1, keepdims=True)
    pshare = np.array([totals[stages[0]]["by_kind"][k]["numel"] for k in kinds]) / n
    im = ax.imshow(share / pshare, cmap="RdBu_r", norm=plt.matplotlib.colors.LogNorm(0.1, 10), aspect="auto")
    for i in range(len(stages)):
        for j in range(len(kinds)):
            ax.text(j, i, f"{share[i, j]:.0%}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(kinds)), [f"{k}\n({p:.0%} of N)" for k, p in zip(kinds, pshare)], fontsize=8)
    ax.set_yticks(range(len(stages)), stages)
    fig.colorbar(im, label="share of ‖Δ‖² ÷ share of parameters")
    ax.set_title("Share of each stage's ‖Δ‖² by module (text); colour = over/under its parameter share")
    fig.tight_layout()
    fig.savefig(out / "modules.png", dpi=150)

    summary = {s: {k: totals[s][k] for k in ("norm_a", "norm_b", "norm_delta", "frac_exact_zero", "frac_le_1e-5")}
               | {"a": totals[s]["a"], "b": totals[s]["b"]} for s in stages}
    summary["extra"] = [{k: r[k] for k in ("a", "b", "norm_delta", "frac_exact_zero")}
                        for r in rows if r["kind"] == "extra"]
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for s in stages:
        t = totals[s]
        print(f"{s:13s} ||D|| {t['norm_delta']:9.4g}   ||D||/||theta|| {t['norm_delta'] / t['norm_a']:.2e}   "
              f"unchanged {t['frac_exact_zero']:.1%}")
    for e in summary["extra"]:
        print("extra", e)


if __name__ == "__main__":
    main()
