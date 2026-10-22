#!/usr/bin/env python3
"""
Histogram of absolute per-parameter changes: RL displacement vs the random unit direction.

Reads the direction files landscape/1d_displacement.py writes in its --work-dir:
    delta_final_minus_base.pt   Delta = theta_final - theta_base (raw)
    random_seed0_unit.pt        Gaussian direction, ||d|| = 1 (what --direction random --direction-norm unit sweeps)

Example:
    python -m directions.displacement_histogram \
        --work-dir /scratch/micah/vm2825/landscape1d_tmp/train300_gs8p32_ep20_ckpt188 \
        --output results/landscape1d/displacement_histogram/base_to_ckpt188.png
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

LOG_MIN, LOG_MAX, NBINS = -14.0, 1.0, 1500  # uniform bins in log10|x|; merged x10 for display


def flatten(path: Path) -> tuple[torch.Tensor, dict[str, int], dict[str, float]]:
    state = torch.load(path, weights_only=True)
    names = sorted(state)
    numel = defaultdict(int)
    sq = defaultdict(float)
    for k in names:
        numel[module_kind(k)] += state[k].numel()
        sq[module_kind(k)] += (state[k].double() ** 2).sum().item()
    return torch.cat([state[k].float().flatten() for k in names]), numel, sq


def module_kind(name: str) -> str:
    if "embed_tokens" in name:
        return "embed"
    if "norm" in name:
        return "norms"
    m = re.search(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.", name)
    return m.group(1) if m else "other"


def log_hist(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Counts and sum of x^2 per log10|x| bin (exact zeros dropped)."""
    a = x.abs()
    a = a[a > 0]
    idx = ((a.log10() - LOG_MIN) / (LOG_MAX - LOG_MIN) * NBINS).long().clamp_(0, NBINS - 1)
    counts = torch.bincount(idx, minlength=NBINS).double().numpy()
    sq = torch.bincount(idx, weights=a.double() ** 2, minlength=NBINS).numpy()
    return counts, sq


def topk_energy_curve(counts: np.ndarray, sq: np.ndarray, n: int, fracs: np.ndarray) -> np.ndarray:
    # Walk bins from largest |x| down; interpolate within a bin.
    c = np.concatenate([[0], np.cumsum(counts[::-1])]) / n
    e = np.concatenate([[0], np.cumsum(sq[::-1])]) / sq.sum()
    return np.interp(fracs, c, e)


def hist_quantile(counts: np.ndarray, q: float) -> float:
    edges = np.linspace(LOG_MIN, LOG_MAX, NBINS + 1)
    c = np.concatenate([[0], np.cumsum(counts)]) / counts.sum()
    return float(10 ** np.interp(q, c, edges))


def stats(x: torch.Tensor, counts: np.ndarray, sq: np.ndarray, fracs: np.ndarray, curve: np.ndarray) -> dict:
    n = x.numel()
    x64 = x.double()
    norm = x64.norm().item()
    return {
        "norm": norm,
        "rms_abs": norm / n ** 0.5,
        "max_abs": x.abs().max().item(),
        "frac_exact_zero": (x == 0).double().mean().item(),
        "abs_quantiles_nonzero": {q: hist_quantile(counts, q) for q in (0.1, 0.5, 0.9, 0.99, 0.999)},
        "energy_in_top_frac": {f"{f:g}": float(np.interp(f, fracs, curve)) for f in (1e-4, 1e-3, 1e-2, 0.1)},
        "participation_ratio_frac": norm ** 4 / (x64 ** 4).sum().item() / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--random-file", default="random_seed0_unit.pt")
    parser.add_argument("--c-max", type=float, default=4.0, help="Largest c swept along the random direction.")
    parser.add_argument("--output", required=True, help="Output .png; a .json with the stats is written next to it.")
    args = parser.parse_args()

    work = Path(args.work_dir)
    norms = json.loads((work / "norms.json").read_text())
    delta, n_by, sq_by_d = flatten(work / "delta_final_minus_base.pt")
    rand, _, sq_by_r = flatten(work / args.random_file)
    n = delta.numel()
    assert rand.numel() == n

    series = {
        "RL displacement Δ (c=1 raw)": delta,
        "random unit d (c=1)": rand,
        f"random, c={args.c_max:g}": rand * args.c_max,
    }
    styles = [dict(fill=True, alpha=0.55, color="C0"), dict(lw=1.8, color="k"), dict(lw=1.5, color="C3", ls="--")]

    fracs = np.logspace(-6, 0, 200)
    hists, summary = {}, {"numel": n, "theta_base_norm": norms["base"],
                          "rms_abs_theta_base": norms["base"] / n ** 0.5}
    for label, x in list(series.items())[:2]:
        counts, sq = log_hist(x)
        curve = topk_energy_curve(counts, sq, n, fracs)
        hists[label] = (counts, sq, curve)
        summary[label] = stats(x, counts, sq, fracs, curve)
    hists[f"random, c={args.c_max:g}"] = hists["random unit d (c=1)"]  # same shape, shifted by log10(c_max)

    edges = 10 ** np.linspace(LOG_MIN, LOG_MAX, NBINS // 10 + 1)
    coarse = lambda v: v.reshape(NBINS // 10, -1).sum(1)  # noqa: E731
    rms_theta = summary["rms_abs_theta_base"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, weight_by_sq in ((axes[0, 0], False), (axes[0, 1], True)):
        for (label, (counts, sq, _)), style in zip(hists.items(), styles):
            shift = args.c_max if label.startswith("random, c=") else 1.0
            v = coarse(sq) / sq.sum() if weight_by_sq else coarse(counts) / n
            ax.stairs(v, edges * shift, label=label, **style)
        ax.axvline(rms_theta, color="gray", ls=":", label=f"RMS |θ_base,i| = {rms_theta:.1e}")
        ax.set_xscale("log")
        ax.set_xlim(1e-10, 1)
        ax.set_xlabel("absolute per-parameter change |x_i|")
        ax.legend(fontsize=9, loc="upper left")
    axes[0, 0].set_ylabel("fraction of parameters")
    axes[0, 0].set_title(f"(a) |x_i| histogram   (Δ: {summary['RL displacement Δ (c=1 raw)']['frac_exact_zero']:.1%}"
                         " exactly 0, not shown)")
    axes[0, 1].set_ylabel("fraction of ‖x‖²")
    axes[0, 1].set_title("(b) same buckets weighted by x_i² (where the norm lives)")

    ax = axes[1, 0]
    ax.plot(fracs, hists["RL displacement Δ (c=1 raw)"][2], color="C0", label="RL displacement Δ")
    ax.plot(fracs, hists["random unit d (c=1)"][2], "k--", label="random Gaussian d")
    ax.plot(fracs, fracs, ":", color="gray", label="all |x_i| equal")
    ax.set_xscale("log")
    ax.set_xlabel("fraction of parameters (largest |x_i| first)")
    ax.set_ylabel("fraction of ‖x‖² captured")
    ax.set_title("(c) concentration (scale-free)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    kinds = sorted(n_by, key=lambda k: -sq_by_d[k])
    x = np.arange(len(kinds))
    tot_d, tot_r = sum(sq_by_d.values()), sum(sq_by_r.values())
    ax.bar(x - 0.27, [sq_by_d[k] / tot_d for k in kinds], 0.27, color="C0", label="Δ: share of ‖Δ‖²")
    ax.bar(x, [sq_by_r[k] / tot_r for k in kinds], 0.27, color="k", label="random d: share of ‖d‖²")
    ax.bar(x + 0.27, [n_by[k] / n for k in kinds], 0.27, color="lightgray", label="share of parameters")
    ax.set_xticks(x, kinds, rotation=30)
    ax.set_title("(d) by module type")
    ax.legend()
    summary["share_sq_by_kind_delta"] = {k: sq_by_d[k] / tot_d for k in kinds}
    summary["share_params_by_kind"] = {k: n_by[k] / n for k in kinds}

    sd, sr = summary["RL displacement Δ (c=1 raw)"], summary["random unit d (c=1)"]
    fig.suptitle(f"Qwen3-0.6B, N = {n / 1e6:.0f}M:  ‖Δ‖ = {sd['norm']:.3f} (RMS |Δ_i| {sd['rms_abs']:.1e}, "
                 f"max {sd['max_abs']:.1e})   vs   ‖d‖ = {sr['norm']:.3f} (RMS |d_i| {sr['rms_abs']:.1e}, "
                 f"max {sr['max_abs']:.1e})")
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
