"""Basin radii and log-volumes from landscape/basin_radius.py sweeps (Huang et al. 2019, sec. 5).

For each checkpoint CSV in a folder and each threshold tau, the radius along direction k is the
first alpha where the linearly interpolated reward R(alpha) drops below tau (R(0) = mean of the
centre repeats). A direction that never drops below tau inside the sweep is right-censored at its
last alpha (counted, and flagged in the table). Thresholds are relative (tau = f x R(0), the basin
of points keeping a fraction f of the checkpoint's own reward) and absolute (tau fixed, as the
paper's loss cutoff; a checkpoint with R(0) < tau has no basin and is skipped).

    log V = log omega_n + log mean_k r_k^n,   omega_n = pi^(n/2) / Gamma(1 + n/2)

with n the number of perturbed parameters. For n ~ 6e8 the mean over directions is dominated by the
largest radius, so the table also lists mean / min / max radius; radii are in alpha units of the
filter-normalized directions (||d_k|| ~ ||theta||).

    python landscape/basin_volume.py results/basin/<folder>
"""

import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REL = (0.9, 0.75, 0.5, 0.25)
ABS = (0.5, 0.4, 0.3, 0.2, 0.1)


def radius(alphas, rewards, r0, tau):
    """First crossing below tau of the piecewise-linear curve through (0, r0), (alphas, rewards).
    Returns (radius, censored)."""
    a = np.concatenate([[0.0], alphas])
    r = np.concatenate([[r0], rewards])
    for i in range(1, len(a)):
        if r[i] < tau:
            return a[i - 1] + (a[i] - a[i - 1]) * (r[i - 1] - tau) / (r[i - 1] - r[i]), False
    return a[-1], True


def log_volume(radii, n):
    """log(omega_n * mean r^n), natural log."""
    radii = np.asarray(radii, dtype=float)
    if (radii <= 0).any():
        return -math.inf
    log_omega = 0.5 * n * math.log(math.pi) - math.lgamma(1 + 0.5 * n)
    x = n * np.log(radii)
    return log_omega + x.max() + math.log(np.mean(np.exp(x - x.max())))


def analyse(folder):
    folder = Path(folder)
    rows = []
    curves = {}
    for csv in sorted(folder.glob("step*_p*_g*.csv")):
        step = int(re.match(r"step(\d+)", csv.name).group(1))
        n = json.loads(Path(str(csv).replace(".csv", "_dims.json")).read_text())["n_perturbed"]
        df = pd.read_csv(csv)
        centre = df[df["dir_seed"] == -1]
        if centre.empty:
            continue
        r0 = float(centre["reward"].mean())
        dirs = {s: g.sort_values("alpha") for s, g in df[df["dir_seed"] >= 0].groupby("dir_seed")}
        curves[step] = (r0, dirs)
        for kind, values in (("rel", REL), ("abs", ABS)):
            for v in values:
                tau = v * r0 if kind == "rel" else v
                if not dirs or tau >= r0:
                    continue
                rs, cens = zip(*(radius(g["alpha"].to_numpy(), g["reward"].to_numpy(), r0, tau)
                                 for g in dirs.values()))
                rs = np.array(rs)
                rows.append({"step": step, "R0": r0, "kind": kind, "level": v, "tau": tau,
                             "directions": len(rs), "censored": int(sum(cens)),
                             "r_mean": rs.mean(), "r_se": rs.std(ddof=1) / np.sqrt(len(rs)) if len(rs) > 1 else np.nan,
                             "r_min": rs.min(), "r_max": rs.max(),
                             "log10_V": log_volume(rs, n) / math.log(10), "n": n})
    table = pd.DataFrame(rows).sort_values(["kind", "level", "step"])
    table.to_csv(folder / "basin_radii.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table.drop(columns=["n"]).round(4).to_string(index=False))
    plot(folder, table, curves)


def plot(folder, table, curves):
    fig, axes = plt.subplots(1, 4, figsize=(24, 4.8))
    steps = sorted(curves)
    axes[0].plot(steps, [curves[s][0] for s in steps], marker="o")
    axes[0].set(title="centre reward R(0) (train prompts)", xlabel="training step", ylabel="reward")
    for ax, kind, title in ((axes[1], "rel", "radius where R < f · R(0)"),
                            (axes[2], "abs", "radius where R < τ (absolute)")):
        for i, (level, g) in enumerate(table[table["kind"] == kind].groupby("level")):
            label = f"f = {level}" if kind == "rel" else f"τ = {level}"
            ax.errorbar(g["step"], g["r_mean"], yerr=g["r_se"], marker="o", capsize=3, label=label)
        ax.set(title=title, xlabel="training step", ylabel="mean radius α (filter-normalized)")
        ax.legend(fontsize=8)
    for level, g in table[table["kind"] == "rel"].groupby("level"):
        axes[3].plot(g["step"], g["log10_V"] - g["log10_V"].iloc[0], marker="o", label=f"f = {level}")
    axes[3].set(title="log10 basin volume, relative to first checkpoint", xlabel="training step",
                ylabel="Δ log10 V")
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Qwen3-0.6B · MATH500 basin along random filter-normalized directions "
                 "(Huang et al. 2019 volume estimate)")
    fig.tight_layout()
    fig.savefig(folder / "basin_vs_step.png", dpi=150)

    # Radial curves per checkpoint: mean over directions.
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(steps):
        r0, dirs = curves[s]
        if not dirs:
            continue
        frame = pd.concat(dirs.values()).groupby("alpha")["reward"].mean()
        ax.plot([0, *frame.index], [r0, *frame.values], marker=".",
                color=cmap(i / max(1, len(steps) - 1)), label=f"step {s}")
    ax.set(xlabel="α along random filter-normalized direction", ylabel="mean reward over directions",
           title="reward vs distance, per checkpoint")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(folder / "basin_radial_curves.png", dpi=150)
    print("Wrote", folder / "basin_vs_step.png", "and", folder / "basin_radial_curves.png")


if __name__ == "__main__":
    analyse(sys.argv[1])
