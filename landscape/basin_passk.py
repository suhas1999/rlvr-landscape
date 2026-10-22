"""pass@k along the basin sweeps of landscape/basin_radius.py, from its saved rollouts.

Per prompt with c correct out of n samples, pass@k = 1 - C(n - c, k) / C(n, k) (unbiased, Chen et al.
2021); averaged over prompts. R(0) is the mean of the centre repeats. Radii use the same first-crossing
rule as basin_volume.py, at relative thresholds f x pass@k(0).

    python landscape/basin_passk.py results/basin/<folder>
"""

import gzip
import json
import re
import sys
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from basin_volume import radius  # noqa: E402

KS = (1, 2, 4, 8)
REL = (0.9, 0.75, 0.5, 0.25)


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def point_passk(path):
    correct = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            correct.setdefault(r["prompt_index"], []).append(r["reward"] > 0.5)
    return {k: float(np.mean([pass_at_k(len(v), sum(v), k) for v in correct.values()])) for k in KS}


def load(folder):
    """{step: DataFrame(dir_seed, alpha, repeat, pass@k ...)}"""
    out = {}
    for d in sorted((folder / "rollouts").glob("step*")):
        step = int(re.match(r"step(\d+)", d.name).group(1))
        rows = []
        for p in d.glob("dir*_a*_r*.jsonl.gz"):
            m = re.match(r"dir([+-]\d+)_a([\d.]+)_r(\d+)", p.name)
            rows.append({"dir_seed": int(m.group(1)), "alpha": float(m.group(2)), "repeat": int(m.group(3)),
                         **{f"pass@{k}": v for k, v in point_passk(p).items()}})
        if rows:
            out[step] = pd.DataFrame(rows)
    return out


def analyse(folder):
    folder = Path(folder)
    data = load(folder)
    steps = sorted(data)
    rows = []
    curves = {}
    for s in steps:
        df = data[s]
        centre = df[df["dir_seed"] == -1]
        dirs = df[df["dir_seed"] >= 0]
        for k in KS:
            col = f"pass@{k}"
            r0 = float(centre[col].mean())
            mean_curve = dirs.groupby("alpha")[col].mean()
            curves[(s, k)] = ([0.0, *mean_curve.index], [r0, *mean_curve.values])
            for f in REL:
                rs = [radius(g["alpha"].to_numpy(), g[col].to_numpy(), r0, f * r0)[0]
                      for _, g in dirs.sort_values("alpha").groupby("dir_seed")]
                rows.append({"step": s, "k": k, "R0": r0, "level": f, "directions": len(rs),
                             "r_mean": np.mean(rs), "r_se": np.std(rs, ddof=1) / np.sqrt(len(rs))})
    table = pd.DataFrame(rows)
    table.to_csv(folder / "basin_passk_radii.csv", index=False)
    print(table[table["level"] == 0.5].round(4).to_string(index=False))

    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, len(KS), figsize=(5.5 * len(KS), 9))
    for j, k in enumerate(KS):
        ax = axes[0, j]
        for i, s in enumerate(steps):
            x, y = curves[(s, k)]
            ax.plot(x, y, marker=".", color=cmap(i / max(1, len(steps) - 1)), label=f"step {s}")
        ax.set(title=f"pass@{k} vs distance", xlabel="α (random filter-normalized)", ylabel=f"pass@{k}",
               ylim=(0, 1))
        ax.grid(alpha=0.3)
        ax = axes[1, j]
        for f, g in table[table["k"] == k].groupby("level"):
            ax.errorbar(g["step"], g["r_mean"], yerr=g["r_se"], marker="o", capsize=3, label=f"f = {f}")
        ax.set(title=f"radius where pass@{k} < f · pass@{k}(0)", xlabel="training step",
               ylabel="mean radius α")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Qwen3-0.6B · MATH500 train prompts (32 × 8 samples): pass@k basin along random directions")
    fig.tight_layout()
    fig.savefig(folder / "basin_passk.png", dpi=150)
    print("Wrote", folder / "basin_passk.png")


if __name__ == "__main__":
    analyse(sys.argv[1])
