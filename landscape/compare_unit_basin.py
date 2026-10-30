"""Compare unit-norm basin sweeps (landscape/basin_radius.py --direction unit) across models.

<folder>/models.json lists the models in plot order:
    [{"label": "base", "csv": "step000_base_train_32_p32_g8_unit.csv",
      "run": "/path/to/output_dir or null", "step": 0}, ...]
"run" (a training output dir) adds that model's train-reward curve with its checkpoint marked.

Writes, in <folder>:
  unit_basin_radii.csv      radius per model x threshold (first crossing, censored flagged)
  unit_radial_mean.png      mean reward over directions vs alpha (absolute and R / R(0))
  unit_radial_dirs.png      every direction, one panel per model
  unit_radius_bars.png      mean radius +- SE per relative threshold
  unit_train_curves.png     train reward vs step, chosen checkpoints marked

    python landscape/compare_unit_basin.py results/basin/<folder>
"""

import glob
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from basin_volume import log_volume, radius  # noqa: E402

REL = (0.9, 0.75, 0.5, 0.25)


def load(folder, model):
    df = pd.read_csv(folder / model["csv"])
    n = json.loads((folder / model["csv"].replace(".csv", "_dims.json")).read_text())["n_perturbed"]
    r0 = float(df[df["dir_seed"] == -1]["reward"].mean())
    dirs = {s: g.sort_values("alpha") for s, g in df[df["dir_seed"] >= 0].groupby("dir_seed")}
    return r0, dirs, n


def train_rewards(run):
    states = sorted(glob.glob(f"{run}/checkpoint-*/trainer_state.json"),
                    key=lambda p: int(Path(p).parent.name.split("-")[1]))
    if not states:
        return None
    log = json.loads(Path(states[-1]).read_text())["log_history"]
    rows = [(e["step"], e["reward"]) for e in log if "reward" in e and "eval_reward" not in e]
    return pd.DataFrame(rows, columns=["step", "reward"]).drop_duplicates("step")


def main(folder):
    folder = Path(folder)
    models = json.loads((folder / "models.json").read_text())
    data = {m["label"]: load(folder, m) for m in models}
    colors = {m["label"]: plt.get_cmap("tab10")(i) for i, m in enumerate(models)}

    rows = []
    for label, (r0, dirs, n) in data.items():
        for f in REL:
            rs, cens = zip(*(radius(g["alpha"].to_numpy(), g["reward"].to_numpy(), r0, f * r0)
                             for g in dirs.values()))
            rs = np.array(rs)
            rows.append({"model": label, "R0": r0, "f": f, "tau": f * r0, "directions": len(rs),
                         "censored": int(sum(cens)), "r_mean": rs.mean(),
                         "r_se": rs.std(ddof=1) / math.sqrt(len(rs)), "r_min": rs.min(),
                         "r_max": rs.max(), "log10_V": log_volume(rs, n) / math.log(10)})
    table = pd.DataFrame(rows)
    table.to_csv(folder / "unit_basin_radii.csv", index=False)
    with pd.option_context("display.width", 200):
        print(table.round(4).to_string(index=False))

    # Mean radial curve: averaged only over alphas every direction reached (a stopped direction has
    # fallen below the stop threshold, so averaging the survivors alone would bias the mean upward).
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, (r0, dirs, _) in data.items():
        common = sorted(set.intersection(*(set(g["alpha"]) for g in dirs.values())))
        mean = [np.mean([g.set_index("alpha").loc[a, "reward"] for g in dirs.values()]) for a in common]
        x, y = [0.0, *common], [r0, *mean]
        axes[0].plot(x, y, marker="o", color=colors[label], label=f"{label} (R0={r0:.2f})")
        axes[1].plot(x, np.array(y) / r0, marker="o", color=colors[label], label=label)
    for ax, ylabel in ((axes[0], "mean reward over directions"), (axes[1], "R(α) / R(0)")):
        ax.set(xlabel="α = weight-space distance along unit random direction", ylabel=ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Reward vs distance along unit-norm random directions (8 directions, mean over those reaching α)")
    fig.tight_layout()
    fig.savefig(folder / "unit_radial_mean.png", dpi=150)

    fig, axes = plt.subplots(1, len(data), figsize=(4.2 * len(data), 4), sharey=True, squeeze=False)
    for ax, (label, (r0, dirs, _)) in zip(axes[0], data.items()):
        for g in dirs.values():
            ax.plot([0.0, *g["alpha"]], [r0, *g["reward"]], marker=".", alpha=0.7)
        ax.axhline(0.5 * r0, color="k", ls=":", lw=1)
        ax.set(title=label, xlabel="α")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("reward (dotted: 0.5 · R0)")
    fig.tight_layout()
    fig.savefig(folder / "unit_radial_dirs.png", dpi=150)

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(data)
    for i, label in enumerate(data):
        g = table[table["model"] == label]
        xs = np.arange(len(REL)) + i * width
        ax.bar(xs, g["r_mean"], width, yerr=g["r_se"], capsize=3, color=colors[label], label=label)
        for x, (_, row) in zip(xs, g.iterrows()):
            if row["censored"]:
                ax.text(x, row["r_mean"], f"{row['censored']}c", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(REL)) + 0.4 - width / 2, [f"R < {f} · R0" for f in REL])
    ax.set(ylabel="mean radius α (unit directions)",
           title="Basin radius per threshold (\"Nc\" = N directions never dropped: radius censored)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(folder / "unit_radius_bars.png", dpi=150)

    runs = [m for m in models if m.get("run")]
    if runs:
        fig, ax = plt.subplots(figsize=(9, 5))
        for m in runs:
            curve = train_rewards(m["run"])
            if curve is None:
                continue
            smooth = curve["reward"].rolling(10, min_periods=1).mean()
            ax.plot(curve["step"], smooth, color=colors[m["label"]], label=f"{m['label']} (10-step mean)")
            ax.axvline(m["step"], color=colors[m["label"]], ls="--", lw=1)
        ax.set(xlabel="training step", ylabel="train reward", title="Training reward; dashed = basin checkpoint")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / "unit_train_curves.png", dpi=150)
    print("Wrote plots to", folder)


if __name__ == "__main__":
    main(sys.argv[1])
