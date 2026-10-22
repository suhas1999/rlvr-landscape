"""Plot landscape/1d_displacement.py results: one panel per split, reward vs c with ±1 SE.

    python landscape/plot_1d.py results/landscape1d/<folder>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def load(folder):
    """{split: DataFrame sorted by c}, merging the shard CSVs of each split."""
    frames = {}
    for csv in sorted(folder.glob("*_reward1d_*_shard*.csv")):
        split = csv.name.split("_reward1d_")[0].rsplit("_", 1)[1]
        frames.setdefault(split, []).append(pd.read_csv(csv))
    return {s: pd.concat(f).drop_duplicates("c").sort_values("c") for s, f in frames.items()}


def plot(folder):
    folder = Path(folder)
    data = load(folder)
    names = [p.name for p in folder.glob("*_reward1d_*_shard*.csv")]
    unit = any("_unit_" in n for n in names)
    random_dir = any("_reward1d_random" in n for n in names)
    final_c = json.loads((folder / "norms.json").read_text())["delta"] if unit else 1.0
    markers = ((0.0, "base"),) if random_dir else ((0.0, "base"), (final_c, "final"))
    direction = "a random Gaussian direction" if random_dir else "the training displacement"
    splits = [s for s in ("train", "test") if s in data] + [s for s in data if s not in ("train", "test")]
    fig, axes = plt.subplots(1, len(splits), figsize=(7 * len(splits), 4.8), squeeze=False)
    for ax, split in zip(axes[0], splits):
        df = data[split]
        ax.plot(df["c"], df["reward"], marker="o", ms=3, lw=1.5, color="C0", label="reward")
        ax.fill_between(df["c"], df["reward"] - df["se"], df["reward"] + df["se"],
                        color="C0", alpha=0.2)
        ax.plot(df["c"], df["truncated_frac"], lw=1, ls=":", color="C3",
                label="fraction truncated at 2048 tokens")
        for c, name in markers:
            ax.axvline(c, color="grey", ls="--", lw=1)
            row = df.iloc[[(df["c"] - c).abs().argmin()]]
            if len(row):
                ax.annotate(f"{name} (c={c:.2f}) {row['reward'].iloc[0]:.3f}", (c, row["reward"].iloc[0]),
                            xytext=(5, 8), textcoords="offset points", fontsize=8)
        ax.set_title(f"{split} prompts ({len(df)} points)")
        if random_dir:
            ax.set_xlabel("c   (θ = θ_base + c · d,  ‖d‖ = 1)" if unit
                          else "c   (θ = θ_base + c · d,  ‖d‖ = ‖θ_final − θ_base‖)")
        else:
            ax.set_xlabel("c   (θ = θ_base + c · Δ/‖Δ‖,  Δ = θ_final − θ_base)" if unit
                          else "c   (θ = θ_base + c · (θ_final − θ_base))")
        ax.set_ylabel("mean reward")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Qwen3-0.6B · MATH500 reward along {direction} "
                 f"({'unit-normalized' if unit else 'raw'}, no canonicalization)")
    fig.tight_layout()
    out = folder / "displacement_1d.png"
    fig.savefig(out, dpi=150)
    print("Wrote", out)


if __name__ == "__main__":
    plot(sys.argv[1])
