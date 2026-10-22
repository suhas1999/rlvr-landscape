"""Group-size sweep (gs 4 / 8 / 16, 32 prompts per step, same steps): training curves, displacement norm,
basin radii (landscape/basin_radius.py) and 1-D base->final sweeps (landscape/1d_displacement.py).

    python landscape/compare_gs.py --out results/gs_sweep
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from basin_volume import radius  # noqa: E402
from plot_1d import load as load_1d  # noqa: E402

CK = "/scratch/micah/vm2825/checkpoints/output_qwen3_0.6b_grpo_trl_train300_seed42"
RUNS = {
    4: f"{CK}_gs4_p32_lr5e-6_max2048_s210_topp1_tokenIS",
    8: f"{CK}_gs8_p32_lr5e-6_max2048_ep30_topp1_tokenIS",
    16: f"{CK}_gs16_p32_lr5e-6_max2048_s210_topp1_tokenIS",
}
COLORS = {4: "C0", 8: "C1", 16: "C2"}
BASIN_GS8 = "results/basin/2026-10-20_train300_gs8p32_ep30"
BASIN_NEW = "results/basin/2026-10-22_gs_sweep"
L1D = "results/landscape1d/2026-10-22_gs_sweep_unit"
REL = (0.9, 0.75, 0.5, 0.25)
ABS = (0.5, 0.4, 0.3, 0.2)


def log_history(run, max_step):
    states = sorted(glob.glob(f"{run}/checkpoint-*/trainer_state.json"),
                    key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    states = [p for p in states if int(re.search(r"checkpoint-(\d+)", p).group(1)) <= max_step]
    return json.load(open(states[-1]))["log_history"] if states else []


def plot_training(out, max_step):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    for gs, run in RUNS.items():
        h = log_history(run, max_step)
        tr = pd.DataFrame([x for x in h if "reward" in x])
        ev = pd.DataFrame([x for x in h if "eval_reward" in x])
        if tr.empty:
            continue
        axes[0].plot(tr["step"], tr["reward"].rolling(10, min_periods=1).mean(), color=COLORS[gs],
                     label=f"gs {gs}")
        axes[0].plot(tr["step"], tr["reward"], color=COLORS[gs], alpha=0.15, lw=0.8)
        if not ev.empty:
            axes[1].plot(ev["step"], ev["eval_reward"], marker="o", color=COLORS[gs], label=f"gs {gs}")
        if "completions/mean_length" in tr:
            axes[2].plot(tr["step"], tr["completions/mean_length"].rolling(10, min_periods=1).mean(),
                         color=COLORS[gs], label=f"gs {gs}")
    axes[0].set(title="train reward (10-step mean)", xlabel="step", ylabel="reward")
    axes[1].set(title="test reward (MATH500 test, every 30 steps)", xlabel="step", ylabel="reward")
    axes[2].set(title="mean completion length", xlabel="step", ylabel="tokens")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"GRPO group size sweep · 32 prompts/step · {max_step} steps")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)


def delta_norms(out, max_step, base_model):
    """||theta_step - theta_base|| every 30 steps (cached in delta_norms.csv)."""
    import torch
    from safetensors.torch import load_file

    cache = out / "delta_norms.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["gs", "step", "delta", "rel"])
    base = {}
    for f in glob.glob(f"{base_model}/*.safetensors"):
        base.update(load_file(f))
    rows = done.to_dict("records")
    have = {(int(r["gs"]), int(r["step"])) for r in rows}
    for gs, run in RUNS.items():
        for step in range(30, max_step + 1, 30):
            if (gs, step) in have or not Path(f"{run}/checkpoint-{step}/model.safetensors").exists():
                continue
            m = load_file(f"{run}/checkpoint-{step}/model.safetensors")
            sq = sum((v.float() - base[k].float()).pow(2).sum().item() for k, v in m.items())
            bn = sum(base[k].float().pow(2).sum().item() for k in m)
            rows.append({"gs": gs, "step": step, "delta": sq ** 0.5, "rel": (sq / bn) ** 0.5})
            print(f"delta gs{gs} step{step}: {sq ** 0.5:.4f}", flush=True)
    df = pd.DataFrame(rows).sort_values(["gs", "step"])
    df.to_csv(cache, index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for gs, g in df.groupby("gs"):
        ax.plot([0, *g["step"]], [0, *g["delta"]], marker="o", color=COLORS[int(gs)], label=f"gs {int(gs)}")
    ax.set(title="‖θ_step − θ_base‖", xlabel="step", ylabel="L2 norm")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "delta_norm_vs_step.png", dpi=150)
    return df


def basin_csvs():
    """{(gs, step): csv path}; step 0 is the shared base model (gs8 folder)."""
    out = {}
    for p in glob.glob(f"{BASIN_GS8}/step*_train_p32_g8.csv"):
        step = int(re.search(r"step(\d+)", p).group(1))
        out[(8, step)] = p
        if step == 0:
            out[(4, 0)] = out[(16, 0)] = p
    for p in glob.glob(f"{BASIN_NEW}/gs*_step*_train_p32_g8.csv"):
        m = re.search(r"gs(\d+)_step(\d+)", Path(p).name)
        out[(int(m.group(1)), int(m.group(2)))] = p
    return out


def plot_basin(out, steps):
    rows, curves = [], {}
    for (gs, step), p in sorted(basin_csvs().items()):
        if step not in steps:
            continue
        df = pd.read_csv(p)
        r0 = float(df[df["dir_seed"] == -1]["reward"].mean())
        dirs = {s: g.sort_values("alpha") for s, g in df[df["dir_seed"] >= 0].groupby("dir_seed")}
        if not dirs:
            continue
        curves[(gs, step)] = (r0, pd.concat(dirs.values()).groupby("alpha")["reward"].mean())
        for kind, levels in (("rel", REL), ("abs", ABS)):
            for v in levels:
                tau = v * r0 if kind == "rel" else v
                if tau >= r0:
                    continue
                rs = np.array([radius(g["alpha"].to_numpy(), g["reward"].to_numpy(), r0, tau)[0]
                               for g in dirs.values()])
                rows.append({"gs": gs, "step": step, "R0": r0, "kind": kind, "level": v, "tau": tau,
                             "r_mean": rs.mean(), "r_se": rs.std(ddof=1) / np.sqrt(len(rs)), "dirs": len(rs)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "basin_radii_gs.csv", index=False)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    for ax, v in zip(axes[0], REL):
        for gs, g in table[(table["kind"] == "rel") & (table["level"] == v)].groupby("gs"):
            ax.errorbar(g["step"], g["r_mean"], yerr=g["r_se"], marker="o", capsize=3, color=COLORS[gs],
                        label=f"gs {gs}")
        ax.set(title=f"radius where R < {v} · R(0)", xlabel="step", ylabel="mean radius α")
    for ax, v in zip(axes[1], ABS):
        for gs, g in table[(table["kind"] == "abs") & (table["level"] == v)].groupby("gs"):
            ax.errorbar(g["step"], g["r_mean"], yerr=g["r_se"], marker="o", capsize=3, color=COLORS[gs],
                        label=f"gs {gs}")
        ax.set(title=f"radius where R < {v} (absolute)", xlabel="step", ylabel="mean radius α")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Basin radius along 8 random filter-normalized directions (32 train prompts × 8 samples)")
    fig.tight_layout()
    fig.savefig(out / "basin_radii_vs_step.png", dpi=150)

    final = max(steps)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for gs in RUNS:
        if (gs, final) in curves:
            r0, c = curves[(gs, final)]
            axes[0].plot([0, *c.index], [r0, *c.values], marker=".", color=COLORS[gs], label=f"gs {gs}")
            axes[1].plot([0, *c.index], np.array([r0, *c.values]) / r0, marker=".", color=COLORS[gs],
                         label=f"gs {gs}")
    if (8, 0) in curves:
        r0, c = curves[(8, 0)]
        for ax, y in ((axes[0], [r0, *c.values]), (axes[1], np.array([r0, *c.values]) / r0)):
            ax.plot([0, *c.index], y, marker=".", color="k", ls="--", label="base")
    axes[0].set(title=f"mean reward vs distance, step {final}", xlabel="α", ylabel="reward")
    axes[1].set(title="normalised by R(0)", xlabel="α", ylabel="R(α) / R(0)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "basin_radial_final.png", dpi=150)
    return table


def load_unit(gs, step):
    """(data by split, ||Delta_t||) for the unit-normalized sweep base -> checkpoint-step, or None."""
    folder = Path(L1D) / f"gs{gs}_step{step}"
    if not (folder / "norms.json").exists():
        return None
    data = load_1d(folder)
    return (data, json.loads((folder / "norms.json").read_text())["delta"]) if data else None


def draw_1d(ax, df, norm, color, label, ls="-"):
    ax.plot(df["c"], df["reward"], marker="o", ms=3, ls=ls, color=color, label=f"{label} (‖Δ‖={norm:.2f})")
    ax.fill_between(df["c"], df["reward"] - df["se"], df["reward"] + df["se"], color=color, alpha=0.12)
    ax.axvline(norm, color=color, ls=":", lw=1)


def plot_1d_compare(out):
    """Unit-normalized sweeps theta_base + c * Delta_t / ||Delta_t||: c is distance in weight space and
    checkpoint t sits at c = ||Delta_t|| (dotted line)."""
    xlabel = "c   (θ = θ_base + c · Δ_t/‖Δ_t‖,  Δ_t = θ_t − θ_base)"
    # Step 210, all group sizes, train and test.
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for gs in RUNS:
        got = load_unit(gs, 210)
        if got is None:
            continue
        data, norm = got
        for ax, split in zip(axes, ("train", "test")):
            if split in data:
                draw_1d(ax, data[split], norm, COLORS[gs], f"gs {gs}")
    for ax, split in zip(axes, ("train", "test")):
        ax.axvline(0, color="grey", ls="--", lw=1)
        ax.set(title=f"{split} prompts (100 × 8), direction base → step 210", xlabel=xlabel,
               ylabel="mean reward", ylim=(0, 1))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Reward along each run's unit-normalized displacement (dotted: the step-210 checkpoint)")
    fig.tight_layout()
    fig.savefig(out / "displacement_1d_unit_step210.png", dpi=150)

    # Per group size: directions to checkpoints 60 / 120 / 210 (train prompts).
    fig, axes = plt.subplots(1, len(RUNS), figsize=(7 * len(RUNS), 4.8), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax, gs in zip(axes[0], RUNS):
        for i, step in enumerate((60, 120, 210)):
            got = load_unit(gs, step)
            if got is None or "train" not in got[0]:
                continue
            draw_1d(ax, got[0]["train"], got[1], cmap(i / 2.5), f"step {step}")
        ax.axvline(0, color="grey", ls="--", lw=1)
        ax.set(title=f"gs {gs}: train prompts, direction base → checkpoint", xlabel=xlabel,
               ylabel="mean reward", ylim=(0, 1))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Unit-normalized displacement to intermediate checkpoints (dotted: where each checkpoint sits)")
    fig.tight_layout()
    fig.savefig(out / "displacement_1d_unit_intermediate.png", dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/gs_sweep")
    p.add_argument("--max-step", type=int, default=210)
    p.add_argument("--base-model", default="/scratch/micah/vm2825/models/Qwen3-0.6B")
    p.add_argument("--skip-delta", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_training(out, a.max_step)
    if not a.skip_delta:
        print(delta_norms(out, a.max_step, a.base_model).to_string(index=False))
    table = plot_basin(out, (0, 60, 120, 180, 210))
    if not table.empty:
        with pd.option_context("display.width", 200):
            print(table[table["level"].isin([0.5, 0.4])].round(4).to_string(index=False))
    plot_1d_compare(out)
    print("Wrote", out)


if __name__ == "__main__":
    main()
