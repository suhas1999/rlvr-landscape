"""Tables and plots for the reward-Hessian runs (reward_hessian.py + basin_radius.py --direction files).

<folder>/runs.json lists the checkpoints:
    [{"name": "gs8_step140", "gs": 8, "step": 140}, {"name": "base", "gs": null, "step": 0}, ...]
and <folder> holds <name>_hessian.json and sweeps/<name>_*_files.csv (direction order: eigneg1,
eigneg2, grad; dir_seed = index).

Writes, in <folder>:
  hessian_table.csv          one row per checkpoint
  hessian_eigs_vs_step.png   most negative / most positive curvature vs step (held-out B and in-sample A)
  hessian_trace_vs_step.png  Hutchinson trace / n (mean curvature of a random unit direction) vs step
  hessian_eig_curv_vs_step.png  measured reward curvature c(a) = [R(a) + R(-a) - 2 R(0)] / a^2 along the
                             top two negative-curvature eigenvectors (fresh rollouts, a = 0.1) vs step
  hessian_sweeps.png         reward vs signed alpha along eigneg1 / eigneg2 / gradient, one panel per run

    python landscape/compare_hessian.py results/hessian/<folder>
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DIRS = ["eigneg1", "eigneg2", "grad"]
CURV_ALPHA = 0.1   # step of the measured second difference along the eigenvectors


def main(folder):
    folder = Path(folder)
    runs = json.loads((folder / "runs.json").read_text())
    rows = []
    for r in runs:
        path = folder / f"{r['name']}_hessian.json"
        if not path.exists():
            continue
        h = json.loads(path.read_text())
        ritz = h["lanczos"]["ritz"]
        row = {"name": r["name"], "gs": r["gs"], "step": r["step"], "J": h["J"], "n_active": sum(h["n_active"]),
               "grad_norm": h["grad_norm"], "grad_cos_AB": h["grad_cos_AB"],
               "lam_neg1_A": ritz[0]["lambda_A_in_sample"], "lam_neg1_B": ritz[0]["lambda_B_held_out"],
               "lam_neg2_A": ritz[1]["lambda_A_in_sample"], "lam_neg2_B": ritz[1]["lambda_B_held_out"],
               "lam_pos1_A": ritz[-1]["lambda_A_in_sample"], "lam_pos1_B": ritz[-1]["lambda_B_held_out"],
               "trace_over_n": h["trace_over_n"]["full"], "trace_over_n_se": h["trace_over_n"]["full_se"],
               "curv_grad": h["curv_grad"]["full"], "curv_grad_B": h["curv_grad"]["B"],
               "lanczos_iters": h["lanczos"]["iters"], "minutes": h["seconds_total"] / 60}
        if "curv_displacement" in h:
            row.update(curv_disp=h["curv_displacement"]["full"], disp_norm=h["displacement_norm"],
                       cos_grad_disp=h["cos_grad_displacement"])
        rows.append(row)
    if not rows:
        print("no results yet")
        return
    for row in rows:
        files = glob.glob(str(folder / "sweeps" / f"{row['name']}_*_files.csv"))
        if not files:
            continue
        d = pd.read_csv(files[0])
        c = d[d["dir_seed"] == -1]
        r0, se0 = c["reward"].mean(), c["se"].mean() / np.sqrt(len(c))
        for i, tag in enumerate(DIRS[:2]):
            g = d[d["dir_seed"] == i].set_index("alpha")
            if CURV_ALPHA in g.index and -CURV_ALPHA in g.index:
                row[f"c_{tag}"] = (g.loc[CURV_ALPHA, "reward"] + g.loc[-CURV_ALPHA, "reward"] - 2 * r0) / CURV_ALPHA**2
                row[f"c_{tag}_se"] = np.sqrt(g.loc[CURV_ALPHA, "se"]**2 + g.loc[-CURV_ALPHA, "se"]**2
                                             + 4 * se0**2) / CURV_ALPHA**2
    table = pd.DataFrame(rows).sort_values(["gs", "step"], na_position="first")
    table.to_csv(folder / "hessian_table.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table.round(4).to_string(index=False))

    base = table[table["gs"].isna()]
    gss = sorted(g for g in table["gs"].dropna().unique())
    colors = {g: plt.get_cmap("tab10")(i) for i, g in enumerate(gss)}

    def per_gs(ax, ycol, label=None, ls="-", marker="o", errcol=None):
        for g in gss:
            t = table[table["gs"] == g].sort_values("step")
            ys, xs = t[ycol], t["step"]
            if not base.empty:          # every run starts from the base model at step 0
                xs, ys = pd.concat([pd.Series([0]), xs]), pd.concat([base[ycol], ys])
            kw = dict(color=colors[g], ls=ls, marker=marker, label=f"gs{int(g)}{label or ''}")
            if errcol:
                es = t[errcol] if base.empty else pd.concat([base[errcol], t[errcol]])
                ax.errorbar(xs, ys, yerr=es, capsize=3, **kw)
            else:
                ax.plot(xs, ys, **kw)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    per_gs(axes[0], "lam_neg1_B", " held-out")
    per_gs(axes[0], "lam_neg1_A", " in-sample", ls=":", marker="x")
    axes[0].set(title="Most negative curvature (sharpest drop direction)", ylabel="eigenvalue of Hess J")
    per_gs(axes[1], "lam_pos1_B", " held-out")
    per_gs(axes[1], "lam_pos1_A", " in-sample", ls=":", marker="x")
    axes[1].set(title="Most positive curvature", ylabel="eigenvalue of Hess J")
    for ax in axes:
        ax.set_xlabel("training step (0 = base)")
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Top Hessian eigenvalues of the expected reward (Lanczos on half A; held-out = curvature on half B)")
    fig.tight_layout()
    fig.savefig(folder / "hessian_eigs_vs_step.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 5))
    per_gs(ax, "trace_over_n", errcol="trace_over_n_se")
    ax.set(xlabel="training step (0 = base)", ylabel="tr(Hess J) / n",
           title="Mean curvature along a random unit direction (Hutchinson, +- SE)")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "hessian_trace_vs_step.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, tag in zip(axes, DIRS[:2]):
        if f"c_{tag}" in table:
            per_gs(ax, f"c_{tag}", errcol=f"c_{tag}_se")
        ax.set(title=f"Measured reward curvature along {tag} (a = {CURV_ALPHA})",
               xlabel="training step (0 = base)", ylabel="[R(a) + R(-a) - 2 R(0)] / a^2")
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Curvature along the top negative-curvature Hessian eigenvectors (fresh rollouts, +- SE)")
    fig.tight_layout()
    fig.savefig(folder / "hessian_eig_curv_vs_step.png", dpi=150)

    sweeps = []
    for r in runs:
        files = glob.glob(str(folder / "sweeps" / f"{r['name']}_*_files.csv"))
        if files:
            sweeps.append((r, pd.read_csv(files[0])))
    if sweeps:
        ncol = min(5, len(sweeps))
        nrow = (len(sweeps) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.8 * nrow), sharey=True, squeeze=False)
        for ax, (r, df) in zip(axes.flat, sweeps):
            r0 = df[df["dir_seed"] == -1]["reward"].mean()
            for i, d in enumerate(DIRS[:2]):
                g = df[df["dir_seed"] == i].sort_values("alpha")
                if g.empty:
                    continue
                xs = pd.concat([g["alpha"], pd.Series([0.0])]).sort_values()
                ys = pd.concat([g.set_index("alpha")["reward"], pd.Series({0.0: r0})]).loc[xs]
                ax.plot(xs, ys, marker=".", label=d)
            ax.set_xlim(-0.35, 0.35)
            ax.set(title=f"{r['name']} (R0={r0:.2f})", xlabel="alpha (signed, unit direction)")
            ax.grid(alpha=0.3)
        for ax in list(axes.flat)[len(sweeps):]:
            ax.axis("off")
        axes[0][0].set_ylabel("reward")
        axes[0][0].legend(fontsize=8)
        fig.suptitle("Reward along the top two negative-curvature Hessian eigenvectors")
        fig.tight_layout()
        fig.savefig(folder / "hessian_sweeps.png", dpi=150)
    print("Wrote", folder)


if __name__ == "__main__":
    main(sys.argv[1])
