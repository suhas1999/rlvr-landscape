"""Side-by-side comparison of several 2-D reward landscapes (landscape/2d_perturb.py runs).

Every panel shares axes, colour scale and contour levels, so the variants can be compared
directly. Unmeasured grid points (a run stopped early) are left blank (grey), not filled.

Writes, in --out-dir:
  compare_contour.png    labelled line contours, one panel per variant
  compare_surface3d.png  shaded surfaces, one panel per variant, same z range
  compare_radial.png     reward vs distance from the centre (all grid points), plus the
                         1-D range probes along +alpha when present

    python -m landscape.compare_2d results/variants --out-dir results/variants
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource, ListedColormap

from landscape.plot_2d import grid_from_csv, smooth

ORDER = ["canon_qkn", "canon", "raw_all", "raw_nonqk", "raw_global", "raw_global_nonqk", "raw_global_none"]
LABELS = {
    "canon_qkn": "canon + q/k-norm perturbed",
    "canon": "canon, norms fixed",
    "raw_all": "raw, all norms perturbed",
    "raw_nonqk": "raw, all norms except q/k",
    "raw_global": "raw, global (un-normalized)",
    "raw_global_nonqk": "raw, global, q/k norms fixed",
    "raw_global_none": "raw, global, all norms fixed",
}


def find_runs(root):
    """{variant: grid csv} for each variant folder under root."""
    runs = {}
    for csv in sorted(Path(root).glob("*/*_r*_n*.csv")):
        variant = csv.parent.name
        if "beta" in pd.read_csv(csv, nrows=1).columns:
            runs[variant] = csv
    return {v: runs[v] for v in ORDER if v in runs} | {
        v: c for v, c in runs.items() if v not in ORDER}


def plot_contours(grids, extent, vmax, path):
    levels = np.linspace(0, vmax, 12)[1:]
    fig, axes = plt.subplots(1, len(grids), figsize=(4.2 * len(grids), 4.6), squeeze=False)
    for ax, (variant, (fa, fb, fz, alphas, betas, z)) in zip(axes[0], grids.items()):
        ax.set_facecolor("#e6e6e6")  # grey = not measured
        ax.pcolormesh(fa, fb, np.ma.masked_invalid(fz * 0), cmap=ListedColormap(["white"]),
                      shading="auto")  # white = measured (interpolated) region
        cs = ax.contour(fa, fb, fz, levels=levels, cmap="viridis", vmin=0, vmax=vmax,
                        linewidths=1.2)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
        a, b = np.meshgrid(alphas, betas)
        ok = ~np.isnan(z)
        ax.scatter(a[ok], b[ok], s=3, c="k", alpha=0.25, linewidths=0)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_title(LABELS.get(variant, variant), fontsize=10)
        ax.set_xlabel("α")
    axes[0][0].set_ylabel("β")
    fig.suptitle("Qwen3-0.6B · MATH500 reward · shared axes and contour levels "
                 "(dots = measured points, white = measured region, grey = not measured)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_surfaces(grids, vmax, path):
    ls = LightSource(azdeg=315, altdeg=35)
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(0, vmax)
    fig = plt.figure(figsize=(4.2 * len(grids), 4.4))
    for k, (variant, (fa, fb, fz, *_)) in enumerate(grids.items()):
        ax = fig.add_subplot(1, len(grids), k + 1, projection="3d")
        a, b = np.meshgrid(fa, fb)
        filled = np.where(np.isnan(fz), 0.0, fz)
        rgb = ls.shade(filled, cmap=cmap, norm=norm, vert_exag=2.0, blend_mode="soft")
        ax.plot_surface(a, b, np.ma.masked_invalid(fz), facecolors=rgb, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False)
        ax.set_zlim(0, vmax)
        ax.set_axis_off()
        ax.view_init(elev=30, azim=-60)
        ax.set_box_aspect((1, 1, 0.6))
        ax.set_title(LABELS.get(variant, variant), fontsize=10, y=0.92)
    fig.suptitle("Same z range (0 to max reward) and colour scale in every panel", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_radial(runs, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = plt.get_cmap("tab10")
    for k, (variant, csv) in enumerate(runs.items()):
        df = pd.read_csv(csv).groupby(["alpha", "beta"])["reward"].mean().reset_index()
        r = np.hypot(df["alpha"], df["beta"]).round(4)
        ax1.scatter(r, df["reward"], s=10, color=colors(k), alpha=0.35)
        prof = df.groupby(r)["reward"].mean()
        ax1.plot(prof.index, prof.values, "-o", ms=3, color=colors(k),
                 label=LABELS.get(variant, variant))
        probes = list(csv.parent.glob("*_probe.csv"))
        if probes:
            p = pd.read_csv(probes[0])
            ax2.plot(p["alpha"], p["reward"], "-o", ms=3, color=colors(k),
                     label=LABELS.get(variant, variant))
    ax1.set_xlabel("distance from centre  √(α²+β²)")
    ax1.set_ylabel("mean reward")
    ax1.set_title("All grid points (dots) and mean per radius (lines)")
    ax1.legend(fontsize=8)
    ax2.set_xscale("symlog", linthresh=0.002)
    ax2.set_xlabel("α along d1 (β = 0), symlog")
    ax2.set_title("1-D range probes")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.02, None)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Folder with one sub-folder per variant.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir or args.root)

    runs = find_runs(args.root)
    grids, extent, vmax = {}, 0.0, 0.0
    for variant, csv in runs.items():
        alphas, betas, z, _ = grid_from_csv(csv)
        fa, fb, fz = smooth(alphas, betas, z)
        grids[variant] = (fa, fb, fz, alphas, betas, z)
        extent = max(extent, np.abs(alphas).max(), np.abs(betas).max())
        vmax = max(vmax, np.nanmax(z))
        print(f"{variant}: {int((~np.isnan(z)).sum())} measured points from {csv.name}")

    plot_contours(grids, extent, vmax, out_dir / "compare_contour.png")
    plot_surfaces(grids, vmax, out_dir / "compare_surface3d.png")
    plot_radial(runs, out_dir / "compare_radial.png")
    print(f"Comparison plots written to {out_dir}/compare_*.png")


if __name__ == "__main__":
    main()
