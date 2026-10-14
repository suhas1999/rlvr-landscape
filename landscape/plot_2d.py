"""3-D surface + contour plots of a 2-D reward landscape CSV (from landscape/2d_perturb.py).

    python -m landscape.plot_2d results/landscape2d/qwen3_0.6b_reward_grid.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def grid_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Average the repeats at each (alpha, beta).
    point = df.groupby(["alpha", "beta"])["reward"].agg(["mean", "min", "max"]).reset_index()
    alphas = np.sort(point["alpha"].unique())
    betas = np.sort(point["beta"].unique())
    z = np.full((len(betas), len(alphas)), np.nan)
    for row in point.itertuples():
        z[np.searchsorted(betas, row.beta), np.searchsorted(alphas, row.alpha)] = row.mean
    return alphas, betas, z, df


def plot(csv_path, title=None, out_dir=None):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir or csv_path.parent)
    alphas, betas, z, df = grid_from_csv(csv_path)
    a, b = np.meshgrid(alphas, betas)
    title = title or csv_path.stem
    n_rep = df["repeat"].nunique()
    stem = csv_path.stem
    cmap = "viridis"
    center = z[np.searchsorted(betas, 0.0), np.searchsorted(alphas, 0.0)] if (
        0.0 in alphas and 0.0 in betas) else np.nan
    zmax = max(1e-6, np.nanmax(z))

    # ---- 3-D surface with contour projected on the floor ----
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(a, b, z, cmap=cmap, vmin=0, vmax=zmax, rstride=1, cstride=1,
                           linewidth=0.3, edgecolor="k", alpha=0.92, antialiased=True)
    ax.contourf(a, b, z, zdir="z", offset=0, levels=15, cmap=cmap, vmin=0, vmax=zmax, alpha=0.6)
    if np.isfinite(center):
        ax.scatter([0], [0], [center], color="red", s=60, depthshade=False, label=f"checkpoint ({center:.3f})")
        ax.legend(loc="upper left")
    ax.set_xlabel("α (direction 1)")
    ax.set_ylabel("β (direction 2)")
    ax.set_zlabel("mean reward")
    ax.set_zlim(0, max(zmax * 1.05, 0.05))
    ax.view_init(elev=28, azim=-55)
    ax.set_title(f"{title}\nfilter-normalized directions, mean of {n_rep} repeat(s)")
    fig.colorbar(surf, shrink=0.55, pad=0.08, label="mean reward")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_surface3d.png", dpi=200)
    plt.close(fig)

    # ---- 2-D filled contour ----
    fig, ax = plt.subplots(figsize=(8, 7))
    cf = ax.contourf(a, b, z, levels=20, cmap=cmap, vmin=0, vmax=zmax)
    cs = ax.contour(a, b, z, levels=10, colors="white", linewidths=0.6, alpha=0.8)
    ax.clabel(cs, fmt="%.2f", fontsize=7)
    ax.plot(0, 0, marker="*", color="red", markersize=14, label="checkpoint")
    ax.set_xlabel("α (direction 1)")
    ax.set_ylabel("β (direction 2)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title(f"{title}\nmean reward")
    fig.colorbar(cf, label="mean reward")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_contour.png", dpi=200)
    plt.close(fig)

    # ---- Interactive 3-D (plotly, if installed) ----
    try:
        import plotly.graph_objects as go

        fig = go.Figure(go.Surface(x=alphas, y=betas, z=z, colorscale="Viridis",
                                   contours={"z": {"show": True, "usecolormap": True,
                                                   "project_z": True}}))
        fig.update_layout(title=title, scene=dict(xaxis_title="α", yaxis_title="β",
                                                  zaxis_title="mean reward"))
        fig.write_html(out_dir / f"{stem}_surface3d.html", include_plotlyjs="cdn")
    except ImportError:
        pass

    print(f"Plots written to {out_dir}/{stem}_*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    plot(args.csv, args.title, args.out_dir)
