"""Paper-style plots (Li et al. 2018) of a 2-D reward landscape CSV from landscape/2d_perturb.py.

Writes, next to the CSV:
  *_contour.png      labelled line contours
  *_surface3d.png    smooth shaded surface
  *_surface3d.html   the same surface, interactive (drag to rotate, scroll to zoom)

Contours and surface are drawn through a cubic-spline interpolation of the measured
grid (Li et al. used a 51x51 grid; ours is coarser). The CSV holds the measured points.

    python -m landscape.plot_2d results/.../qwen3_0.6b_reward2d_....csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource
from scipy.interpolate import RectBivariateSpline


def grid_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    point = df.groupby(["alpha", "beta"])["reward"].mean().reset_index()
    alphas = np.sort(point["alpha"].unique())
    betas = np.sort(point["beta"].unique())
    z = np.full((len(betas), len(alphas)), np.nan)
    for row in point.itertuples():
        z[np.searchsorted(betas, row.beta), np.searchsorted(alphas, row.alpha)] = row.reward
    return alphas, betas, z, df


def smooth(alphas, betas, z, n=121):
    """Cubic spline through the grid (linear if the grid is too small), for rendering.

    A partial grid (run stopped early) is interpolated from the measured points only;
    the unmeasured area outside them stays empty.
    """
    if np.isnan(z).any():
        from scipy.interpolate import griddata

        a, b = np.meshgrid(alphas, betas)
        ok = ~np.isnan(z)
        fa = np.linspace(alphas[0], alphas[-1], n)
        fb = np.linspace(betas[0], betas[-1], n)
        fa_grid, fb_grid = np.meshgrid(fa, fb)
        fz = griddata((a[ok], b[ok]), z[ok], (fa_grid, fb_grid), method="cubic")
        return fa, fb, np.clip(fz, 0.0, 1.0)

    k = 3 if min(len(alphas), len(betas)) >= 4 else 1
    spline = RectBivariateSpline(betas, alphas, z, kx=k, ky=k)
    fa = np.linspace(alphas[0], alphas[-1], n)
    fb = np.linspace(betas[0], betas[-1], n)
    fz = np.clip(spline(fb, fa), 0.0, 1.0)  # reward is in [0, 1]
    return fa, fb, fz


def plot_contour(alphas, betas, z, path, title):
    fig, ax = plt.subplots(figsize=(6, 5.4))
    levels = np.linspace(np.nanmin(z), np.nanmax(z), 16)[1:-1]
    cs = ax.contour(alphas, betas, z, levels=levels, cmap="viridis", linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.3f")
    ax.set_xlim(alphas[0], alphas[-1])
    ax.set_ylim(betas[0], betas[-1])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_surface(fa, fb, fz, path, title):
    a, b = np.meshgrid(fa, fb)
    ls = LightSource(azdeg=315, altdeg=35)
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(np.nanmin(fz), np.nanmax(fz))
    filled = np.where(np.isnan(fz), np.nanmin(fz), fz)
    rgb = ls.shade(filled, cmap=cmap, norm=norm, vert_exag=2.0, blend_mode="soft")
    fz = np.ma.masked_invalid(fz)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(a, b, fz, facecolors=rgb, rstride=1, cstride=1, linewidth=0,
                    edgecolor="none", antialiased=False, shade=False)
    ax.set_axis_off()
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect((1, 1, 0.6))
    ax.set_title(title, fontsize=12, y=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=250, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_interactive(fa, fb, fz, path, title):
    import plotly.graph_objects as go

    fig = go.Figure(go.Surface(
        x=fa, y=fb, z=fz,
        colorscale="RdBu_r",
        showscale=True,
        colorbar=dict(title="reward"),
        lighting=dict(ambient=0.45, diffuse=0.8, specular=0.35, roughness=0.5, fresnel=0.2),
        lightposition=dict(x=-1000, y=-1000, z=2000),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=2)),
        hovertemplate="α=%{x:.2f}<br>β=%{y:.2f}<br>reward=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="α", yaxis_title="β", zaxis_title="reward",
            aspectratio=dict(x=1, y=1, z=0.6),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.write_html(path, include_plotlyjs="cdn")


def plot(csv_path, title=None, out_dir=None):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir or csv_path.parent)
    alphas, betas, z, df = grid_from_csv(csv_path)
    title = title or csv_path.stem
    stem = csv_path.stem
    fa, fb, fz = smooth(alphas, betas, z)

    plot_contour(fa, fb, fz, out_dir / f"{stem}_contour.png", title)
    plot_surface(fa, fb, fz, out_dir / f"{stem}_surface3d.png", title)
    try:
        plot_interactive(fa, fb, fz, out_dir / f"{stem}_surface3d.html", title)
    except ImportError:
        print("plotly not installed; skipping the interactive plot")
    print(f"Plots written to {out_dir}/{stem}_*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    plot(args.csv, args.title, args.out_dir)
