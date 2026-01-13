#!/usr/bin/env python3
"""
Plot violin plots of alongshore-averaged beach width across all surveys.
"""

import argparse
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_widths(path):
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=True) as data:
            try:
                width_along = np.asarray(data["width_along_transect_m"])
                width_euclid = np.asarray(data["width_euclid_m"])
            except KeyError as exc:
                raise KeyError(
                    "Input file must contain width_along_transect_m and width_euclid_m arrays"
                ) from exc
    elif path.lower().endswith(".mat"):
        try:
            from scipy.io import loadmat
        except Exception as exc:
            raise RuntimeError("scipy is required to read .mat files. Install with: pip install scipy") from exc
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        if "width_along_transect_m" not in mat or "width_euclid_m" not in mat:
            raise KeyError("MAT file missing width matrices: width_along_transect_m and width_euclid_m")
        width_along = np.asarray(mat["width_along_transect_m"])
        width_euclid = np.asarray(mat["width_euclid_m"])
    else:
        raise ValueError("Input must be a .npz or .mat file")
    return width_along, width_euclid


def _mean_alongshore(width_matrix):
    arr = np.asarray(width_matrix)
    if arr.ndim != 2:
        raise ValueError(f"Width matrix must be 2-D (time x transect). Got shape {arr.shape}")
    finite = np.isfinite(arr)
    count = finite.sum(axis=1)
    summed = np.where(finite, arr, 0.0).sum(axis=1)
    return np.divide(summed, count, out=np.full(arr.shape[0], np.nan), where=count > 0)


def _plot_violin(ax, values, title, color, show_ylabel):
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("No finite values available to plot")

    center = 0.0
    violin = ax.violinplot(
        finite_values,
        positions=[center],
        widths=0.7,
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor(color)
        body.set_linewidth(1.2)
    if "cmeans" in violin:
        violin["cmeans"].set_color(color)
        violin["cmeans"].set_linewidth(2.2)
    if "cmedians" in violin:
        violin["cmedians"].set_color("#2f2f2f")
        violin["cmedians"].set_linewidth(2.2)

    # Jitter points give a sense of distribution without hiding the violin.
    rng = np.random.default_rng(0)
    jitter = center + rng.normal(loc=0.0, scale=0.04, size=finite_values.size)
    ax.scatter(jitter, finite_values, color="#2f2f2f", s=10, alpha=0.6, linewidth=0.3)

    mean_val = float(np.nanmean(finite_values))
    median_val = float(np.nanmedian(finite_values))
    xmin, xmax = center - 0.9, center + 0.9
    ax.axhline(mean_val, xmin=0.05, xmax=0.95, color=color, linestyle="--", linewidth=1.8, alpha=0.9)
    ax.axhline(median_val, xmin=0.05, xmax=0.95, color="#2f2f2f", linestyle="-", linewidth=2.0, alpha=0.95)

    ax.set_title(title)
    ax.set_xticks([])
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_xlim(xmin, xmax)
    if show_ylabel:
        ax.set_ylabel("Alongshore-averaged beach width (m)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot violin plots of alongshore-averaged back beach width for each survey."
    )
    parser.add_argument(
        "--input",
        default="data/processed/back_beach_widths.npz",
        help="Path to width output file (.npz or .mat)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("figures", "alongshore_width_violin.png"),
        help="Path to save the output figure",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for the saved figure",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    width_along, width_euclid = _load_widths(args.input)
    along_means = _mean_alongshore(width_along)
    euclid_means = _mean_alongshore(width_euclid)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=args.dpi, sharey=True)
    _plot_violin(axes[0], along_means, "Along-transect width", "#1f77b4", show_ylabel=True)
    _plot_violin(axes[1], euclid_means, "Euclidean width", "#d95f02", show_ylabel=False)

    fig.suptitle("Alongshore-averaged back beach width across surveys")
    fig.tight_layout()

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved violin plot to {args.output}")


if __name__ == "__main__":
    main()
