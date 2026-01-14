#!/usr/bin/env python3
"""
Plot violin plots of alongshore-averaged beach width across all surveys.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle


def _load_widths(path):
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=True) as data:
            try:
                width_along = np.asarray(data["width_along_transect_m"])
            except KeyError as exc:
                raise KeyError(
                    "Input file must contain width_along_transect_m"
                ) from exc
    elif path.lower().endswith(".mat"):
        try:
            from scipy.io import loadmat
        except Exception as exc:
            raise RuntimeError("scipy is required to read .mat files. Install with: pip install scipy") from exc
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        if "width_along_transect_m" not in mat:
            raise KeyError("MAT file missing width_along_transect_m")
        width_along = np.asarray(mat["width_along_transect_m"])
    else:
        raise ValueError("Input must be a .npz or .mat file")
    return width_along


def _mean_alongshore(width_matrix):
    arr = np.asarray(width_matrix)
    if arr.ndim != 2:
        raise ValueError(f"Width matrix must be 2-D (time x transect). Got shape {arr.shape}")
    finite = np.isfinite(arr)
    count = finite.sum(axis=1)
    summed = np.where(finite, arr, 0.0).sum(axis=1)
    return np.divide(summed, count, out=np.full(arr.shape[0], np.nan), where=count > 0, dtype=float)


def _plot_violin(ax, values, title, color, show_ylabel):
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("No finite values available to plot")

    sns.violinplot(
        y=finite_values,
        ax=ax,
        inner=None,
        color=color,
        linewidth=1.2,
        cut=0,
        saturation=0.9,
        width=0.6,
    )

    q1, q3 = np.nanpercentile(finite_values, [25, 75])
    median_val = float(np.nanmedian(finite_values))
    iqr = q3 - q1
    whisker_low = q1 - 1.5 * iqr
    whisker_high = q3 + 1.5 * iqr
    finite_sorted = np.sort(finite_values)
    whisker_min = finite_sorted[finite_sorted >= whisker_low].min(initial=finite_sorted.min())
    whisker_max = finite_sorted[finite_sorted <= whisker_high].max(initial=finite_sorted.max())

    x_center = 0
    whisker_width = 0.07
    box_width = 0.16
    ax.vlines(x_center, whisker_min, whisker_max, color="#111111", linewidth=2.0, zorder=3)
    ax.hlines(whisker_min, x_center - whisker_width, x_center + whisker_width, color="#111111", linewidth=2.0, zorder=3)
    ax.hlines(whisker_max, x_center - whisker_width, x_center + whisker_width, color="#111111", linewidth=2.0, zorder=3)
    ax.add_patch(
        Rectangle(
            (x_center - box_width / 2, q1),
            box_width,
            q3 - q1,
            facecolor="white",
            edgecolor="#111111",
            linewidth=1.8,
            zorder=4,
        )
    )
    ax.plot(x_center, median_val, marker="o", color="#111111", markersize=6.5, zorder=5)

    ax.set_title(title, pad=10, fontweight="semibold")
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_xlim(-0.5, 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if show_ylabel:
        ax.set_ylabel("Alongshore-averaged beach width (m)")
    else:
        ax.set_ylabel("")
    ymin_data, ymax_data = np.nanpercentile(finite_values, [1, 99])
    ymin = min(ymin_data, whisker_min, q1)
    ymax = max(ymax_data, whisker_max, q3)
    pad = (ymax - ymin) * 0.08 if np.isfinite(ymax - ymin) else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a publication-ready violin plot of alongshore-averaged back beach width."
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
        default=400,
        help="DPI for the saved figure",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    sns.set_theme(style="whitegrid", context="talk")

    width_along = _load_widths(args.input)
    along_means = _mean_alongshore(width_along)

    fig, ax = plt.subplots(figsize=(6, 7.5), dpi=args.dpi)
    _plot_violin(ax, along_means, "Along-transect back beach width", "#1f77b4", show_ylabel=True)

    fig.suptitle("Alongshore-averaged back beach width (all surveys)", y=0.98, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved violin plot to {args.output}")


if __name__ == "__main__":
    main()
