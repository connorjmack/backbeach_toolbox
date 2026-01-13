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
        width=0.5,
    )
    sns.stripplot(
        y=finite_values,
        ax=ax,
        color="#2f2f2f",
        size=4.5,
        alpha=0.4,
        jitter=0.12,
        linewidth=0,
    )

    mean_val = float(np.nanmean(finite_values))
    median_val = float(np.nanmedian(finite_values))
    ax.axhline(mean_val, color=color, linestyle="--", linewidth=2.0, alpha=0.9, label=f"Mean: {mean_val:.2f} m")
    ax.axhline(median_val, color="#222222", linestyle="-", linewidth=2.2, alpha=0.95, label=f"Median: {median_val:.2f} m")

    ax.set_title(title, pad=10, fontweight="semibold")
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if show_ylabel:
        ax.set_ylabel("Alongshore-averaged beach width (m)")
    else:
        ax.set_ylabel("")
    ax.legend(frameon=False, loc="lower right")
    ymin, ymax = np.nanpercentile(finite_values, [1, 99])
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
