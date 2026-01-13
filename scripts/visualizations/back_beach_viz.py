#!/usr/bin/env python3
"""
Create visualization outputs from back_beach_finder results.

Outputs:
  - GIF of cliff toe and high tide line through time
  - Time series plot of beach width
"""

import argparse
import os
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point

try:
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover
    raise RuntimeError("imageio is required for GIF output. Install with: pip install imageio") from exc


def _as_str_list(arr):
    values = np.asarray(arr).squeeze()
    if values.dtype.kind in ("U", "S"):
        return [str(v) for v in values.tolist()]
    if values.dtype == object:
        out = []
        for item in values.flatten():
            if isinstance(item, bytes):
                out.append(item.decode("utf-8", errors="ignore"))
            else:
                out.append(str(item))
        return out
    return [str(v) for v in values.tolist()]


def _load_results(path):
    width_signed = None
    tide_east = None
    tide_north = None
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=True) as data:
            width_along = data["width_along_transect_m"]
            width_euclid = data["width_euclid_m"]
            survey_dates = _as_str_list(data["survey_dates"])
            transect_ids = np.asarray(data["transect_ids"]).tolist()
            if "width_along_transect_signed_m" in data:
                width_signed = np.asarray(data["width_along_transect_signed_m"])
            if "tide_line_east_m" in data and "tide_line_north_m" in data:
                tide_east = np.asarray(data["tide_line_east_m"])
                tide_north = np.asarray(data["tide_line_north_m"])
    elif path.lower().endswith(".mat"):
        try:
            from scipy.io import loadmat
        except Exception as exc:
            raise RuntimeError("scipy is required to read .mat files. Install with: pip install scipy") from exc
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        width_along = np.asarray(data["width_along_transect_m"])
        width_euclid = np.asarray(data["width_euclid_m"])
        survey_dates = _as_str_list(data["survey_dates"])
        transect_ids = np.asarray(data["transect_ids"]).tolist()
        width_signed = data.get("width_along_transect_signed_m")
        if width_signed is not None:
            width_signed = np.asarray(width_signed)
        tide_east = data.get("tide_line_east_m")
        tide_north = data.get("tide_line_north_m")
        if tide_east is not None and tide_north is not None:
            tide_east = np.asarray(tide_east)
            tide_north = np.asarray(tide_north)
    else:
        raise ValueError("Input must be a .npz or .mat file")

    survey_dt = [datetime.fromisoformat(d) for d in survey_dates]
    return width_along, width_euclid, survey_dt, transect_ids, width_signed, tide_east, tide_north


def _load_cliff_toe(mat_path):
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise RuntimeError("scipy is required to read .mat files. Install with: pip install scipy") from exc
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    east = np.asarray(mat.get("CliffToe_East_Corrected"))
    north = np.asarray(mat.get("CliffToe_North_Corrected"))
    if east.size == 0 or north.size == 0:
        raise KeyError("CliffToe_East_Corrected or CliffToe_North_Corrected not found in MAT file")
    return east, north


def _resolve_transect_column_indices(transect_ids, n_columns):
    ids = np.asarray(transect_ids)
    try:
        ids_num = ids.astype(float)
    except Exception:
        return None
    if np.isnan(ids_num).any():
        return None
    ids_int = ids_num.astype(int)
    if np.any(np.abs(ids_num - ids_int) > 1e-6):
        return None

    min_id = int(ids_int.min())
    max_id = int(ids_int.max())
    if min_id == 0 and max_id == n_columns - 1:
        return ids_int.tolist()
    if min_id == 1 and max_id == n_columns:
        return (ids_int - 1).tolist()
    unique_sorted = np.unique(ids_int)
    if unique_sorted.size == n_columns:
        diffs = np.diff(np.sort(unique_sorted))
        if np.all(diffs == 1):
            return (ids_int - unique_sorted.min()).tolist()
    return None


def _choose_id_field(gdf, preferred=None):
    if preferred and preferred in gdf.columns:
        return preferred
    candidates = ["Id", "ID", "TransectID", "TRANSECTID", "transect_id", "fid"]
    return next((c for c in candidates if c in gdf.columns), None)


def _order_transects(gdf, transect_ids, id_field):
    if id_field is None:
        return list(gdf.geometry)
    mapping = {row[id_field]: row.geometry for _, row in gdf.iterrows()}
    return [mapping.get(tid) for tid in transect_ids]


def _tide_point_from_width(line, cliff_point, width, direction="auto"):
    if line is None or cliff_point is None:
        return None
    if not np.isfinite(width):
        return None
    length = line.length
    if length <= 0:
        return None
    s_cliff = line.project(cliff_point)
    if direction == "signed":
        s_tide = s_cliff + width
    else:
        width = abs(width)
        if direction == "toward-start":
            s_tide = s_cliff - width
        elif direction == "toward-end":
            s_tide = s_cliff + width
        else:
            toward_end = (length - s_cliff) >= s_cliff
            s_tide = s_cliff + width if toward_end else s_cliff - width
    s_tide = max(0.0, min(length, s_tide))
    return line.interpolate(s_tide)


def _make_gif(
    width_along,
    width_signed,
    survey_dates,
    transect_geoms,
    cliff_east,
    cliff_north,
    tide_east,
    tide_north,
    col_indices,
    output_path,
    bounds,
    fps=4,
    dpi=150,
    frame_step=1,
    tide_direction="auto",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    x_min, y_min, x_max, y_max = bounds
    pad_x = (x_max - x_min) * 0.02
    pad_y = (y_max - y_min) * 0.02

    use_signed = width_signed is not None and np.shape(width_signed) == np.shape(width_along)
    use_tide_xy = (
        tide_east is not None
        and tide_north is not None
        and np.shape(tide_east) == np.shape(width_along)
        and np.shape(tide_north) == np.shape(width_along)
    )

    with imageio.get_writer(output_path, mode="I", duration=1 / fps) as writer:
        for idx in range(0, len(survey_dates), frame_step):
            fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=dpi)
            ax.set_aspect("equal", "box")
            ax.set_facecolor("#f7f7f4")

            tide_x = []
            tide_y = []
            cliff_x = []
            cliff_y = []

            for t_idx, geom in enumerate(transect_geoms):
                if geom is None:
                    tide_x.append(np.nan)
                    tide_y.append(np.nan)
                    cliff_x.append(np.nan)
                    cliff_y.append(np.nan)
                    continue
                col_idx = col_indices[t_idx]
                if col_idx < 0 or col_idx >= cliff_east.shape[1]:
                    tide_x.append(np.nan)
                    tide_y.append(np.nan)
                    cliff_x.append(np.nan)
                    cliff_y.append(np.nan)
                    continue
                cx = cliff_east[idx, col_idx]
                cy = cliff_north[idx, col_idx]
                if not np.isfinite(cx) or not np.isfinite(cy):
                    tide_x.append(np.nan)
                    tide_y.append(np.nan)
                    cliff_x.append(np.nan)
                    cliff_y.append(np.nan)
                    continue
                cliff_point = Point(cx, cy)
                if use_tide_xy:
                    tx = tide_east[idx, col_idx]
                    ty = tide_north[idx, col_idx]
                    if not np.isfinite(tx) or not np.isfinite(ty):
                        tide_x.append(np.nan)
                        tide_y.append(np.nan)
                    else:
                        tide_x.append(tx)
                        tide_y.append(ty)
                else:
                    if use_signed:
                        tide_point = _tide_point_from_width(
                            geom, cliff_point, width_signed[idx, col_idx], direction="signed"
                        )
                    else:
                        tide_point = _tide_point_from_width(
                            geom, cliff_point, width_along[idx, col_idx], direction=tide_direction
                        )
                    if tide_point is None:
                        tide_x.append(np.nan)
                        tide_y.append(np.nan)
                    else:
                        tide_x.append(tide_point.x)
                        tide_y.append(tide_point.y)
                cliff_x.append(cx)
                cliff_y.append(cy)

            ax.plot(cliff_x, cliff_y, color="#3a3a3a", linewidth=2.2, label="Cliff toe")
            ax.plot(tide_x, tide_y, color="#1f77b4", linewidth=2.2, label="Mean tide line")

            ax.set_xlim(x_min - pad_x, x_max + pad_x)
            ax.set_ylim(y_min - pad_y, y_max + pad_y)
            ax.set_xlabel("Easting")
            ax.set_ylabel("Northing")
            ax.set_title(f"Cliff Toe and Mean Tide Line - {survey_dates[idx].date().isoformat()}")
            ax.legend(loc="upper right", frameon=False)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())
            frame = frame[..., :3]  # drop alpha channel
            writer.append_data(frame)
            plt.close(fig)


def _make_timeseries(width_matrix, survey_dates, output_path, dpi=200):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mean_vals = np.nanmean(width_matrix, axis=1)
    median_vals = np.nanmedian(width_matrix, axis=1)
    q25 = np.nanpercentile(width_matrix, 25, axis=1)
    q75 = np.nanpercentile(width_matrix, 75, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=dpi)
    ax.fill_between(survey_dates, q25, q75, color="#c7d9e8", alpha=0.6, label="IQR")
    ax.plot(survey_dates, median_vals, color="#1f77b4", linewidth=1.8, label="Median")
    ax.plot(survey_dates, mean_vals, color="#2c3e50", linewidth=1.6, label="Mean")

    ax.set_ylabel("Beach width (m)")
    ax.set_xlabel("Survey date")
    ax.set_title("Backbeach Width Through Time")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate figures from back beach width outputs.")
    parser.add_argument(
        "--input",
        default="data/processed/back_beach_widths.npz",
        help="Path to width output file (.npz or .mat)",
    )
    parser.add_argument(
        "--mat-file",
        default="data/raw/BachBeach_and_tides.mat",
        help="Path to MAT file with cliff toe coordinates",
    )
    parser.add_argument(
        "--transects",
        default=(
            "data/shp_files/DelMarTransects595to620at1m/"
            "DelMarTransects595to620at1m.shp"
        ),
        help="Path to transects shapefile",
    )
    parser.add_argument(
        "--transect-id-field",
        default=None,
        help="Transect ID field name (defaults to auto-detect, e.g., Id)",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--gif-name",
        default="back_beach_lines.gif",
        help="GIF filename",
    )
    parser.add_argument(
        "--timeseries-name",
        default="back_beach_width_timeseries.png",
        help="Time series PNG filename",
    )
    parser.add_argument(
        "--width-type",
        choices=["along", "euclid"],
        default="along",
        help="Width matrix for time series (along or euclid)",
    )
    parser.add_argument(
        "--tide-direction",
        choices=["auto", "toward-start", "toward-end"],
        default="auto",
        help="Direction for placing tide points along transects",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Frames per second for GIF",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Step size for surveys when building GIF (e.g., 2 = every other survey)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output figures",
    )
    args = parser.parse_args()

    (
        width_along,
        width_euclid,
        survey_dates,
        transect_ids,
        width_signed,
        tide_east,
        tide_north,
    ) = _load_results(args.input)
    cliff_east, cliff_north = _load_cliff_toe(args.mat_file)

    gdf = gpd.read_file(args.transects)
    if gdf.empty:
        raise ValueError("Transects shapefile is empty")
    id_field = _choose_id_field(gdf, args.transect_id_field)
    transect_geoms = _order_transects(gdf, transect_ids, id_field)

    col_indices = _resolve_transect_column_indices(transect_ids, cliff_east.shape[1])
    if col_indices is None:
        if cliff_east.shape[1] != len(transect_geoms):
            raise ValueError(
                "Transect IDs do not map to cliff toe columns and counts do not match. "
                f"Cliff columns: {cliff_east.shape[1]}, Transects: {len(transect_geoms)}"
            )
        col_indices = list(range(len(transect_geoms)))

    bounds = gdf.total_bounds
    gif_path = os.path.join(args.output_dir, args.gif_name)
    print(f"Writing GIF to {gif_path}")
    _make_gif(
        width_along,
        width_signed,
        survey_dates,
        transect_geoms,
        cliff_east,
        cliff_north,
        tide_east,
        tide_north,
        col_indices,
        gif_path,
        bounds=bounds,
        fps=args.fps,
        dpi=args.dpi,
        frame_step=max(1, args.frame_step),
        tide_direction=args.tide_direction,
    )

    width_matrix = width_along if args.width_type == "along" else width_euclid
    ts_path = os.path.join(args.output_dir, args.timeseries_name)
    print(f"Writing time series to {ts_path}")
    _make_timeseries(width_matrix, survey_dates, ts_path, dpi=args.dpi)


if __name__ == "__main__":
    main()
