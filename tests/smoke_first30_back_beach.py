#!/usr/bin/env python3
"""
Quick smoke run on the first N transects to produce a small NPZ and GIF.

This limits computation to a subset of transects/columns to keep runtime down
while exercising the full width + visualization pipeline.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
FINDER_PATH = REPO_ROOT / "scripts" / "tools" / "back_beach_finder.py"
VIZ_PATH = REPO_ROOT / "scripts" / "visualizations" / "back_beach_viz.py"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bbf = _load_module(FINDER_PATH, "back_beach_finder")
viz = _load_module(VIZ_PATH, "back_beach_viz")


def compute_subset(args):
    mat_data = bbf.load_mat_file(args.mat_file)

    cliff_east = np.asarray(mat_data.get("CliffToe_East_Corrected"))
    cliff_north = np.asarray(mat_data.get("CliffToe_North_Corrected"))
    if cliff_east.size == 0 or cliff_north.size == 0:
        raise KeyError("CliffToe_East_Corrected or CliffToe_North_Corrected not found")

    if cliff_east.shape[1] > args.max_transects:
        cliff_east = cliff_east[:, : args.max_transects]
        cliff_north = cliff_north[:, : args.max_transects]

    dates_num, dem_files = bbf.extract_dem_list(mat_data)
    survey_dates = [bbf.matlab_datenum_to_datetime(dn) for dn in dates_num]

    sl_record = np.asarray(mat_data.get("SL_Record"))
    if sl_record.size == 0:
        raise KeyError("SL_Record not found in .mat file")
    mht = mat_data.get("MHT")
    if mht is None:
        raise KeyError("MHT not found in .mat file (required for mean high tide protocol)")
    mht_vals = np.asarray(mht).squeeze()
    if mht_vals.size != sl_record.shape[0]:
        raise ValueError(
            "MHT length does not match SL_Record rows. "
            f"MHT size: {mht_vals.size}, SL_Record rows: {sl_record.shape[0]}"
        )

    tide_times = np.array([bbf.matlab_datenum_to_datetime(dn) for dn in sl_record[:, 0]], dtype=object)
    tide_levels = pd.to_numeric(mht_vals, errors="coerce")
    valid_mask = pd.notna(tide_times) & np.isfinite(tide_levels)
    tide_times_valid = tide_times[valid_mask]
    tide_levels_valid = tide_levels[valid_mask]

    daily_tide = bbf.compute_daily_high_tide(tide_times_valid, tide_levels_valid, method="mean")
    if daily_tide.empty:
        raise RuntimeError("No valid daily tide values after filtering NaNs in MHT/SL_Record.")

    import geopandas as gpd
    import rasterio

    transects_gdf = gpd.read_file(args.transects)
    if transects_gdf.empty:
        raise ValueError("Transects shapefile is empty")
    transects_gdf = transects_gdf.head(args.max_transects)
    transect_geoms = list(transects_gdf.geometry)

    col_indices = list(range(cliff_east.shape[1]))

    width_along = np.full((len(survey_dates), cliff_east.shape[1]), np.nan, dtype=float)
    width_along_signed = np.full_like(width_along, np.nan)
    width_euclid = np.full_like(width_along, np.nan)
    tide_line_east = np.full_like(width_along, np.nan)
    tide_line_north = np.full_like(width_along, np.nan)

    dem_base = args.dem_base_dir or args.mat_file.parent

    for survey_idx, (survey_date, dem_path) in enumerate(zip(survey_dates, dem_files)):
        tide_date = survey_date.date()
        tide_level = daily_tide.get(tide_date)
        if tide_level is None:
            print(f"Skipping survey {survey_idx} ({tide_date}): no tide data")
            continue

        if not Path(dem_path).is_absolute():
            dem_path = dem_base / dem_path

        if not Path(dem_path).exists():
            print(f"Skipping survey {survey_idx} ({tide_date}): DEM not found at {dem_path}")
            continue

        with rasterio.open(dem_path) as ds:
            if ds.crs and transects_gdf.crs and transects_gdf.crs != ds.crs:
                transects_local = transects_gdf.to_crs(ds.crs)
            else:
                transects_local = transects_gdf

            for tran_idx, geom in enumerate(transects_local.geometry):
                if geom is None:
                    continue
                col_idx = col_indices[tran_idx]
                if col_idx >= cliff_east.shape[1]:
                    continue
                xs, ys, _ = bbf.sample_transect(geom, args.spacing_m)
                samples = np.array([val[0] for val in ds.sample(zip(xs, ys))], dtype=float)
                if ds.nodata is not None:
                    samples[samples == ds.nodata] = np.nan

                cliff_x = cliff_east[survey_idx, col_idx]
                cliff_y = cliff_north[survey_idx, col_idx]
                if not np.isfinite(cliff_x) or not np.isfinite(cliff_y):
                    continue

                cliff_point = bbf._point_from_xy((cliff_x, cliff_y))
                cross = bbf.find_tide_intersection(xs, ys, samples, tide_level, line=geom, cliff_point=cliff_point)
                if cross is None:
                    continue

                cross_point = bbf._point_from_xy(cross)
                s_cliff = geom.project(cliff_point)
                s_cross = geom.project(cross_point)
                signed_width = s_cross - s_cliff
                width_along_val = abs(signed_width)
                width_euclid_val = np.hypot(cross[0] - cliff_x, cross[1] - cliff_y)

                width_along[survey_idx, col_idx] = width_along_val
                width_along_signed[survey_idx, col_idx] = signed_width
                width_euclid[survey_idx, col_idx] = width_euclid_val
                tide_line_east[survey_idx, col_idx] = cross[0]
                tide_line_north[survey_idx, col_idx] = cross[1]

    return (
        survey_dates,
        transect_geoms,
        col_indices,
        cliff_east,
        cliff_north,
        width_along,
        width_along_signed,
        width_euclid,
        tide_line_east,
        tide_line_north,
    )


def main():
    parser = argparse.ArgumentParser(description="Smoke test: compute first N transects and plot GIF.")
    parser.add_argument(
        "--mat-file",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "BachBeach_and_tides.mat",
        help="Path to MATLAB file with cliff/tide data",
    )
    parser.add_argument(
        "--transects",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "shp_files"
        / "DelMarTransects595to620at1m"
        / "DelMarTransects595to620at1m.shp",
        help="Path to transects shapefile",
    )
    parser.add_argument(
        "--max-transects",
        type=int,
        default=30,
        help="Number of alongshore transects to process",
    )
    parser.add_argument(
        "--spacing-m",
        type=float,
        default=1.0,
        help="Sampling spacing along transects (meters)",
    )
    parser.add_argument(
        "--dem-base-dir",
        type=Path,
        default=None,
        help="Base directory for DEM paths if they are relative",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "artifacts" / "first30",
        help="Output directory for NPZ/GIF",
    )
    parser.add_argument("--gif-name", default="back_beach_lines_first30.gif", help="GIF filename")
    parser.add_argument("--npz-name", default="back_beach_widths_first30.npz", help="NPZ filename")
    parser.add_argument("--fps", type=float, default=4.0, help="Frames per second for GIF")
    parser.add_argument("--frame-step", type=int, default=1, help="Survey step when building GIF")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for output figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        survey_dates,
        transect_geoms,
        col_indices,
        cliff_east,
        cliff_north,
        width_along,
        width_along_signed,
        width_euclid,
        tide_line_east,
        tide_line_north,
    ) = compute_subset(args)

    npz_path = args.output_dir / args.npz_name
    np.savez(
        npz_path,
        width_along_transect_m=width_along,
        width_along_transect_signed_m=width_along_signed,
        width_euclid_m=width_euclid,
        tide_line_east_m=tide_line_east,
        tide_line_north_m=tide_line_north,
        survey_dates=np.array([d.date().isoformat() for d in survey_dates]),
        transect_ids=np.arange(width_along.shape[1]),
    )
    print(f"Wrote subset NPZ to {npz_path}")

    gif_path = args.output_dir / args.gif_name
    bounds = transects_gdf_total_bounds(transect_geoms)
    viz._make_gif(
        width_along,
        width_along_signed,
        survey_dates,
        transect_geoms,
        cliff_east,
        cliff_north,
        tide_line_east,
        tide_line_north,
        col_indices,
        gif_path,
        bounds=bounds,
        fps=args.fps,
        dpi=args.dpi,
        frame_step=max(1, args.frame_step),
        tide_direction="signed",
    )
    print(f"Wrote subset GIF to {gif_path}")


def transects_gdf_total_bounds(transect_geoms):
    xs = []
    ys = []
    for geom in transect_geoms:
        if geom is None:
            continue
        x, y = geom.xy
        xs.extend(x)
        ys.extend(y)
    return (min(xs), min(ys), max(xs), max(ys))


if __name__ == "__main__":
    main()
