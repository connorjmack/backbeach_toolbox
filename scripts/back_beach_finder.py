#!/usr/bin/env python3
"""
Compute beach widths by intersecting daily mean tide elevations with transect profiles
and measuring distance to cliff toe locations.

Requirements:
  - numpy
  - pandas
  - scipy (for .mat loading) or mat73 (optional fallback)
  - geopandas
  - shapely
  - rasterio
"""

import argparse
import os
import platform
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def matlab_datenum_to_datetime(datenum):
    """Convert a MATLAB datenum (days since 0000-01-00) to datetime."""
    return datetime.fromordinal(int(datenum)) + timedelta(days=float(datenum) % 1) - timedelta(days=366)


def _to_py_str(value):
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("U", "S"):
            if value.ndim == 0:
                return str(value.item())
            if value.ndim == 1:
                return "".join(value.tolist()).strip()
            if value.ndim == 2:
                return "".join(value.flatten().tolist()).strip()
        if value.dtype.kind in ("i", "u"):
            try:
                return bytes(value.tolist()).decode("utf-8", errors="ignore").strip()
            except Exception:
                pass
        if value.size == 1:
            return _to_py_str(value.item())
    return str(value)


def _cellstr_to_list(cell):
    arr = np.asarray(cell)
    if arr.dtype == object:
        return [_to_py_str(item) for item in arr.flatten()]
    if arr.dtype.kind in ("U", "S"):
        if arr.ndim == 2:
            return ["".join(row).strip() for row in arr]
        return [_to_py_str(arr)]
    return [_to_py_str(item) for item in arr.flatten()]


def _mat_field_names(obj):
    if isinstance(obj, dict):
        return list(obj.keys())
    if hasattr(obj, "_fieldnames"):
        return list(obj._fieldnames)
    return []


def _get_field(obj, name):
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    if hasattr(obj, name):
        return getattr(obj, name)
    raise KeyError(name)


def load_mat_file(mat_path):
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise RuntimeError("scipy is required to read .mat files. Install with: pip install scipy") from exc
    return loadmat(mat_path, squeeze_me=True, struct_as_record=False)


def extract_dem_list(mat_data):
    dem_list = mat_data.get("DEM_list_Corrected")
    if dem_list is None:
        raise KeyError("DEM_list_Corrected not found in .mat file")

    try:
        dem_full = _get_field(dem_list, "FULL")
    except KeyError as exc:
        available = ", ".join(_mat_field_names(dem_list))
        raise KeyError(f"DEM_list_Corrected is missing FULL. Available fields: {available}") from exc

    def _normalize_dem_path(path):
        raw = _to_py_str(path).strip()
        if not raw:
            return raw
        normalized = raw.replace("\\", "/")
        if normalized.lower().startswith("z:/"):
            rel = normalized[2:]
            if not rel.startswith("/"):
                rel = "/" + rel
            if platform.system() == "Darwin":
                base = "/Volumes/group"
            elif platform.system() == "Linux":
                base = "/project/group"
            else:
                base = None
            if base:
                return os.path.normpath(base + rel)
        return os.path.normpath(normalized)

    def _build_path(folder, name):
        folder = _to_py_str(folder).strip()
        name = _to_py_str(name).strip()
        if not folder:
            return _normalize_dem_path(name)
        joined = f"{folder.rstrip('/\\\\')}/{name.lstrip('/\\\\')}"
        return _normalize_dem_path(joined)

    def _expand_columnar(item):
        names = _get_field(item, "name")
        folders = _get_field(item, "folder") if "folder" in _mat_field_names(item) else None
        dates = _get_field(item, "dates_num")

        name_list = _cellstr_to_list(names)
        folder_list = _cellstr_to_list(folders) if folders is not None else [""] * len(name_list)
        dates_arr = np.asarray(dates).astype(float).flatten()

        if dates_arr.size == 1 and len(name_list) > 1:
            dates_arr = np.full(len(name_list), dates_arr.item(), dtype=float)

        if len(name_list) != len(folder_list) or dates_arr.size != len(name_list):
            raise ValueError(
                "DEM_list_Corrected.FULL field lengths do not align. "
                f"name={len(name_list)}, folder={len(folder_list)}, dates={dates_arr.size}"
            )

        paths = [_build_path(folder, name) for folder, name in zip(folder_list, name_list)]
        return dates_arr, paths

    if isinstance(dem_full, np.ndarray) and dem_full.dtype == object:
        struct_items = dem_full.flatten().tolist()
    elif isinstance(dem_full, list):
        struct_items = dem_full
    else:
        struct_items = [dem_full]

    if len(struct_items) == 1 and hasattr(struct_items[0], "_fieldnames"):
        first_item = struct_items[0]
        try:
            names = _get_field(first_item, "name")
            if np.asarray(names).size > 1:
                return _expand_columnar(first_item)
        except Exception:
            pass

    dates = []
    paths = []
    for item in struct_items:
        if not hasattr(item, "_fieldnames"):
            continue
        try:
            name = _get_field(item, "name")
        except KeyError as exc:
            available = ", ".join(_mat_field_names(item))
            raise KeyError(f"DEM_list_Corrected.FULL entries missing name. Available fields: {available}") from exc
        folder = _get_field(item, "folder") if "folder" in _mat_field_names(item) else ""
        try:
            date_val = _get_field(item, "dates_num")
        except KeyError as exc:
            available = ", ".join(_mat_field_names(item))
            raise KeyError(
                "DEM_list_Corrected.FULL entries missing dates_num. "
                f"Available fields: {available}"
            ) from exc

        dates.append(float(np.asarray(date_val).squeeze()))
        paths.append(_build_path(folder, name))

    if not dates:
        available = ", ".join(_mat_field_names(dem_full))
        raise KeyError(
            "Could not parse DEM_list_Corrected.FULL entries. "
            f"Available fields: {available}"
        )

    return np.asarray(dates, dtype=float), paths


def compute_daily_high_tide(tide_times, tide_levels, method="max"):
    tide_times = pd.to_datetime(tide_times)
    df = pd.DataFrame({"time": tide_times, "level": tide_levels})
    df = df.dropna()
    df["date"] = df["time"].dt.date

    if method == "max":
        daily = df.groupby("date")["level"].max()
    elif method == "mean":
        daily = df.groupby("date")["level"].mean()
    else:
        raise ValueError(f"Unknown method: {method}")
    return daily


def sample_transect(line, spacing):
    length = line.length
    distances = np.arange(0, length + spacing, spacing)
    points = [line.interpolate(dist) for dist in distances]
    xs = np.array([pt.x for pt in points])
    ys = np.array([pt.y for pt in points])
    return xs, ys, distances


def find_tide_intersection(xs, ys, zs, tide_level, line=None, cliff_point=None):
    mask = np.isfinite(zs)
    if mask.sum() < 2:
        return None

    xs = xs[mask]
    ys = ys[mask]
    zs = zs[mask]

    diffs = zs - tide_level
    sign = np.sign(diffs)
    cross_idx = np.where(sign[:-1] * sign[1:] <= 0)[0]
    if cross_idx.size == 0:
        return None

    candidates = []
    for idx in cross_idx:
        z1 = zs[idx]
        z2 = zs[idx + 1]
        if z2 == z1:
            frac = 0.0
        else:
            frac = (tide_level - z1) / (z2 - z1)
        x = xs[idx] + frac * (xs[idx + 1] - xs[idx])
        y = ys[idx] + frac * (ys[idx + 1] - ys[idx])
        candidates.append((x, y))

    if cliff_point is None:
        return candidates[0]

    if line is None:
        def dist(p):
            return np.hypot(p[0] - cliff_point.x, p[1] - cliff_point.y)
        return min(candidates, key=dist)

    cliff_dist = line.project(cliff_point)

    def line_dist(p):
        return abs(line.project(_point_from_xy(p)) - cliff_dist)

    return min(candidates, key=line_dist)


def _point_from_xy(xy):
    from shapely.geometry import Point

    return Point(xy[0], xy[1])


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
    return None


def main():
    parser = argparse.ArgumentParser(description="Compute beach widths from DEMs, tides, and cliff toe data.")
    parser.add_argument(
        "--mat-file",
        default="data/raw/BachBeach_and_tides.mat",
        help="Path to the MATLAB .mat file with cliff toe, DEM list, and tide data",
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
        "--output-npz",
        default="data/processed/back_beach_widths.npz",
        help="Output NPZ path for width matrices",
    )
    parser.add_argument(
        "--output-mat",
        default="data/processed/back_beach_widths.mat",
        help="Output MAT path for width matrices",
    )
    parser.add_argument(
        "--spacing-m",
        type=float,
        default=1.0,
        help="Sampling spacing along transects (meters)",
    )
    parser.add_argument(
        "--tide-method",
        choices=["max", "mean"],
        default="mean",
        help="Daily tide aggregation: max (daily high tide) or mean",
    )
    parser.add_argument(
        "--allow-nearest-tide-day",
        action="store_true",
        help="Use nearest available tide day when survey date is missing",
    )
    parser.add_argument(
        "--transect-id-field",
        default=None,
        help="Optional transect ID field name from shapefile attributes",
    )
    parser.add_argument(
        "--dem-base-dir",
        default=None,
        help="Base directory for DEM paths if they are relative",
    )

    args = parser.parse_args()

    mat_data = load_mat_file(args.mat_file)

    cliff_east = np.asarray(mat_data.get("CliffToe_East_Corrected"))
    cliff_north = np.asarray(mat_data.get("CliffToe_North_Corrected"))
    if cliff_east.size == 0 or cliff_north.size == 0:
        raise KeyError("CliffToe_East_Corrected or CliffToe_North_Corrected not found")

    dates_num, dem_files = extract_dem_list(mat_data)
    survey_dates = [matlab_datenum_to_datetime(dn) for dn in dates_num]

    sl_record = np.asarray(mat_data.get("SL_Record"))
    if sl_record.size == 0:
        raise KeyError("SL_Record not found in .mat file")
    tide_times = [matlab_datenum_to_datetime(dn) for dn in sl_record[:, 0]]
    tide_levels = sl_record[:, 1]

    daily_tide = compute_daily_high_tide(tide_times, tide_levels, method=args.tide_method)

    n_surveys = len(survey_dates)
    if cliff_east.shape[0] != n_surveys and cliff_east.shape[1] == n_surveys:
        cliff_east = cliff_east.T
        cliff_north = cliff_north.T
    if cliff_east.shape[0] != n_surveys:
        raise ValueError(
            "Cliff toe arrays do not match the number of survey dates. "
            f"Surveys: {n_surveys}, Cliff shape: {cliff_east.shape}"
        )

    import geopandas as gpd
    import rasterio

    transects_gdf = gpd.read_file(args.transects)
    if transects_gdf.empty:
        raise ValueError("Transects shapefile is empty")

    if args.transect_id_field:
        transect_ids = transects_gdf[args.transect_id_field].tolist()
    else:
        id_field_candidates = ["Id", "ID", "TransectID", "TRANSECTID", "transect_id", "fid"]
        id_field = next((f for f in id_field_candidates if f in transects_gdf.columns), None)
        transect_ids = transects_gdf[id_field].tolist() if id_field else list(range(len(transects_gdf)))

    width_along = np.full((n_surveys, cliff_east.shape[1]), np.nan, dtype=float)
    width_euclid = np.full((n_surveys, cliff_east.shape[1]), np.nan, dtype=float)

    dem_base = args.dem_base_dir or os.path.dirname(os.path.abspath(args.mat_file))

    for survey_idx, (survey_date, dem_path) in enumerate(zip(survey_dates, dem_files)):
        tide_date = survey_date.date()
        tide_level = daily_tide.get(tide_date)
        if tide_level is None and args.allow_nearest_tide_day:
            nearest_date = min(daily_tide.index, key=lambda d: abs(d - tide_date))
            tide_level = daily_tide.get(nearest_date)
            tide_date = nearest_date
        if tide_level is None:
            print(f"Skipping survey {survey_idx} ({survey_date.date()}): no tide data")
            continue

        if not os.path.isabs(dem_path):
            dem_path = os.path.join(dem_base, dem_path)

        if not os.path.exists(dem_path):
            print(f"Skipping survey {survey_idx}: DEM not found at {dem_path}")
            continue

        with rasterio.open(dem_path) as ds:
            dem_crs = ds.crs
            if dem_crs and transects_gdf.crs and transects_gdf.crs != dem_crs:
                transects = transects_gdf.to_crs(dem_crs)
            else:
                transects = transects_gdf

            col_indices = _resolve_transect_column_indices(transect_ids, cliff_east.shape[1])
            if col_indices is None:
                if cliff_east.shape[1] != len(transects):
                    raise ValueError(
                        "Cliff toe alongshore count does not match transect count and "
                        "transect IDs do not map to cliff columns. "
                        f"Cliff: {cliff_east.shape[1]}, Transects: {len(transects)}"
                    )
                col_indices = list(range(len(transects)))

            for tran_idx, (tran_id, geom) in enumerate(zip(transect_ids, transects.geometry)):
                if geom is None:
                    continue
                col_idx = col_indices[tran_idx]
                if col_idx < 0 or col_idx >= cliff_east.shape[1]:
                    continue
                xs, ys, _ = sample_transect(geom, args.spacing_m)
                samples = np.array([val[0] for val in ds.sample(zip(xs, ys))], dtype=float)
                if ds.nodata is not None:
                    samples[samples == ds.nodata] = np.nan

                cliff_x = cliff_east[survey_idx, col_idx]
                cliff_y = cliff_north[survey_idx, col_idx]
                if not np.isfinite(cliff_x) or not np.isfinite(cliff_y):
                    continue

                cliff_point = _point_from_xy((cliff_x, cliff_y))
                cross = find_tide_intersection(xs, ys, samples, tide_level, line=geom, cliff_point=cliff_point)
                if cross is None:
                    continue

                cross_point = _point_from_xy(cross)
                width_along_val = abs(geom.project(cross_point) - geom.project(cliff_point))
                width_euclid_val = np.hypot(cross[0] - cliff_x, cross[1] - cliff_y)

                width_along[survey_idx, col_idx] = width_along_val
                width_euclid[survey_idx, col_idx] = width_euclid_val

    if np.isnan(width_along).all() and np.isnan(width_euclid).all():
        raise RuntimeError("No beach widths were computed. Check inputs and parameters.")

    os.makedirs(os.path.dirname(args.output_npz), exist_ok=True)
    np.savez(
        args.output_npz,
        width_along_transect_m=width_along,
        width_euclid_m=width_euclid,
        survey_dates=np.array([d.date().isoformat() for d in survey_dates]),
        transect_ids=np.asarray(transect_ids),
    )
    print(f"Wrote width matrices to {args.output_npz}")

    try:
        from scipy.io import savemat
    except Exception as exc:
        raise RuntimeError("scipy is required to write .mat files. Install with: pip install scipy") from exc

    os.makedirs(os.path.dirname(args.output_mat), exist_ok=True)
    savemat(
        args.output_mat,
        {
            "width_along_transect_m": width_along,
            "width_euclid_m": width_euclid,
            "survey_dates": np.array([d.date().isoformat() for d in survey_dates], dtype=object),
            "transect_ids": np.asarray(transect_ids),
        },
    )
    print(f"Wrote width matrices to {args.output_mat}")



if __name__ == "__main__":
    main()
