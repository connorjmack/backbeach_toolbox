import importlib.util
from datetime import date
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "back_beach_finder.py"
spec = importlib.util.spec_from_file_location("back_beach_finder", MODULE_PATH)
bbf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bbf)


class FakeStruct:
    def __init__(self, **kwargs):
        self._fieldnames = list(kwargs.keys())
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_matlab_datenum_to_datetime_epoch():
    dt = bbf.matlab_datenum_to_datetime(719529)
    assert dt.date() == date(1970, 1, 1)


def test_compute_daily_high_tide_mean():
    times = [
        "2020-01-01 00:00:00",
        "2020-01-01 01:00:00",
        "2020-01-02 00:00:00",
        "2020-01-02 01:00:00",
    ]
    levels = [1.0, 3.0, 4.0, 6.0]
    daily = bbf.compute_daily_high_tide(times, levels, method="mean")
    assert daily.loc[date(2020, 1, 1)] == 2.0
    assert daily.loc[date(2020, 1, 2)] == 5.0


def test_extract_dem_list_struct_array_darwin(monkeypatch):
    monkeypatch.setattr(bbf.platform, "system", lambda: "Darwin")
    dem_full = np.array(
        [
            FakeStruct(
                name="dem1.tif",
                folder="Z:\\LiDAR\\VMZ2000_Truck\\DEM",
                dates_num=730486.0,
            ),
            FakeStruct(
                name="dem2.tif",
                folder="Z:\\LiDAR\\VMZ2000_Truck\\DEM",
                dates_num=730487.0,
            ),
        ],
        dtype=object,
    )
    dem_list = FakeStruct(FULL=dem_full)
    dates, paths = bbf.extract_dem_list({"DEM_list_Corrected": dem_list})
    assert np.allclose(dates, [730486.0, 730487.0])
    assert paths[0].endswith("LiDAR/VMZ2000_Truck/DEM/dem1.tif")
    assert paths[0].startswith("/Volumes/group/")


def test_extract_dem_list_join_paths():
    dem_full = np.array(
        [
            FakeStruct(
                name="/dem3.tif",
                folder="/data/dems/",
                dates_num=730488.0,
            ),
        ],
        dtype=object,
    )
    dem_list = FakeStruct(FULL=dem_full)
    _, paths = bbf.extract_dem_list({"DEM_list_Corrected": dem_list})
    assert paths[0].endswith("data/dems/dem3.tif")


def test_resolve_transect_column_indices():
    ids_zero = list(range(5))
    ids_one = list(range(1, 6))
    assert bbf._resolve_transect_column_indices(ids_zero, 5) == ids_zero
    assert bbf._resolve_transect_column_indices(ids_one, 5) == [0, 1, 2, 3, 4]
    assert bbf._resolve_transect_column_indices([10, 11], 5) is None
