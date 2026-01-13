# Backbeach Toolbox

## Project Overview
The **Backbeach Toolbox** is a Python-based utility for coastal data processing, specifically designed to compute beach widths by intersecting daily mean tide elevations with transect profiles and measuring the distance to cliff toe locations. It integrates geospatial data (shapefiles, DEMs) with time-series data (tides) to produce analytical outputs.

## Architecture & Technologies
*   **Language:** Python 3.11
*   **Core Libraries:**
    *   `numpy`, `pandas`, `scipy`: Data manipulation and scientific computing.
    *   `geopandas`, `shapely`: Geospatial vector data handling.
    *   `rasterio`: Raster data (DEM) processing.
    *   `pyproj`: Coordinate reference system management.
*   **Data Formats:**
    *   Inputs: MATLAB `.mat` files (legacy data), Shapefiles (`.shp`), GeoTIFFs (DEMs).
    *   Outputs: `.npz` (NumPy archives), `.mat` (MATLAB compatibility).

## Setup and Installation

### Conda (Recommended)
This project includes a Conda environment file for easy setup.

```bash
conda env create -f environment.yml
conda activate backbeach_toolbox
```

### Pip
Alternatively, dependencies can be installed via pip:

```bash
pip install -r requirements.txt
```

## Key Scripts

### `scripts/back_beach_finder.py`
This is the primary utility for calculating beach widths. It processes survey data, tide records, and transects to compute distances.

**Usage:**

```bash
python scripts/back_beach_finder.py [options]
```

**Key Arguments:**
*   `--mat-file`: Path to the MATLAB file containing cliff toe data, DEM lists, and tide records (default: `data/raw/BachBeach_and_tides.mat`).
*   `--transects`: Path to the transect shapefile (default: `data/shp_files/DelMarTransects595to620at1m/DelMarTransects595to620at1m.shp`).
*   `--output-npz`: Output path for NumPy archive (default: `data/processed/back_beach_widths.npz`).
*   `--output-mat`: Output path for MATLAB file (default: `data/processed/back_beach_widths.mat`).
*   `--spacing-m`: Sampling spacing along transects in meters (default: `1.0`).
*   `--tide-method`: Aggregation method for daily tides (`max` or `mean`).

**Example:**

```bash
python scripts/back_beach_finder.py \
  --mat-file data/raw/BachBeach_and_tides.mat \
  --transects data/shp_files/MyTransects/transects.shp \
  --output-npz data/processed/results.npz
```

## Data Directory Structure
*   `data/raw/`: Contains original datasets, including the master `.mat` file and shapefiles.
*   `data/processed/`: Destination for cleaned data and script outputs.
*   `data/shp_files/`: Geospatial vector files (transects).

## Development
*   **Source Code:** Core logic is located in `src/` (currently empty/placeholder) and `scripts/`.
*   **Testing:** Tests should be placed in the `tests/` directory.
