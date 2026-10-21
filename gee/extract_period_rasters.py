"""Download actual gridded (raster) climate surfaces, one per growing season,
for the continuous-variable period maps in cdh.maps.

Why this exists, separate from extract_zimbabwe_climate.py: that script
extracts DISTRICT MEANS (reduceRegions against the district polygons) for the
panel the regression runs on -- correct for the analysis, but a district mean
choropleth of a continuous field like rainfall implies the value is uniform
within a district and jumps discontinuously at its border, which is not what
the physical field actually does. This script instead downloads the native
pixel grid itself (CHIRPS ~5.5 km, ERA5-Land ~11 km) for each growing season,
so the map can show the real continuous surface, with district/province
boundaries drawn as reference lines on top rather than as the fill itself.

Run:  python gee/extract_period_rasters.py
Writes one GeoTIFF per (variable, season-year) to data/processed/rasters/.
Takes a few minutes: ~16 seasons x 2 variables, a few seconds each.
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import ee
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cdh import config as C  # noqa: E402

EE_PROJECT = "ee-cheaqi"
OUT_DIR = C.PROC / "rasters"
SCALE_RAIN = 5566
SCALE_TEMP = 11132


def _download(url: str, out: Path, retries=5):
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, out)
            return
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"      retry {attempt + 1}/{retries} after {type(e).__name__}, "
                  f"waiting {wait}s")
            time.sleep(wait)


def season_rainfall_raster(region, start: str, end: str) -> str:
    img = (ee.ImageCollection("UCSB-CHC/CHIRPS/V3/DAILY_RNL")
           .select("precipitation").filterDate(start, end).sum())
    return img.getDownloadURL({"region": region, "scale": SCALE_RAIN,
                               "format": "GEO_TIFF", "crs": "EPSG:4326"})


def season_tmax_raster(region, start: str, end: str) -> str:
    img = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
           .select("temperature_2m_max").filterDate(start, end).mean()
           .subtract(273.15))
    return img.getDownloadURL({"region": region, "scale": SCALE_TEMP,
                               "format": "GEO_TIFF", "crs": "EPSG:4326"})


def main(season_years):
    ee.Initialize(project=EE_PROJECT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use the SAME boundary source the maps mask/overlay against (cdh.maps'
    # geoBoundaries-derived country shape), not the separately-processed FEWS
    # boundary used for the district-mean extraction. Using two independently
    # simplified boundary datasets for "the region to download" versus "the
    # region to display" left the raster not quite reaching the edge the
    # displayed black outline showed -- a real, visible cropping artefact
    # caught by looking at the rendered map, not by any check on the data.
    # A small buffer (0.02 deg, ~2 km) on the download region is cheap
    # insurance against the same mismatch recurring in the other direction.
    from cdh import maps as M
    districts, _ = M.load_boundaries()
    country_4326 = districts.to_crs(4326).union_all()
    region = ee.Geometry(country_4326.buffer(0.02).__geo_interface__).bounds()

    for i, y in enumerate(season_years):
        start, end = f"{y - 1}-10-01", f"{y}-04-01"

        rain_out = OUT_DIR / f"rain_{y}.tif"
        if not rain_out.exists():
            t0 = time.time()
            _download(season_rainfall_raster(region, start, end), rain_out)
            print(f"   [{i + 1}/{len(season_years)}] rain {y}: {time.time() - t0:.1f}s")

        tmax_out = OUT_DIR / f"tmax_{y}.tif"
        if not tmax_out.exists():
            t0 = time.time()
            _download(season_tmax_raster(region, start, end), tmax_out)
            print(f"   [{i + 1}/{len(season_years)}] tmax {y}: {time.time() - t0:.1f}s")

    print(f"\nWrote rasters to {OUT_DIR}")


if __name__ == "__main__":
    panel = pd.read_csv(C.PROC / "panel_real.csv")
    years = sorted(panel["season_year"].unique().tolist())
    main(years)
