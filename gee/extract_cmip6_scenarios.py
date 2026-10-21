"""District-level climatological growing-season rainfall and Tmax under future
climate scenarios (NASA NEX-GDDP-CMIP6), for the projection in cdh.scenarios.

Method: the "delta method", standard in agricultural climate-impact work when
a trusted long observational baseline (here, our own CHIRPS/ERA5-Land panel)
exists but the GCM's own absolute values should not be trusted directly (GCM
precipitation in particular carries model-specific biases even after
NEX-GDDP's own downscaling/bias-correction against a different reference
product). Only the GCM's projected CHANGE is used:

    delta = climatological_mean(future_scenario) - climatological_mean(historical_GCM)

cdh.scenarios then applies that delta on top of our own observed historical
climatology, not the GCM's absolute future value. This is why the script
extracts THREE climatological periods (historical GCM, and two future
scenarios), not the future periods alone.

Ensemble: all models available in NEX-GDDP-CMIP6 for Zimbabwe (33 as of
extraction), equally weighted. For a SUM aggregation (growing-season total
rainfall), the ensemble mean is computed as sum-over-all-model-days divided
by the model count, which is exactly equal to (mean across models of each
model's own seasonal sum) because both sum and mean are linear operators --
this avoids an explicit 33-way loop. For a MEAN aggregation (growing-season
mean Tmax), ee's own .mean() over the pooled model x day collection is
already the correct ensemble mean, for the same reason.

Scope: ONE time horizon (2046-2065, "2050s", the standard mid-century
horizon in this literature) x two scenarios (ssp245 moderate, ssp585 high
emissions), not a full multi-horizon sweep -- a deliberate scope decision
given the size of a full sweep, stated here and in the manuscript rather than
left implicit.

Run:  python gee/extract_cmip6_scenarios.py
Writes data/processed/cmip6_district_climatology.csv:
  district, period (historical/ssp245_2050s/ssp585_2050s), rain_mm, tmax_c
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import ee
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cdh import config as C  # noqa: E402

EE_PROJECT = "ee-cheaqi"
SCALE = 27830  # NEX-GDDP-CMIP6 native, ~0.25 deg

HIST_YEARS = range(1995, 2015)     # historical GCM reference, 20 seasons
FUTURE_YEARS = range(2046, 2066)   # 2050s horizon, 20 seasons
SCENARIOS = ["ssp245", "ssp585"]

# 3 of the 33 models in NEX-GDDP-CMIP6 (CESM2, CESM2-WACCM, IITM-ESM) do not
# carry a tasmax band, only tas (mean temperature) -- confirmed by checking
# actual band presence, not assumed from one sample image. Excluded from
# BOTH variables' ensembles, not just temperature's, so "the ensemble" means
# the same 30 models throughout rather than a different set per variable.
EXCLUDE_MODELS = ["CESM2", "CESM2-WACCM", "IITM-ESM"]


def districts_fc():
    import json
    gj = json.loads((C.PROC / "_zw_districts_normalized.geojson").read_text())
    feats = []
    for f in gj["features"]:
        geom = ee.Geometry(f["geometry"])
        feats.append(ee.Feature(geom, {"district": f["properties"]["district"]}))
    return ee.FeatureCollection(feats)


def _season_rain_sum(scenario: str, y: int, districts) -> ee.FeatureCollection:
    """Ensemble-mean growing-season total rainfall for one season-year."""
    t0 = ee.Date.fromYMD(y - 1, 10, 1)
    t1 = ee.Date.fromYMD(y, 4, 1)
    coll = (ee.ImageCollection("NASA/GDDP-CMIP6")
            .filter(ee.Filter.eq("scenario", scenario))
            .filter(ee.Filter.inList("model", EXCLUDE_MODELS).Not())
            .select("pr").filterDate(t0, t1))
    n_days = t1.difference(t0, "day")
    n_models = ee.Number(coll.size()).divide(n_days)
    total = coll.sum().multiply(86400).divide(n_models)  # kg/m2/s -> mm/day, sum -> mm
    # A single-band image's reduceRegions names its output after the reducer
    # ("mean"), not the band -- pin it explicitly (see
    # extract_zimbabwe_climate.py, which hit the same issue).
    reducer = ee.Reducer.mean().setOutputs(["rain_mm"])
    return total.rename("rain_mm").reduceRegions(
        collection=districts, reducer=reducer, scale=SCALE)


def _season_tmax_mean(scenario: str, y: int, districts) -> ee.FeatureCollection:
    """Ensemble-mean growing-season mean Tmax for one season-year."""
    t0 = ee.Date.fromYMD(y - 1, 10, 1)
    t1 = ee.Date.fromYMD(y, 4, 1)
    coll = (ee.ImageCollection("NASA/GDDP-CMIP6")
            .filter(ee.Filter.eq("scenario", scenario))
            .filter(ee.Filter.inList("model", EXCLUDE_MODELS).Not())
            .select("tasmax").filterDate(t0, t1))
    mean_k = coll.mean()
    mean_c = mean_k.subtract(273.15)
    reducer = ee.Reducer.mean().setOutputs(["tmax_c"])
    return mean_c.rename("tmax_c").reduceRegions(
        collection=districts, reducer=reducer, scale=SCALE)


CACHE_DIR = None  # set in main() once C.PROC is known


def _getinfo_retry(fc, retries=5):
    """A background run this size (60 season-years x 2 variables) is long
    enough to outlast a transient network blip -- confirmed the hard way
    when one killed an unchecked-pointed first attempt after ~30 minutes.
    Retry with backoff rather than let one dropped connection cost the
    whole run."""
    for attempt in range(retries):
        try:
            return fc.getInfo()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"      retry {attempt + 1}/{retries} after {type(e).__name__}, "
                  f"waiting {wait}s")
            time.sleep(wait)


def _year_climatology(scenario: str, y: int, districts) -> pd.DataFrame:
    """One season-year's district table, cached to disk so a failure later
    in a 20-year loop does not require re-fetching years already done."""
    cache = CACHE_DIR / f"{scenario}_{y}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    r = _getinfo_retry(_season_rain_sum(scenario, y, districts))
    t = _getinfo_retry(_season_tmax_mean(scenario, y, districts))
    rain = pd.DataFrame([{"district": f["properties"]["district"],
                         "rain_mm": f["properties"]["rain_mm"]} for f in r["features"]])
    tmax = pd.DataFrame([{"district": f["properties"]["district"],
                         "tmax_c": f["properties"]["tmax_c"]} for f in t["features"]])
    out = rain.merge(tmax, on="district")
    out.to_csv(cache, index=False)
    return out


def climatology(scenario: str, years, districts) -> pd.DataFrame:
    """Mean-across-years district table: one row per district, rain_mm/tmax_c
    averaged over every season-year in `years`."""
    frames = []
    for i, y in enumerate(years):
        t0 = time.time()
        frames.append(_year_climatology(scenario, y, districts))
        print(f"      [{i + 1}/{len(years)}] {scenario} {y}: {time.time() - t0:.1f}s")
    all_years = pd.concat(frames, ignore_index=True)
    return all_years.groupby("district", as_index=False)[["rain_mm", "tmax_c"]].mean()


def main():
    global CACHE_DIR
    ee.Initialize(project=EE_PROJECT)
    CACHE_DIR = C.PROC / "cmip6_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    districts = districts_fc()

    out_rows = []
    print("[1/3] Historical GCM reference (1995-2014)")
    hist = climatology("historical", HIST_YEARS, districts)
    hist["period"] = "historical"
    out_rows.append(hist)

    for scen in SCENARIOS:
        print(f"\n[{SCENARIOS.index(scen) + 2}/3] {scen} 2050s (2046-2065)")
        fut = climatology(scen, FUTURE_YEARS, districts)
        fut["period"] = f"{scen}_2050s"
        out_rows.append(fut)

    out = pd.concat(out_rows, ignore_index=True)[["district", "period", "rain_mm", "tmax_c"]]
    dst = C.PROC / "cmip6_district_climatology.csv"
    out.to_csv(dst, index=False)
    print(f"\nWrote {dst} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
