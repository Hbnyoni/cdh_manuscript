"""Python equivalent of extract_zimbabwe_climate.js, run directly against the
Earth Engine Python API instead of pasted into the Code Editor.

Written because this machine turned out to have real network access to
earthengine.googleapis.com (verified: `ee.Initialize(project=...)` succeeds),
unlike the restricted Linux VM CLAUDE.md describes. Same sources (CHIRPS v3
DAILY_RNL, ERA5-Land DAILY_AGGR), same definitions, same growing-season window.

Two differences from the .js reference, both performance-only (no change to
any number the analysis would compute):

1. Boundaries. The districts here come from the SAME FEWS NET admin
   boundaries shapefile the outcome is measured on (fetch_data.sh ->
   data/raw/fewsnet_boundaries/), dissolved from livelihood-zone polygons up
   to district level, with the Rural/Urban naming fixed against the
   geoBoundaries admin2 layer (see config.HFID_DISTRICT_NAME_FIXES). This is
   what the .js script's "BOUNDARY_ASSET" comment asks for; it is supplied
   here as a literal FeatureCollection (84 districts, ~0.9 MB simplified)
   rather than an uploaded EE asset, because at that size a literal is simpler
   and avoids an asset-ingestion step for a one-off extraction.

2. Hot-day counting. The .js script's seasonHot computation nests a per-day
   reduceRegion call inside a per-district map, i.e. one server call per
   (day, district) pair -- tens of thousands of calls that are slow enough to
   need Code Editor's long-running export rather than a direct API call. This
   script instead exports RAW district-day Tmax (reduceRegions ONCE per day,
   vectorised across all districts), matching the compact script's
   zw_daily_tmax_gs.csv path: run_pipeline.py already supports that as the
   (larger, ~15-20 MB) fallback input, and CLAUDE.md records that the two
   paths were checked against each other and agree. Thresholding and
   season-day counting then happen in cdh.indices, exactly as they do for the
   compact path; no threshold logic is duplicated here.

3. Chunked synchronous download, not a batch export task. The first version of
   this script used ee.batch.Export.table.toAsset and polled for completion --
   the documented, intended way to pull a table this size out of Earth Engine.
   In practice the batch queue for this project never picked the task up (even
   a trivial 1-feature diagnostic export sat at PENDING/READY indefinitely),
   while a direct synchronous compute request against the SAME already-
   authenticated session returned promptly. Two limits apply to that
   synchronous path, both worked around here: getInfo() aborts past 5,000
   elements, and a single request spanning many months/seasons at once (the
   whole 1981-2026 range built in one graph) draws a "Too many concurrent
   aggregations" 429. So the extraction is chunked -- one calendar year per
   request for the monthly table, one growing season per request for the daily
   table -- and each chunk is fetched with getDownloadURL(), which is not
   subject to the 5,000-element getInfo() cap, with retry/backoff for any 429
   that still occurs under load. If Earth Engine's batch queue is healthy again
   by the time this runs elsewhere, ee.batch.Export.table.toAsset is the
   simpler code path and chunking is unnecessary -- this fallback exists
   because that queue was not responding here.

Run:  python gee/extract_zimbabwe_climate.py
Writes data/raw/zw_monthly_climate.csv and data/raw/zw_daily_tmax_gs.csv.
Takes roughly 10-20 minutes: ~46 sequential requests for the monthly table,
~35 for the daily table, a few seconds to tens of seconds each.
"""
from __future__ import annotations

import json
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
DISTRICTS_GEOJSON = ROOT / "data" / "processed" / "_zw_districts_normalized.geojson"
START = "1981-01-01"
END = "2026-07-31"
DAILY_START = "1991-01-01"         # covers the 1991-2020 baseline + full panel
GS_MONTHS = [10, 11, 12, 1, 2, 3]
SCALE = 5566                       # CHIRPS native, ~0.05 deg


def districts_fc() -> ee.FeatureCollection:
    gj = json.loads(DISTRICTS_GEOJSON.read_text())
    feats = []
    for f in gj["features"]:
        geom = ee.Geometry(f["geometry"])
        name = f["properties"]["district"]
        c = geom.centroid(1000).coordinates()
        feats.append(ee.Feature(geom, {
            "district": name,
            "lat": c.get(1),
            "lon": c.get(0),
        }))
    return ee.FeatureCollection(feats)


def monthly_table(districts: ee.FeatureCollection, start=None, end=None) -> ee.FeatureCollection:
    """start/end narrow the actual computation graph, not just the output: a
    ee.List.sequence(...).map(...) still evaluates every month it lists even if
    the caller filters the result afterward, which is what blew through Earth
    Engine's concurrent-aggregation limit on the first attempt at this (540
    months x 84 districts in one graph). Chunk by passing start/end here.
    """
    start = start or START
    end = end or END
    chirps = (ee.ImageCollection("UCSB-CHC/CHIRPS/V3/DAILY_RNL")
              .select("precipitation").filterDate(start, end))
    era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .select(["temperature_2m_max", "temperature_2m_min"])
            .filterDate(start, end))
    era5p = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
             .select("total_precipitation_sum").filterDate(start, end))

    # ee.Date.difference(..., "month") is a floating-point elapsed-time ratio,
    # not an integer calendar-month count: for a non-leap-year chunk it comes
    # back fractionally short of 12 (e.g. 11.9997), and .floor() truncated that
    # to 11 -- silently dropping December from every non-leap year. Compute the
    # month count in Python instead, where it is exact.
    n_months = (pd.Timestamp(end).year - pd.Timestamp(start).year) * 12 \
        + (pd.Timestamp(end).month - pd.Timestamp(start).month)
    months = ee.List.sequence(0, n_months - 1)

    def per_month(m):
        t0 = ee.Date(start).advance(m, "month")
        t1 = t0.advance(1, "month")
        pr = chirps.filterDate(t0, t1).sum().rename("precip_mm")
        tt = (era5.filterDate(t0, t1).mean().subtract(273.15)
              .rename(["tmax_c", "tmin_c"]))
        pr_alt = (era5p.filterDate(t0, t1).sum().multiply(1000)
                  .rename("precip_alt_mm"))
        img = pr.addBands(tt).addBands(pr_alt)
        return img.reduceRegions(collection=districts, reducer=ee.Reducer.mean(),
                                  scale=SCALE).map(lambda f: ee.Feature(None, {
            "district": f.get("district"), "lat": f.get("lat"), "lon": f.get("lon"),
            "date": t0.format("YYYY-MM-01"),
            "precip_mm": f.get("precip_mm"), "precip_alt_mm": f.get("precip_alt_mm"),
            "tmax_c": f.get("tmax_c"), "tmin_c": f.get("tmin_c"),
        }))

    return ee.FeatureCollection(months.map(per_month)).flatten()


def daily_gs_tmax_table(districts: ee.FeatureCollection, start=None, end=None) -> ee.FeatureCollection:
    """start/end scope the graph itself, same reasoning as monthly_table."""
    start = start or DAILY_START
    end = end or END
    gs_filter = ee.Filter.Or(
        ee.Filter.calendarRange(10, 12, "month"),
        ee.Filter.calendarRange(1, 3, "month"))
    tmax = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .select("temperature_2m_max")
            .filterDate(start, end).filter(gs_filter)
            .map(lambda img: img.subtract(273.15)
                 .rename("tmax_c").copyProperties(img, ["system:time_start"])))

    # A single-band image's reduceRegions names its output after the reducer
    # ("mean"), not the band ("tmax_c") -- unlike a multi-band image, which
    # uses band names. Pin the output name explicitly rather than rely on
    # that inconsistency (it silently produced an all-null tmax_c column,
    # which the CSV exporter then dropped from the header entirely).
    reducer = ee.Reducer.mean().setOutputs(["tmax_c"])

    def per_day(img):
        d = ee.Date(img.get("system:time_start"))
        return img.reduceRegions(collection=districts, reducer=reducer,
                                  scale=SCALE).map(lambda f: ee.Feature(None, {
            "district": f.get("district"),
            "date": d.format("YYYY-MM-dd"),
            "tmax_c": f.get("tmax_c"),
        }))

    return ee.FeatureCollection(tmax.map(per_day)).flatten()


def _year_ranges(start: str, end: str):
    """[(chunk_start, chunk_end), ...] covering [start, end) in whole years."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for y in range(s.year, e.year + 1):
        cs = max(pd.Timestamp(f"{y}-01-01"), s)
        ce = min(pd.Timestamp(f"{y + 1}-01-01"), e)
        if cs < ce:
            out.append((cs.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")))
    return out


def _season_ranges(y0: int, y1: int):
    """[(start, end), ...] one growing season each, Oct(Y-1) through Mar(Y)."""
    return [(f"{y - 1}-10-01", f"{y}-04-01") for y in range(y0, y1 + 1)]


def _download_chunk(fc: ee.FeatureCollection, retries=5) -> pd.DataFrame:
    """getDownloadURL a chunk small enough to avoid both the 5000-element
    interactive cap and (mostly) the concurrent-aggregations 429; retry with
    backoff for the 429s that still happen under load."""
    for attempt in range(retries):
        try:
            url = fc.getDownloadURL("csv")
            with urllib.request.urlopen(url) as r:
                return pd.read_csv(r)
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"      retry {attempt + 1}/{retries} after {type(e).__name__}, "
                  f"waiting {wait}s")
            time.sleep(wait)


def extract_monthly(districts: ee.FeatureCollection) -> pd.DataFrame:
    chunks = _year_ranges(START, END)
    frames = []
    for i, (cs, ce) in enumerate(chunks):
        t0 = time.time()
        df = _download_chunk(monthly_table(districts, start=cs, end=ce))
        frames.append(df)
        print(f"   [{i + 1}/{len(chunks)}] {cs[:4]}: {len(df):,} rows "
              f"({time.time() - t0:.1f}s)")
    out = pd.concat(frames, ignore_index=True)
    return out.drop(columns=[c for c in ("system:index", ".geo") if c in out.columns])


def extract_daily(districts: ee.FeatureCollection) -> pd.DataFrame:
    y0, y1 = int(DAILY_START[:4]), int(END[:4])
    chunks = _season_ranges(y0, y1)
    frames = []
    for i, (cs, ce) in enumerate(chunks):
        t0 = time.time()
        df = _download_chunk(daily_gs_tmax_table(districts, start=cs, end=ce))
        frames.append(df)
        print(f"   [{i + 1}/{len(chunks)}] season {cs[:4]}-{ce[:4]}: {len(df):,} rows "
              f"({time.time() - t0:.1f}s)")
    out = pd.concat(frames, ignore_index=True)
    return out.drop(columns=[c for c in ("system:index", ".geo") if c in out.columns])


def main():
    ee.Initialize(project=EE_PROJECT)
    districts = districts_fc()
    print(f"Districts: {districts.size().getInfo()}")

    print("\n[1/2] Monthly climate table (CHIRPS v3 + ERA5-Land), by year")
    monthly = extract_monthly(districts)
    out1 = C.RAW / "zw_monthly_climate.csv"
    monthly.to_csv(out1, index=False)
    print(f"   wrote {out1} ({len(monthly):,} rows, {out1.stat().st_size / 1e6:.2f} MB)")

    print("\n[2/2] Daily growing-season Tmax table (ERA5-Land), by season")
    daily = extract_daily(districts)
    out2 = C.RAW / "zw_daily_tmax_gs.csv"
    daily.to_csv(out2, index=False)
    print(f"   wrote {out2} ({len(daily):,} rows, {out2.stat().st_size / 1e6:.2f} MB)")

    print("\nDone. Both CSVs are in data/raw/.")
    print("run_pipeline.py will use zw_daily_tmax_gs.csv (no zw_season_hotdays.csv "
          "was generated) and compute hot-day thresholds/counts in cdh.indices.")


if __name__ == "__main__":
    main()
