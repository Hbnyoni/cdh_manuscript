"""Generate synthetic inputs with the EXACT schemas the real files use.

Purpose: prove the pipeline runs end to end and recovers a known effect, before
any real data exists. The data generating process embeds a true compound
interaction of TRUE_INTERACTION; a working pipeline should recover an estimate
near it, attenuated by the discretisation of the outcome into IPC phases.

Run:  python make_synthetic.py --outdir data/synthetic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260914)

N_DISTRICTS = 60
YEARS = (1991, 2026)
TRUE_DRY = 0.35
TRUE_HEAT = 0.15
TRUE_INTERACTION = 0.30

BOUNDARIES = Path(__file__).resolve().parent / "data" / "boundaries" / "zwe_adm2.geojson"


def _district_frame():
    """Real district names and centroids where the boundary layer is available.

    Using the real geometry means the synthetic run exercises the same name join
    and the same maps as the real run, so a broken join shows up in testing rather
    than on the first real execution.
    """
    if BOUNDARIES.exists():
        import geopandas as gpd
        g = gpd.read_file(BOUNDARIES).rename(columns={"shapeName": "district"})
        # same urban-council rule the map join uses, so the two never disagree
        sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
        from cdh.maps import URBAN_RE
        g = g[~g["district"].str.contains(URBAN_RE, na=False)].copy()
        cen = g.geometry.representative_point()
        d = pd.DataFrame({"district": g["district"].to_numpy(),
                          "lat": cen.y.to_numpy(), "lon": cen.x.to_numpy()})
        d = d.sort_values("district").reset_index(drop=True)
    else:
        print("  ! no boundary layer; falling back to synthetic district names")
        d = pd.DataFrame({
            "district": [f"ZW_D{i:02d}" for i in range(1, N_DISTRICTS + 1)],
            "lat": RNG.uniform(-22.2, -15.8, N_DISTRICTS),
            "lon": RNG.uniform(25.3, 33.0, N_DISTRICTS)})

    # Rainfall falls to the south and west, as the Natural Regions do
    d["rain_mean"] = np.clip(
        980 + (d["lat"] + 19) * 55 + (d["lon"] - 29) * 18
        + RNG.normal(0, 60, len(d)), 380, 1250)
    return d


def _daily(dinfo):
    y0, y1 = YEARS
    dates = pd.date_range(f"{y0-1}-07-01", f"{y1}-07-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    # Southern-hemisphere seasonal cycle, warmest around late October
    seas = np.cos(2 * np.pi * (doy - 300) / 365.25)
    yr = dates.year.to_numpy()
    warming = 0.025 * (yr - 1991)                          # ~0.9 C over the record
    year_anom = {y: RNG.normal(0, 0.8) for y in range(y0 - 1, y1 + 2)}
    ya = np.array([year_anom[y] for y in yr])

    frames = []
    for _, r in dinfo.iterrows():
        base = 29.5 - 0.55 * (r["lat"] + 19) - 0.004 * (r["rain_mean"] - 800)
        tmax = base + 4.5 * seas + warming + ya + RNG.normal(0, 2.4, len(dates))
        tmin = tmax - (11.0 + RNG.normal(0, 1.2, len(dates)))
        frames.append(pd.DataFrame({"district": r["district"], "date": dates,
                                    "tmax_c": tmax, "tmin_c": tmin}))
    return pd.concat(frames, ignore_index=True)


def _monthly_precip(dinfo):
    """Unimodal wet season, normalised WITHIN the 12-month cycle.

    Monthly shares approximate the Zimbabwean climatology: about 70 per cent of
    annual rainfall falls December to February, and May to September is
    effectively dry.
    """
    y0, y1 = YEARS
    months = pd.date_range(f"{y0-1}-07-01", f"{y1}-07-01", freq="MS")
    m = months.month.to_numpy()

    # index 0 = January; shares sum to 1 across the calendar year
    SHARE = np.array([0.200, 0.170, 0.110, 0.040, 0.010, 0.005,
                      0.003, 0.004, 0.008, 0.030, 0.120, 0.200])
    SHARE = SHARE / SHARE.sum()
    share_m = SHARE[m - 1]

    yr = months.year.to_numpy()
    season_year = np.where(m >= 10, yr + 1, yr)
    # Season-wide anomaly, with strong dry years standing in for El Nino
    sa = {y: RNG.normal(0, 0.22) for y in range(y0 - 1, y1 + 2)}
    for y in (1992, 2016, 2019, 2024):
        if y in sa:
            sa[y] -= 0.38

    frames = []
    for _, r in dinfo.iterrows():
        mult = np.clip(np.array([1 + sa[y] for y in season_year]), 0.15, None)
        mu = r["rain_mean"] * share_m * mult
        k = 2.2                                   # gamma shape -> CV ~0.67
        precip = RNG.gamma(shape=k, scale=np.maximum(mu, 0.2) / k)
        frames.append(pd.DataFrame({"district": r["district"], "date": months,
                                    "precip_mm": precip}))
    return pd.concat(frames, ignore_index=True)


def _outcome(monthly, daily, dinfo):
    """Build IPC-like phases from a known structural model."""
    m = monthly.copy()
    m["season_year"] = np.where(m["date"].dt.month >= 10,
                                m["date"].dt.year + 1, m["date"].dt.year)
    gs = m[m["date"].dt.month.isin([10, 11, 12, 1, 2, 3])]
    rain = gs.groupby(["district", "season_year"])["precip_mm"].sum().reset_index()

    d = daily[daily["date"].dt.month.isin([10, 11, 12, 1, 2, 3])].copy()
    d["season_year"] = np.where(d["date"].dt.month >= 10,
                                d["date"].dt.year + 1, d["date"].dt.year)
    thr = (d[d["date"].dt.year.between(1991, 2020)]
           .groupby("district")["tmax_c"].quantile(0.90))
    d["hot"] = (d["tmax_c"] > d["district"].map(thr)).astype(float)
    hot = d.groupby(["district", "season_year"])["hot"].sum().reset_index(name="hot_days")

    p = rain.merge(hot, on=["district", "season_year"])
    g = p.groupby("district")
    p["dry"] = g["precip_mm"].transform(lambda s: -(s - s.mean()) / s.std(ddof=0))
    p["heat"] = g["hot_days"].transform(lambda s: (s - s.mean()) / s.std(ddof=0))

    names = list(dinfo["district"])
    dfe = dict(zip(names, RNG.normal(0, 0.45, len(names))))
    yfe = {y: RNG.normal(0, 0.25) for y in p["season_year"].unique()}
    latent = (1.85
              + TRUE_DRY * p["dry"]
              + TRUE_HEAT * p["heat"]
              + TRUE_INTERACTION * p["dry"] * p["heat"]
              + p["district"].map(dfe)
              + p["season_year"].map(yfe)
              + RNG.normal(0, 0.30, len(p)))
    p["phase"] = np.clip(np.rint(latent), 1, 5)

    # Emit in FEWS-like long form: three lean-season observations per season-year,
    # measured in the year AFTER the season, plus projection rows to be filtered out.
    rows = []
    pop = dict(zip(names, RNG.integers(35_000, 320_000, len(names))))
    for _, r in p.iterrows():
        oy = int(r["season_year"]) + 1
        if oy > YEARS[1]:
            continue
        for mo in (1, 2, 3):
            rows.append({
                "country": "Zimbabwe",
                "fnid": f"ZW{names.index(r['district']):03d}",
                "admin2": r["district"],
                "reporting_date": f"{oy}-{mo:02d}-01",
                "scenario": "CS",
                "overall_phase": int(r["phase"]),
                "population": int(pop[r["district"]]),
            })
            rows.append({                      # projection row: must be dropped
                "country": "Zimbabwe",
                "fnid": f"ZW{names.index(r['district']):03d}",
                "admin2": r["district"],
                "reporting_date": f"{oy}-{mo:02d}-01",
                "scenario": "ML1",
                "overall_phase": int(min(5, r["phase"] + 1)),
                "population": int(pop[r["district"]]),
            })
    # A second country, to verify the country filter works
    for i in range(200):
        rows.append({"country": "Malawi", "fnid": f"MW{i:03d}", "admin2": f"MW_D{i%20:02d}",
                     "reporting_date": "2015-01-01", "scenario": "CS",
                     "overall_phase": 2, "population": 50_000})
    return pd.DataFrame(rows)


def main(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    dinfo = _district_frame()
    daily = _daily(dinfo)
    monthly_p = _monthly_precip(dinfo)

    mt = (daily.assign(ym=daily["date"].values.astype("datetime64[M]"))
               .groupby(["district", "ym"])[["tmax_c", "tmin_c"]].mean().reset_index()
               .rename(columns={"ym": "date"}))
    monthly = (monthly_p.merge(mt, on=["district", "date"], how="inner")
                        .merge(dinfo[["district", "lat", "lon"]], on="district", how="left"))
    # Alternate rainfall product: correlated with the primary but wet-biased and
    # noisier, mimicking how ERA5-Land precipitation behaves against CHIRPS here.
    monthly["precip_alt_mm"] = np.maximum(
        0.0, monthly["precip_mm"] * RNG.normal(1.18, 0.20, len(monthly))
        + RNG.normal(0, 6, len(monthly)))
    monthly = monthly[["district", "date", "precip_mm", "precip_alt_mm",
                       "tmax_c", "tmin_c", "lat", "lon"]]

    gs_daily = daily[daily["date"].dt.month.isin([10, 11, 12, 1, 2, 3])][
        ["district", "date", "tmax_c"]]

    # Compact season-level hot-day table, matching what the Earth Engine export
    # now produces. This is the file the pipeline prefers, because it is ~120 KB
    # rather than ~15 MB.
    gd = gs_daily.copy()
    gd["season_year"] = np.where(gd["date"].dt.month >= 10,
                                 gd["date"].dt.year + 1, gd["date"].dt.year)
    base = gd[gd["date"].dt.year.between(1991, 2020)]
    thr = {q: base.groupby("district")["tmax_c"].quantile(q / 100.0)
           for q in (85, 90, 95)}
    hot = (gd.groupby(["district", "season_year"], observed=True)
             .agg(n_days=("tmax_c", "size"), tmax_mean=("tmax_c", "mean"))
             .reset_index())
    for q in (85, 90, 95):
        gd[f"_h{q}"] = (gd["tmax_c"] > gd["district"].map(thr[q])).astype(float)
        hot[f"hot_days_p{q}"] = (gd.groupby(["district", "season_year"], observed=True)
                                   [f"_h{q}"].sum().reset_index(drop=True))

    hfid = _outcome(monthly_p, daily, dinfo)

    monthly.to_csv(outdir / "zw_monthly_climate.csv", index=False)
    gs_daily.to_csv(outdir / "zw_daily_tmax_gs.csv", index=False)
    hot.to_csv(outdir / "zw_season_hotdays.csv", index=False)
    hfid.to_csv(outdir / "FEWS_June_2026_Update_TrueBoundaries_08-13-26.csv", index=False)

    print(f"Synthetic inputs written to {outdir}")
    print(f"  monthly climate : {len(monthly):,} rows")
    print(f"  daily Tmax (GS) : {len(gs_daily):,} rows  "
          f"({(outdir / 'zw_daily_tmax_gs.csv').stat().st_size/1e6:.1f} MB, optional)")
    print(f"  season hot days : {len(hot):,} rows  "
          f"({(outdir / 'zw_season_hotdays.csv').stat().st_size/1e3:.0f} KB, preferred)")
    print(f"  HFID-like       : {len(hfid):,} rows")
    print(f"\nTrue parameters embedded: dry={TRUE_DRY}, heat={TRUE_HEAT}, "
          f"interaction={TRUE_INTERACTION}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/synthetic")
    a = ap.parse_args()
    main(Path(a.outdir))
