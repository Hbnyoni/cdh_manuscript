"""Drought, heat and compound-event indices.

Implements SPI, SPEI (Hargreaves PET), growing-season hot-day counts and two
operationalisations of a compound drought-heat event.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


# ---------------------------------------------------------------- PET & balance
def extraterrestrial_radiation(lat_deg: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """Ra in MJ m-2 day-1 (FAO-56 eq. 21)."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    doy = np.asarray(doy, dtype=float)
    dr = 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    x = np.clip(-np.tan(lat) * np.tan(decl), -1.0, 1.0)
    ws = np.arccos(x)
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.sin(ws)
    )


def hargreaves_pet_monthly(tmax_c, tmin_c, lat_deg, doy_mid, days_in_month):
    """Monthly PET (mm) by Hargreaves-Samani.

    Chosen over Penman-Monteith deliberately: it needs only temperature, so it can
    be computed from the same ERA5-Land extraction as the heat variables and does
    not introduce wind/radiation products with different error structures.
    """
    tmax_c = np.asarray(tmax_c, float)
    tmin_c = np.asarray(tmin_c, float)
    ra = extraterrestrial_radiation(lat_deg, doy_mid)          # MJ m-2 d-1
    ra_mm = ra / 2.45                                          # -> mm d-1 equivalent
    tmean = (tmax_c + tmin_c) / 2.0
    trange = np.maximum(tmax_c - tmin_c, 0.0)
    pet_d = 0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(trange)
    pet_d = np.maximum(pet_d, 0.0)
    return pet_d * np.asarray(days_in_month, float)


# ---------------------------------------------------------------- standardising
def _gringorten_z(values: np.ndarray) -> np.ndarray:
    """Nonparametric standardisation via Gringorten plotting position.

    Robust for ~45-year records where fitting a 3-parameter log-logistic is
    unstable, and recommended where the accumulation window contains zeros.
    """
    v = np.asarray(values, float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    n = int(ok.sum())
    if n < 10:
        return out
    ranks = stats.rankdata(v[ok], method="average")
    p = (ranks - 0.44) / (n + 0.12)
    out[ok] = stats.norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))
    return out


def _loglogistic_z(values: np.ndarray) -> np.ndarray:
    """Classical parametric SPEI: 3-parameter log-logistic, then inverse normal."""
    v = np.asarray(values, float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.sum() < 15:
        return out
    x = v[ok]
    try:
        params = stats.fisk.fit(x - x.min() + 1e-3)
        p = stats.fisk.cdf(x - x.min() + 1e-3, *params)
    except Exception:
        return _gringorten_z(v)
    out[ok] = stats.norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))
    return out


def _standardise(values, method=None):
    method = method or C.STANDARDISE
    return _loglogistic_z(values) if method == "loglogistic" else _gringorten_z(values)


def standardised_index(df: pd.DataFrame, value_col: str, scale: int,
                       group_col: str = "district", out_col: str = "z") -> pd.DataFrame:
    """Accumulate `value_col` over `scale` months, then standardise within
    (district, calendar month) so the index is anomaly-relative to that unit's own
    climatology for that time of year. This is what makes SPI/SPEI comparable
    across Zimbabwe's very different Natural Regions.
    """
    d = df.sort_values([group_col, "date"]).copy()
    d["_acc"] = (
        d.groupby(group_col, observed=True)[value_col]
         .transform(lambda s: s.rolling(scale, min_periods=scale).sum())
    )
    d["_cmonth"] = d["date"].dt.month
    d[out_col] = np.nan
    for (_g, _m), idx in d.groupby([group_col, "_cmonth"], observed=True).groups.items():
        d.loc[idx, out_col] = _standardise(d.loc[idx, "_acc"].to_numpy())
    return d.drop(columns=["_acc", "_cmonth"])


# ---------------------------------------------------------------- heat
def hot_day_threshold(daily: pd.DataFrame, pctl: int = None,
                      baseline: tuple[int, int] = None) -> pd.Series:
    """District-specific hot-day threshold: the `pctl` percentile of growing-season
    daily Tmax over the baseline period. District-specific rather than a fixed
    absolute cut because a 34 C day means something different in Natural Region I
    than in the Zambezi valley.
    """
    pctl = pctl or C.HOT_PCTL
    lo, hi = baseline or C.CLIM_BASELINE
    base = daily[(daily["date"].dt.year >= lo) & (daily["date"].dt.year <= hi)]
    base = base[base["date"].dt.month.isin(C.GROWING_SEASON_MONTHS)]
    return base.groupby("district", observed=True)["tmax_c"].quantile(pctl / 100.0)


def season_hot_days(daily: pd.DataFrame, thresholds: pd.Series) -> pd.DataFrame:
    """Count growing-season days above each district's threshold, by season-year.

    Season-year Y spans Oct(Y-1) to Mar(Y), so Oct-Dec observations are assigned
    to the FOLLOWING calendar year.
    """
    d = daily[daily["date"].dt.month.isin(C.GROWING_SEASON_MONTHS)].copy()
    d["season_year"] = np.where(d["date"].dt.month >= 10,
                                d["date"].dt.year + 1, d["date"].dt.year)
    d["thr"] = d["district"].map(thresholds)
    d["is_hot"] = (d["tmax_c"] > d["thr"]).astype(float)
    out = (d.groupby(["district", "season_year"], observed=True)
             .agg(hot_days=("is_hot", "sum"),
                  n_days=("is_hot", "size"),
                  tmax_mean=("tmax_c", "mean"))
             .reset_index())
    # Guard against partial seasons (e.g. the final, incomplete year).
    out["hot_days"] = np.where(out["n_days"] >= 150, out["hot_days"], np.nan)
    return out


# ---------------------------------------------------------------- compound
def add_compound_terms(panel: pd.DataFrame) -> pd.DataFrame:
    """Two operationalisations, reported side by side because the compound-event
    literature has not settled on one and reviewers will ask about the other.

    A. Binary co-occurrence: SPEI-6 below DROUGHT_THRESHOLD *and* hot-day count
       above the district's own HOT_SEASON_PCTL.
    B. Continuous interaction: z(-SPEI) x z(hot days), which keeps the full
       gradient rather than discarding it at a cut point.
    """
    d = panel.copy()

    # within-district standardisation, so effects are "relative to this district's
    # own normal" and are not driven by cross-sectional climate differences that
    # the district fixed effect already absorbs.
    g = d.groupby("district", observed=True)
    d["dry_z"] = g["spei6"].transform(lambda s: -(s - s.mean()) / s.std(ddof=0))
    d["heat_z"] = g["hot_days"].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    d["spi_dry_z"] = g["spi6"].transform(lambda s: -(s - s.mean()) / s.std(ddof=0))

    # A. binary
    hot_cut = g["hot_days"].transform(lambda s: s.quantile(C.HOT_SEASON_PCTL / 100.0))
    d["is_drought"] = (d["spei6"] < C.DROUGHT_THRESHOLD).astype(float)
    d["is_hot_season"] = (d["hot_days"] > hot_cut).astype(float)
    d["compound_bin"] = d["is_drought"] * d["is_hot_season"]
    d["drought_only"] = d["is_drought"] * (1 - d["is_hot_season"])
    d["heat_only"] = (1 - d["is_drought"]) * d["is_hot_season"]

    # B. continuous
    d["compound_cont"] = d["dry_z"] * d["heat_z"]
    return d
