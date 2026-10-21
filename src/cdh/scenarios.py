"""Climate-change scenario projection: delta method.

Applies the CMIP6 multi-model ensemble's projected CHANGE (not its absolute
values -- see gee/extract_cmip6_scenarios.py's docstring for why) on top of
this project's own observed CHIRPS/ERA5-Land historical climatology, then
propagates that shift through the SAME nonparametric standardisation and the
SAME fitted model coefficients the rest of the paper uses. Nothing here is a
fresh statistical fit: it is the existing fitted relationship, evaluated at
a shifted input.

Two deliberate simplifying assumptions, stated here once rather than
scattered through the code, and reported in the manuscript:

1. Tmin is assumed to shift by the same amount as Tmax (the diurnal range is
   unchanged). This lets the Hargreaves PET response to warming be computed
   in closed form: PET is proportional to (Tmean + 17.8) x sqrt(Trange), and
   if Trange is unchanged, PET_future / PET_hist = (Tmean_hist + dT + 17.8)
   / (Tmean_hist + 17.8). Tmean is approximated by each district's own
   historical growing-season mean Tmax (not the true Tmax/Tmin average,
   which this pipeline does not retain per district after PET is computed;
   the approximation changes the implied PET sensitivity by a few tenths of
   a percentage point per degree, not the qualitative result).
2. Population is held at its current (HFID-reported) level in every
   scenario: this is a climate-only projection, not a combined climate x
   demographic-growth projection. Stated explicitly wherever a population
   burden number is reported.

Moisture: rather than assume a parametric shape for how a climatological
shift maps to a district's dry_z, this interpolates the future water balance
against that district's OWN historical (water_balance, dry_z) pairs --
consistent with the rest of the paper's preference for nonparametric
standardisation (CLAUDE.md; config.STANDARDISE = "empirical").

Heat: rather than assume a Gaussian daily Tmax distribution, this reuses
indices.season_hot_days on the REAL historical daily Tmax records already in
data/raw/, with each district's threshold shifted by -delta_tmax -- exactly
equivalent to shifting the whole future daily distribution up by
delta_tmax and re-counting against the original threshold, using the
distribution's actual empirical shape rather than an assumed one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import indices as IX


def _empirical_map(x_hist: np.ndarray, z_hist: np.ndarray, x_future: float) -> float:
    """Where would x_future fall on this district's own historical (x, z)
    relationship? Linear interpolation within the historical range; linear
    extrapolation beyond it, from the boundary point to the nearest point
    with a DIFFERENT x value (flagged by the caller via the returned flag,
    not silently produced as if interpolated) -- a future climatology can
    genuinely fall outside 16-20 seasons of history.

    Skipping to the nearest DISTINCT x, rather than just the adjacent sorted
    point, matters because a short real record sometimes ties at its own
    extreme (two seasons landing on the same water-balance or hot-day count
    to the unit used). Differencing two tied x's divides by zero, which
    silently produces NaN; project()'s isfinite() check then drops that
    district-scenario row from the whole output with no error. That is
    exactly what happened to five district-scenario rows (Kadoma Urban under
    SSP5-8.5; Gokwe South Urban, Goromonzi, Ruwa and Zvimba under both
    scenarios) before this was traced and fixed.
    """
    order = np.argsort(x_hist)
    x_sorted, z_sorted = x_hist[order], z_hist[order]
    n = len(x_sorted)
    if x_future <= x_sorted[0]:
        k = 1
        while k < n and x_sorted[k] == x_sorted[0]:
            k += 1
        if k == n:
            return float(z_sorted[0]), True  # every historical value tied: no slope, hold flat
        slope = (z_sorted[k] - z_sorted[0]) / (x_sorted[k] - x_sorted[0])
        return z_sorted[0] + slope * (x_future - x_sorted[0]), True
    if x_future >= x_sorted[-1]:
        k = n - 2
        while k >= 0 and x_sorted[k] == x_sorted[-1]:
            k -= 1
        if k < 0:
            return float(z_sorted[-1]), True
        slope = (z_sorted[-1] - z_sorted[k]) / (x_sorted[-1] - x_sorted[k])
        return z_sorted[-1] + slope * (x_future - x_sorted[-1]), True
    return float(np.interp(x_future, x_sorted, z_sorted)), False


def future_dry_z(panel: pd.DataFrame, district: str, delta_rain_pct: float,
                 delta_tmax_c: float):
    g = panel[panel["district"] == district]
    if len(g) < 4:
        return np.nan, True
    tmean_hist = g["tmax_mean"].mean()
    pet_ratio = (tmean_hist + delta_tmax_c + 17.8) / (tmean_hist + 17.8)
    d_hist = (g["season_rain_mm"] - g["season_pet_mm"]).to_numpy()
    rain_future = g["season_rain_mm"] * (1 + delta_rain_pct)
    pet_future = g["season_pet_mm"] * pet_ratio
    d_future_mean = float((rain_future - pet_future).mean())
    return _empirical_map(d_hist, g["dry_z"].to_numpy(), d_future_mean)


def future_heat_z(panel: pd.DataFrame, daily: pd.DataFrame, thresholds: pd.Series,
                  district: str, delta_tmax_c: float):
    g = panel[panel["district"] == district]
    if len(g) < 4 or district not in thresholds.index:
        return np.nan, True
    shifted_thr = thresholds[[district]] - delta_tmax_c
    d = daily[daily["district"] == district]
    hot_future = IX.season_hot_days(d, shifted_thr)
    if hot_future.empty or hot_future["hot_days"].isna().all():
        return np.nan, True
    hd_future_mean = float(hot_future["hot_days"].mean())
    return _empirical_map(g["hot_days"].to_numpy(), g["heat_z"].to_numpy(), hd_future_mean)


def project(panel: pd.DataFrame, daily: pd.DataFrame, cmip6: pd.DataFrame,
           coefs: dict, verbose=True) -> pd.DataFrame:
    """Project future mean lean-season IPC phase under each CMIP6 scenario.

    `coefs`: {stratum_or_None: {"dry_z": b, "heat_z": b, "compound_cont": b}},
    keyed by the panel's `stratum` column; a None key is the pooled/fallback
    coefficient set, used for any district whose stratum has no separate
    fitted coefficients.
    """
    thresholds = IX.hot_day_threshold(daily)
    hist = cmip6[cmip6["period"] == "historical"].set_index("district")
    scenarios = sorted(s for s in cmip6["period"].unique() if s != "historical")

    rows = []
    for scen in scenarios:
        fut = cmip6[cmip6["period"] == scen].set_index("district")
        for district in panel["district"].unique():
            if district not in hist.index or district not in fut.index:
                continue
            delta_rain_pct = (fut.loc[district, "rain_mm"] - hist.loc[district, "rain_mm"]) \
                / hist.loc[district, "rain_mm"]
            delta_tmax_c = fut.loc[district, "tmax_c"] - hist.loc[district, "tmax_c"]

            dz, dz_extrap = future_dry_z(panel, district, delta_rain_pct, delta_tmax_c)
            hz, hz_extrap = future_heat_z(panel, daily, thresholds, district, delta_tmax_c)
            if not (np.isfinite(dz) and np.isfinite(hz)):
                continue

            g = panel[panel["district"] == district]
            stratum = g["stratum"].iloc[0] if "stratum" in g.columns and len(g) else None
            b = coefs.get(stratum, coefs.get(None))
            if b is None:
                continue
            delta_phase = (b["dry_z"] * dz + b["heat_z"] * hz
                          + b["compound_cont"] * dz * hz)
            hist_phase = g["phase"].mean()
            future_phase = float(np.clip(hist_phase + delta_phase, 1, 5))
            population = g["population"].mean() if "population" in g.columns else np.nan

            rows.append({
                "scenario": scen, "district": district, "stratum": stratum,
                "delta_rain_pct": delta_rain_pct, "delta_tmax_c": delta_tmax_c,
                "delta_dry_z": dz, "delta_heat_z": hz, "delta_phase": delta_phase,
                "historical_phase": hist_phase, "future_phase": future_phase,
                "population": population,
                "extrapolated": bool(dz_extrap or hz_extrap),
            })

    out = pd.DataFrame(rows)
    if verbose and not out.empty:
        n_extrap = int(out["extrapolated"].sum())
        print(f"   Scenario projection: {len(out)} district-scenario rows "
              f"({n_extrap} beyond the historical range, extrapolated linearly)")
    return out


ELEVATED_RISK_THRESHOLD = 0.5   # phase points; see population_burden docstring


def population_burden(proj: pd.DataFrame) -> pd.DataFrame:
    """Population-weighted burden by scenario.

    NOT "share of population reaching Crisis+ (phase >= 3)": historical_phase
    and future_phase are each a district's MULTI-SEASON MEAN, which a badly
    affected district can still keep well under 3 even when individual
    seasons within it cross the crisis threshold -- averaging over ~16
    seasons smooths out exactly the acute seasons "Crisis+" is meant to
    flag. Comparing that mean against an absolute level of 3 produced a
    share of 0.0 in every scenario tried and would misreport the result as
    "no burden" rather than as the wrong statistic for what was modelled.

    Reported instead: the population-weighted mean projected phase change
    (the quantity this module actually estimates), and separately, the
    population living in districts crossing a stated COMPARATIVE threshold
    (delta_phase >= 0.5, half an IPC phase point) -- "elevated future risk"
    relative to that district's own historical level, not an absolute
    Crisis+ classification, which the multi-season means here cannot support.
    """
    rows = []
    for scen, g in proj.groupby("scenario"):
        total_pop = g["population"].sum()
        w = g["population"]
        mean_delta_phase_pw = (g["delta_phase"] * w).sum() / w.sum() if w.sum() else np.nan
        elevated = g.loc[g["delta_phase"] >= ELEVATED_RISK_THRESHOLD, "population"].sum()
        rows.append({
            "scenario": scen,
            "total_population": total_pop,
            "mean_delta_phase": g["delta_phase"].mean(),
            "mean_delta_phase_population_weighted": mean_delta_phase_pw,
            "elevated_risk_threshold": ELEVATED_RISK_THRESHOLD,
            "elevated_risk_population": elevated,
            "elevated_risk_share": elevated / total_pop if total_pop else np.nan,
        })
    return pd.DataFrame(rows)
