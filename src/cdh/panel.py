"""Assemble the district x season-year analysis panel."""
from __future__ import annotations

import calendar
import numpy as np
import pandas as pd

from . import config as C
from . import indices as IX


def build_climate_panel(monthly: pd.DataFrame, daily: pd.DataFrame = None,
                        precip_col: str = None,
                        hotdays: pd.DataFrame = None) -> pd.DataFrame:
    """Monthly water balance -> SPEI-6/SPI-6 at season end, plus hot-day counts."""
    precip_col = precip_col or C.PRECIP_COL
    m = monthly.copy()
    if precip_col != "precip_mm":
        m = m.drop(columns=["precip_mm"]).rename(columns={precip_col: "precip_mm"})
    m["doy_mid"] = m["date"].dt.dayofyear + 14
    m["days"] = m["date"].dt.days_in_month
    m["pet_mm"] = IX.hargreaves_pet_monthly(
        m["tmax_c"], m["tmin_c"], m["lat"], m["doy_mid"], m["days"]
    )
    m["wb_mm"] = m["precip_mm"] - m["pet_mm"]

    m = IX.standardised_index(m, "wb_mm", C.SPEI_SCALE, out_col="spei6")
    m = IX.standardised_index(m, "precip_mm", C.SPI_SCALE, out_col="spi6")

    # Take the index value at season end (March), label it with that season-year.
    season = m[m["date"].dt.month == C.SEASON_END_MONTH].copy()
    season["season_year"] = season["date"].dt.year
    season = season[["district", "season_year", "spei6", "spi6"]]

    # Season rainfall total, kept as an interpretable descriptive covariate.
    gs = m[m["date"].dt.month.isin(C.GROWING_SEASON_MONTHS)].copy()
    gs["season_year"] = np.where(gs["date"].dt.month >= 10,
                                 gs["date"].dt.year + 1, gs["date"].dt.year)
    rain = (gs.groupby(["district", "season_year"], observed=True)
              .agg(season_rain_mm=("precip_mm", "sum"),
                   season_pet_mm=("pet_mm", "sum"),
                   n_months=("precip_mm", "size"))
              .reset_index())
    rain = rain[rain["n_months"] == len(C.GROWING_SEASON_MONTHS)].drop(columns="n_months")

    if hotdays is not None:
        hot = hotdays[["district", "season_year", "hot_days"]].copy()
        if "tmax_mean" in hotdays.columns:
            hot["tmax_mean"] = hotdays["tmax_mean"]
    elif daily is not None:
        thr = IX.hot_day_threshold(daily)
        hot = IX.season_hot_days(daily, thr)
    else:
        raise ValueError("Provide either `daily` Tmax or a precomputed `hotdays` table.")

    geo = m.groupby("district", observed=True)[["lat", "lon"]].first().reset_index() \
        if "lon" in m.columns else m.groupby("district", observed=True)[["lat"]].first().reset_index()
    clim = (season.merge(rain, on=["district", "season_year"], how="inner")
                  .merge(hot, on=["district", "season_year"], how="inner")
                  .merge(geo, on="district", how="left"))
    return clim


def build_outcome_panel(hfid: pd.DataFrame) -> pd.DataFrame:
    """Lean-season food insecurity outcome, by district and season-year.

    The outcome for season-year Y is measured in the lean season of Y+1 (Jan-Mar),
    which is the period the harvest of year Y must carry households through. This
    ordering is what makes the exposure temporally prior to the outcome.

    HFID reporting units can be finer than a district (the 2026 release reports
    FEWS NET livelihood zones within a district). Those are first collapsed to
    one row per unit per season (mean over the lean-season months), then to one
    row per district per season by a population-weighted mean, because the
    two-way fixed-effects estimator needs a unique (district, season_year) index.
    For a release where unit_id already IS the district (one row per district),
    this second step is a no-op.
    """
    d = hfid.copy()
    d = d[d["date"].dt.month.isin(C.LEAN_SEASON_MONTHS)]
    d["season_year"] = d["date"].dt.year - 1          # outcome year Y+1 -> season Y

    unit = (d.groupby(["unit_id", "district", "season_year"], observed=True)
              .agg(phase=("phase", "mean"),
                   population=("population", "mean"))
              .reset_index())

    w = unit["population"].fillna(0)
    unit["_wphase"] = unit["phase"] * w
    unit["_w"] = w

    def _collapse(g):
        wsum = g["_w"].sum()
        phase = g["_wphase"].sum() / wsum if wsum > 0 else g["phase"].mean()
        return pd.Series({
            "phase": phase,
            "phase_max": g["phase"].max(),
            "population": g["population"].sum(min_count=1),
            "n_obs": len(g),
        })

    out = (unit.groupby(["district", "season_year"], observed=True)
               .apply(_collapse, include_groups=False)
               .reset_index())
    out["crisis"] = (out["phase_max"] >= 3).astype(float)
    return out


def assemble(hfid: pd.DataFrame, monthly: pd.DataFrame, daily: pd.DataFrame = None,
             precip_col: str = None, hotdays: pd.DataFrame = None,
             verbose=True) -> pd.DataFrame:
    clim = build_climate_panel(monthly, daily, precip_col=precip_col, hotdays=hotdays)
    out = build_outcome_panel(hfid)

    panel = out.merge(clim, on=["district", "season_year"], how="inner")
    panel = panel.dropna(subset=["phase", "spei6", "spi6", "hot_days"])

    # Drop thin districts: a within-district estimator needs variation over time.
    counts = panel.groupby("district", observed=True)["season_year"].nunique()
    keep = counts[counts >= C.MIN_YEARS_PER_DISTRICT].index
    dropped = sorted(set(counts.index) - set(keep))
    panel = panel[panel["district"].isin(keep)].copy()

    panel = IX.add_compound_terms(panel)
    panel = panel.sort_values(["district", "season_year"]).reset_index(drop=True)

    if verbose:
        print(f"\nPanel: {len(panel):,} district-seasons | "
              f"{panel['district'].nunique()} districts | "
              f"{panel['season_year'].min()}-{panel['season_year'].max()}")
        if dropped:
            print(f"   dropped {len(dropped)} districts with < {C.MIN_YEARS_PER_DISTRICT} "
                  f"seasons: {dropped[:6]}{' ...' if len(dropped) > 6 else ''}")
        print(f"   mean phase {panel['phase'].mean():.2f} | "
              f"crisis-year share {panel['crisis'].mean():.1%} | "
              f"compound (binary) share {panel['compound_bin'].mean():.1%}")
    return panel
