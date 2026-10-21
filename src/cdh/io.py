"""Loaders. The HFID loader is deliberately schema-tolerant: the FEWS NET tabular
export has renamed columns between releases and this pipeline was written before
the file could be inspected. It resolves each field against a candidate list and
prints what it chose, so a wrong guess is visible rather than silent.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from . import config as C


def _resolve(columns, candidates, field):
    lower = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # fall back to substring match
    for cand in candidates:
        for lc, orig in lower.items():
            if cand.lower() in lc:
                return orig
    raise KeyError(
        f"Could not find a column for '{field}'. Tried {candidates}.\n"
        f"Columns present: {list(columns)[:40]}\n"
        f"Fix by editing HFID_COLUMNS['{field}'] in src/cdh/config.py."
    )


def _resolve_district(unit_name: pd.Series) -> pd.Series:
    """District name from the HFID unit_name field.

    Some HFID releases report one row per admin2 district (unit_name IS the
    district, no commas). The 2026 release reports one row per FEWS NET
    livelihood zone within a district: unit_name is
    "<livelihood zone>, <district>, <province>, <country>". In that case the
    district is the third-from-last comma-separated token. Either way, apply
    HFID_DISTRICT_NAME_FIXES for the handful of Rural/Urban naming mismatches
    against the geoBoundaries admin2 layer; a fix of None (e.g. "parks", which
    is not an administrative district) drops the row.
    """
    def _token(v):
        s = str(v).strip()
        parts = [p.strip() for p in s.split(",")]
        d = parts[-3] if len(parts) >= 3 else s
        return C.HFID_DISTRICT_NAME_FIXES.get(d, d)
    return unit_name.map(_token)


def load_hfid(path=None, country=C.COUNTRY, verbose=True) -> pd.DataFrame:
    """Return tidy observed (not projected) IPC-compatible phases for one country.

    Output columns: unit_id, district, date, phase, population
    """
    path = path or C.HFID_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run ./fetch_data.sh first (see README)."
        )

    head = pd.read_csv(path, nrows=5)
    cols = {}
    for k, v in C.HFID_COLUMNS.items():
        try:
            cols[k] = _resolve(head.columns, v, k)
        except KeyError:
            if k in C.HFID_OPTIONAL_FIELDS:
                continue
            raise
    if verbose:
        print("HFID column mapping resolved as:")
        for k, v in cols.items():
            print(f"   {k:11s} -> {v}")

    usecols = list(dict.fromkeys(cols.values()))
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.rename(columns={v: k for k, v in cols.items()})

    df = df[df["country"].astype(str).str.strip().str.lower() == country.lower()].copy()
    if df.empty:
        raise ValueError(f"No rows for country={country!r}. Check spelling in the file.")

    # Keep only observed current-situation rows. Projections (ML1/ML2) are
    # forecasts and would make the outcome partly a function of the forecaster's
    # own view of the rainfall season, which is exactly the circularity to avoid.
    if "scenario" in df.columns:
        s = df["scenario"].astype(str).str.strip().str.lower()
        keep = s.isin([k.lower() for k in C.HFID_SCENARIO_KEEP])
        if keep.any():
            df = df[keep].copy()
        elif verbose:
            print(f"   ! no scenario value matched {C.HFID_SCENARIO_KEEP}; "
                  f"found {sorted(s.unique())[:8]}. Keeping all rows.")

    dt = pd.to_datetime(df["date"], errors="coerce")
    if dt.isna().mean() > 0.5:
        # "YYYY_MM" string (the 2026 HFID release's year_month column), not an
        # ISO date pandas recognises unprompted.
        dt = pd.to_datetime(df["date"].astype(str) + "_01", format="%Y_%m_%d",
                            errors="coerce")
    df["date"] = dt
    df["phase"] = pd.to_numeric(df["phase"], errors="coerce")
    df["population"] = pd.to_numeric(df.get("population"), errors="coerce")
    df["district"] = _resolve_district(df["unit_name"])

    df = df.dropna(subset=["date", "phase", "district"])
    df = df[df["phase"].between(1, 5)]

    out = df[["unit_id", "district", "date", "phase", "population"]].copy()
    if verbose:
        print(f"   {len(out):,} observed rows | {out['district'].nunique()} units | "
              f"{out['date'].min():%Y-%m} to {out['date'].max():%Y-%m}")
    return out


def load_monthly_climate(path=None, verbose=True) -> pd.DataFrame:
    """District x month climate from the Earth Engine export.

    Expected columns: district, date (YYYY-MM-01), precip_mm, tmax_c, tmin_c, lat
    """
    path = path or C.CLIM_MONTHLY_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run gee/extract_zimbabwe_climate.js in the Earth "
            f"Engine Code Editor and place the exported CSV here."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    need = {"district", "date", "precip_mm", "tmax_c", "tmin_c", "lat"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    if "lon" not in df.columns:
        print("   ! no 'lon' column: Conley spatial standard errors will be skipped. "
              "Re-run the Earth Engine export to add it.")
    if C.PRECIP_ALT_COL not in df.columns:
        print(f"   ! no '{C.PRECIP_ALT_COL}' column: the rainfall-product "
              f"sensitivity analysis will be skipped.")
    if verbose:
        print(f"Monthly climate: {len(df):,} rows | {df['district'].nunique()} districts | "
              f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}")
    return df.sort_values(["district", "date"]).reset_index(drop=True)


def load_daily_tmax(path=None, verbose=True) -> pd.DataFrame:
    """District x day growing-season Tmax from the Earth Engine export.

    Expected columns: district, date, tmax_c
    """
    path = path or C.CLIM_DAILY_TMAX_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See gee/ and the README.")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if verbose:
        print(f"Daily Tmax: {len(df):,} rows | {df['district'].nunique()} districts | "
              f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    return df.sort_values(["district", "date"]).reset_index(drop=True)


def load_season_hotdays(path=None, verbose=True) -> pd.DataFrame:
    """Season-level hot-day counts precomputed in Earth Engine.

    Preferred over the daily Tmax table because it is ~120 KB rather than ~15 MB,
    which is the difference between data that can be attached to a message and
    data that cannot. Carries counts at three percentiles so the threshold can
    still be varied here.
    """
    path = path or C.CLIM_HOTDAYS_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path)
    need = {"district", "season_year", C.HOT_PCTL_COL}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    df = df.rename(columns={C.HOT_PCTL_COL: "hot_days"})
    # Guard against partial seasons, the same rule the daily path applies
    if "n_days" in df.columns:
        df.loc[df["n_days"] < 150, "hot_days"] = np.nan
    if verbose:
        alts = [c for c in df.columns if c.startswith("hot_days_p")]
        print(f"Season hot days: {len(df):,} rows | {df['district'].nunique()} districts "
              f"| using {C.HOT_PCTL_COL}"
              f"{f' (also available: {alts})' if alts else ''}")
    return df


def harmonise_names(a: pd.Series, b: pd.Series, verbose=True):
    """Report unit-name mismatches between the outcome and climate tables.

    The FEWS NET units are the authoritative geometry because the outcome is
    measured on them; the Earth Engine extraction should be run against the SAME
    boundary asset. If you did that, this should report zero mismatches. If it
    reports many, you extracted against GAUL or GADM instead and the join is
    not safe.
    """
    sa, sb = set(a.unique()), set(b.unique())
    only_a, only_b = sorted(sa - sb), sorted(sb - sa)
    if verbose:
        print(f"Name check: {len(sa & sb)} matched | {len(only_a)} outcome-only | "
              f"{len(only_b)} climate-only")
        if only_a:
            print(f"   outcome-only e.g. {only_a[:8]}")
        if only_b:
            print(f"   climate-only e.g. {only_b[:8]}")
    return only_a, only_b
