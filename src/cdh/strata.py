"""Agro-ecological stratification for the heterogeneity analysis.

The substantive question is whether compound drought-heat effects concentrate in
Zimbabwe's marginal Natural Regions, where communal-area households cultivate
maize on land classified as suitable only for extensive livestock.

Two routes, because the authoritative boundary file is not openly downloadable:

1. TRUE Natural Regions. Supply data/raw/zw_natural_regions.csv with columns
   `district,natural_region` (values I, II, III, IV, V). Districts spanning more
   than one region should be assigned their MODAL region by cropped area, and how
   you did that belongs in the methods. This is the preferred route.

2. ARIDITY TERCILE fallback. Computed from each district's own CHIRPS growing-
   season rainfall climatology, which the pipeline already has. This needs no
   extra file and is a defensible stratifier in its own right, but it is NOT the
   Natural Region classification and must not be described as one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

NR_ORDER = ["I", "II", "III", "IV", "V"]


def assign_strata(panel: pd.DataFrame, verbose=True) -> tuple[pd.DataFrame, str]:
    """Attach a `stratum` column. Returns (panel, source_label)."""
    d = panel.copy()

    if C.NATURAL_REGION_CSV.exists():
        lut = pd.read_csv(C.NATURAL_REGION_CSV)
        need = {"district", "natural_region"}
        if not need.issubset(lut.columns):
            raise ValueError(
                f"{C.NATURAL_REGION_CSV.name} must have columns {sorted(need)}; "
                f"found {list(lut.columns)}")
        lut["natural_region"] = lut["natural_region"].astype(str).str.strip().str.upper()
        d["stratum"] = d["district"].map(dict(zip(lut["district"], lut["natural_region"])))
        missing = d["stratum"].isna().sum()
        if verbose:
            print(f"   Strata: Natural Regions from {C.NATURAL_REGION_CSV.name} "
                  f"({d['stratum'].notna().sum()} matched, {missing} unmatched)")
        if missing:
            unmatched = sorted(d.loc[d["stratum"].isna(), "district"].unique())[:8]
            print(f"   ! unmatched districts e.g. {unmatched}")
        d["stratum"] = pd.Categorical(
            d["stratum"], categories=[r for r in NR_ORDER if r in set(d["stratum"].dropna())],
            ordered=True)
        return d, "Natural Region"

    # Fallback: district long-run growing-season rainfall, split into terciles.
    clim = (d.groupby("district", observed=True)["season_rain_mm"].mean()
              .rename("clim_rain"))
    q = clim.quantile([1 / 3, 2 / 3]).to_numpy()
    lab = pd.cut(clim, [-np.inf, q[0], q[1], np.inf],
                 labels=["Driest third", "Middle third", "Wettest third"])
    d["stratum"] = pd.Categorical(
        d["district"].map(dict(zip(clim.index, lab))),
        categories=["Driest third", "Middle third", "Wettest third"], ordered=True)
    if verbose:
        print(f"   Strata: aridity terciles from CHIRPS climatology "
              f"(no {C.NATURAL_REGION_CSV.name} found). Cutpoints "
              f"{q[0]:.0f} and {q[1]:.0f} mm per season.")
        print("   NOTE: this is a rainfall proxy, not the Natural Region classification.")
    return d, "Aridity tercile"


def heterogeneity(panel: pd.DataFrame, outcome="phase", verbose=True) -> pd.DataFrame:
    """Estimate the compound specification separately within each stratum.

    Split-sample rather than a three-way interaction: with five strata a fully
    interacted model is hard to read and the district fixed effects already make
    each split an internally valid within estimator. The cost is that strata
    cannot be formally compared; a pooled interaction model is reported alongside
    for that purpose.
    """
    from . import models as M

    rows = []
    for s, g in panel.groupby("stratum", observed=True):
        n_d = g["district"].nunique()
        if len(g) < 60 or n_d < 5:
            rows.append({"stratum": str(s), "n": len(g), "n_districts": n_d,
                         "note": "too few observations"})
            continue
        try:
            r = M.fit_spec(g, M.SPECS["M4_compound"], outcome=outcome)
            rows.append({
                "stratum": str(s), "n": int(r.nobs), "n_districts": n_d,
                "dry_z": r.params["dry_z"],
                "heat_z": r.params["heat_z"],
                "interaction": r.params["compound_cont"],
                "se_interaction": r.std_errors["compound_cont"],
                "p_interaction": r.pvalues["compound_cont"],
                "within_r2": float(r.rsquared),  # two-way demeaned; see models.fit_table
            })
        except Exception as e:
            rows.append({"stratum": str(s), "n": len(g), "n_districts": n_d,
                         "note": str(e)[:70]})
    out = pd.DataFrame(rows)
    if verbose and "interaction" in out.columns:
        print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return out


def pooled_interaction(panel: pd.DataFrame, outcome="phase"):
    """Three-way interaction against the most-humid stratum as reference.

    This is the specification that formally tests whether the compound effect
    DIFFERS across strata, which the split-sample estimates cannot do.
    """
    from . import models as M

    d = panel.dropna(subset=["stratum"]).copy()
    cats = list(d["stratum"].cat.categories)
    if len(cats) < 2:
        return None
    ref = cats[-1]          # wettest / Natural Region V last in NR_ORDER is driest,
    if set(cats) & set(NR_ORDER):
        ref = cats[0]       # for Natural Regions use NR I (most humid) as reference
    regs = ["dry_z", "heat_z", "compound_cont"]
    for c in cats:
        if c == ref:
            continue
        key = "s_" + str(c).replace(" ", "_")
        d[key] = (d["stratum"] == c).astype(float)
        d[key + "_x_comp"] = d[key] * d["compound_cont"]
        d[key + "_x_dry"] = d[key] * d["dry_z"]
        regs += [key + "_x_dry", key + "_x_comp"]
    try:
        return M.fit_spec(d, regs, outcome=outcome), ref
    except Exception:
        return None
