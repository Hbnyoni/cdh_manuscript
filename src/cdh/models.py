"""Estimation: nested two-way fixed effects models and out-of-sample comparison.

The paper's claim is comparative (does adding heat and its interaction with
moisture deficit improve on a rainfall-only model?), so the models are estimated
as a nested ladder and compared on both in-sample fit and leave-one-year-out
predictive error. Predictive error is the harder test and is reported first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from . import config as C

OUTCOME = "phase"

SPECS = {
    "M1_rain_only":  ["spi_dry_z"],
    "M2_spei_only":  ["dry_z"],
    "M3_additive":   ["dry_z", "heat_z"],
    "M4_compound":   ["dry_z", "heat_z", "compound_cont"],
    "M5_binary":     ["drought_only", "heat_only", "compound_bin"],
}

LABELS = {
    "spi_dry_z":    "Rainfall deficit (SPI-6, reversed, z)",
    "dry_z":        "Moisture deficit (SPEI-6, reversed, z)",
    "heat_z":       "Growing-season hot days (z)",
    "compound_cont": "Moisture deficit x hot days",
    "drought_only": "Drought, not hot",
    "heat_only":    "Hot, not drought",
    "compound_bin": "Compound drought-heat",
}


def _prep(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    d["district"] = d["district"].astype("category")
    d["year"] = d["season_year"].astype(int)
    return d.set_index(["district", "year"])


def fit_spec(panel: pd.DataFrame, regressors, outcome=OUTCOME,
             entity_fe=True, time_fe=True, cluster_entity=True):
    d = _prep(panel).dropna(subset=[outcome] + list(regressors))
    rhs = " + ".join(regressors)
    parts = [rhs]
    if entity_fe:
        parts.append("EntityEffects")
    if time_fe:
        parts.append("TimeEffects")
    formula = f"{outcome} ~ " + " + ".join(parts)
    mod = PanelOLS.from_formula(formula, d, drop_absorbed=True, check_rank=False)
    return mod.fit(cov_type="clustered", cluster_entity=cluster_entity)


def fit_ladder(panel: pd.DataFrame, outcome=OUTCOME, specs=None) -> dict:
    specs = specs or SPECS
    return {name: fit_spec(panel, regs, outcome=outcome)
            for name, regs in specs.items()}


def coef_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        for term in res.params.index:
            rows.append({
                "model": name,
                "term": LABELS.get(term, term),
                "coef": res.params[term],
                "se": res.std_errors[term],
                "t": res.tstats[term],
                "p": res.pvalues[term],
                "ci_lo": res.conf_int().loc[term].iloc[0],
                "ci_hi": res.conf_int().loc[term].iloc[1],
            })
    return pd.DataFrame(rows)


def fit_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        rows.append({
            "model": name,
            "n": int(res.nobs),
            # R2 of the two-way demeaned regression. linearmodels' rsquared_within
            # only removes entity means, so with TimeEffects it is not the fit of
            # this model and need not rise along the nested ladder.
            "within_r2": float(res.rsquared),
            "overall_r2": float(res.rsquared_overall),
            "loglik": float(res.loglik),
        })
    return pd.DataFrame(rows)


def leave_one_year_out(panel: pd.DataFrame, specs=None, outcome=OUTCOME) -> pd.DataFrame:
    """Hold out one season-year at a time, refit, predict.

    Leave-one-YEAR-out rather than random k-fold: the question is whether the
    model generalises to a season it has not seen, which is how it would be used.
    Random folds would leak same-year information across districts through the
    year fixed effect and flatter every specification equally.
    """
    specs = specs or SPECS
    years = sorted(panel["season_year"].unique())
    rows = []
    for name, regs in specs.items():
        errs, base_errs = [], []
        for y in years:
            tr = panel[panel["season_year"] != y]
            te = panel[panel["season_year"] == y]
            cols = [outcome] + list(regs)
            tr = tr.dropna(subset=cols)
            te = te.dropna(subset=cols)
            if len(te) < 5 or tr["season_year"].nunique() < 4:
                continue
            try:
                res = fit_spec(tr, regs, outcome=outcome, time_fe=False)
            except Exception:
                continue
            # district effects from training; year effect unavailable out of sample,
            # so prediction uses district intercept + slopes, which is the honest
            # forecasting position.
            fe = res.estimated_effects.groupby(level=0, observed=True).first().iloc[:, 0]
            te = te.copy()
            te["_fe"] = te["district"].map(fe)
            pred = te["_fe"].fillna(0.0) + res.params.get("Intercept", 0.0)
            for r in regs:
                pred = pred + res.params[r] * te[r].to_numpy()
            resid = te[outcome].to_numpy() - np.asarray(pred, float)
            errs.extend(resid[np.isfinite(resid)])
            base = te[outcome].to_numpy() - tr[outcome].mean()
            base_errs.extend(base[np.isfinite(base)])
        if not errs:
            continue
        errs = np.asarray(errs); base_errs = np.asarray(base_errs)
        rows.append({
            "model": name,
            "n_pred": len(errs),
            "rmse": float(np.sqrt(np.mean(errs ** 2))),
            "mae": float(np.mean(np.abs(errs))),
            "rmse_skill_vs_mean": float(
                1 - np.sqrt(np.mean(errs ** 2)) / np.sqrt(np.mean(base_errs ** 2))),
        })
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def marginal_effect_of_drought(res, heat_grid=None) -> pd.DataFrame:
    """dY/d(drought) at different heat levels, from the M4 interaction.

    This is the quantity the paper is actually about: whether a moisture deficit
    costs more when the season is also hot.
    """
    heat_grid = np.linspace(-2, 2, 41) if heat_grid is None else np.asarray(heat_grid)
    b_d = res.params["dry_z"]
    b_i = res.params.get("compound_cont", 0.0)
    V = res.cov
    v_dd = V.loc["dry_z", "dry_z"]
    v_ii = V.loc["compound_cont", "compound_cont"] if "compound_cont" in V.index else 0.0
    v_di = V.loc["dry_z", "compound_cont"] if "compound_cont" in V.index else 0.0
    me = b_d + b_i * heat_grid
    var = v_dd + (heat_grid ** 2) * v_ii + 2 * heat_grid * v_di
    se = np.sqrt(np.maximum(var, 0))
    return pd.DataFrame({
        "heat_z": heat_grid, "effect": me,
        "lo": me - 1.96 * se, "hi": me + 1.96 * se,
    })


def robustness(panel: pd.DataFrame, outcome=OUTCOME) -> pd.DataFrame:
    """Specifications a referee will ask for, run up front."""
    regs = SPECS["M4_compound"]
    out = []

    def _row(label, df, **kw):
        try:
            r = fit_spec(df, regs, outcome=outcome, **kw)
            out.append({
                "variant": label, "n": int(r.nobs),
                "dry_z": r.params["dry_z"],
                "heat_z": r.params["heat_z"],
                "interaction": r.params["compound_cont"],
                "p_interaction": r.pvalues["compound_cont"],
            })
        except Exception as e:
            out.append({"variant": label, "n": np.nan, "dry_z": np.nan,
                        "heat_z": np.nan, "interaction": np.nan,
                        "p_interaction": np.nan, "note": str(e)[:60]})

    _row("Main (district + year FE)", panel)
    _row("District FE only", panel, time_fe=False)
    _row("Excluding COVID years", panel[~panel["season_year"].isin(C.DROP_COVID_YEARS)])
    _row("Outcome = max phase", panel)  # placeholder replaced below
    out = [o for o in out if o["variant"] != "Outcome = max phase"]

    try:
        r = fit_spec(panel, regs, outcome="crisis")
        out.append({"variant": "Outcome = crisis indicator (LPM)", "n": int(r.nobs),
                    "dry_z": r.params["dry_z"], "heat_z": r.params["heat_z"],
                    "interaction": r.params["compound_cont"],
                    "p_interaction": r.pvalues["compound_cont"]})
    except Exception:
        pass

    return pd.DataFrame(out)
