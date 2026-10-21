"""End-to-end run: inputs -> panel -> models -> tables and figures.

  python run_pipeline.py                 # real data in data/raw/
  python run_pipeline.py --synthetic     # synthetic data in data/synthetic/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cdh import config as C          # noqa: E402
from cdh import io, panel, models, figures, strata, spatial, maps, scenarios   # noqa: E402


def main(synthetic: bool = False):
    src = C.ROOT / "data" / ("synthetic" if synthetic else "raw")
    tag = "SYNTHETIC" if synthetic else "REAL"
    print("=" * 72)
    print(f"Compound drought-heat and food insecurity in Zimbabwe  [{tag} DATA]")
    print("=" * 72)

    hfid = io.load_hfid(src / C.HFID_CSV.name)
    monthly = io.load_monthly_climate(src / C.CLIM_MONTHLY_CSV.name)
    hot_path = src / C.CLIM_HOTDAYS_CSV.name
    daily_path = src / C.CLIM_DAILY_TMAX_CSV.name
    daily = hotdays = None
    if hot_path.exists():
        hotdays = io.load_season_hotdays(hot_path)
    elif daily_path.exists():
        daily = io.load_daily_tmax(daily_path)
    else:
        raise FileNotFoundError(
            f"Need either {hot_path.name} (compact, preferred) or "
            f"{daily_path.name} (large). See gee/ and the README.")
    io.harmonise_names(hfid["district"], monthly["district"])

    print("\n--- Building panel " + "-" * 52)
    pan = panel.assemble(hfid, monthly, daily, hotdays=hotdays)
    pan.to_csv(C.PROC / f"panel_{tag.lower()}.csv", index=False)

    print("\n--- Estimating " + "-" * 56)
    res = models.fit_ladder(pan)
    coefs = models.coef_table(res)
    fits = models.fit_table(res)
    cv = models.leave_one_year_out(pan)
    rob = models.robustness(pan)

    for name, r in res.items():
        print(f"\n{name}  (n={int(r.nobs)}, within R2={r.rsquared_within:.3f})")
        for t in r.params.index:
            star = "***" if r.pvalues[t] < .01 else "**" if r.pvalues[t] < .05 \
                else "*" if r.pvalues[t] < .10 else ""
            print(f"   {models.LABELS.get(t, t):42s} "
                  f"{r.params[t]:+.4f} ({r.std_errors[t]:.4f}){star}")

    print("\n--- Leave-one-year-out prediction " + "-" * 37)
    print(cv.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- Robustness (M4 interaction) " + "-" * 39)
    print(rob.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    me = models.marginal_effect_of_drought(res["M4_compound"])

    # ---- Heterogeneity by agro-ecological stratum ---------------------------
    print("\n--- Heterogeneity " + "-" * 53)
    pan, strat_label = strata.assign_strata(pan)
    het = strata.heterogeneity(pan)

    # Marginal effect within whichever stratum carries the largest, significant
    # compound interaction, so fig_marginal_effect can show that estimate next
    # to the pooled one rather than only the pooled (diluted) curve. See
    # CLAUDE.md's requirement that a null/mixed result be reported honestly:
    # the pooled effect alone would understate what the stratified test shows.
    me_strat, strat_name = None, None
    if not het.empty and "interaction" in het.columns:
        h_nz = het.dropna(subset=["interaction"])
        if len(h_nz):
            cand = h_nz.sort_values("interaction", ascending=False).iloc[0]
            if cand["p_interaction"] < 0.05:
                strat_name = str(cand["stratum"])
                sub = pan[pan["stratum"].astype(str) == strat_name]
                try:
                    res_strat = models.fit_spec(sub, models.SPECS["M4_compound"])
                    me_strat = models.marginal_effect_of_drought(res_strat)
                except Exception as e:
                    print(f"   ! could not fit the stratum-specific marginal "
                          f"effect ({strat_name}): {e}")

    pooled = strata.pooled_interaction(pan)
    if pooled:
        pres, ref = pooled
        print(f"\n   Pooled three-way interaction (reference stratum: {ref})")
        for t in pres.params.index:
            if "_x_comp" in t:
                print(f"     {t:34s} {pres.params[t]:+.4f} "
                      f"({pres.std_errors[t]:.4f})  p={pres.pvalues[t]:.3f}")
        # Saved, not only printed: the manuscript cites these numbers.
        comp = [t for t in pres.params.index if t.endswith("_x_comp")]
        ptab = pd.DataFrame({"term": comp, "coef": pres.params[comp].to_numpy(),
                             "se": pres.std_errors[comp].to_numpy(),
                             "p": pres.pvalues[comp].to_numpy(), "reference": ref})
        if len(comp) > 1:
            w = pres.wald_test(formula=", ".join(f"{t} = 0" for t in comp))
            print(f"     joint Wald chi2({w.df}) = {w.stat:.2f}, p = {w.pval:.3f}")
            ptab = pd.concat([ptab, pd.DataFrame([{
                "term": "joint_wald_chi2", "coef": float(w.stat), "se": float(w.df),
                "p": float(w.pval), "reference": ref}])], ignore_index=True)
        ptab.to_csv(C.TAB / f"pooled_interaction_{tag.lower()}.csv", index=False)

    # ---- Conley spatial HAC standard errors ---------------------------------
    conley = pd.DataFrame()
    if "lon" in pan.columns and pan["lon"].notna().any():
        print("\n--- Conley spatial standard errors " + "-" * 36)
        b = res["M4_compound"].params
        ctab = spatial.conley_se(
            pan, models.SPECS["M4_compound"],
            dist_km=C.CONLEY_DIST_KM, time_lag=C.CONLEY_TIME_LAG,
            validate_against={k: b[k] for k in models.SPECS["M4_compound"]})
        cl = res["M4_compound"].std_errors
        print(f"   cutoff {C.CONLEY_DIST_KM:.0f} km, serial lag {C.CONLEY_TIME_LAG} years")
        for _, r in ctab.iterrows():
            print(f"     {models.LABELS.get(r['term'], r['term']):42s} "
                  f"{r['coef']:+.4f}  clustered SE {cl[r['term']]:.4f} -> "
                  f"Conley SE {r['se_conley']:.4f}  (p={r['p']:.3f})")
        conley = spatial.conley_sensitivity(
            pan, models.SPECS["M4_compound"], cutoffs=C.CONLEY_CUTOFFS,
            time_lag=C.CONLEY_TIME_LAG)
        ctab.to_csv(C.TAB / f"conley_{tag.lower()}.csv", index=False)
        conley.to_csv(C.TAB / f"conley_sensitivity_{tag.lower()}.csv", index=False)
    else:
        print("\n   Conley SEs skipped: no district longitudes in the climate export.")

    # ---- Rainfall product sensitivity ---------------------------------------
    alt = pd.DataFrame()
    if C.PRECIP_ALT_COL in monthly.columns:
        print("\n--- Rainfall product sensitivity " + "-" * 38)
        pan_alt = panel.assemble(hfid, monthly, daily, hotdays=hotdays,
                                 precip_col=C.PRECIP_ALT_COL, verbose=False)
        res_alt = models.fit_ladder(pan_alt)
        rows = []
        for name in ["M1_rain_only", "M4_compound"]:
            for prod, rr in (("CHIRPS v3", res[name]), ("alternate", res_alt[name])):
                for t in rr.params.index:
                    rows.append({"model": name, "product": prod, "term": t,
                                 "coef": rr.params[t], "se": rr.std_errors[t],
                                 "p": rr.pvalues[t]})
        alt = pd.DataFrame(rows)
        piv = (alt[alt.term == "compound_cont"]
               .set_index("product")[["coef", "se", "p"]])
        print(piv.to_string(float_format=lambda x: f"{x:.4f}"))
        alt.to_csv(C.TAB / f"rainfall_sensitivity_{tag.lower()}.csv", index=False)

    het.to_csv(C.TAB / f"heterogeneity_{tag.lower()}.csv", index=False)

    coefs.to_csv(C.TAB / f"coefficients_{tag.lower()}.csv", index=False)
    fits.to_csv(C.TAB / f"model_fit_{tag.lower()}.csv", index=False)
    cv.to_csv(C.TAB / f"cv_{tag.lower()}.csv", index=False)
    rob.to_csv(C.TAB / f"robustness_{tag.lower()}.csv", index=False)
    me.to_csv(C.TAB / f"marginal_effects_{tag.lower()}.csv", index=False)

    # ---- Climate-change scenario projection ----------------------------------
    proj = burden = pd.DataFrame()
    cmip6_path = C.PROC / "cmip6_district_climatology.csv"
    if cmip6_path.exists() and daily is not None:
        print("\n--- Climate-change scenario projection " + "-" * 31)
        cmip6 = pd.read_csv(cmip6_path)
        strat_coefs = {None: {
            "dry_z": res["M4_compound"].params["dry_z"],
            "heat_z": res["M4_compound"].params["heat_z"],
            "compound_cont": res["M4_compound"].params["compound_cont"]}}
        if not het.empty and "interaction" in het.columns:
            for _, r in het.dropna(subset=["interaction"]).iterrows():
                strat_coefs[r["stratum"]] = {
                    "dry_z": r["dry_z"], "heat_z": r["heat_z"],
                    "compound_cont": r["interaction"]}
        proj = scenarios.project(pan, daily, cmip6, strat_coefs)
        if not proj.empty:
            burden = scenarios.population_burden(proj)
            print(burden.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
            proj.to_csv(C.TAB / f"scenario_projection_{tag.lower()}.csv", index=False)
            burden.to_csv(C.TAB / f"scenario_burden_{tag.lower()}.csv", index=False)
    else:
        print("\n   Scenario projection skipped: run "
              "gee/extract_cmip6_scenarios.py first (needs "
              f"{cmip6_path.name} and daily Tmax records).")

    print("\n--- Figures " + "-" * 59)
    figures.make_all(pan, res, me, cv, tag=tag.lower(), synthetic=synthetic,
                     me_strat=me_strat, strat_name=strat_name)
    if not proj.empty:
        figures.fig_scenario_burden(burden, tag=tag.lower(), synthetic=synthetic)
    figures.fig_heterogeneity(het, strat_label, tag.lower(), synthetic)
    if not conley.empty:
        figures.fig_conley(conley, tag=tag.lower(), synthetic=synthetic)
    figures.fig_binned_interaction(pan, tag=tag.lower(), synthetic=synthetic)
    placebo_path = C.TAB / f"placebo_permutation_{tag.lower()}.csv"
    placebo_summary_path = C.TAB / f"placebo_summary_{tag.lower()}.csv"
    if placebo_path.exists() and placebo_summary_path.exists():
        perm_coefs = pd.read_csv(placebo_path)["perm_coef"].to_numpy()
        true_coef = pd.read_csv(placebo_summary_path).iloc[0]["true_coef"]
        figures.fig_placebo(perm_coefs, true_coef, tag=tag.lower(), synthetic=synthetic)
    else:
        print("\n   Placebo figure skipped: run run_placebo_test.py first.")

    print("\n--- Maps " + "-" * 62)
    # Annual small multiples stay legible up to ~20 panels; beyond that (the
    # much longer synthetic record) fall back to 5-year bins.
    n_seasons = pan["season_year"].nunique()
    period_years = 1 if n_seasons <= 20 else 5
    maps.make_all(pan, tag=tag.lower(), synthetic=synthetic, strat_label=strat_label,
                  period_years=period_years)
    if not proj.empty:
        maps.map_scenario_delta(proj, tag=tag.lower(), synthetic=synthetic)

    print(f"\nTables -> {C.TAB}")
    print(f"Figures -> {C.FIG}")
    if synthetic:
        b = res["M4_compound"].params
        print("\nRECOVERY CHECK (synthetic truth: dry=0.35, heat=0.15, interaction=0.30)")
        print(f"   estimated dry={b['dry_z']:+.3f}  heat={b['heat_z']:+.3f}  "
              f"interaction={b['compound_cont']:+.3f}")
        print("   Attenuation is expected: the outcome is rounded to integer IPC phases.")
    return pan, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    main(synthetic=a.synthetic)
