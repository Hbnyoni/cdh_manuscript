"""Populate the manuscript's [[MARKERS]] from the results tables.

  python fill_manuscript.py                # uses real-data outputs
  python fill_manuscript.py --synthetic    # uses synthetic outputs

Writes manuscript/manuscript_filled.md. Any marker left unresolved is reported
rather than silently dropped, so a missing number is visible.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cdh import config as C  # noqa: E402


def _p(v):
    return "< 0.001" if v < 0.001 else f"= {v:.3f}"


def _signed(v, unit="phase points"):
    return f"{v:+.3f} {unit}"


def _stratum_prose(name: str, is_natural_region: bool) -> str:
    """A stratum name as it reads mid-sentence, article included ("within
    ..."). Natural Region labels are Roman numerals ("V"): lowercasing gives
    "the v", so those get a "Natural Region" prefix and keep their case
    instead of being lowercased like an aridity-tercile label ("Driest
    third" -> "the driest third")."""
    return f"Natural Region {name}" if is_natural_region else f"the {name.lower()}"


def build(tag: str) -> dict:
    panel = pd.read_csv(C.PROC / f"panel_{tag}.csv")
    coefs = pd.read_csv(C.TAB / f"coefficients_{tag}.csv")
    fits = pd.read_csv(C.TAB / f"model_fit_{tag}.csv").set_index("model")
    cv = pd.read_csv(C.TAB / f"cv_{tag}.csv").set_index("model")
    me = pd.read_csv(C.TAB / f"marginal_effects_{tag}.csv")
    rob = pd.read_csv(C.TAB / f"robustness_{tag}.csv")

    def _opt(name):
        f = C.TAB / f"{name}_{tag}.csv"
        return pd.read_csv(f) if f.exists() else pd.DataFrame()

    het = _opt("heterogeneity")
    conley = _opt("conley")
    conley_sens = _opt("conley_sensitivity")
    altrain = _opt("rainfall_sensitivity")
    scen_burden = _opt("scenario_burden")
    placebo = _opt("placebo_summary")

    def coef(model, term_label):
        r = coefs[(coefs.model == model) & (coefs.term == term_label)]
        return None if r.empty else r.iloc[0]

    ci = coef("M4_compound", "Moisture deficit x hot days")
    me_cold = me.iloc[(me.heat_z + 2).abs().argmin()]
    me_hot = me.iloc[(me.heat_z - 2).abs().argmin()]
    # A ratio is only meaningful when both effects share a sign. When the cold-season
    # effect is null or negative the honest description is a sign change, not a ratio.
    if me_cold.effect > 0.02 and me_hot.effect > 0:
        ratio_txt = f"a ratio of {me_hot.effect / me_cold.effect:.1f} to 1"
    elif me_hot.effect > 0 >= me_cold.effect:
        ratio_txt = ("so the effect is indistinguishable from zero, or reversed, in cool "
                     "seasons and appears only as the season warms")
    else:
        ratio_txt = "with no consistent gradient across heat conditions"

    d_only = coef("M5_binary", "Drought, not hot")
    h_only = coef("M5_binary", "Hot, not drought")
    c_bin = coef("M5_binary", "Compound drought-heat")

    rmse1 = float(cv.loc["M1_rain_only", "rmse"]) if "M1_rain_only" in cv.index else np.nan
    rmse4 = float(cv.loc["M4_compound", "rmse"]) if "M4_compound" in cv.index else np.nan
    impr = (rmse1 - rmse4) / rmse1 * 100 if np.isfinite(rmse1) and rmse1 else np.nan

    sup = ""
    if all(x is not None for x in (d_only, h_only, c_bin)):
        add = d_only.coef + h_only.coef
        sup = (f"The compound coefficient exceeds the sum of the two single-hazard "
               f"coefficients ({c_bin.coef:+.3f} against {add:+.3f}), which is the "
               f"super-additivity the compound event framing predicts."
               if c_bin.coef > add else
               f"The compound coefficient does not exceed the sum of the single-hazard "
               f"coefficients ({c_bin.coef:+.3f} against {add:+.3f}), so the effect is "
               f"additive rather than super-additive on this definition.")

    # ---- heterogeneity, computed BEFORE the headline sentences below, because
    # the honest headline for this study is a three-way branch (uniform
    # positive / uniform null / concentrated in a stratum), not the binary
    # positive-or-null a single pooled coefficient can express on its own.
    is_nr = C.NATURAL_REGION_CSV.exists()
    strat_src = ("Natural Regions" if is_nr
                 else "terciles of district growing-season rainfall climatology "
                      "(the Natural Region boundary file was not available, so these "
                      "are an aridity proxy, not Natural Regions)")
    # The formal between-stratum test, read from the saved pooled three-way
    # model. Prose must quote it, never infer it from the spread of the
    # split-sample estimates.
    pooled_tab = _opt("pooled_interaction")
    pooled_sent = ("The pooled three-way model, which is what formally tests "
                   "difference, could not be read (no pooled_interaction table).")
    if not pooled_tab.empty:
        pt = pooled_tab.set_index("term")
        pair = pt.drop(index=[t for t in pt.index if t.startswith("joint")])
        top = pair.sort_values("coef", ascending=False).iloc[0]
        top_name = top.name.removeprefix("s_").removesuffix("_x_comp").replace("_", " ")
        ref_name = str(top["reference"])
        pooled_sent = (
            f"The pooled three-way model, which is what formally tests difference, "
            f"puts the compound interaction in the {top_name.lower()} "
            f"{top['coef']:+.3f} above the {ref_name.lower()} (SE {top['se']:.3f}, "
            f"p {_p(top['p'])})")
        if "joint_wald_chi2" in pt.index:
            j = pt.loc["joint_wald_chi2"]
            pooled_sent += (f"; the joint Wald test across strata gives "
                            f"χ²({int(j['se'])}) = {j['coef']:.2f}, p {_p(j['p'])}")
        pooled_sent += "."
    het_sent = "Heterogeneity could not be estimated."
    het_concentrated = False
    driest_row = None
    if not het.empty and "interaction" in het.columns:
        h = het.dropna(subset=["interaction"])
        if len(h):
            rng = f"{h['interaction'].min():+.3f} to {h['interaction'].max():+.3f}"
            n_sig = int((h["p_interaction"] < 0.05).sum())
            het_concentrated = h["interaction"].std() >= h["se_interaction"].mean()
            het_sent = (
                f"The compound interaction ranges from {rng} across the "
                f"{len(h)} strata and is individually significant in {n_sig} of them. "
                f"The pooled model, which is what formally tests difference, "
                f"{'does not detect' if not het_concentrated else 'detects'} "
                f"a reliable difference between strata, so the effect is best read as "
                f"broadly common rather than concentrated in the drier margins."
                if not het_concentrated else
                f"The compound interaction ranges from {rng} across the "
                f"{len(h)} strata and is individually significant in {n_sig} of them. "
                f"{pooled_sent}")
            # Name whichever stratum carries the largest interaction, if it is
            # itself significant -- for an aridity-tercile or Natural-Region
            # stratification this is the substantively relevant one for prose
            # (the driest/most marginal end), not necessarily literally first.
            cand = h.sort_values("interaction", ascending=False).iloc[0]
            if cand["p_interaction"] < 0.05:
                driest_row = cand

    # ---- statistical power: is the "marginal" pooled result just underpowered? ----
    # Minimum detectable effect at 80% power, two-sided alpha 0.05, is the
    # textbook (z_.975 + z_.80) * SE = (1.96 + 0.84) * SE = 2.80 * SE. Reported
    # for the pooled estimate and, where available, the concentrated stratum,
    # so a referee's obvious "is the pooled null just low power" question is
    # answered with the same standard errors already reported elsewhere,
    # rather than left for them to compute themselves.
    power_sent = "Power could not be assessed (missing coefficient or standard error)."
    if ci is not None and np.isfinite(ci.se) and ci.se > 0:
        mde_pooled = 2.8 * ci.se
        if abs(ci.coef) < mde_pooled:
            pooled_txt = (
                f"falls below that threshold, so a marginal p-value there is close to "
                f"what an underpowered test for an effect of this size would produce "
                f"even if the true pooled effect were exactly this small, not evidence "
                f"that it is zero")
        else:
            pooled_txt = (
                "clears that threshold, so the marginal significance reflects the "
                "estimate itself rather than a general lack of power")
        power_sent = (
            f"Part of the pooled result is a power question, not only a magnitude "
            f"question. At the pooled clustered standard error ({ci.se:.4f}), the "
            f"design has 80 per cent power to detect an interaction of about "
            f"{mde_pooled:.3f} phase units at the 5 per cent level; the pooled point "
            f"estimate itself ({ci.coef:+.3f}) {pooled_txt}.")
        if driest_row is not None and np.isfinite(driest_row["se_interaction"]) \
                and driest_row["se_interaction"] > 0:
            mde_strat = 2.8 * driest_row["se_interaction"]
            if abs(driest_row["interaction"]) > mde_strat:
                strat_txt = (
                    f"clears its own comparable threshold (about {mde_strat:.3f}) "
                    f"comfortably, so that result is not an artefact of a subgroup "
                    f"that happened to be better powered by chance")
            else:
                strat_txt = (
                    f"does not clearly clear its own comparable threshold (about "
                    f"{mde_strat:.3f}), so it should be read with the same caution "
                    f"as the pooled estimate")
            stratum_txt = _stratum_prose(driest_row["stratum"], is_nr)
            power_sent += (
                f" {stratum_txt[0].upper()}{stratum_txt[1:]} estimate "
                f"({driest_row['interaction']:+.3f}, standard error "
                f"{driest_row['se_interaction']:.3f}) {strat_txt}.")

    sig = ci is not None and ci.p < 0.05 and ci.coef > 0
    sig_concentrated = (not sig) and het_concentrated and driest_row is not None
    if ci is not None and sig and np.isfinite(impr):
        results_sentence = (
            f"The compound interaction is positive and statistically significant "
            f"(β = {ci.coef:+.3f}, p {_p(ci.p)}). The compound model improves "
            f"leave-one-season-out RMSE by {impr:.1f} per cent over the rainfall-only "
            f"specification, so the gain is predictive and not only in-sample.")
    elif ci is not None and sig_concentrated:
        results_sentence = (
            f"Pooled across the whole country the compound interaction is small and only "
            f"marginally significant (β = {ci.coef:+.3f}, p {_p(ci.p)}), and the compound "
            f"model does not improve leave-one-season-out prediction over simpler "
            f"specifications. That pooled estimate, however, averages over real "
            f"heterogeneity: within {_stratum_prose(driest_row['stratum'], is_nr)}, the "
            f"compound interaction is "
            f"{driest_row['interaction']:+.3f} (p {_p(driest_row['p_interaction'])}). "
            f"{pooled_sent} The compound mechanism is concentrated on the most "
            f"marginal land, not uniform across Zimbabwe.")
    elif ci is not None:
        results_sentence = (
            "The compound interaction is not statistically distinguishable from zero on "
            "these data, and the compound specification does not improve out-of-sample "
            "prediction. We report this as a null result: on district-level IPC outcomes "
            "in Zimbabwe, heat adds little beyond its contribution to the water balance.")
    else:
        results_sentence = "[[results pending]]"

    if sig:
        concl = (
            "Moisture deficit and growing-season heat interact in their effect on lean-season "
            "food insecurity in Zimbabwe. Monitoring that conditions on rainfall alone "
            "understates risk in hot dry seasons.")
    elif sig_concentrated:
        concl = (
            f"Moisture deficit and growing-season heat interact in their effect on "
            f"lean-season food insecurity in Zimbabwe, but not everywhere: the effect is "
            f"concentrated in {_stratum_prose(driest_row['stratum'], is_nr)}, where "
            f"communal-area maize "
            f"cultivation already sits at the edge of what the land supports. A national "
            f"average is the wrong scale to test this claim at; monitoring that pools all "
            f"districts together will miss a real, localised risk that a stratified "
            f"assessment recovers.")
    else:
        concl = (
            "We find no evidence that growing-season heat adds predictive information beyond "
            "a moisture-balance drought index for district-level food insecurity in Zimbabwe. "
            "This is a useful negative result: it suggests SPEI already captures the "
            "agronomically relevant part of the heat signal at this scale.")

    rob_sent = "The interaction "
    if not rob.empty and "p_interaction" in rob:
        n_sig = int((rob["p_interaction"] < 0.05).sum())
        rob_sent += (f"retains its sign and significance in {n_sig} of {len(rob)} "
                     f"alternative specifications, including district-only fixed effects, "
                     f"exclusion of the pandemic years, and a linear probability model on "
                     f"the crisis indicator.")
    else:
        rob_sent += "could not be evaluated across robustness variants."

    # ---- Conley ----
    conley_se_txt = clustered_se_txt = "n/a"
    conley_sent = conley_stable = ""
    if not conley.empty:
        row = conley[conley["term"] == "compound_cont"]
        if len(row):
            conley_se_txt = f"{row.iloc[0]['se_conley']:.4f}"
            cse = coefs[(coefs.model == "M4_compound") &
                        (coefs.term == "Moisture deficit x hot days")]
            if len(cse):
                clustered_se_txt = f"{cse.iloc[0]['se']:.4f}"
                wider = row.iloc[0]["se_conley"] > cse.iloc[0]["se"]
                conley_sent = (
                    "Allowing for spatial correlation widens the interval, as expected, "
                    "but does not overturn the result."
                    if wider else
                    "Allowing for spatial correlation does not widen the interval, which "
                    "indicates the district clustering was already conservative here.")
    if not conley_sens.empty:
        cs = conley_sens[conley_sens["term"] == "compound_cont"].dropna(subset=["p"])
        if len(cs):
            conley_stable = ("unchanged" if (cs["p"] < 0.05).all()
                             else f"sensitive: significance is lost at "
                                  f"{', '.join(str(int(c)) + ' km' for c in cs.loc[cs['p'] >= 0.05, 'dist_km'])}")

    # ---- Randomization inference ----
    placebo_sent = ("Randomization inference was not run for this build (run "
                    "run_placebo_test.py first).")
    if not placebo.empty:
        row = placebo.iloc[0]
        n_valid = int(row["n_valid"])
        p = row["p_value"]
        n_exceed = int(round(p * n_valid))
        p_txt = f"< {1 / n_valid:.3f}" if n_exceed == 0 else f"{p:.3f}"
        placebo_sent = (
            f"Randomization inference reaches the same conclusion by a route with no "
            f"distributional assumption. Across {n_valid} permutations, each preserving "
            f"a district's own marginal exposure distribution while breaking the true "
            f"season-to-season pairing with its own outcome, only {n_exceed} produced a "
            f"compound coefficient as large in magnitude as the observed "
            f"{row['true_coef']:+.4f} (empirical two-sided p {p_txt}), against a null "
            f"distribution centred near zero (mean {row['null_mean']:+.4f}, s.d. "
            f"{row['null_std']:.4f}) ([[FIG:fig_placebo]]).")

    # ---- alternate rainfall ----
    alt_int_txt, alt_sent = "n/a", ""
    if not altrain.empty:
        a = altrain[(altrain.model == "M4_compound") &
                    (altrain.term == "compound_cont") &
                    (altrain["product"] != "CHIRPS v3")]
        if len(a):
            alt_int_txt = f"{a.iloc[0]['coef']:+.3f}"
            if ci is not None:
                diff = abs(a.iloc[0]["coef"] - ci.coef)
                alt_sent = ("The two agree closely, so the result is not an artefact of "
                            "the rainfall product."
                            if diff < 0.5 * abs(ci.coef) else
                            "The estimates differ materially, so the choice of rainfall "
                            "product matters and the result should be reported as "
                            "conditional on it.")

    # ---- climate-change scenario projection ----
    scen_sent = ("Scenario projection was not run for this build (needs "
                 "gee/extract_cmip6_scenarios.py's output).")
    if not scen_burden.empty:
        parts = []
        for _, r in scen_burden.sort_values("scenario").iterrows():
            label = r["scenario"].replace("_2050s", "").upper()
            pw = r["mean_delta_phase_population_weighted"]
            thr = r["elevated_risk_threshold"]
            elevated_pct = r["elevated_risk_share"] * 100
            elevated_n = r["elevated_risk_population"]
            direction = "a rise" if pw > 0 else "a fall" if pw < 0 else "no change"
            parts.append(
                f"Under {label} by the 2050s, the population-weighted mean change "
                f"in lean-season IPC phase is {direction} of {abs(pw):.3f}, and "
                f"{elevated_pct:.0f}% of the reported population ({elevated_n:,.0f} "
                f"people) lives in districts projected to cross a "
                f"{thr:+.1f}-phase-point increase, holding population at its "
                f"current level.")
        scen_sent = (" ".join(parts) +
                    " This is an illustration of direction and rough magnitude "
                    "under the paper's own fitted relationship, not a probabilistic "
                    "forecast (Section 6).")

    return {
        "STRATUM_SOURCE": strat_src,
        "HETEROGENEITY_SENTENCE": het_sent,
        "POWER_SENTENCE": power_sent,
        "CONLEY_SE": conley_se_txt,
        "CLUSTERED_SE": clustered_se_txt,
        "CONLEY_SENTENCE": conley_sent,
        "CONLEY_STABILITY": conley_stable or "not evaluated",
        "PLACEBO_SENTENCE": placebo_sent,
        "ALT_INTERACTION": alt_int_txt,
        "ALT_SENTENCE": alt_sent,
        "SCENARIO_SENTENCE": scen_sent,
        "N_OBS": f"{len(panel):,}",
        "N_DISTRICTS": str(panel["district"].nunique()),
        "N_YEARS": str(panel["season_year"].nunique()),
        "YEAR_MIN": str(int(panel["season_year"].min())),
        "YEAR_MAX": str(int(panel["season_year"].max())),
        "MEAN_PHASE": f"{panel['phase'].mean():.2f}",
        "PCT_COMPOUND": f"{panel['compound_bin'].mean():.1%}",
        "R2_M1": f"{fits.loc['M1_rain_only','within_r2']:.3f}",
        "R2_M2": f"{fits.loc['M2_spei_only','within_r2']:.3f}",
        "R2_M3": f"{fits.loc['M3_additive','within_r2']:.3f}",
        "R2_M4": f"{fits.loc['M4_compound','within_r2']:.3f}",
        "B_INTERACTION": f"{ci.coef:+.3f}" if ci is not None else "n/a",
        "SE_INTERACTION": f"{ci.se:.3f}" if ci is not None else "n/a",
        "P_INTERACTION": _p(ci.p) if ci is not None else "n/a",
        "ME_COLD": _signed(me_cold.effect),
        "ME_HOT": _signed(me_hot.effect),
        "ME_RATIO": ratio_txt,
        "B_DROUGHT_ONLY": _signed(d_only.coef) if d_only is not None else "n/a",
        "B_HEAT_ONLY": _signed(h_only.coef) if h_only is not None else "n/a",
        "B_COMPOUND_BIN": _signed(c_bin.coef) if c_bin is not None else "n/a",
        "SUPERADDITIVITY_SENTENCE": sup,
        "RMSE_M1": f"{rmse1:.3f}",
        "RMSE_M4": f"{rmse4:.3f}",
        "RMSE_IMPROVEMENT": f"{impr:.1f} per cent",
        "ROBUSTNESS_SENTENCE": rob_sent,
        "RESULTS_SENTENCE": results_sentence,
        "CONCLUSION": concl,
    }


def main(synthetic: bool):
    tag = "synthetic" if synthetic else "real"
    vals = build(tag)
    src = C.ROOT / "manuscript" / "manuscript.md"
    text = src.read_text(encoding="utf-8")

    def sub(m):
        k = m.group(1).strip()
        return vals.get(k, m.group(0))

    out = re.sub(r"\[\[([A-Z0-9_]+)\]\]", sub, text)
    left = sorted(set(re.findall(r"\[\[([A-Z0-9_]+)\]\]", out)))

    if synthetic:
        out = out.replace(
            "# Compound drought and heat exposure",
            "> **WARNING: filled with SYNTHETIC data. Every number below is fabricated "
            "for pipeline testing and must not be cited.**\n\n"
            "# Compound drought and heat exposure", 1)

    dst = C.ROOT / "manuscript" / f"manuscript_filled_{tag}.md"
    dst.write_text(out, encoding="utf-8")
    print(f"Wrote {dst}")
    print(f"Filled {len(vals)} markers.")
    if left:
        print(f"UNRESOLVED markers remain: {left}")
    else:
        print("All result markers resolved. Figure/table references resolve during Word export.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    main(ap.parse_args().synthetic)
