"""Manuscript figures.

Design decisions, made once here rather than per figure:

* Two accent colours only, validated for colour-vision deficiency separation
  (protan dE 17.5, normal dE 26.5, both >= 3:1 against the surface). Blue carries
  the comparison models, rust carries the paper's own specification, so the
  reader's eye lands on the claim being made.
* The four compound categories are ORDERED by expected severity, so they get a
  single-hue sequential ramp rather than categorical hues. Ordering is information
  and a categorical palette would throw it away.
* No dual axes anywhere. Where two quantities of different scale appear, they are
  separate panels.
* Every series is identified by a label as well as a colour, so nothing depends on
  colour alone in print or greyscale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config as C
from . import models as M

BLUE = "#1f6f9c"      # comparison / reference
RUST = "#c8622a"      # the paper's specification
INK = "#1c2321"
MUTED = "#6b7a76"
GRID = "#dde3e1"
SURFACE = "#fcfcfb"
# Single-hue sequential ramp, light -> dark, for the ordered compound categories.
# Used for FILLED marks (bars), where the pale end sits on a large area and reads fine.
SEQ = ["#d3e3ee", "#9cc2d8", "#5695b9", "#1f6f9c"]
# Compressed ramp for THIN marks (lines, dots), where the pale end of SEQ falls below
# 3:1 against the surface. Every step here is >= 3:1. Darkest = most severe, so the
# ordering runs driest-first. Identity is also carried by the axis labels.
SEQ_LINE = ["#083d63", "#13507a", "#22638c", "#35779f", "#4a8ab0"]

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    # Times New Roman: matches the manuscript body text (manuscript/template/
    # reference_style.docx), so figures and prose read as one document rather
    # than a sans-serif figure dropped into a serif Word manuscript.
    "font.family": "Times New Roman", "font.size": 12,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300,
})


def _note(fig, synthetic):
    if synthetic:
        fig.text(0.99, 0.01, "SYNTHETIC DATA — not real results",
                 ha="right", va="bottom", fontsize=10.5, color=RUST, style="italic")


def fig_coefficients(results, tag, synthetic):
    """Dot-and-whisker across the nested model ladder."""
    rows = []
    for name in ["M1_rain_only", "M2_spei_only", "M3_additive", "M4_compound"]:
        r = results.get(name)
        if r is None:
            continue
        for t in r.params.index:
            ci = r.conf_int().loc[t]
            rows.append({"model": name, "term": M.LABELS.get(t, t),
                         "b": r.params[t], "lo": ci.iloc[0], "hi": ci.iloc[1]})
    d = pd.DataFrame(rows)
    if d.empty:
        return
    d["y"] = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(d) + 1.5))
    for _, r in d.iterrows():
        hi = r["model"] == "M4_compound"
        col = RUST if hi else BLUE
        ax.plot([r["lo"], r["hi"]], [r["y"], r["y"]], color=col,
                lw=2.0, solid_capstyle="round", alpha=.95)
        ax.plot([r["b"]], [r["y"]], "o", ms=6.5, color=col,
                mec=SURFACE, mew=1.6, zorder=3)
    ax.axvline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.set_yticks(d["y"])
    ax.set_yticklabels([f"{r.term}" for r in d.itertuples()], fontsize=12.5)
    ax.set_xlabel("Change in mean lean-season IPC phase per 1 SD")
    ax.grid(axis="y", visible=False)

    ax.set_ylim(top=d["y"].max() + 0.75)   # headroom for the "M1 rain only"-style
                                            # labels sitting above the top row
    prev = None
    for _, r in d.iterrows():
        if r["model"] != prev:
            ax.text(ax.get_xlim()[0], r["y"] + 0.42, r["model"].replace("_", " "),
                    fontsize=11, color=INK, va="bottom",
                    fontweight="bold" if r["model"] == "M4_compound" else "normal")
            prev = r["model"]
    _note(fig, synthetic)
    fig.tight_layout()
    fig.savefig(C.FIG / f"fig1_coefficients_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def _plot_me_panel(ax, me, title, ymin, ymax):
    ax.fill_between(me["heat_z"], me["lo"], me["hi"], color=RUST, alpha=.16, lw=0)
    ax.plot(me["heat_z"], me["effect"], color=RUST, lw=2.0, solid_capstyle="round")
    ax.axhline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)))

    x0, x1 = me["heat_z"].iloc[0], me["heat_z"].iloc[-1]
    ax.plot([x1], [me["effect"].iloc[-1]], "o", ms=7, color=RUST,
            mec=SURFACE, mew=1.8, zorder=4)
    ax.annotate(f"{me['effect'].iloc[-1]:+.2f} phase\nat +2 SD hot days",
                xy=(x1, me["effect"].iloc[-1]), xytext=(-8, 14),
                textcoords="offset points", ha="right", fontsize=12.5, color=INK)
    ax.annotate(f"{me['effect'].iloc[0]:+.2f} at -2 SD",
                xy=(x0, me["effect"].iloc[0]), xytext=(12, -18),
                textcoords="offset points", ha="left", fontsize=12.5, color=INK)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Growing-season hot days (SD from district mean)")
    ax.set_title(title, loc="left", fontsize=14, color=INK, pad=8)


def fig_marginal_effect(me: pd.DataFrame, tag, synthetic, me_strat=None, strat_name=None):
    """The headline: cost of a moisture deficit as the season gets hotter.

    When a stratum-specific estimate is available (the pooled national effect
    was too diluted to be the whole story -- see run_pipeline.py), shows it
    alongside the pooled curve on a shared y-axis, so the reader sees the
    concentration directly rather than only the diluted pooled estimate. When
    there is no such stratum (the pooled effect is itself the strongest
    estimate available), a single panel is enough.
    """
    if me_strat is not None and strat_name:
        both = pd.concat([me[["lo", "hi"]], me_strat[["lo", "hi"]]])
        pad = (both["hi"].max() - both["lo"].min()) * 0.16
        ymin, ymax = both["lo"].min() - pad, both["hi"].max() + pad

        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
        _plot_me_panel(axes[0], me, "a. Pooled, whole country (marginal)", ymin, ymax)
        _plot_me_panel(axes[1], me_strat, f"b. {strat_name} (robust)", ymin, ymax)
        axes[0].set_ylabel("Effect of 1 SD moisture deficit on IPC phase")
        _note(fig, synthetic)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        both = me
        pad = (both["hi"].max() - both["lo"].min()) * 0.14
        _plot_me_panel(ax, me, "", both["lo"].min() - pad, both["hi"].max() + pad)
        ax.set_ylabel("Effect of 1 SD moisture deficit on IPC phase")
        _note(fig, synthetic)
        fig.tight_layout()

    fig.savefig(C.FIG / f"fig2_marginal_effect_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def fig_categories(panel: pd.DataFrame, tag, synthetic):
    """Observed outcome by ordered compound category, with 95% CIs."""
    d = panel.copy()
    d["cat"] = np.select(
        [(d.is_drought == 0) & (d.is_hot_season == 0),
         (d.is_drought == 0) & (d.is_hot_season == 1),
         (d.is_drought == 1) & (d.is_hot_season == 0),
         (d.is_drought == 1) & (d.is_hot_season == 1)],
        ["Neither", "Hot only", "Drought only", "Compound"], default="Neither")
    order = ["Neither", "Hot only", "Drought only", "Compound"]
    g = d.groupby("cat", observed=True)["phase"].agg(["mean", "sem", "size"]).reindex(order)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(len(order))
    for i, (cat, r) in enumerate(g.iterrows()):
        if not np.isfinite(r["mean"]):
            continue
        ax.bar(i, r["mean"], width=0.62, color=SEQ[i], zorder=2,
               edgecolor=SURFACE, linewidth=2.0)
        ax.errorbar(i, r["mean"], yerr=1.96 * r["sem"], color=INK,
                    lw=1.2, capsize=4, zorder=3)
        ax.text(i, r["mean"] + 1.96 * r["sem"] + .045, f"{r['mean']:.2f}",
                ha="center", fontsize=13, color=INK)
        ax.text(i, .06, f"n={int(r['size'])}", ha="center", fontsize=11, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=13)
    ax.set_ylabel("Mean lean-season IPC phase")
    ax.set_ylim(0, max(2.2, float(np.nanmax(g["mean"])) * 1.28))
    ax.grid(axis="x", visible=False)
    _note(fig, synthetic)
    fig.tight_layout()
    fig.savefig(C.FIG / f"fig3_categories_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def fig_cv(cv: pd.DataFrame, tag, synthetic):
    """Leave-one-year-out predictive error. Lower is better."""
    if cv.empty:
        return
    d = cv.sort_values("rmse", ascending=False).reset_index(drop=True)
    best = d["rmse"].idxmin()
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(d) + 1.6))
    for i, r in d.iterrows():
        col = RUST if i == best else BLUE
        ax.barh(i, r["rmse"], height=0.58, color=col, zorder=2,
                edgecolor=SURFACE, linewidth=2.0)
        ax.text(r["rmse"] + 0.004, i, f"{r['rmse']:.3f}", va="center",
                fontsize=12.5, color=INK)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([m.replace("_", " ") for m in d["model"]], fontsize=12.5)
    ax.set_xlabel("Out-of-sample RMSE, leave-one-season-out (lower is better)")
    ax.set_xlim(0, d["rmse"].max() * 1.16)
    ax.grid(axis="y", visible=False)
    _note(fig, synthetic)
    fig.tight_layout()
    fig.savefig(C.FIG / f"fig4_cv_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def fig_spacetime_heatmap(panel: pd.DataFrame, tag="real", synthetic=False):
    """District x season-year matrix of lean-season food insecurity, with
    compound drought-heat seasons marked. The whole record at a glance: rows
    are ordered by each district's own rainfall climatology (driest at top,
    consistent with the driest-first ordering used for the heterogeneity
    ramp), so a season that hit the whole country reads as a horizontal band
    and a chronically dry district reads as a vertical one.
    """
    from . import maps as MP

    d = panel.dropna(subset=["phase"]).copy()
    if d.empty or d["season_year"].nunique() < 2:
        return

    order = (d.groupby("district", observed=True)["season_rain_mm"].mean()
               .sort_values().index.tolist())
    years = sorted(d["season_year"].unique())
    mat = (d.pivot_table(index="district", columns="season_year", values="phase")
             .reindex(index=order, columns=years))
    comp = (d.pivot_table(index="district", columns="season_year", values="compound_bin")
              .reindex(index=order, columns=years))

    fig_h = max(4.2, 0.155 * len(order) + 1.6)
    fig_w = max(7.2, 0.34 * len(years) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = MP.RAMP_IPC.copy()
    cmap.set_bad("#e8ebea")
    vals = np.ma.masked_invalid(mat.to_numpy(dtype=float))
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=1, vmax=5)

    # A single rust "x" nearly vanished against RAMP_IPC's own orange mid-tones.
    # A white halo underneath a dark mark stays legible against every cell
    # colour in the ramp, from pale green to dark maroon.
    cy, cx = np.where(comp.to_numpy(dtype=float) == 1)
    ax.scatter(cx, cy, marker="x", s=34, color=SURFACE, linewidths=2.6, zorder=3)
    ax.scatter(cx, cy, marker="x", s=16, color=INK, linewidths=1.2, zorder=4)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=90, fontsize=9.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Season-year")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cb.set_label("Mean lean-season IPC phase", fontsize=12.5, color=INK)
    cb.outline.set_visible(False)
    _note(fig, synthetic)
    fig.tight_layout()
    fig.savefig(C.FIG / f"fig8_spacetime_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def fig_national_trend(panel: pd.DataFrame, tag="real", synthetic=False):
    """National time series: how the outcome and the exposure have moved,
    season by season, across the whole record. The headline "change over
    time" figure that a period-averaged map cannot show by itself -- the
    map answers "where", this answers "when" and "how much".

    Panel a: mean lean-season IPC phase per season-year, population-weighted
    across districts, with the interquartile range across districts as a band
    (so the reader sees both the national average and how much districts
    disagreed with it that season). Panel b: share of districts classified
    compound drought-heat that season-year, the paper's own exposure measure.
    """
    d = panel.dropna(subset=["phase", "season_year"]).copy()
    if d.empty or d["season_year"].nunique() < 3:
        return

    w = d["population"].fillna(0)
    d["_wphase"] = d["phase"] * w
    d["_w"] = w

    def _year_stats(g):
        wsum = g["_w"].sum()
        mean = g["_wphase"].sum() / wsum if wsum > 0 else g["phase"].mean()
        return pd.Series({
            "mean_phase": mean,
            "q25": g["phase"].quantile(0.25),
            "q75": g["phase"].quantile(0.75),
            "compound_share": g["compound_bin"].mean() * 100
            if "compound_bin" in g.columns else np.nan,
            "n_districts": g["district"].nunique(),
        })

    yr = (d.groupby("season_year", observed=True)
            .apply(_year_stats, include_groups=False)
            .reset_index().sort_values("season_year"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.4, 1]})

    ax1.fill_between(yr["season_year"], yr["q25"], yr["q75"], color=BLUE,
                     alpha=.16, lw=0, label="Interquartile range across districts")
    ax1.plot(yr["season_year"], yr["mean_phase"], color=BLUE, lw=2.0,
             marker="o", ms=4.5, mec=SURFACE, mew=1.0, solid_capstyle="round",
             label="National mean (population-weighted)")
    ax1.set_ylabel("Lean-season IPC phase")
    ax1.set_title("a.", loc="left", fontsize=14, color=INK)
    ax1.legend(loc="upper left", fontsize=12, ncol=1)

    ax2.bar(yr["season_year"], yr["compound_share"], color=RUST, width=0.68,
            zorder=2, edgecolor=SURFACE, linewidth=1.2)
    ax2.set_ylabel("Districts compound\ndrought-heat (%)")
    ax2.set_xlabel("Season-year")
    ax2.set_title("b.", loc="left", fontsize=14, color=INK)
    ax2.grid(axis="x", visible=False)
    ax2.set_ylim(0, max(20, float(yr["compound_share"].max()) * 1.25))

    years = yr["season_year"].to_numpy()
    step = max(1, len(years) // 14)
    ax2.set_xticks(years[::step])
    ax2.set_xticklabels([str(int(y)) for y in years[::step]], rotation=45, ha="right")

    _note(fig, synthetic)
    fig.tight_layout()
    fig.savefig(C.FIG / f"fig0_national_trend_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def fig_study_design(tag="real", synthetic=False):
    """Season-timing schematic: how exposure (growing season Y) precedes
    outcome (lean season Y+1) by nine to fourteen months, and why that
    ordering -- fixed by the agricultural calendar, not chosen to fit -- is
    what makes exposure temporally prior to outcome. A methods figure, not a
    data figure: no panel dependency, so it is identical across tag/synthetic
    other than the watermark.
    """
    months = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]

    fig, ax = plt.subplots(figsize=(9.6, 3.4))

    # Growing season: Oct(Y-1) - Mar(Y), months 0-5
    # Headers anchored va="bottom" and captions va="top" with an explicit
    # 0.25-unit gap between their edges, rather than two va="baseline" (the
    # default) texts 0.24 apart: at the bigger 12-13.5pt sizes the old
    # center-to-center spacing put the caption's ascenders inside the bold
    # header's descenders.
    ax.axvspan(0, 6, color=BLUE, alpha=.16, lw=0, zorder=1)
    ax.text(3, 1.55, "Growing season $Y$", ha="center", va="bottom",
            fontsize=13.5, color=INK, fontweight="bold")
    ax.text(3, 1.30, "exposure measured here\nSPEI-6, growing-season hot days",
            ha="center", va="top", fontsize=12, color=INK)

    # Harvest: Apr-May(Y), months 6-8
    ax.axvspan(6, 8, color=MUTED, alpha=.12, lw=0, zorder=1)
    ax.text(7, 1.55, "Harvest", ha="center", va="bottom", fontsize=13, color=INK)

    # Lean season: Jan-Mar(Y+1), months 15-18
    ax.axvspan(15, 18, color=RUST, alpha=.18, lw=0, zorder=1)
    ax.text(16.5, 1.55, "Lean season $Y+1$", ha="center", va="bottom",
            fontsize=13.5, color=INK, fontweight="bold")
    ax.text(16.5, 1.30, "outcome measured here\nmean IPC phase",
            ha="center", va="top", fontsize=12, color=INK)

    # Lag arrow: mid-growing-season to mid-lean-season
    ax.annotate("", xy=(16.5, 0.55), xytext=(3, 0.55),
               arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4,
                               shrinkA=0, shrinkB=0))
    ax.text(9.75, 0.72, "9 to 14 months: exposure precedes outcome",
            ha="center", fontsize=12.5, color=INK, style="italic")

    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(0, 1.85)
    ax.set_xticks(range(18))
    ax.set_xticklabels(months, fontsize=11, color=INK)
    # Only the three boundaries that separate a labelled band from the next
    # (growing season / harvest / lean season); no unexplained mid-gap line.
    for i in (0, 6, 8, 15):
        ax.axvline(i, color=GRID, lw=0.8, zorder=0)
    # Season-year (Y-1/Y/Y+1) strip beneath the month labels
    for lab, span in (("$Y-1$", (0, 3)), ("$Y$", (3, 15)), ("$Y+1$", (15, 18))):
        ax.text((span[0] + span[1]) / 2, -0.32, lab, ha="center", fontsize=12,
                color=INK, transform=ax.get_xaxis_transform())

    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.grid(False)
    _note(fig, synthetic)
    fig.tight_layout()
    fig.savefig(C.FIG / f"fig_study_design_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def make_all(panel, results, me, cv, tag="real", synthetic=False,
            me_strat=None, strat_name=None):
    fig_study_design(tag, synthetic)
    fig_national_trend(panel, tag, synthetic)
    fig_coefficients(results, tag, synthetic)
    fig_marginal_effect(me, tag, synthetic, me_strat=me_strat, strat_name=strat_name)
    fig_categories(panel, tag, synthetic)
    fig_cv(cv, tag, synthetic)
    fig_spacetime_heatmap(panel, tag, synthetic)
    print(f"   core figures written to {C.FIG}")


def fig_heterogeneity(het: pd.DataFrame, label: str, tag, synthetic):
    """Compound interaction estimated within each agro-ecological stratum.

    Strata are ordered, so they take the sequential ramp rather than categorical
    hues. Sample size is printed against each because split-sample estimates in
    thin strata are the obvious weak point and hiding n would be evasive.
    """
    d = het.dropna(subset=["interaction"]).reset_index(drop=True) \
        if "interaction" in het.columns else pd.DataFrame()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 0.66 * len(d) + 1.7))
    ramp = (SEQ_LINE * (len(d) // len(SEQ_LINE) + 1))[:len(d)]
    lo_all = (d["interaction"] - 1.96 * d["se_interaction"]).min()
    hi_all = (d["interaction"] + 1.96 * d["se_interaction"]).max()
    span = max(hi_all - min(lo_all, 0), 1e-6)
    xmax = hi_all + span * 0.42          # reserve a column for the sample sizes
    for i, r in d.iterrows():
        y = len(d) - 1 - i
        lo = r["interaction"] - 1.96 * r["se_interaction"]
        hi = r["interaction"] + 1.96 * r["se_interaction"]
        ax.plot([lo, hi], [y, y], color=ramp[i], lw=2.6, solid_capstyle="round")
        ax.plot([r["interaction"]], [y], "o", ms=7.5, color=ramp[i],
                mec=SURFACE, mew=1.8, zorder=3)
        ax.text(xmax, y, f"n={int(r['n'])}  ({int(r['n_districts'])} districts)",
                fontsize=11, color=INK, ha="right", va="center")
    ax.axvline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.set_xlim(min(lo_all, 0) - span * 0.08, xmax + span * 0.02)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(list(d["stratum"])[::-1], fontsize=13)
    ax.set_xlabel("Compound interaction coefficient (95% CI)")
    ax.grid(axis="y", visible=False)
    _note(fig, synthetic)
    fig.tight_layout()
    fig.savefig(C.FIG / f"fig5_heterogeneity_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def fig_conley(sens: pd.DataFrame, term="compound_cont", tag="real", synthetic=False):
    """Inference across Conley spatial cutoffs. The point estimate is fixed; only
    the interval moves, so this shows whether the conclusion depends on a choice
    with no data-driven answer."""
    d = sens[sens["term"] == term].dropna(subset=["se_conley"]).reset_index(drop=True)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    x = np.arange(len(d))
    lo = d["coef"] - 1.96 * d["se_conley"]
    hi = d["coef"] + 1.96 * d["se_conley"]
    for i in x:
        ax.plot([i, i], [lo[i], hi[i]], color=BLUE, lw=2.2, solid_capstyle="round")
        ax.plot([i], [d["coef"][i]], "o", ms=7, color=BLUE, mec=SURFACE, mew=1.8, zorder=3)
    ax.axhline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)} km" for c in d["dist_km"]], fontsize=13)
    ax.set_xlabel("Conley spatial correlation cutoff")
    ax.set_ylabel("Compound interaction (95% CI)")
    ax.set_xlim(-0.6, len(d) - 0.4)
    ax.grid(axis="x", visible=False)
    _note(fig, synthetic)
    fig.tight_layout()
    fig.savefig(C.FIG / f"fig6_conley_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def fig_placebo(perm_coefs: np.ndarray, true_coef: float, tag="real", synthetic=False):
    """Randomization-inference null: the compound interaction obtained from
    many within-district season-label permutations, against the true
    estimate. A different kind of check than Conley/clustered SEs -- no
    asymptotic-normality assumption -- so it earns its own figure rather
    than folding into fig6_conley."""
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.hist(perm_coefs, bins=30, color=BLUE, alpha=0.55, edgecolor=SURFACE, linewidth=0.6,
            label=f"Permuted (n={len(perm_coefs)})")
    ax.axvline(true_coef, color=RUST, lw=2.2, label=f"Observed ({true_coef:+.3f})")
    ax.axvline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.set_xlabel("Compound interaction coefficient")
    ax.set_ylabel("Permutations")
    ax.legend(loc="upper left" if true_coef > 0 else "upper right", fontsize=11.5)
    _note(fig, synthetic)
    fig.tight_layout()
    fig.savefig(C.FIG / f"fig_placebo_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def fig_scenario_burden(burden: pd.DataFrame, tag="real", synthetic=False):
    """Population-weighted projected phase change, and population in
    districts crossing a stated comparative threshold (delta_phase >= 0.5),
    under each CMIP6 scenario. NOT a Crisis+ (phase >= 3) share: these are
    multi-season MEANS, which even a badly affected district keeps well
    under 3 (see scenarios.population_burden's docstring for why an absolute
    threshold on a multi-season mean is the wrong statistic here). Population
    is held constant across scenarios -- stated again in-figure so this
    cannot be read as including demographic change.
    """
    if burden.empty:
        return
    d = burden.sort_values("scenario").reset_index(drop=True)
    labels = [s.replace("_2050s", "").upper() for s in d["scenario"]]
    thr = float(d["elevated_risk_threshold"].iloc[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.0))

    vals = d["mean_delta_phase_population_weighted"]
    colors = [RUST if v >= 0 else BLUE for v in vals]
    ax1.axhline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax1.bar(range(len(vals)), vals, color=colors, width=0.55, zorder=2,
           edgecolor=SURFACE, linewidth=1.5)
    for i, v in enumerate(vals):
        ax1.text(i, v + np.sign(v) * max(abs(vals)) * 0.06, f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=13, color=INK)
    ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, fontsize=13)
    ax1.set_ylabel("Population-weighted mean\nchange in IPC phase by 2050s")
    ax1.grid(axis="x", visible=False)
    ax1.set_title("a. Mean projected change", loc="left", fontsize=14, color=INK, pad=8)
    ax1.set_ylim(min(0, float(vals.min()) * 1.3), max(0.01, float(vals.max()) * 1.3))

    shares = d["elevated_risk_share"] * 100
    ax2.bar(range(len(shares)), shares, color=RUST, width=0.55, zorder=2,
           edgecolor=SURFACE, linewidth=1.5)
    for i, v in enumerate(shares):
        ax2.text(i, v + max(shares, default=1) * 0.03, f"{v:.0f}%", ha="center",
                fontsize=13, color=INK)
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, fontsize=13)
    ax2.set_ylabel(f"Population in districts with\nprojected change ≥ {thr:+.1f} phase, %")
    ax2.set_ylim(0, max(shares.max() * 1.25, 10))
    ax2.grid(axis="x", visible=False)
    ax2.set_title("b. Population at elevated risk", loc="left", fontsize=14, color=INK, pad=8)

    _note(fig, synthetic)
    fig.tight_layout()
    fig.savefig(C.FIG / f"fig9_scenario_burden_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def fig_binned_interaction(panel: pd.DataFrame, tag="real", synthetic=False):
    """Nonparametric view of the interaction, as three small multiples.

    Why facets and not three coloured lines on one axis: three ordered series need
    a sequential ramp, and one hue cannot span three steps that are both mutually
    distinguishable (normal-vision dE >= 15) and all legible against the surface.
    Faceting removes colour from the encoding altogether; panel order carries the
    ordering instead, and the steepening slope reads directly.

    Both axes are two-way demeaned, so the panels show exactly the variation the
    fixed effects estimator uses, not raw cross-sectional differences.
    """
    from . import spatial as SP

    d = panel.dropna(subset=["phase", "dry_z", "heat_z"]).copy()
    if len(d) < 60:
        return
    dm = SP.two_way_demean(d, ["phase", "dry_z"])
    d["_y"], d["_x"] = dm[:, 0], dm[:, 1]
    d["_tercile"] = pd.qcut(d["heat_z"], 3, labels=["Coolest third",
                                                    "Middle third", "Hottest third"])

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.7), sharey=True, sharex=True)
    ymin, ymax = np.inf, -np.inf
    for ax, (lab, g) in zip(axes, d.groupby("_tercile", observed=True)):
        g = g.copy()
        g["_bin"] = pd.qcut(g["_x"], 6, duplicates="drop")
        b = (g.groupby("_bin", observed=True)
               .agg(x=("_x", "mean"), y=("_y", "mean"),
                    se=("_y", "sem"), n=("_y", "size")).dropna())
        ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
        ax.errorbar(b["x"], b["y"], yerr=1.96 * b["se"], fmt="o", ms=6,
                    color=RUST, ecolor=RUST, elinewidth=1.3, capsize=3,
                    mec=SURFACE, mew=1.4, zorder=3)
        if len(b) >= 2:
            k = np.polyfit(g["_x"], g["_y"], 1)
            xs = np.linspace(b["x"].min(), b["x"].max(), 50)
            ax.plot(xs, np.polyval(k, xs), color=RUST, lw=1.8, alpha=.55, zorder=2)
            ax.text(0.04, 0.93, f"slope {k[0]:+.3f}", transform=ax.transAxes,
                    fontsize=13, color=INK, va="top", fontweight="bold")
        ax.set_title(str(lab), loc="left", fontsize=13.5, color=INK, pad=6)
        ax.set_xlabel("Moisture deficit (demeaned)")
        ymin = min(ymin, (b["y"] - 1.96 * b["se"]).min())
        ymax = max(ymax, (b["y"] + 1.96 * b["se"]).max())
    axes[0].set_ylabel("Lean-season IPC phase (demeaned)")
    pad = (ymax - ymin) * 0.18
    axes[0].set_ylim(ymin - pad, ymax + pad)
    _note(fig, synthetic)
    fig.tight_layout()
    fig.savefig(C.FIG / f"fig7_binned_interaction_{tag}.png", bbox_inches="tight")
    plt.close(fig)
