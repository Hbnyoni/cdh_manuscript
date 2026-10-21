"""Randomization inference on the compound interaction.

Independent of the Conley and clustered standard errors reported elsewhere:
those are analytic, asymptotic-normality-based corrections for spatial and
serial correlation. This is a different kind of check entirely, with no
distributional assumption. It asks a narrower, sharper question: if a
district's own exposure values were attached to the wrong season within that
same district (same marginal distribution of dry_z/heat_z/compound_cont for
that district, wrong pairing with the outcome), how often would a compound
interaction this large appear by chance alone?

Permutation is WITHIN each district, independently, rather than across the
whole panel: shuffling globally would also scramble which values face which
year's national shock, contaminating the test with the year fixed effect's
own structure. Permuting only the within-district time index leaves each
district's marginal exposure distribution and the year fixed effects' cross-
sectional composition untouched, so the only thing broken is the true
season-to-season pairing of that district's own exposure with its own
outcome -- exactly the null a compound-timing claim needs to be tested
against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import models as M


def permutation_test(panel: pd.DataFrame, term: str = "compound_cont",
                      regressors=None, n_perm: int = 1000, seed: int = 0,
                      verbose: bool = True):
    """Return (true_coef, perm_coefs, p_value) for `term` in the M4-style
    specification `regressors` (defaults to M4_compound's own three terms).

    p_value is two-sided empirical: the share of permutations whose |coef|
    meets or exceeds the true |coef|, i.e. (1 + count) / (1 + n_valid) is
    NOT used -- a plain share is reported and is exactly zero only when the
    true value exceeds every permutation, which is stated as such rather
    than implied to be impossibly small.
    """
    regressors = list(regressors or M.SPECS["M4_compound"])
    d = panel.dropna(subset=[M.OUTCOME] + regressors).reset_index(drop=True)

    true_res = M.fit_spec(d, regressors)
    true_coef = float(true_res.params[term])

    rng = np.random.default_rng(seed)
    group_idx = {g: np.asarray(idx) for g, idx in d.groupby("district").groups.items()}
    perm_coefs = np.full(n_perm, np.nan)
    for i in range(n_perm):
        dp = d.copy()
        for idx in group_idx.values():
            if len(idx) < 2:
                continue
            dp.loc[idx, regressors] = d.loc[rng.permutation(idx), regressors].to_numpy()
        try:
            perm_coefs[i] = M.fit_spec(dp, regressors).params[term]
        except Exception:
            pass

    valid = perm_coefs[np.isfinite(perm_coefs)]
    p_value = float(np.mean(np.abs(valid) >= abs(true_coef))) if len(valid) else float("nan")
    if verbose:
        n_failed = n_perm - len(valid)
        print(f"   Permutation test on {term}: true coef {true_coef:+.4f}, "
              f"{len(valid)}/{n_perm} valid permutations"
              f"{f' ({n_failed} failed to fit)' if n_failed else ''}, "
              f"null range [{valid.min():+.4f}, {valid.max():+.4f}], "
              f"empirical two-sided p = {p_value:.4f}")
    return true_coef, valid, p_value


def block_permutation_test(panel: pd.DataFrame, term: str = "compound_cont",
                           regressors=None, n_perm: int = 1000, seed: int = 0,
                           verbose: bool = True):
    """Season-block variant of permutation_test.

    One permutation of the season-years is drawn and applied to EVERY
    district, so each season's cross-district pattern of exposure moves as a
    block. Unlike the within-district scheme above, this keeps the spatial
    dependence of weather shocks intact, so its null is comparable to the one
    behind the Conley errors (spatial.py). Rows whose source season is missing
    for that district are dropped for that permutation.
    """
    regressors = list(regressors or M.SPECS["M4_compound"])
    d = panel.dropna(subset=[M.OUTCOME] + regressors).reset_index(drop=True)
    true_coef = float(M.fit_spec(d, regressors).params[term])

    years = np.array(sorted(d["season_year"].unique()))
    lut = d.set_index(["district", "season_year"])[regressors]
    rng = np.random.default_rng(seed)
    perm_coefs = np.full(n_perm, np.nan)
    for i in range(n_perm):
        mapping = dict(zip(years, rng.permutation(years)))
        src = pd.MultiIndex.from_arrays([d["district"], d["season_year"].map(mapping)])
        dp = d.copy()
        dp[regressors] = lut.reindex(src).to_numpy()
        try:
            perm_coefs[i] = M.fit_spec(dp.dropna(subset=regressors), regressors).params[term]
        except Exception:
            pass

    valid = perm_coefs[np.isfinite(perm_coefs)]
    p_value = float(np.mean(np.abs(valid) >= abs(true_coef))) if len(valid) else float("nan")
    if verbose:
        print(f"   Season-block permutation on {term}: true coef {true_coef:+.4f}, "
              f"{len(valid)}/{n_perm} valid, null s.d. {valid.std():.4f}, "
              f"empirical two-sided p = {p_value:.4f}")
    return true_coef, valid, p_value
