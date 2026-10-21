"""Conley spatial HAC standard errors for the two-way fixed effects panel.

Why this is needed. Clustering at district allows arbitrary serial correlation
within a district but assumes independence ACROSS districts. That assumption is
untenable here: weather is spatially correlated at scales far larger than a
Zimbabwean district, so the shocks driving the regressor are correlated across
neighbouring units in the same season. Year fixed effects absorb the national
component but not the regional one. Conley (1999) standard errors allow
correlation that decays with distance, and are the standard fix.

Implementation follows the separable space-time kernel of Hsiang (2010):

    meat = X' (W_S kron-elementwise W_T * u u') X
    V    = (X'X)^-1 meat (X'X)^-1

with Bartlett kernels in both space (cutoff DIST_KM) and time (lag TIME_LAG).
Setting TIME_LAG = 0 gives contemporaneous spatial correlation only.

The two-way within transformation is done here by alternating projections, which
handles unbalanced panels. The function validates itself: the OLS coefficients on
the demeaned data must reproduce the PanelOLS point estimates, and it raises if
they do not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distance in km."""
    la = np.radians(np.asarray(lat, float))
    lo = np.radians(np.asarray(lon, float))
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def two_way_demean(df: pd.DataFrame, cols, entity="district", time="season_year",
                   tol=1e-10, max_iter=200) -> np.ndarray:
    """Remove entity and time means by alternating projections.

    Exact for balanced panels; converges to the correct within transformation for
    unbalanced ones, where a single pass of subtract-both-means is biased.
    """
    X = df[cols].to_numpy(float).copy()
    e = df[entity].to_numpy()
    t = df[time].to_numpy()
    for _ in range(max_iter):
        prev = X.copy()
        X -= pd.DataFrame(X).groupby(e, sort=False).transform("mean").to_numpy()
        X -= pd.DataFrame(X).groupby(t, sort=False).transform("mean").to_numpy()
        if np.max(np.abs(X - prev)) < tol:
            break
    return X


def conley_se(panel: pd.DataFrame, regressors, outcome="phase",
              dist_km: float = 200.0, time_lag: int = 3,
              entity="district", time="season_year",
              lat_col="lat", lon_col="lon",
              validate_against=None, verbose=True) -> pd.DataFrame:
    """Conley spatial HAC standard errors for a two-way FE specification.

    Parameters
    ----------
    dist_km : spatial Bartlett cutoff. 200 km is a common default for weather
        shocks; report sensitivity to it, because the choice is not innocuous.
    time_lag : serial Bartlett lag in years. 0 = contemporaneous spatial only.
    validate_against : optional dict of {regressor: coefficient} from PanelOLS.
        If given, the demeaned-OLS coefficients must match it.
    """
    cols = list(regressors)
    d = panel.dropna(subset=[outcome] + cols + [lat_col, lon_col]).copy()
    d = d.sort_values([entity, time]).reset_index(drop=True)

    Y = two_way_demean(d, [outcome], entity, time).ravel()
    X = two_way_demean(d, cols, entity, time)

    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ Y)
    u = Y - X @ beta

    if validate_against:
        for k, v in validate_against.items():
            if k in cols:
                got = beta[cols.index(k)]
                if not np.isfinite(got) or abs(got - v) > 1e-4:
                    raise AssertionError(
                        f"Demeaned OLS does not reproduce PanelOLS for '{k}': "
                        f"{got:.6f} vs {v:.6f}. The within transformation is wrong.")
        if verbose:
            print(f"   Conley: within transformation validated against PanelOLS "
                  f"({len(validate_against)} coefficients match)")

    # Unique units and their centroids
    units = d[entity].to_numpy()
    uniq = pd.Index(pd.unique(units))
    coords = (d.groupby(entity, observed=True)[[lat_col, lon_col]].first()
                .reindex(uniq))
    D = haversine_matrix(coords[lat_col].to_numpy(), coords[lon_col].to_numpy())
    W_space_u = np.maximum(0.0, 1.0 - D / float(dist_km))     # Bartlett in space

    ui = uniq.get_indexer(units)
    W_space = W_space_u[np.ix_(ui, ui)]

    tvals = d[time].to_numpy(float)
    dt = np.abs(tvals[:, None] - tvals[None, :])
    if time_lag and time_lag > 0:
        W_time = np.maximum(0.0, 1.0 - dt / (float(time_lag) + 1.0))
    else:
        W_time = (dt == 0).astype(float)

    W = W_space * W_time
    meat = X.T @ (W * np.outer(u, u)) @ X
    V = XtX_inv @ meat @ XtX_inv

    # Small-sample correction, as in standard practice for clustered/HAC panels
    n, k = X.shape
    n_units = len(uniq)
    dof = max(n - k - n_units - d[time].nunique() + 1, 1)
    V *= n / dof

    se = np.sqrt(np.maximum(np.diag(V), 0))
    tstat = beta / np.where(se > 0, se, np.nan)
    from scipy import stats as _st
    p = 2 * (1 - _st.norm.cdf(np.abs(tstat)))

    out = pd.DataFrame({
        "term": cols, "coef": beta, "se_conley": se,
        "t": tstat, "p": p,
        "ci_lo": beta - 1.96 * se, "ci_hi": beta + 1.96 * se,
    })
    out.attrs["dist_km"] = dist_km
    out.attrs["time_lag"] = time_lag
    out.attrs["n"] = n
    return out


def conley_sensitivity(panel, regressors, outcome="phase",
                       cutoffs=(100, 150, 200, 300, 500), time_lag=3,
                       **kw) -> pd.DataFrame:
    """Conley SEs across spatial cutoffs.

    The cutoff is a researcher choice with no data-driven answer, so the defence
    is to show the inference does not hinge on it.
    """
    rows = []
    for c in cutoffs:
        try:
            r = conley_se(panel, regressors, outcome=outcome,
                          dist_km=c, time_lag=time_lag, verbose=False, **kw)
            for _, x in r.iterrows():
                rows.append({"dist_km": c, "term": x["term"], "coef": x["coef"],
                             "se_conley": x["se_conley"], "p": x["p"]})
        except Exception as e:
            rows.append({"dist_km": c, "term": "ERROR", "coef": np.nan,
                         "se_conley": np.nan, "p": np.nan, "note": str(e)[:70]})
    return pd.DataFrame(rows)
