"""Randomization inference on the compound interaction: a one-off, slow
check (roughly 2 seconds per permutation, so minutes for a few hundred),
kept out of run_pipeline.py the same way gee/extract_cmip6_scenarios.py is:
run it once, cache the result, let the main pipeline read the cache.

  python run_placebo_test.py             # real data, 500 permutations
  python run_placebo_test.py --n 1000
  python run_placebo_test.py --synthetic
  python run_placebo_test.py --block --n 1000            # season-block scheme
  python run_placebo_test.py --block --stratum "Driest third"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cdh import config as C, placebo, strata  # noqa: E402


def main(synthetic: bool, n_perm: int, seed: int, block: bool = False, stratum=None):
    tag = "synthetic" if synthetic else "real"
    panel = pd.read_csv(C.PROC / f"panel_{tag}.csv")
    stem = "placebo_block" if block else "placebo"
    if stratum:
        panel, _ = strata.assign_strata(panel, verbose=False)
        panel = panel[panel["stratum"].astype(str) == stratum].copy()
        panel["district"] = panel["district"].astype(str)
        stem += "_" + stratum.split()[0].lower()
    test = placebo.block_permutation_test if block else placebo.permutation_test
    true_coef, perm_coefs, p_value = test(panel, n_perm=n_perm, seed=seed)

    perm_name = f"{stem}_permutation_{tag}.csv" if block or stratum else f"placebo_permutation_{tag}.csv"
    pd.DataFrame({"perm_coef": perm_coefs}).to_csv(C.TAB / perm_name, index=False)
    summary = pd.DataFrame([{
        "true_coef": true_coef, "n_valid": len(perm_coefs), "p_value": p_value,
        "null_min": float(perm_coefs.min()), "null_max": float(perm_coefs.max()),
        "null_mean": float(perm_coefs.mean()), "null_std": float(perm_coefs.std()),
    }])
    summary.to_csv(C.TAB / f"{stem}_summary_{tag}.csv", index=False)
    print(f"Wrote {C.TAB / perm_name} and {C.TAB / f'{stem}_summary_{tag}.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", action="store_true",
                    help="permute whole season-years jointly across districts")
    ap.add_argument("--stratum", default=None, help='e.g. "Driest third"')
    a = ap.parse_args()
    main(a.synthetic, a.n, a.seed, a.block, a.stratum)
