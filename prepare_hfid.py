"""Cut the global HFID file down to one country.

The World Bank HFID release is a 52 MB global CSV. Filtered to one country and
to observed (current-situation) rows it is typically under 1 MB, which is small
enough to attach to a message or read through a Drive connector.

Run this once on a machine that has the big file:

    python prepare_hfid.py --input FEWS_June_2026_Update_TrueBoundaries_08-13-26.csv
    python prepare_hfid.py --input <file> --country Malawi --out mwi_hfid.csv

It streams the file in chunks, so it does not need 52 MB of memory, and it
reports what it found rather than assuming the schema.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cdh import config as C  # noqa: E402
from cdh.io import _resolve  # noqa: E402


def main(inp: Path, country: str, out: Path, keep_projections: bool, chunk: int):
    if not inp.exists():
        raise SystemExit(f"{inp} not found.")

    head = pd.read_csv(inp, nrows=5)
    cols = {}
    for k, v in C.HFID_COLUMNS.items():
        try:
            cols[k] = _resolve(head.columns, v, k)
        except KeyError:
            if k in C.HFID_OPTIONAL_FIELDS:
                continue
            raise
    print("Resolved columns:")
    for k, v in cols.items():
        print(f"   {k:11s} -> {v}")

    country_col = cols["country"]
    scen_col = cols.get("scenario")
    usecols = list(dict.fromkeys(cols.values()))

    kept, seen, scen_seen = [], 0, set()
    for part in pd.read_csv(inp, usecols=usecols, chunksize=chunk, low_memory=False):
        seen += len(part)
        sel = part[part[country_col].astype(str).str.strip().str.lower()
                   == country.lower()]
        if scen_col and len(sel):
            scen_seen.update(sel[scen_col].astype(str).str.strip().unique())
            if not keep_projections:
                s = sel[scen_col].astype(str).str.strip().str.lower()
                sel = sel[s.isin([k.lower() for k in C.HFID_SCENARIO_KEEP])]
        if len(sel):
            kept.append(sel)

    if not kept:
        raise SystemExit(
            f"No rows for country={country!r} in {seen:,} scanned. "
            f"Check the spelling used in the file.")

    df = pd.concat(kept, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    mb_in = inp.stat().st_size / 1e6
    mb_out = out.stat().st_size / 1e6
    print(f"\nScanned {seen:,} rows, kept {len(df):,} for {country}.")
    if scen_col:
        print(f"Scenario values present for this country: {sorted(scen_seen)}")
        if not keep_projections:
            print("Kept current-situation rows only (projections dropped).")
    print(f"{inp.name}  {mb_in:.1f} MB  ->  {out.name}  {mb_out:.2f} MB "
          f"({mb_out / mb_in * 100:.1f}% of the original)")
    print(f"\nThat file is small enough to attach to a message or drop in Drive.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="the global HFID CSV from the World Bank Data Catalog")
    ap.add_argument("--country", default=C.COUNTRY)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--keep-projections", action="store_true",
                    help="keep ML1/ML2 forecast rows (not used as an outcome)")
    ap.add_argument("--chunksize", type=int, default=200_000)
    a = ap.parse_args()
    out = a.out or Path(f"{a.country.lower().replace(' ', '_')}_hfid.csv")
    main(a.input, a.country, out, a.keep_projections, a.chunksize)
