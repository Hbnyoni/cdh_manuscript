#!/usr/bin/env bash
# Download manifest for the compound drought-heat / food insecurity analysis.
#
# Run this on a machine with normal internet access (not in the Claude sandbox,
# whose egress policy blocks these hosts). Total download is about 122 MB.
#
#   bash fetch_data.sh
#
# Then follow gee/extract_zimbabwe_climate.js for the two climate tables, which
# are produced server-side in Earth Engine rather than downloaded as rasters.

set -euo pipefail
DEST="$(cd "$(dirname "$0")" && pwd)/data/raw"
mkdir -p "$DEST"
cd "$DEST"

echo "Downloading to $DEST"

# ---------------------------------------------------------------------------
# 1. OUTCOME: Harmonized sub-national food security data (HFID)
#    World Bank Data Catalog entry 0064614. IPC-compatible phases and population
#    by sub-national unit, harmonised across IPC, Cadre Harmonise and FEWS NET.
#    Licence: CC BY 4.0. Updated through June 2026.
# ---------------------------------------------------------------------------
echo "[1/4] HFID phase table (52 MB)"
curl -fL --retry 3 -o "FEWS_June_2026_Update_TrueBoundaries_08-13-26.csv" \
  "https://datacatalogfiles.worldbank.org/ddh-published/0064614/DR0091743/FEWS_June_2026_Update_TrueBoundaries_08-13-26.csv"

echo "[2/4] HFID column descriptions (11 KB)"
curl -fL --retry 3 -o "HFID_column_descriptions.xlsx" \
  "https://datacatalogfiles.worldbank.org/ddh-published/0064614/DR0091811/Column%20Descriptions.xlsx"

# ---------------------------------------------------------------------------
# 2. GEOMETRY: FEWS NET admin boundaries
#    These are the units the outcome is measured on. Upload the Zimbabwe admin-2
#    layer to Earth Engine as an asset and extract the climate against it, so the
#    exposure and the outcome share a geometry exactly.
# ---------------------------------------------------------------------------
echo "[3/4] FEWS NET admin boundaries (70 MB)"
curl -fL --retry 3 -o "FEWS_NET_Admin_Boundaries.zip" \
  "https://datacatalogfiles.worldbank.org/ddh-published/0064614/DR0091744/FEWS%20NET%20Admin%20Boundaries.zip"
unzip -o -q "FEWS_NET_Admin_Boundaries.zip" -d fewsnet_boundaries || \
  echo "  ! unzip failed; extract manually"

# ---------------------------------------------------------------------------
# 3. CONTEXT (optional): ZimLAC Rural Livelihoods Assessment 2025
#    PDF only. Used for the study-area description and to sanity-check the HFID
#    series against the national assessment, not as a model input.
# ---------------------------------------------------------------------------
echo "[4/5] ZimLAC 2025 Rural Livelihoods Assessment (PDF, optional)"
curl -fL --retry 2 -o "ZimLAC_2025_RLA.pdf" \
  "https://www.fnc.org.zw/wp-content/uploads/2025/09/2025-ZimLAC-Rural-Livelihoods-Assessment-Report_web.pdf" \
  || echo "  ! optional download failed, continuing"

# ---------------------------------------------------------------------------
# 4. DISPLAY GEOMETRY: geoBoundaries open admin boundaries (CC BY 4.0)
#    Used for the maps only. Served through GitHub's LFS media endpoint.
# ---------------------------------------------------------------------------
# ISO3 must match COUNTRY_ISO3 in src/cdh/config.py. Note the media. host: the
# raw. host returns a 132-byte Git LFS pointer rather than the data.
ISO3="${ISO3:-ZWE}"
iso3_lc=$(echo "$ISO3" | tr "A-Z" "a-z")
echo "[5/5] geoBoundaries $ISO3 ADM1 and ADM2 (~1.2 MB)"
mkdir -p "$DEST/../boundaries"
GB="https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/main/releaseData/gbOpen/$ISO3"
for LVL in 1 2; do
  curl -fL --retry 3 -o "$DEST/../boundaries/${iso3_lc}_adm${LVL}.geojson" \
    "$GB/ADM${LVL}/geoBoundaries-${ISO3}-ADM${LVL}_simplified.geojson" \
    || echo "  ! ADM${LVL} download failed"
done

echo
echo "Done. Now:"
echo "  1. Upload the Zimbabwe admin-2 layer from fewsnet_boundaries/ to Earth Engine."
echo "  2. Set BOUNDARY_ASSET in gee/extract_zimbabwe_climate.js and run all exports."
echo "  3. Put zw_monthly_climate.csv and zw_daily_tmax_gs.csv in data/raw/."
echo "  4. python run_pipeline.py"
