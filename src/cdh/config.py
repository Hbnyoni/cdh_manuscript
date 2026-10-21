"""Configuration for the Zimbabwe compound drought-heat / food insecurity analysis.

Everything a reviewer would want to change lives here, not scattered in the code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
for _p in (RAW, PROC, FIG, TAB):
    _p.mkdir(parents=True, exist_ok=True)

COUNTRY = "Zimbabwe"
# ISO3 code, used to fetch admin boundaries from geoBoundaries. Changing COUNTRY
# and COUNTRY_ISO3 together is all that is needed to move this analysis to
# another of the ~20 countries HFID covers (see the season settings below, which
# DO need changing for bimodal-rainfall countries).
COUNTRY_ISO3 = "ZWE"

# --- Season definition -------------------------------------------------------
# Zimbabwe has a single rainfed season. The agricultural year labelled Y runs
# from October of Y-1 through March of Y (harvest Apr-May of Y). Food insecurity
# outcomes are read in the lean season that FOLLOWS that harvest, i.e. Jan-Mar of
# Y+1, which is the period the harvest of year Y has to carry the household through.
GROWING_SEASON_MONTHS = [10, 11, 12, 1, 2, 3]   # Oct(Y-1) .. Mar(Y)
SEASON_END_MONTH = 3                             # SPEI/SPI accumulation ends here
LEAN_SEASON_MONTHS = [1, 2, 3]                   # outcome window, in year Y+1

# --- Index parameters --------------------------------------------------------
SPEI_SCALE = 6           # months; Oct-Mar accumulation
SPI_SCALE = 6
CLIM_BASELINE = (1991, 2020)   # WMO standard normal for the hot-day threshold
HOT_PCTL = 90            # percentile of growing-season daily Tmax defining a hot day

# Binary compound event thresholds
DROUGHT_THRESHOLD = -0.8     # SPEI-6 below this = drought (moderate, McKee scale)
HOT_SEASON_PCTL = 80         # season hot-day count above this within-district pctl

# Standardisation method for SPI/SPEI: "empirical" (Gringorten plotting position,
# robust for ~45-year records) or "loglogistic" (parametric, classical SPEI).
STANDARDISE = "empirical"

# --- Analysis ----------------------------------------------------------------
MIN_YEARS_PER_DISTRICT = 8     # drop districts with sparse outcome coverage
CLUSTER = "district"           # SE clustering
DROP_COVID_YEARS = [2020, 2021]  # for a robustness specification only

# --- Input files -------------------------------------------------------------
HFID_CSV = RAW / "FEWS_June_2026_Update_TrueBoundaries_08-13-26.csv"
CLIM_MONTHLY_CSV = RAW / "zw_monthly_climate.csv"
CLIM_DAILY_TMAX_CSV = RAW / "zw_daily_tmax_gs.csv"      # optional, large
CLIM_HOTDAYS_CSV = RAW / "zw_season_hotdays.csv"        # preferred, compact

# Which percentile column to use from the compact hot-days table. The export
# carries 85/90/95 so the definition can be varied without re-running GEE.
HOT_PCTL_COL = "hot_days_p90"

# Optional: district -> Natural Region lookup for the heterogeneity analysis.
# Columns: district,natural_region (I..V). If absent, the pipeline falls back to
# aridity terciles computed from the CHIRPS climatology and says so.
NATURAL_REGION_CSV = RAW / "zw_natural_regions.csv"

BOUNDARY_DIR = ROOT / "data" / "boundaries"
BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)

# --- Rainfall products -------------------------------------------------------
# Primary is CHIRPS v3. The alternate is an independent product used only for a
# sensitivity analysis; if the Earth Engine export supplies it, the whole ladder
# is re-run on it. ERA5-Land precipitation is model reanalysis and is known to be
# wet-biased over southern Africa, so it tests robustness to product LINEAGE, not
# accuracy. For a gauge-calibrated independent check, download TAMSAT v3 from the
# University of Reading (no Earth Engine asset exists) and supply it as this column.
PRECIP_COL = "precip_mm"
PRECIP_ALT_COL = "precip_alt_mm"

# --- Conley spatial HAC ------------------------------------------------------
CONLEY_DIST_KM = 200.0     # Bartlett cutoff; sensitivity reported across 100-500
CONLEY_TIME_LAG = 3        # years of serial correlation allowed
CONLEY_CUTOFFS = (100, 150, 200, 300, 500)

# --- HFID schema tolerance ---------------------------------------------------
# The FEWS NET tabular export has changed column names between releases, and this
# pipeline cannot see the file until you download it. Each entry below is an
# ordered list of candidate column names; the loader takes the first that exists
# and PRINTS what it chose so you can verify it. Override here if it guesses wrong.
HFID_COLUMNS = {
    "country":    ["country", "COUNTRY", "country_name", "adm0_name"],
    "unit_id":    ["fnid", "FNID", "unit_id", "area_id", "admin_code"],
    "unit_name":  ["admin2", "ADMIN2", "admin_name", "area_name", "adm2_name", "name"],
    "date":       ["reporting_date", "collection_date", "date", "start_date",
                   "period_date", "year_month"],
    "scenario":   ["scenario", "period", "product", "classification_type"],
    "phase":      ["fews_ipc", "overall_phase", "ipc_phase", "phase",
                   "classification", "value"],
    "population": ["population", "pop", "total_population", "population_estimate"],
}
# Fields that a given HFID release may simply not have. The 2026 World Bank HFID
# release is wide (current situation, near-term and medium-term projections as
# separate columns) rather than long (one phase column plus a scenario
# discriminator), so there is no "scenario" column at all: picking "phase" ->
# "fews_ipc" above already selects current-situation only, and there is nothing
# left for scenario to filter.
HFID_OPTIONAL_FIELDS = {"scenario"}

# Which scenario rows to keep, for releases that DO use the long format with an
# explicit scenario column. "CS" = current situation (observed), as opposed to
# ML1/ML2 which are forward projections and must not be used as an outcome.
HFID_SCENARIO_KEEP = ["CS", "current", "Current Situation", "current situation"]

# --- HFID reporting-unit district names, real-data normalisation ------------
# The 2026 World Bank HFID release reports at FEWS NET livelihood-zone-within-
# district granularity: unit_name is a string like
# "Agrofisheries, Binga, Matabeleland North, Zimbabwe". io.load_hfid takes the
# third-from-last comma-separated token as the district. That token mostly
# matches the geoBoundaries admin2 names used for climate extraction and maps,
# except for a handful of Rural/Urban council naming differences and one merged
# district, checked by hand against data/boundaries/zwe_adm2.geojson. Anything
# mapped to None is not a real administrative district and is dropped.
HFID_DISTRICT_NAME_FIXES = {
    "Beitbridge Rural": "Beitbridge",
    "Bindura Rural": "Bindura",
    "Centenary": "Centenary/ Muzarabani",
    "Chegutu Rural": "Chegutu",
    "Chinhoyi Urban": "Chinhoyi",
    "Chipinge Rural": "Chipinge",
    "Chiredzi Rural": "Chiredzi",
    "Gokwe South Rural": "Gokwe South",
    "Gwanda Rural": "Gwanda",
    "Gweru Rural": "Gweru",
    "Harare Urban": "Harare",
    "Hwange Rural": "Hwange",
    "Karoi Urban": "Karoi",
    "Kwekwe Rural": "Kwekwe",
    "Masvingo Rural": "Masvingo",
    "Mutare Rural": "Mutare",
    "Rusape Urban": "Rusape",
    "Shurugwi Rural": "Shurugwi",
    "Shurugwi Urban": "Shurugwi Town",
    "Zvishavane Rural": "Zvishavane",
    "parks": None,
}
