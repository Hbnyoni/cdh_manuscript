/**
 * Zimbabwe district climate extraction for the compound drought-heat analysis.
 * Paste into the Earth Engine Code Editor (https://code.earthengine.google.com)
 * and run. Exports two CSVs to your Google Drive folder "cdh_zimbabwe".
 *
 * WHY EARTH ENGINE: the alternative is downloading roughly 40 GB of CHIRPS and
 * ERA5-Land rasters to extract about 4 MB of district means. The reduction is
 * done server-side; only the table leaves.
 *
 * BOUNDARIES - READ THIS FIRST
 * The outcome variable (IPC phase) is measured on FEWS NET units, so the climate
 * MUST be extracted on those same units or the join is unsafe. Steps:
 *   1. Download "FEWS NET Admin Boundaries.zip" (see fetch_data.sh).
 *   2. Unzip, take the Zimbabwe admin-2 layer.
 *   3. Assets -> New -> Shape files, upload it.
 *   4. Put the asset path in BOUNDARY_ASSET below and set NAME_FIELD.
 * FAO GAUL is left as a fallback so the script runs before you have done that,
 * but results from the fallback are for checking the plumbing only: GAUL and
 * FEWS NET units do not align, and run_pipeline.py will report the mismatches.
 */

// ----------------------------------------------------------------- settings
var BOUNDARY_ASSET = '';            // e.g. 'users/you/fewsnet_zw_admin2'
var NAME_FIELD     = 'admin2';      // the district-name field in that asset
var START          = '1981-01-01';
var END            = '2026-07-31';  // CHIRPS v3 DAILY_RNL currently ends 2026-07-31
var GS_MONTHS      = [10, 11, 12, 1, 2, 3];
var DRIVE_FOLDER   = 'cdh_zimbabwe';
var SCALE          = 5566;          // CHIRPS native, ~0.05 deg

// ----------------------------------------------------------------- boundaries
var districts;
if (BOUNDARY_ASSET) {
  districts = ee.FeatureCollection(BOUNDARY_ASSET);
} else {
  print('WARNING: using FAO GAUL fallback. Units will NOT match the HFID outcome.');
  districts = ee.FeatureCollection('FAO/GAUL/2015/level2')
                .filter(ee.Filter.eq('ADM0_NAME', 'Zimbabwe'));
  NAME_FIELD = 'ADM2_NAME';
}
districts = districts.map(function (f) {
  var c = f.geometry().centroid(1000).coordinates();
  return f.set({
    district: ee.String(f.get(NAME_FIELD)),
    lat: ee.Number(c.get(1)),
    lon: ee.Number(c.get(0))
  });
});
print('Districts:', districts.size(), districts.limit(3));
Map.addLayer(districts, {color: 'c8622a'}, 'Districts');
Map.centerObject(districts, 6);

// ----------------------------------------------------------------- sources
// CHIRPS v3 daily reanalysis: ERA5 used only to partition CHIRPS pentadal totals
// into days, so the monthly totals are CHIRPS, not reanalysis rainfall.
var chirps = ee.ImageCollection('UCSB-CHC/CHIRPS/V3/DAILY_RNL')
               .select('precipitation').filterDate(START, END);

var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
             .select(['temperature_2m_max', 'temperature_2m_min'])
             .filterDate(START, END);

// Alternate rainfall product for the sensitivity analysis. ERA5-Land precipitation
// is model reanalysis: independent of CHIRPS in lineage, but wet-biased over
// southern Africa. It tests robustness to product choice, not accuracy. For a
// gauge-calibrated independent check use TAMSAT v3, which has no Earth Engine
// asset and must be downloaded from the University of Reading.
var era5p = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
              .select('total_precipitation_sum').filterDate(START, END);

var months = ee.List.sequence(0, ee.Date(END).difference(ee.Date(START), 'month').floor());

// ----------------------------------------------------- Table 1: monthly panel
var monthly = ee.FeatureCollection(months.map(function (m) {
  var t0 = ee.Date(START).advance(m, 'month');
  var t1 = t0.advance(1, 'month');

  var pr = chirps.filterDate(t0, t1).sum().rename('precip_mm');
  var tt = era5.filterDate(t0, t1).mean()
               .subtract(273.15)                       // K -> C
               .rename(['tmax_c', 'tmin_c']);

  var pr_alt = era5p.filterDate(t0, t1).sum()
                    .multiply(1000)                        // m -> mm
                    .rename('precip_alt_mm');

  var img = pr.addBands(tt).addBands(pr_alt);
  return img.reduceRegions({
    collection: districts,
    reducer: ee.Reducer.mean(),
    scale: SCALE
  }).map(function (f) {
    return ee.Feature(null, {
      district:      f.get('district'),
      lat:           f.get('lat'),
      lon:           f.get('lon'),
      date:          t0.format('YYYY-MM-01'),
      precip_mm:     f.get('precip_mm'),
      precip_alt_mm: f.get('precip_alt_mm'),
      tmax_c:        f.get('tmax_c'),
      tmin_c:        f.get('tmin_c')
    });
  });
})).flatten();

Export.table.toDrive({
  collection: monthly,
  description: 'zw_monthly_climate',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'zw_monthly_climate',
  fileFormat: 'CSV',
  selectors: ['district', 'date', 'precip_mm', 'precip_alt_mm',
              'tmax_c', 'tmin_c', 'lat', 'lon']
});

// --------------------------------- Table 2: season hot-day counts (COMPACT)
// The earlier version of this script exported district x DAY Tmax, about 15 MB.
// That is fine on a workstation but too large to move through a chat attachment
// or a Drive read, which is how this analysis actually gets its data.
//
// So the exceedance counting is done server-side here and only the season totals
// leave: 75 districts x 45 seasons is roughly 2,600 rows, about 120 KB.
//
// The cost is that the hot-day threshold moves from Python into Earth Engine.
// To keep the definition explorable without re-running GEE, counts are exported
// at THREE percentiles (85th, 90th, 95th) of each district's own growing-season
// daily Tmax over the 1991-2020 baseline. The pipeline uses the 90th by default
// and the other two are there for sensitivity analysis.

var BASE_START = ee.Date.fromYMD(1991, 1, 1);
var BASE_END   = ee.Date.fromYMD(2021, 1, 1);
var PCTLS      = [85, 90, 95];
var SEASONS    = ee.List.sequence(1991, 2026);

var gsFilter = ee.Filter.or(
  ee.Filter.calendarRange(10, 12, 'month'),
  ee.Filter.calendarRange(1, 3, 'month'));

var tmaxDaily = era5.select('temperature_2m_max').subtract(273.15);

// District-level baseline percentiles of growing-season daily Tmax
var baseline = tmaxDaily.filterDate(BASE_START, BASE_END).filter(gsFilter);
var thresholds = ee.Image(baseline.toBands()).reduceRegions({
  collection: districts,
  reducer: ee.Reducer.percentile(PCTLS),
  scale: 11132
});
// reduceRegions over a banded stack is unwieldy; reduce per district instead
var thrByDistrict = districts.map(function (f) {
  var vals = baseline.map(function (img) {
    return ee.Feature(null, img.reduceRegion({
      reducer: ee.Reducer.mean(), geometry: f.geometry(), scale: 11132,
      maxPixels: 1e9, bestEffort: true
    }));
  }).aggregate_array('temperature_2m_max');
  var arr = ee.Array(ee.List(vals).removeAll([null]));
  var pct = ee.List(PCTLS).map(function (q) {
    return arr.reduce(ee.Reducer.percentile([ee.Number(q)]), [0]).get([0]);
  });
  return f.set({thr85: pct.get(0), thr90: pct.get(1), thr95: pct.get(2)});
});

var seasonHot = ee.FeatureCollection(SEASONS.map(function (y) {
  y = ee.Number(y);
  var t0 = ee.Date.fromYMD(y.subtract(1), 10, 1);
  var t1 = ee.Date.fromYMD(y, 4, 1);
  var col = tmaxDaily.filterDate(t0, t1);

  return thrByDistrict.map(function (f) {
    var series = col.map(function (img) {
      return ee.Feature(null, img.reduceRegion({
        reducer: ee.Reducer.mean(), geometry: f.geometry(), scale: 11132,
        maxPixels: 1e9, bestEffort: true
      }));
    }).aggregate_array('temperature_2m_max');
    var clean = ee.List(series).removeAll([null]);
    var arr = ee.Array(clean);

    var count = function (thr) {
      return arr.gt(ee.Number(thr)).reduce(ee.Reducer.sum(), [0]).get([0]);
    };
    return ee.Feature(null, {
      district:      f.get('district'),
      season_year:   y,
      n_days:        clean.size(),
      tmax_mean:     arr.reduce(ee.Reducer.mean(), [0]).get([0]),
      hot_days_p85:  count(f.get('thr85')),
      hot_days_p90:  count(f.get('thr90')),
      hot_days_p95:  count(f.get('thr95'))
    });
  });
})).flatten();

Export.table.toDrive({
  collection: seasonHot,
  description: 'zw_season_hotdays',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'zw_season_hotdays',
  fileFormat: 'CSV',
  selectors: ['district', 'season_year', 'n_days', 'tmax_mean',
              'hot_days_p85', 'hot_days_p90', 'hot_days_p95']
});

print('Two exports to run: zw_monthly_climate (~3 MB) and zw_season_hotdays (~120 KB).');
print('Total is small enough to attach to a chat or read from Drive directly.');
