# Compound drought and heat exposure and lean-season food insecurity in Zimbabwe

Analysis pipeline for:

> Compound drought and heat exposure and lean-season food insecurity in Zimbabwe:
> agro-ecological heterogeneity in a 2009-2025 district panel

District by season-year panel, Zimbabwe, 2009 to 2025 (84 district and urban-council
units, 1,342 district-seasons). Two-way fixed effects, nested model ladder,
leave-one-season-out validation, Conley spatial HAC standard errors, season-block
permutation inference.

## Layout

```
src/cdh/config.py       every analytical choice, in one place
src/cdh/io.py            schema-tolerant HFID loader, climate loaders, name checks
src/cdh/indices.py       Hargreaves PET, SPI, SPEI, hot days, compound terms
src/cdh/panel.py         district x season-year assembly
src/cdh/models.py        nested FE ladder, leave-one-season-out CV, robustness
src/cdh/spatial.py       Conley spatial HAC standard errors
src/cdh/placebo.py       season-block permutation test
src/cdh/strata.py        agro-ecological stratification and heterogeneity
src/cdh/scenarios.py     CMIP6 (SSP2-4.5 / SSP5-8.5) scenario projection
src/cdh/maps.py          choropleths on real geometry (geoBoundaries)
src/cdh/figures.py       manuscript figures
run_pipeline.py          the whole analysis
run_placebo_test.py      the season-block permutation test, standalone
make_synthetic.py        synthetic data generator with a known data-generating process,
                         used to validate the pipeline recovers known parameters
prepare_hfid.py          cuts the global HFID file down to one country
gee/                     Earth Engine extraction scripts (CHIRPS, ERA5-Land, CMIP6)
fill_manuscript.py       substitutes results into the manuscript template
export_docx.py           Word export with figures and tables inline
```

## Data

All input data is public:

- Harmonised food-insecurity classifications and FEWS NET administrative boundaries:
  World Bank Data Catalog, "Harmonized Sub-National Food Security Data" (entry 0064614),
  CC BY 4.0. https://datacatalog.worldbank.org/search/dataset/0064614
- CHIRPS v3 precipitation: https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHC_CHIRPS_V3_DAILY_RNL
- ERA5-Land reanalysis: https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR

Earth Engine access and the HFID/CHIRPS/ERA5-Land downloads need network access to
`earthengine.googleapis.com` and related hosts; the pipeline itself runs with no
external network once the inputs are in `data/raw/`.

## Quick start

```
pip install -r requirements.txt
python prepare_hfid.py          # cut the global HFID file to Zimbabwe
python run_pipeline.py          # full analysis on real data
python run_pipeline.py --synthetic   # verify against a known data-generating process
```

`run_pipeline.py --synthetic` is a self-check: `make_synthetic.py` embeds known true
parameters and the pipeline recovers them with the attenuation expected from rounding
the outcome to integer IPC phases.

## License

Code license: **not yet chosen** -- add a LICENSE file before the first public release
(MIT and Apache-2.0 are the common permissive choices for academic code; GitHub can add
one at repo-creation time). Data: see the licenses of the original sources linked above.
