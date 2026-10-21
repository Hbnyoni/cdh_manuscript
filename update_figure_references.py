"""One-time migration of manuscript callouts to stable asset identifiers."""
from pathlib import Path
import re

path = Path(__file__).parent / "manuscript" / "manuscript.md"
text = path.read_text(encoding="utf-8")
stems = ["map1_study_area", "fig0_national_trend", "map4_trends",
         "map_rainfall_periods", "map_temperature_periods", "fig_study_design",
         "map_moisture_deficit_periods", "map_heat_periods", "fig1_coefficients",
         "fig2_marginal_effect", "fig7_binned_interaction", "fig3_categories",
         "map2_exposure_outcome", "map3_hazard_frequency", "map_phase_periods",
         "map_compound_periods", "fig8_spacetime", "fig4_cv", "map5_strata",
         "fig5_heterogeneity", "fig6_conley", "map_scenario_delta", "fig9_scenario_burden"]
tables = ["coefficients", "cv", "robustness", "heterogeneity", "conley_sensitivity",
          "rainfall_sensitivity", "scenario_burden"]
if "[[FIG:map0_locator]]" not in text:
    text = re.sub(r"Figures (\d+) and (\d+)", r"Figure \1 and Figure \2", text)
    text = re.sub(r"Figure (\d+)", lambda m: "[[FIG:" + stems[int(m[1])-1] + "]]", text)
    text = re.sub(r"Table (\d+)", lambda m: "[[TAB:" + tables[int(m[1])-1] + "]]", text)
    text = text.replace("Zimbabwe covers approximately", "Zimbabwe is situated in southern Africa, bordered by Zambia, Mozambique, South Africa and Botswana ([[FIG:map0_locator]]). Zimbabwe covers approximately", 1)
    text = text.replace("[[FIG:map1_study_area]] shows the rainfall and temperature gradients the rest of the paper works against.",
                        "District rainfall and temperature gradients provide the climatic context for the analysis ([[FIG:map1_study_area]]).")
    path.write_text(text, encoding="utf-8")
    print("Migrated 24 figure and 7 table callouts to stable identifiers.")
else:
    print("Stable callouts already present; no changes.")
