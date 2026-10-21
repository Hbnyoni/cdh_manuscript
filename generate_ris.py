"""Export manuscript.md's reference list as a .ris file for import into
Zotero or Mendeley (File > Import in either).

Kept as a small standalone list here, matching the References section in
manuscript.md exactly, rather than parsed out of the markdown: the markdown
is prose meant for a human reader and a co-author to edit directly, and
round-tripping it through a parser risks a silent mismatch that is easy to
miss. Update both places together if a reference changes.

  python generate_ris.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ARTICLES = [
    # (authors, year, title, journal, volume, issue, pages, doi)
    (["AghaKouchak, Amir", "Chiang, Felicia", "Huning, Laurie S.", "Love, Charlotte A.",
      "Mallakpour, Iman", "Mazdiyasni, Omid", "Moftakhari, Hamed", "Papalexiou, Simon Michael",
      "Ragno, Elisa", "Sadegh, Mojtaba"],
     2020, "Climate extremes and compound hazards in a warming world",
     "Annual Review of Earth and Planetary Sciences", "48", "", "519-548",
     "10.1146/annurev-earth-071719-055228"),
    (["Cohen, Idan", "Zandalinas, Sara I.", "Huck, Christian", "Fritschi, Felix B.",
      "Mittler, Ron"],
     2021, "Meta-analysis of drought and heat stress combination impact on crop yield and yield components",
     "Physiologia Plantarum", "171", "1", "66-76", "10.1111/ppl.13203"),
    (["Funk, Chris", "Peterson, Pete", "Landsfeld, Martin", "Pedreros, Diego", "Verdin, James",
      "Shukla, Shraddhanand", "Husak, Gregory", "Rowland, James", "Harrison, Laura",
      "Hoell, Andrew", "Michaelsen, Joel"],
     2015, "The climate hazards infrared precipitation with stations: A new environmental record for monitoring extremes",
     "Scientific Data", "2", "", "150066", "10.1038/sdata.2015.66"),
    (["Funk, Chris", "Peterson, Pete", "Harrison, Laura", "Saldivar, Rocio", "Landsfeld, Martin",
      "Pedreros, Diego", "Shukla, Shraddhanand", "Fink, Andreas H.", "Davenport, Frank",
      "Peterson, Seth", "Turner, Will", "Sonnier, Alex", "Budde, Michael", "Tabor, Karen",
      "Verdin, James", "Hauzaree, Dilkushi", "Naim, Mizna", "Alaso, Diriba", "Husak, Gregory"],
     2026, "The Climate Hazards Center Infrared Precipitation with Stations, Version 3",
     "Scientific Data", "13", "", "718", "10.1038/s41597-026-07096-4"),
    (["Hargreaves, George H.", "Samani, Zohrab A."],
     1985, "Reference crop evapotranspiration from temperature",
     "Applied Engineering in Agriculture", "1", "2", "96-99", "10.13031/2013.26773"),
    (["Machefer, Myriam", "Ronco, Marc", "Thomas, Anne-Claire", "Assouline, Michael",
      "Rabier, Marion", "Corbane, Christina", "Rembold, Felix"],
     2025, "A monthly sub-national Harmonized Food Insecurity Dataset for comprehensive analysis and predictive modeling",
     "Scientific Data", "12", "", "741", "10.1038/s41597-025-05034-4"),
    (["Maidment, Ross I.", "Quaife, Tristan", "Pinnington, Ewan", "Black, Emily",
      "Ejigu, Adisu", "Kumar, Dhruv", "Thomas, Stephen"],
     2026, "A new, long-term root zone soil moisture dataset for operational agricultural drought monitoring over Africa",
     "Scientific Data", "13", "", "260", "10.1038/s41597-026-06585-w"),
    (["Manatsa, Desmond", "Mushore, Terrence", "Lenouo, Andre"],
     2017, "Improved predictability of droughts over southern Africa using the standardized precipitation evapotranspiration index and ENSO",
     "Theoretical and Applied Climatology", "127", "1-2", "259-274", "10.1007/s00704-015-1632-6"),
    (["Muñoz-Sabater, Joaquín", "Dutra, Emanuel", "Agustí-Panareda, Anna", "Albergel, Clément",
      "Arduini, Gabriele", "Balsamo, Gianpaolo", "Boussetta, Souhail", "Choulga, Margarita",
      "Harrigan, Shaun", "Hersbach, Hans", "Martens, Brecht", "Miralles, Diego G.",
      "Piles, Maria", "Rodríguez-Fernández, Nemesio J.", "Zsoter, Ervin", "Buontempo, Carlo",
      "Thépaut, Jean-Noël"],
     2021, "ERA5-Land: A state-of-the-art global reanalysis dataset for land applications",
     "Earth System Science Data", "13", "", "4349-4383", "10.5194/essd-13-4349-2021"),
    (["Peng, Jian", "Dadson, Simon", "Hirpa, Feyera", "Dyer, Ellen", "Lees, Thomas",
      "Miralles, Diego G.", "Vicente-Serrano, Sergio M.", "Funk, Chris"],
     2020, "A pan-African high-resolution drought index dataset",
     "Earth System Science Data", "12", "", "753-769", "10.5194/essd-12-753-2020"),
    (["Schlenker, Wolfram", "Lobell, David B."],
     2010, "Robust negative impacts of climate change on African agriculture",
     "Environmental Research Letters", "5", "1", "014010", "10.1088/1748-9326/5/1/014010"),
    (["Vicente-Serrano, Sergio M.", "Beguería, Santiago", "López-Moreno, Juan I."],
     2010, "A multiscalar drought index sensitive to global warming: The standardized precipitation evapotranspiration index",
     "Journal of Climate", "23", "7", "1696-1718", "10.1175/2009JCLI2909.1"),
    (["Zscheischler, Jakob", "Westra, Seth", "van den Hurk, Bart J. J. M.",
      "Seneviratne, Sonia I.", "Ward, Philip J.", "Pitman, Andy", "AghaKouchak, Amir",
      "Bresch, David N.", "Leonard, Michael", "Wahl, Thomas", "Zhang, Xuebin"],
     2018, "Future climate risk from compound events",
     "Nature Climate Change", "8", "6", "469-477", "10.1038/s41558-018-0156-3"),
    (["Zscheischler, Jakob", "Martius, Olivia", "Westra, Seth", "Bevacqua, Emanuele",
      "Raymond, Colin", "Horton, Radley M.", "van den Hurk, Bart", "AghaKouchak, Amir",
      "Jézéquel, Aglaé", "Mahecha, Miguel D.", "Maraun, Douglas", "Ramos, Alexandre M.",
      "Ridder, Nina N.", "Thiery, Wim", "Vignotto, Edoardo"],
     2020, "A typology of compound weather and climate events",
     "Nature Reviews Earth & Environment", "1", "", "333-347", "10.1038/s43017-020-0060-z"),
]

CONFERENCE = [
    (["McKee, Thomas B.", "Doesken, Nolan J.", "Kleist, John"],
     1993, "The relationship of drought frequency and duration to time scales",
     "Proceedings of the 8th Conference on Applied Climatology", "179-184"),
]

REPORTS = [
    (["Zimbabwe Livelihoods Assessment Committee"],
     2025, "2025 Rural Livelihoods Assessment Report", "Food and Nutrition Council"),
]


def _authors(names):
    return "\n".join(f"AU  - {n}" for n in names)


def build():
    entries = []
    for authors, year, title, journal, vol, issue, pages, doi in ARTICLES:
        sp, _, ep = pages.partition("-")
        lines = ["TY  - JOUR", _authors(authors), f"PY  - {year}", f"TI  - {title}",
                f"JO  - {journal}"]
        if vol:
            lines.append(f"VL  - {vol}")
        if issue:
            lines.append(f"IS  - {issue}")
        if sp:
            lines.append(f"SP  - {sp}")
        if ep:
            lines.append(f"EP  - {ep}")
        lines.append(f"DO  - {doi}")
        lines.append("ER  - ")
        entries.append("\n".join(lines))

    for authors, year, title, source, pages in CONFERENCE:
        sp, _, ep = pages.partition("-")
        lines = ["TY  - CONF", _authors(authors), f"PY  - {year}", f"TI  - {title}",
                f"T2  - {source}"]
        if sp:
            lines.append(f"SP  - {sp}")
        if ep:
            lines.append(f"EP  - {ep}")
        lines.append("ER  - ")
        entries.append("\n".join(lines))

    for authors, year, title, publisher in REPORTS:
        lines = ["TY  - RPRT", _authors(authors), f"PY  - {year}", f"TI  - {title}",
                f"PB  - {publisher}", "ER  - "]
        entries.append("\n".join(lines))

    out = ROOT / "submission" / "references.ris"
    out.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    print(f"Wrote {out}  ({len(entries)} entries)")
    print("Import into Zotero or Mendeley via File > Import.")


if __name__ == "__main__":
    build()
