"""Refresh submission assets in exactly the manuscript's numbered order.

Run after run_pipeline.py, fill_manuscript.py and export_docx.py.
Existing numbered figure files are archived outside the submission folder.
"""
from pathlib import Path
from datetime import datetime
import shutil
import hashlib
import json

from export_docx import (C, FIGURES, TABLES, ordered_entries, resolve_references,
                         _split_main_supp)


def main():
    root = C.ROOT
    dest = root / "submission"
    text = (root / "manuscript/manuscript_filled_real.md").read_text(encoding="utf-8")
    figures = ordered_entries(text, FIGURES, C.FIG, "png", "real")
    tables = ordered_entries(text, TABLES, C.TAB, "csv", "real")
    resolve_references(text, figures, tables)  # fail before touching the package
    fig_main, fig_supp = _split_main_supp(figures)

    archive = root / "outputs/submission_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    for folder, pattern in (("figures", "Figure_*.png"), ("tables", "Table_*.csv")):
        for old in (dest / folder).glob(pattern):
            target = archive / folder / old.name
            target.parent.mkdir(parents=True, exist_ok=True)
            old.rename(target)

    records = []
    for kind, entries, directory, extension, folder, fmt in (
        ("Figure", fig_main, C.FIG, "png", "figures", "{:02d}"),
        ("Figure", fig_supp, C.FIG, "png", "figures", "S{:02d}"),
        ("Table", tables, C.TAB, "csv", "tables", "{:02d}"),
    ):
        for n, entry in enumerate(entries, 1):
            source = directory / f"{entry[1]}_real.{extension}"
            label = fmt.format(n)
            target = dest / folder / f"{kind}_{label}_{entry[1]}.{extension}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records.append({"kind": kind, "label": label, "file": str(target.relative_to(dest)),
                            "caption": resolve_references(entry[2], figures, tables),
                            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})

    for docx_name, dest_name in (("manuscript_real.docx", "manuscript.docx"),
                                 ("manuscript_supplementary_real.docx", "manuscript_supplementary.docx")):
        src_doc = root / "manuscript" / docx_name
        if not src_doc.exists():
            continue
        old_doc = dest / dest_name
        if old_doc.exists():
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_doc, archive / old_doc.name)
        shutil.copy2(src_doc, old_doc)

    for name in ("export_docx.py", "fill_manuscript.py", "run_pipeline.py", "package_submission.py",
                 "run_placebo_test.py", "generate_ris.py", "requirements.txt",
                 "verify_manuscript_assets.py", "test_export_references.py"):
        shutil.copy2(root / name, dest / "code" / name)
    shutil.copytree(root / "src", dest / "code/src", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(root / "manuscript/manuscript.md", dest / "code/manuscript/manuscript.md")
    shutil.copytree(root / "data/boundaries", dest / "data/boundaries", dirs_exist_ok=True)
    shutil.copy2(root / "data/processed/panel_real.csv", dest / "data/processed/panel_real.csv")
    (dest / "asset_manifest.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Packaged {len(fig_main)} main + {len(fig_supp)} supplementary figures and "
          f"{len(tables)} tables; old assets saved to {archive}")


if __name__ == "__main__":
    main()
