"""Numbering regressions: document order, optional assets, invalid anchors,
and the main-text / Supplementary Material split."""
import tempfile
import unittest
from pathlib import Path

from export_docx import ordered_entries, resolve_references, _split_main_supp


class ReferencesTest(unittest.TestCase):
    def test_numbering_tracks_document_order_and_existing_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            entries = [("## B", "later", "", True), ("## A", "earlier", "", True),
                      ("## A", "missing", "", True)]
            for stem in ("earlier", "later"):
                (folder / f"{stem}_real.png").touch()
            result = ordered_entries("## A\ntext\n## B", entries, folder, "png", "real")
            self.assertEqual([e[1] for e in result], ["earlier", "later"])
            self.assertEqual(resolve_references("[[FIG:later]]; [[FIG:earlier]]", result, []),
                             "Figure 2; Figure 1")
            with self.assertRaisesRegex(ValueError, "missing"):
                resolve_references("[[FIG:missing]]", result, [])

    def test_invalid_anchor_fails_instead_of_appending_at_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "figure_real.png").touch()
            for text in ("## Other", "## A\n## A"):
                with self.assertRaisesRegex(ValueError, "Expected one anchor"):
                    ordered_entries(text, [("## A", "figure", "", True)], folder, "png", "real")

    def test_caption_cross_reference_and_tables_use_separate_sequences(self):
        figures = [("", "locator", "", True), ("", "exposure", "", True)]
        tables = [("", "coefficients", "")]
        self.assertEqual(resolve_references("[[FIG:exposure]]a; [[TAB:coefficients]]", figures, tables),
                         "Figure 2a; Table 1")

    def test_supplementary_figures_number_in_their_own_sequence(self):
        # Document order: main, supp, main, supp -- each tier numbers only
        # within itself, still counting from that tier's own position.
        figures = [("", "m1", "", True), ("", "s1", "", False),
                   ("", "m2", "", True), ("", "s2", "", False)]
        main, supp = _split_main_supp(figures)
        self.assertEqual([e[1] for e in main], ["m1", "m2"])
        self.assertEqual([e[1] for e in supp], ["s1", "s2"])
        self.assertEqual(
            resolve_references("[[FIG:m1]] [[FIG:s1]] [[FIG:m2]] [[FIG:s2]]", figures, []),
            "Figure 1 Figure S1 Figure 2 Figure S2")


if __name__ == "__main__":
    unittest.main()
