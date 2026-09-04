"""Unit tests for src/validate.py -- the extraction-quality gate."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validate import check_document, measure_document, summarise

GOOD_TEXT = "\n".join(
    f"In fiscal {2000 + i} the segment reported growth across every major geography."
    for i in range(40)
)


class TestMeasureDocument(unittest.TestCase):
    def test_scores_a_healthy_document_well(self) -> None:
        metrics = measure_document(GOOD_TEXT, source_bytes=len(GOOD_TEXT) * 4)
        self.assertGreater(metrics["alpha_ratio"], 0.6)
        self.assertGreater(metrics["mean_words_per_line"], 3.0)
        self.assertEqual(metrics["lines"], 40)

    def test_empty_text_scores_zero_without_dividing_by_zero(self) -> None:
        # The scanned-PDF case: extraction returns "" and must not crash.
        metrics = measure_document("", source_bytes=2_000_000)
        self.assertEqual(metrics["chars"], 0)
        self.assertEqual(metrics["yield_ratio"], 0.0)
        self.assertEqual(metrics["alpha_ratio"], 0.0)

    def test_zero_source_bytes_does_not_divide_by_zero(self) -> None:
        self.assertEqual(measure_document("some text", source_bytes=0)["yield_ratio"], 0.0)

    def test_detects_repeated_lines_as_boilerplate(self) -> None:
        text = "\n".join(["Apple Inc. Form 10-K"] * 10 + ["unique line"])
        self.assertGreater(measure_document(text, 1000)["boilerplate_ratio"], 0.8)

    def test_detects_table_rows(self) -> None:
        text = "| a | b |\n| --- | --- |\n| 1 | 2 |\nprose line"
        self.assertAlmostEqual(measure_document(text, 100)["table_row_ratio"], 0.75)

    def test_punctuation_heavy_text_scores_low_alpha_ratio(self) -> None:
        # A contents page left behind by preprocessing.
        text = "\n".join(["Section .......................... 6"] * 20)
        self.assertLess(measure_document(text, 1000)["alpha_ratio"], 0.6)


class TestCheckDocument(unittest.TestCase):
    def test_healthy_document_passes(self) -> None:
        status, issues = check_document(measure_document(GOOD_TEXT, len(GOOD_TEXT) * 4))
        self.assertEqual(status, "ok")
        self.assertEqual(issues, [])

    def test_short_email_passes_because_thresholds_are_type_aware(self) -> None:
        # A 200-character email is a normal email. The first full-corpus run
        # failed 16 healthy emails against a floor calibrated for filings.
        email = "Subject: Meeting confirmation\nPlease confirm Tuesday at 9am. Thanks."
        status, issues = check_document(measure_document(email, len(email)), "email")
        self.assertEqual(status, "ok", issues)

    def test_same_short_text_still_fails_for_a_filing(self) -> None:
        # The identical length must remain a failure where it means the
        # extractor gave up rather than the document being genuinely brief.
        text = "Subject: Meeting confirmation\nPlease confirm Tuesday at 9am. Thanks."
        status, _ = check_document(measure_document(text, len(text)), "report")
        self.assertEqual(status, "fail")

    def test_scanned_pdf_with_no_text_layer_fails(self) -> None:
        # The failure mode the whole module exists for: a 2 MB PDF that
        # extracts to nothing must be surfaced, not indexed silently.
        status, issues = check_document(measure_document("", source_bytes=2_000_000))
        self.assertEqual(status, "fail")
        self.assertTrue(any("OCR" in issue for issue in issues))

    def test_document_with_almost_no_text_for_its_size_fails(self) -> None:
        status, _ = check_document(measure_document("x" * 600, source_bytes=5_000_000))
        self.assertEqual(status, "fail")

    def test_punctuation_heavy_document_warns_but_does_not_fail(self) -> None:
        text = "\n".join(["Section .......................... 6"] * 40)
        status, issues = check_document(measure_document(text, len(text) * 2))
        self.assertEqual(status, "warn")
        self.assertTrue(any("alpha_ratio" in issue for issue in issues))

    def test_fragmented_single_word_lines_warn(self) -> None:
        # Symptom of multi-column PDF extraction going wrong.
        text = "\n".join(["word"] * 200)
        status, issues = check_document(measure_document(text, len(text) * 2))
        self.assertIn(status, {"warn", "fail"})
        self.assertTrue(any("fragmented" in issue for issue in issues))

    def test_fail_outranks_warn_when_both_apply(self) -> None:
        status, _ = check_document(measure_document("...", source_bytes=1_000_000))
        self.assertEqual(status, "fail")


class TestSummarise(unittest.TestCase):
    def setUp(self) -> None:
        self.reports = [
            {"doc_id": "a", "doc_type": "report", "status": "ok", "metrics": self._m()},
            {"doc_id": "b", "doc_type": "report", "status": "warn", "metrics": self._m()},
            {"doc_id": "c", "doc_type": "email", "status": "fail", "metrics": self._m()},
        ]

    @staticmethod
    def _m() -> dict:
        return {
            "yield_ratio": 0.1,
            "alpha_ratio": 0.8,
            "boilerplate_ratio": 0.1,
            "table_row_ratio": 0.2,
        }

    def test_counts_by_status_and_lists_problem_documents(self) -> None:
        summary = summarise(self.reports)
        self.assertEqual(summary["by_status"], {"ok": 1, "warn": 1, "fail": 1})
        self.assertEqual(summary["failed"], ["c"])
        self.assertEqual(summary["warned"], ["b"])

    def test_breaks_status_down_by_document_type(self) -> None:
        summary = summarise(self.reports)
        self.assertEqual(summary["by_doc_type"]["report"], {"ok": 1, "warn": 1})
        self.assertEqual(summary["by_doc_type"]["email"], {"fail": 1})

    def test_empty_report_list_does_not_crash(self) -> None:
        self.assertEqual(summarise([])["documents"], 0)


if __name__ == "__main__":
    unittest.main()
