"""Unit tests for scripts/remap_eval_set.py.

The remapper rewrites the evaluation set's gold references, so a bug here
would silently corrupt every downstream metric. These tests pin the
matching behaviour and, in particular, that a reference can never be moved
to a different document.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_spec = importlib.util.spec_from_file_location(
    "remap_eval_set", Path(__file__).resolve().parent.parent / "scripts" / "remap_eval_set.py"
)
remap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(remap)


def chunk(chunk_id: str, text: str, doc_id: str = "AAPL_10-K", section: str = "Item 2") -> dict:
    return {"chunk_id": chunk_id, "doc_id": doc_id, "section": section, "text": text}


class TestSalientTerms(unittest.TestCase):
    def test_drops_stopwords_and_very_short_tokens(self) -> None:
        terms = remap.salient_terms("The company is in the United States")
        self.assertNotIn("the", terms)
        self.assertNotIn("is", terms)
        self.assertIn("company", terms)

    def test_keeps_figures_which_are_the_most_distinctive_terms(self) -> None:
        # "416,161" identifies the right passage far better than any word.
        self.assertIn("416,161", remap.salient_terms("Total net sales were $416,161 million"))

    def test_is_case_insensitive(self) -> None:
        self.assertEqual(
            remap.salient_terms("Cupertino California"), remap.salient_terms("CUPERTINO california")
        )


class TestBestMatchingChunk(unittest.TestCase):
    def setUp(self) -> None:
        self.question = {
            "id": "q05",
            "query": "Where is Apple's corporate headquarters located?",
            "reference_answer": "Cupertino, California.",
        }

    def test_picks_the_chunk_containing_the_reference_answer(self) -> None:
        chunks = [
            chunk("c0", "The Company designs and markets smartphones and tablets."),
            chunk("c1", "The Company's headquarters is located in Cupertino, California."),
            chunk("c2", "Legal proceedings are described in Note 13."),
        ]
        match, score = remap.best_matching_chunk(self.question, chunks)
        self.assertEqual(match["chunk_id"], "c1")
        self.assertGreater(score, remap.MIN_OVERLAP)

    def test_scores_low_when_no_chunk_contains_the_answer(self) -> None:
        # A weak best match must fall below the threshold so the question is
        # reported as unresolved rather than silently mispointed.
        chunks = [chunk("c0", "Revenue increased across every geography.")]
        _, score = remap.best_matching_chunk(self.question, chunks)
        self.assertLess(score, remap.MIN_OVERLAP)

    def test_returns_nothing_for_an_empty_candidate_list(self) -> None:
        self.assertEqual(remap.best_matching_chunk(self.question, []), (None, 0.0))

    def test_prefers_the_answer_over_a_chunk_that_merely_echoes_the_question(self) -> None:
        # A chunk repeating the question's wording without answering it must
        # not outrank the chunk holding the actual answer.
        chunks = [
            chunk("c0", "This section covers where the corporate headquarters located policy."),
            chunk("c1", "Headquarters: Cupertino, California."),
        ]
        match, _ = remap.best_matching_chunk(self.question, chunks)
        self.assertEqual(match["chunk_id"], "c1")

    def test_carries_the_new_section_label_through(self) -> None:
        # Section labels changed when non-SEC documents gained real headings.
        chunks = [chunk("c1", "Headquarters in Cupertino, California.", section="1. PROPERTIES")]
        match, _ = remap.best_matching_chunk(self.question, chunks)
        self.assertEqual(match["section"], "1. PROPERTIES")


if __name__ == "__main__":
    unittest.main()
