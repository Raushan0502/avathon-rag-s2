"""Unit tests for src/evaluate.py -- the retrieval scoring metrics."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate_retrieval, score_retrieval

GOLD_DOC = "AAPL_10-K_2025-10-31"
GOLD_SECTION = "Item 1A. Risk Factors"


def chunk(doc_id: str = GOLD_DOC, section: str = GOLD_SECTION) -> dict:
    return {"doc_id": doc_id, "section": section, "text": "irrelevant for scoring"}


class FakeIndex:
    """Stands in for RetrievalIndex: returns a canned result list per query."""

    def __init__(self, results_by_query: dict[str, list[dict]]):
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int, str]] = []

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[dict]:
        self.calls.append((query, k, mode))
        return self.results_by_query[query][:k]


class TestScoreRetrieval(unittest.TestCase):
    def test_all_retrieved_chunks_relevant(self) -> None:
        precision, recall = score_retrieval([chunk(), chunk()], GOLD_DOC, GOLD_SECTION)
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)

    def test_no_retrieved_chunk_relevant(self) -> None:
        retrieved = [chunk(section="Item 2. Properties"), chunk(doc_id="TSLA_10-K_2026-01-29")]
        precision, recall = score_retrieval(retrieved, GOLD_DOC, GOLD_SECTION)
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)

    def test_partial_match_gives_fractional_precision_but_full_recall(self) -> None:
        retrieved = [chunk(), chunk(section="Item 2. Properties"), chunk(section="Item 2. Properties")]
        precision, recall = score_retrieval(retrieved, GOLD_DOC, GOLD_SECTION)
        self.assertAlmostEqual(precision, 1 / 3)
        self.assertEqual(recall, 1.0, "one relevant chunk anywhere in top-k is a hit")

    def test_same_section_name_in_different_document_is_not_relevant(self) -> None:
        # Section labels repeat across filers ("Item 2. Properties" exists in
        # every 10-K), so doc_id must be part of the relevance check.
        retrieved = [chunk(doc_id="MSFT_10-K_2026-07-29")]
        precision, recall = score_retrieval(retrieved, GOLD_DOC, GOLD_SECTION)
        self.assertEqual((precision, recall), (0.0, 0.0))

    def test_empty_retrieval_scores_zero_without_dividing_by_zero(self) -> None:
        self.assertEqual(score_retrieval([], GOLD_DOC, GOLD_SECTION), (0.0, 0.0))


class TestEvaluateRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.eval_set = [
            {"id": "q01", "query": "hit", "gold_doc_id": GOLD_DOC, "gold_section": GOLD_SECTION},
            {"id": "q02", "query": "miss", "gold_doc_id": GOLD_DOC, "gold_section": GOLD_SECTION},
        ]
        self.index = FakeIndex(
            {
                "hit": [chunk(), chunk()],
                "miss": [chunk(section="Item 2. Properties"), chunk(section="Item 2. Properties")],
            }
        )

    def test_aggregates_means_across_questions(self) -> None:
        result = evaluate_retrieval(self.index, self.eval_set, k=2, mode="hybrid")
        self.assertEqual(result["mean_precision_at_k"], 0.5, "means of 1.0 and 0.0")
        self.assertEqual(result["mean_recall_at_k"], 0.5)

    def test_reports_mode_k_and_per_query_scores(self) -> None:
        result = evaluate_retrieval(self.index, self.eval_set, k=2, mode="dense")
        self.assertEqual(result["mode"], "dense")
        self.assertEqual(result["k"], 2)
        self.assertEqual([row["id"] for row in result["per_query"]], ["q01", "q02"])
        self.assertEqual(result["per_query"][0]["hit_at_k"], 1.0)
        self.assertEqual(result["per_query"][1]["hit_at_k"], 0.0)

    def test_passes_k_and_mode_through_to_the_index(self) -> None:
        evaluate_retrieval(self.index, self.eval_set, k=1, mode="bm25")
        self.assertEqual(self.index.calls, [("hit", 1, "bm25"), ("miss", 1, "bm25")])


if __name__ == "__main__":
    unittest.main()
