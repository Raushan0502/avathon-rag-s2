"""Unit tests for src/retrieval.py -- RRF fusion and the three search modes.

A ``RetrievalIndex`` is assembled by hand from a tiny FAISS index, a real
BM25 index over three short documents, and a stub embedder, so the whole
suite runs offline without the sentence-transformers model.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embed_index import build_index
from src.retrieval import TOKEN_RE, RetrievalIndex, reciprocal_rank_fusion

CHUNK_TEXTS = [
    "competition is intense and margins are under pressure",
    "the cybertruck is a full size electric pickup truck",
    "headquarters are located in cupertino california",
]


class StubEmbedder:
    """Returns a fixed query vector so dense ranking is deterministic."""

    def __init__(self, query_vector: list[float]):
        self.query_vector = query_vector

    def encode(self, inputs, **kwargs) -> np.ndarray:
        return np.array([self.query_vector] * len(inputs), dtype="float32")


def make_index(query_vector: list[float] | None = None) -> RetrievalIndex:
    chunk_dicts = [
        {"chunk_id": f"DOC::{i}", "doc_id": "DOC", "section": f"Item {i}", "text": text}
        for i, text in enumerate(CHUNK_TEXTS)
    ]
    return RetrievalIndex(
        embedder=StubEmbedder(query_vector or [1.0, 0.0, 0.0]),
        faiss_index=build_index(np.eye(3, dtype="float32")),
        bm25_index=BM25Okapi([TOKEN_RE.findall(t.lower()) for t in CHUNK_TEXTS]),
        chunk_dicts=chunk_dicts,
    )


class TestReciprocalRankFusion(unittest.TestCase):
    def test_item_ranked_by_both_retrievers_outranks_single_list_leaders(self) -> None:
        # 7 is only 2nd on each list, but appears on both; 1 and 2 lead one
        # list apiece. Agreement across retrievers should win.
        fused = reciprocal_rank_fusion([[1, 7, 3], [2, 7, 4]])
        self.assertEqual(fused[0][0], 7)

    def test_scores_decrease_with_rank_position(self) -> None:
        fused = dict(reciprocal_rank_fusion([[10, 11, 12]]))
        self.assertGreater(fused[10], fused[11])
        self.assertGreater(fused[11], fused[12])

    def test_results_are_sorted_best_first(self) -> None:
        scores = [score for _, score in reciprocal_rank_fusion([[1, 2, 3], [3, 2, 1]])]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_input_returns_no_results(self) -> None:
        self.assertEqual(reciprocal_rank_fusion([]), [])


class TestSearchModes(unittest.TestCase):
    def test_rejects_an_unknown_mode(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            make_index().search("anything", mode="semantic")
        self.assertIn("semantic", str(ctx.exception))

    def test_dense_mode_ranks_by_the_embedding_vector(self) -> None:
        # Query vector points at row 1, so chunk 1 must come back first.
        results = make_index(query_vector=[0.0, 1.0, 0.0]).search("any text", k=3, mode="dense")
        self.assertEqual(results[0]["chunk_id"], "DOC::1")

    def test_bm25_mode_ranks_by_exact_term_overlap(self) -> None:
        # "cybertruck" appears only in chunk 1; dense scoring is irrelevant here.
        results = make_index().search("cybertruck", k=3, mode="bm25")
        self.assertEqual(results[0]["chunk_id"], "DOC::1")

    def test_hybrid_mode_returns_fused_results(self) -> None:
        results = make_index().search("cupertino headquarters", k=3, mode="hybrid")
        self.assertEqual(results[0]["retrieval_mode"], "hybrid")
        self.assertIn("DOC::2", [r["chunk_id"] for r in results])

    def test_respects_k_and_annotates_each_result(self) -> None:
        results = make_index().search("competition", k=2, mode="hybrid")
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn("score", result)
            self.assertEqual(result["retrieval_mode"], "hybrid")
            self.assertIn("text", result, "chunk metadata is carried through")

    def test_search_does_not_mutate_the_stored_chunks(self) -> None:
        index = make_index()
        index.search("competition", k=3, mode="hybrid")
        self.assertNotIn("score", index.chunk_dicts[0], "results must be copies")


if __name__ == "__main__":
    unittest.main()
