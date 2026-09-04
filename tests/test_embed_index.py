"""Unit tests for src/embed_index.py.

These build tiny vectors by hand rather than loading the real embedding
model, so the suite stays offline and fast; ``embed_texts`` is exercised
against a stub encoder that records how it was called.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embed_index import QUERY_INSTRUCTION, build_index, embed_texts, load_artifacts, save_artifacts


class StubEncoder:
    """Minimal stand-in for SentenceTransformer: records the texts it was
    given and returns fixed-width vectors."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.seen_inputs: list[str] = []

    def encode(self, inputs, **kwargs) -> np.ndarray:
        self.seen_inputs = list(inputs)
        self.encode_kwargs = kwargs
        return np.ones((len(inputs), self.dim), dtype="float64")


class TestEmbedTexts(unittest.TestCase):
    def test_passages_are_embedded_without_the_query_instruction(self) -> None:
        encoder = StubEncoder()
        embed_texts(encoder, ["a passage"], is_query=False)
        self.assertEqual(encoder.seen_inputs, ["a passage"])

    def test_queries_are_prefixed_with_the_bge_instruction(self) -> None:
        # BGE is trained asymmetrically -- getting this backwards silently
        # degrades retrieval, so it is worth pinning down.
        encoder = StubEncoder()
        embed_texts(encoder, ["a question"], is_query=True)
        self.assertEqual(encoder.seen_inputs, [QUERY_INSTRUCTION + "a question"])

    def test_requests_normalized_vectors_and_returns_float32(self) -> None:
        encoder = StubEncoder()
        vectors = embed_texts(encoder, ["x", "y"], is_query=False)
        self.assertTrue(encoder.encode_kwargs["normalize_embeddings"])
        self.assertEqual(vectors.dtype, np.dtype("float32"))
        self.assertEqual(vectors.shape, (2, 4))


class TestBuildIndex(unittest.TestCase):
    def setUp(self) -> None:
        # Three orthogonal unit vectors: each is its own nearest neighbour.
        self.vectors = np.eye(3, dtype="float32")

    def test_index_holds_every_vector_at_matching_dimension(self) -> None:
        index = build_index(self.vectors)
        self.assertEqual(index.ntotal, 3)
        self.assertEqual(index.d, 3)

    def test_search_ranks_the_matching_vector_first(self) -> None:
        index = build_index(self.vectors)
        scores, indices = index.search(np.array([[0.0, 1.0, 0.0]], dtype="float32"), k=3)
        self.assertEqual(indices[0][0], 1)
        self.assertAlmostEqual(float(scores[0][0]), 1.0, places=5)


class TestArtifactRoundTrip(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.index_path = Path(self.tmpdir.name) / "nested" / "index.faiss"
        self.chunks_path = Path(self.tmpdir.name) / "nested" / "chunks.jsonl"
        self.chunks = [
            {"chunk_id": "DOC::0", "doc_id": "DOC", "section": "Item 1", "text": "first"},
            {"chunk_id": "DOC::1", "doc_id": "DOC", "section": "Item 2", "text": "second"},
        ]
        self.index = build_index(np.eye(2, dtype="float32"))

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_round_trip_preserves_index_and_chunk_order(self) -> None:
        save_artifacts(self.index, self.chunks, self.index_path, self.chunks_path)
        loaded_index, loaded_chunks = load_artifacts(self.index_path, self.chunks_path)
        self.assertEqual(loaded_index.ntotal, 2)
        self.assertEqual(loaded_chunks, self.chunks, "chunk row order maps FAISS ids back to text")

    def test_creates_missing_parent_directories(self) -> None:
        save_artifacts(self.index, self.chunks, self.index_path, self.chunks_path)
        self.assertTrue(self.index_path.exists() and self.chunks_path.exists())

    def test_missing_artifacts_raise_with_a_rebuild_hint(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            load_artifacts(self.index_path, self.chunks_path)
        self.assertIn("build_index.py", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
