"""Unit tests for src/embed_cache.py.

The cache decides what gets re-embedded, so a bug here either wastes hours
of CPU or -- far worse -- silently returns a vector belonging to different
text or a different model. These tests pin both directions.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import embed_cache


class RecordingEncoder:
    """Stands in for the real encoder and records what it was asked to embed."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[list[str]] = []

    def __call__(self, model, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        # Deterministic per-text vector so identity can be asserted.
        return np.array([[float(len(t)), 1.0, 2.0, 3.0] for t in texts], dtype="float32")

    @property
    def embedded(self) -> list[str]:
        return [t for call in self.calls for t in call]


class EmbedCacheTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        patcher = mock.patch.object(embed_cache, "CACHE_DIR", self.tmp / "cache")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.encoder = RecordingEncoder()


class TestTextKey(EmbedCacheTestCase):
    def test_same_text_gives_the_same_key(self) -> None:
        self.assertEqual(embed_cache.text_key("hello"), embed_cache.text_key("hello"))

    def test_different_text_gives_a_different_key(self) -> None:
        self.assertNotEqual(embed_cache.text_key("hello"), embed_cache.text_key("hello "))

    def test_contextual_prefix_changes_the_key(self) -> None:
        # The prefix changes the vector, so it must change the key -- otherwise
        # an enriched chunk would reuse the unenriched vector.
        plain = "Competition is intense."
        prefixed = "Apple Inc. | 10-K | Item 1A\nCompetition is intense."
        self.assertNotEqual(embed_cache.text_key(plain), embed_cache.text_key(prefixed))


class TestEmbedCached(EmbedCacheTestCase):
    def test_first_call_embeds_everything(self) -> None:
        out = embed_cache.embed_cached(None, "m", ["a", "bb", "ccc"], self.encoder)
        self.assertEqual(out.shape, (3, 4))
        self.assertEqual(self.encoder.embedded, ["a", "bb", "ccc"])

    def test_second_identical_call_embeds_nothing(self) -> None:
        embed_cache.embed_cached(None, "m", ["a", "bb"], self.encoder)
        self.encoder.calls.clear()
        out = embed_cache.embed_cached(None, "m", ["a", "bb"], self.encoder)
        self.assertEqual(self.encoder.embedded, [], "cache hit must skip the encoder entirely")
        self.assertEqual(out.shape, (2, 4))

    def test_only_new_text_is_embedded_when_corpus_grows(self) -> None:
        # Adding documents must not re-embed the existing ones.
        embed_cache.embed_cached(None, "m", ["a", "bb"], self.encoder)
        self.encoder.calls.clear()
        embed_cache.embed_cached(None, "m", ["a", "bb", "ccc"], self.encoder)
        self.assertEqual(self.encoder.embedded, ["ccc"])

    def test_vectors_are_returned_in_the_requested_order(self) -> None:
        first = embed_cache.embed_cached(None, "m", ["a", "bb", "ccc"], self.encoder)
        reordered = embed_cache.embed_cached(None, "m", ["ccc", "a", "bb"], self.encoder)
        np.testing.assert_array_equal(reordered[0], first[2])
        np.testing.assert_array_equal(reordered[1], first[0])

    def test_duplicate_text_in_one_batch_is_embedded_once(self) -> None:
        embed_cache.embed_cached(None, "m", ["same", "same", "same"], self.encoder)
        self.assertEqual(self.encoder.embedded, ["same"])

    def test_a_different_model_does_not_reuse_cached_vectors(self) -> None:
        # Vectors from different models live in different spaces; reusing them
        # across models would silently produce meaningless similarities.
        embed_cache.embed_cached(None, "model-a", ["a"], self.encoder)
        self.encoder.calls.clear()
        embed_cache.embed_cached(None, "model-b", ["a"], self.encoder)
        self.assertEqual(self.encoder.embedded, ["a"], "must be a cache miss for a new model")

    def test_cache_survives_a_process_restart(self) -> None:
        embed_cache.embed_cached(None, "m", ["a", "bb"], self.encoder)
        key_to_row, matrix = embed_cache.load_cache("m")
        self.assertEqual(len(key_to_row), 2)
        self.assertEqual(matrix.shape, (2, 4))


class TestCacheStats(EmbedCacheTestCase):
    def test_reports_a_cold_cache(self) -> None:
        stats = embed_cache.cache_stats("m", ["a", "bb"])
        self.assertEqual((stats["cached"], stats["to_embed"]), (0, 2))
        self.assertEqual(stats["hit_rate"], 0.0)

    def test_reports_a_warm_cache_without_embedding(self) -> None:
        embed_cache.embed_cached(None, "m", ["a", "bb"], self.encoder)
        self.encoder.calls.clear()
        stats = embed_cache.cache_stats("m", ["a", "bb", "new"])
        self.assertEqual((stats["cached"], stats["to_embed"]), (2, 1))
        self.assertEqual(self.encoder.embedded, [], "stats must not trigger encoding")

    def test_counts_unique_texts_not_repeats(self) -> None:
        self.assertEqual(embed_cache.cache_stats("m", ["a", "a", "b"])["unique"], 2)


if __name__ == "__main__":
    unittest.main()
