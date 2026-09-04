"""Unit tests for src/generation.py -- prompt building and faithfulness detection.

No live LLM calls: ``call_llm`` is patched where the fallback chain itself is
under test, so the suite needs no API keys and no network.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import generation
from src.generation import CANNOT_ANSWER_PHRASE, annotate_faithfulness, build_prompt, generate_answer


def chunk(index: int = 1, text: str = "Some filing text.") -> dict:
    return {
        "chunk_id": f"AAPL_10-K_2025-10-31::{index}",
        "doc_id": "AAPL_10-K_2025-10-31",
        "ticker": "AAPL",
        "form": "10-K",
        "filing_date": "2025-10-31",
        "section": "Item 1A. Risk Factors",
        "text": text,
    }


class TestBuildPrompt(unittest.TestCase):
    def test_numbers_sources_from_one(self) -> None:
        prompt = build_prompt("Why?", [chunk(1, "first"), chunk(2, "second")])
        self.assertIn("[1] (AAPL 10-K, 2025-10-31, section: Item 1A. Risk Factors)", prompt)
        self.assertIn("[2] (AAPL 10-K, 2025-10-31, section: Item 1A. Risk Factors)", prompt)

    def test_includes_chunk_text_and_the_question(self) -> None:
        prompt = build_prompt("What is the risk?", [chunk(1, "Competition is intense.")])
        self.assertIn("Competition is intense.", prompt)
        self.assertIn("QUESTION: What is the risk?", prompt)

    def test_handles_empty_retrieval_without_crashing(self) -> None:
        # Retrieval can legitimately return nothing; the model should then be
        # steered to refuse rather than the pipeline raising.
        prompt = build_prompt("Anything?", [])
        self.assertIn("QUESTION: Anything?", prompt)


class TestAnnotateFaithfulness(unittest.TestCase):
    def test_plain_ascii_citation_counts_as_grounded(self) -> None:
        result = annotate_faithfulness("Revenue rose [1].", num_sources=3)
        self.assertEqual(result["faithfulness_flag"], "cited")
        self.assertEqual(result["citations"], [1])

    def test_full_width_bracket_citation_counts_as_grounded(self) -> None:
        # Observed live from Groq in Step 5 testing.
        result = annotate_faithfulness("Revenue rose【2】.", num_sources=3)
        self.assertEqual(result["faithfulness_flag"], "cited")
        self.assertEqual(result["citations"], [2])

    def test_browsing_style_citation_with_line_range_counts_as_grounded(self) -> None:
        # Observed live in Step 6's eval; this format caused 4 false
        # UNGROUNDED flags before the regex was widened.
        result = annotate_faithfulness("Two segments【1†L1-L4】【2†L1-L3】", num_sources=3)
        self.assertEqual(result["faithfulness_flag"], "cited")
        self.assertEqual(result["citations"], [1, 2])

    def test_answer_without_any_citation_is_flagged_ungrounded(self) -> None:
        result = annotate_faithfulness("Revenue rose sharply last year.", num_sources=3)
        self.assertEqual(result["faithfulness_flag"], "UNGROUNDED")
        self.assertFalse(result["has_citation"])

    def test_explicit_refusal_is_flagged_refused_not_ungrounded(self) -> None:
        result = annotate_faithfulness(CANNOT_ANSWER_PHRASE, num_sources=3)
        self.assertEqual(result["faithfulness_flag"], "refused")
        self.assertTrue(result["refused_unsupported"])

    def test_citation_numbers_outside_the_source_range_are_discarded(self) -> None:
        # A citation to source [9] when only 3 were supplied is itself a
        # hallucination and must not be accepted as grounding.
        result = annotate_faithfulness("Claim [9].", num_sources=3)
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["faithfulness_flag"], "UNGROUNDED")

    def test_repeated_citations_are_deduplicated_and_sorted(self) -> None:
        result = annotate_faithfulness("A [2]. B [1]. C [2].", num_sources=3)
        self.assertEqual(result["citations"], [1, 2])


class TestGenerateAnswer(unittest.TestCase):
    def test_returns_answer_provider_and_faithfulness(self) -> None:
        with mock.patch.object(generation, "call_llm", return_value=("groq", "Answer [1].")):
            result = generate_answer("Why?", [chunk(1)])
        self.assertEqual(result["provider"], "groq")
        self.assertEqual(result["answer"], "Answer [1].")
        self.assertEqual(result["faithfulness"]["faithfulness_flag"], "cited")
        self.assertEqual(result["query"], "Why?")

    def test_carries_the_retrieved_context_through_for_auditing(self) -> None:
        chunks = [chunk(1), chunk(2)]
        with mock.patch.object(generation, "call_llm", return_value=("mistral", "Answer [1][2].")):
            result = generate_answer("Why?", chunks)
        self.assertEqual(result["retrieved_chunks"], chunks)


class TestProviderFallback(unittest.TestCase):
    def test_raises_when_no_provider_key_is_configured(self) -> None:
        with mock.patch.multiple(
            generation, GROQ_API_KEY=None, MISTRAL_API_KEY=None, GEMINI_API_KEY=None
        ):
            with self.assertRaises(RuntimeError) as ctx:
                generation.call_llm("prompt")
        self.assertIn("failed or unconfigured", str(ctx.exception))

    def test_error_message_names_each_provider_that_failed(self) -> None:
        # Keys present but the SDK call fails -> the error must say which
        # providers were tried, so a silent total failure is diagnosable.
        with mock.patch.multiple(
            generation, GROQ_API_KEY="k", MISTRAL_API_KEY="k", GEMINI_API_KEY=None
        ):
            with mock.patch.dict(sys.modules, {"groq": None, "mistralai.client": None}):
                with self.assertRaises(RuntimeError) as ctx:
                    generation.call_llm("prompt")
        message = str(ctx.exception)
        self.assertIn("groq:", message)
        self.assertIn("mistral:", message)


if __name__ == "__main__":
    unittest.main()
