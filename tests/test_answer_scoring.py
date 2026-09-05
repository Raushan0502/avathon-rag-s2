"""Unit tests for src/answer_scoring.py.

The lexical scorer decides what counts as a correct answer, so its known
weakness -- punishing paraphrase -- is pinned here rather than left as a
footnote, along with the rule that a refusal is not a wrong answer.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import answer_scoring
from src.answer_scoring import key_fact_recall, key_terms, score_answers
from src.generation import CANNOT_ANSWER_PHRASE


def record(generated: str, reference: str, ident: str = "q01") -> dict:
    return {"id": ident, "generated_answer": generated, "reference_answer": reference}


class TestKeyTerms(unittest.TestCase):
    def test_keeps_figures_which_identify_a_passage(self) -> None:
        self.assertIn("416,161", key_terms("Total net sales were $416,161 million"))

    def test_drops_stopwords_and_short_tokens(self) -> None:
        terms = key_terms("The company is in the United States")
        self.assertNotIn("the", terms)
        self.assertIn("company", terms)


class TestKeyFactRecall(unittest.TestCase):
    def test_exact_answer_scores_one(self) -> None:
        self.assertEqual(key_fact_recall("Cupertino, California.", "Cupertino, California."), 1.0)

    def test_unrelated_answer_scores_zero(self) -> None:
        self.assertEqual(key_fact_recall("Austin, Texas.", "revenue grew twelve percent"), 0.0)

    def test_empty_reference_does_not_divide_by_zero(self) -> None:
        self.assertEqual(key_fact_recall("anything", ""), 0.0)

    def test_paraphrase_is_under_credited_and_this_is_known(self) -> None:
        # The documented weakness: these two say the same thing, and the
        # lexical scorer cannot tell. It is a floor, not a verdict -- which
        # is why the LLM judge is reported alongside it.
        reference = "Competitors aggressively cut prices and lowered product margins."
        paraphrase = "Rivals repeatedly slashed prices, driving down product margins."
        self.assertLess(key_fact_recall(paraphrase, reference), 0.6)


class TestScoreAnswers(unittest.TestCase):
    def test_refusals_are_excluded_from_accuracy_not_counted_as_wrong(self) -> None:
        # Declining when context is missing is correct behaviour; scoring it
        # as a wrong answer would penalise the safest possible response.
        scores = score_answers([
            record(CANNOT_ANSWER_PHRASE, "Cupertino, California."),
            record("Cupertino, California.", "Cupertino, California.", "q02"),
        ])
        self.assertEqual(scores["refused"], 1)
        self.assertEqual(scores["answered"], 1)
        self.assertEqual(scores["lexical_accuracy"], 1.0)

    def test_all_refused_does_not_divide_by_zero(self) -> None:
        scores = score_answers([record(CANNOT_ANSWER_PHRASE, "anything")])
        self.assertEqual(scores["answered"], 0)
        self.assertEqual(scores["lexical_accuracy"], 0.0)

    def test_reports_per_question_rows(self) -> None:
        scores = score_answers([record("Cupertino, California.", "Cupertino, California.")])
        row = scores["per_query"][0]
        self.assertEqual(row["id"], "q01")
        self.assertTrue(row["lexical_correct"])
        self.assertFalse(row["refused"])

    def test_judge_is_not_called_unless_requested(self) -> None:
        with mock.patch.object(answer_scoring, "call_llm") as called:
            score_answers([record("some answer", "some reference")], use_judge=False)
        called.assert_not_called()

    def test_judge_verdicts_are_tallied_when_requested(self) -> None:
        with mock.patch.object(answer_scoring, "call_llm", return_value=("groq", "CORRECT")):
            scores = score_answers([record("a", "b"), record("c", "d", "q02")], use_judge=True)
        self.assertEqual(scores["judge_verdicts"], {"CORRECT": 2})

    def test_judge_is_skipped_for_refusals(self) -> None:
        # No point asking a judge to grade a declined answer.
        with mock.patch.object(answer_scoring, "call_llm") as called:
            score_answers([record(CANNOT_ANSWER_PHRASE, "x")], use_judge=True)
        called.assert_not_called()

    def test_unrecognised_judge_reply_is_flagged_rather_than_guessed(self) -> None:
        with mock.patch.object(answer_scoring, "call_llm", return_value=("groq", "maybe?")):
            scores = score_answers([record("a", "b")], use_judge=True)
        self.assertEqual(scores["judge_verdicts"], {"UNPARSED": 1})


if __name__ == "__main__":
    unittest.main()
