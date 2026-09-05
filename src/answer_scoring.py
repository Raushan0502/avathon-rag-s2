"""
Answer-correctness scoring against the gold reference answers.

The faithfulness flag says an answer *cited* retrieved context. It does not
say the answer is *right*: an answer can faithfully cite a passage that does
not address the question, which happened here on an earlier corpus and
produced a fluent, cited, wrong answer. Correctness therefore needs a second
signal, measured against ``reference_answer``.

Two scorers, deliberately kept separate:

``key_fact_recall`` is lexical and free. It checks whether the reference
answer's load-bearing tokens -- figures, proper nouns, distinctive terms --
survive into the generated answer. Cheap, deterministic, no API, and it
cannot be gamed by fluent phrasing. It under-credits correct paraphrase
("Cupertino, California" answered as "in Cupertino"), so it is a floor
rather than a verdict.

``judge_answer`` asks an LLM whether the generated answer conveys the same
facts as the reference. It handles paraphrase, but costs an API call per
question and inherits the judge's own biases, so it is reported alongside
the lexical score rather than instead of it. Where the two disagree is
usually where a human should look.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.generation import CANNOT_ANSWER_PHRASE, call_llm

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9,.$%-]*")
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "have", "has", "are", "was",
    "were", "its", "their", "which", "when", "what", "does", "did", "not", "will",
    "would", "can", "any", "all", "may", "such", "than", "then", "there", "these",
    "those", "been", "being", "into", "over", "under", "about", "also", "other",
}
KEY_FACT_PASS = 0.60

JUDGE_PROMPT = """Compare a CANDIDATE answer against a REFERENCE answer.

Reply with exactly one word:
CORRECT   - the candidate conveys the same facts as the reference
PARTIAL   - some facts right, but incomplete or partly wrong
WRONG     - contradicts the reference or answers a different question
REFUSED   - the candidate declines to answer

Ignore wording, length and style. Judge only whether the facts match.

REFERENCE: {reference}

CANDIDATE: {candidate}

VERDICT:"""


def key_terms(text: str) -> set[str]:
    """Load-bearing tokens of an answer: figures, names, distinctive words."""
    return {t for t in TOKEN_RE.findall(text.lower()) if len(t) > 2 and t not in STOPWORDS}


def key_fact_recall(generated: str, reference: str) -> float:
    """Fraction of the reference answer's key terms present in the generated
    one."""
    wanted = key_terms(reference)
    if not wanted:
        return 0.0
    return len(wanted & key_terms(generated)) / len(wanted)


def score_answers(records: list[dict], use_judge: bool = False) -> dict:
    """Score generated answers for correctness, not just grounding."""
    per_query, verdicts = [], {}
    for record in records:
        generated = record.get("generated_answer", "")
        reference = record.get("reference_answer", "")
        refused = CANNOT_ANSWER_PHRASE.lower() in generated.lower()
        recall = key_fact_recall(generated, reference)

        row = {
            "id": record.get("id"),
            "refused": refused,
            "key_fact_recall": recall,
            "lexical_correct": (not refused) and recall >= KEY_FACT_PASS,
        }

        if use_judge and not refused:
            verdict, _ = judge_answer(generated, reference)
            row["judge"] = verdict
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        per_query.append(row)

    answered = [r for r in per_query if not r["refused"]]
    return {
        "questions": len(per_query),
        "refused": sum(1 for r in per_query if r["refused"]),
        "answered": len(answered),
        "mean_key_fact_recall": (
            sum(r["key_fact_recall"] for r in answered) / len(answered) if answered else 0.0
        ),
        "lexical_accuracy": (
            sum(r["lexical_correct"] for r in answered) / len(answered) if answered else 0.0
        ),
        "judge_verdicts": verdicts,
        "per_query": per_query,
    }


def judge_answer(generated: str, reference: str) -> tuple[str, str]:
    """Ask an LLM whether the generated answer matches the reference."""
    _, reply = call_llm(JUDGE_PROMPT.format(reference=reference, candidate=generated))
    upper = reply.strip().upper()
    for verdict in ("CORRECT", "PARTIAL", "WRONG", "REFUSED"):
        if verdict in upper:
            return verdict, reply
    return "UNPARSED", reply
