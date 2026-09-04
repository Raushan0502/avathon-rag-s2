"""
Step 6 — Retrieval evaluation metrics.

Ground truth shape: each question in data/eval/qa_eval.json is authored
against one specific known chunk (and its (doc_id, section)) that the
question's answer actually comes from -- not an exhaustively relevance-
judged set covering every possibly-relevant chunk in the corpus. This is
the standard, practical simplification for a small hand-built eval set:
with exactly one known-relevant item per query, Recall@k reduces to a
binary hit rate ("was the relevant section found anywhere in the top-k
results?"), and Precision@k is the fraction of the top-k slots that were
"spent" on that relevant section. This is documented here rather than
silently treated as a full relevance-judged IR benchmark, since that
distinction matters for how the numbers should be read.

Relevance is checked at (doc_id, section) granularity rather than exact
chunk_id, since adjacent overlapping chunks from the same section are
equally valid evidence for the same question -- exact-chunk-id matching
would understate retrieval quality by penalizing a chunk one window over
from the specific one a question happened to be authored against.

Questions carry their own ``eval_k``, sized to how many chunks genuinely
answer them, because Precision@k measures the size of the answer set as
much as the retriever when fewer than k chunks are relevant. Results
therefore report three complementary figures: raw Precision@k, the
attainable ceiling, and MRR for "how close did it get".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Depth at which reciprocal rank is measured, independent of a question's
# own k, so MRR stays comparable between a k=1 and a k=20 question.
MRR_DEPTH = 20


def reciprocal_rank(retrieved: list[dict], gold_doc_id: str, gold_section: str) -> float:
    """Reciprocal of the rank at which the gold section first appears.

    Precision@k is a blunt instrument for known-item questions: a question
    answered by exactly one chunk scores 0.0 at k=1 whether the gold sat at
    rank 2 or rank 500. Reciprocal rank distinguishes those -- rank 2 scores
    0.5, rank 3 scores 0.33 -- which is what "how close did it get?" needs.

    Args:
        retrieved: Ranked chunk dicts, best first.
        gold_doc_id: Document holding the answer.
        gold_section: Section within that document.

    Returns:
        ``1 / rank`` for the first relevant hit, or 0.0 if none is present.
    """
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk["doc_id"] == gold_doc_id and chunk["section"] == gold_section:
            return 1.0 / rank
    return 0.0


def score_retrieval(retrieved: list[dict], gold_doc_id: str, gold_section: str) -> tuple[float, float]:
    """Score one query's retrieved chunks against its single gold section.

    A retrieved chunk counts as relevant when both its ``doc_id`` and
    ``section`` match the gold pair (see module docstring for why relevance
    is judged at section rather than chunk granularity).

    Args:
        retrieved: Top-k chunk dicts returned by ``RetrievalIndex.search``.
        gold_doc_id: ``doc_id`` of the document the answer comes from.
        gold_section: Section label within that document.

    Returns:
        ``(precision_at_k, recall_at_k)``. Precision is the fraction of the
        top-k slots filled by the gold section; recall is 1.0 if the gold
        section appears anywhere in the top-k and 0.0 otherwise (a binary
        hit rate, since exactly one section is known-relevant per query).
        An empty ``retrieved`` list scores ``(0.0, 0.0)``.
    """
    if not retrieved:
        return 0.0, 0.0
    hits = sum(
        1 for c in retrieved if c["doc_id"] == gold_doc_id and c["section"] == gold_section
    )
    return hits / len(retrieved), (1.0 if hits else 0.0)


def evaluate_retrieval(
    index,
    eval_set: list[dict],
    k: int,
    mode: str,
    gold_section_sizes: dict[tuple[str, str], int] | None = None,
) -> dict:
    """Run every eval question through one retrieval mode and aggregate scores.

    Precision@k is only interpretable when at least k chunks in the corpus
    genuinely answer the question. "Where is Apple headquartered?" is
    answered by exactly one chunk, so at k=5 it can never score above 0.20
    however good retrieval is -- the other four slots have nothing correct
    to hold. Questions therefore carry their own ``eval_k``, sized to their
    real answer set, and results report attainment against the achievable
    ceiling alongside the raw figure.

    Args:
        index: A ``RetrievalIndex`` (anything exposing ``search(query, k, mode)``).
        eval_set: Questions loaded from ``data/eval/qa_eval.json``.
        k: Default retrieval depth for questions without their own ``eval_k``.
        mode: Retrieval mode -- ``"dense"``, ``"bm25"``, or ``"hybrid"``.
        gold_section_sizes: Chunk count per ``(doc_id, section)``, used to
            compute each question's precision ceiling. Ceilings are reported
            as zero when omitted.

    Returns:
        Dict with the mode, k, ``mean_precision_at_k``, ``mean_recall_at_k``,
        ``mean_max_precision_at_k``, ``precision_attainment``, and a
        ``per_query`` list of per-question scores.
    """
    gold_section_sizes = gold_section_sizes or {}
    precisions, recalls, ceilings, rrs, per_query = [], [], [], [], []
    for item in eval_set:
        # Each question carries its own k, matched to how many chunks in the
        # corpus genuinely answer it. Scoring a one-chunk answer at k=5 caps
        # precision at 0.20 no matter how good retrieval is, which measures
        # the answer's size rather than the retriever.
        question_k = item.get("eval_k", k)
        retrieved = index.search(item["query"], k=question_k, mode=mode)
        precision, recall = score_retrieval(
            retrieved, item["gold_doc_id"], item["gold_section"]
        )
        # The best precision physically attainable for this question, used to
        # report attainment rather than a raw figure that can never reach 1.0.
        gold_size = gold_section_sizes.get((item["gold_doc_id"], item["gold_section"]), 0)
        ceiling = min(gold_size, question_k) / question_k if question_k else 0.0

        # MRR is measured over a fixed depth so it stays comparable across
        # tiers: a k=1 question would otherwise be scored on one result only.
        rr = reciprocal_rank(
            index.search(item["query"], k=MRR_DEPTH, mode=mode),
            item["gold_doc_id"],
            item["gold_section"],
        )

        precisions.append(precision)
        recalls.append(recall)
        ceilings.append(ceiling)
        rrs.append(rr)
        per_query.append(
            {
                "id": item["id"],
                "k": question_k,
                "question_type": item.get("question_type", "known-item"),
                "precision_at_k": precision,
                "hit_at_k": recall,
                "max_precision_at_k": ceiling,
                "precision_attainment": precision / ceiling if ceiling else 0.0,
                "reciprocal_rank": rr,
            }
        )

    total_ceiling = sum(ceilings)
    return {
        "mode": mode,
        "k": k,
        "mean_precision_at_k": sum(precisions) / len(precisions),
        "mean_recall_at_k": sum(recalls) / len(recalls),
        # Mean attainable precision (< 1.0 whenever a gold section holds
        # fewer than k chunks) and how much of it was actually achieved.
        "mean_max_precision_at_k": total_ceiling / len(ceilings),
        "precision_attainment": sum(precisions) / total_ceiling if total_ceiling else 0.0,
        "mrr": sum(rrs) / len(rrs),
        "per_query": per_query,
    }
