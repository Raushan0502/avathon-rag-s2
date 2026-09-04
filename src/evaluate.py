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
"""

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


def evaluate_retrieval(index, eval_set: list[dict], k: int, mode: str) -> dict:
    """Run every eval question through one retrieval mode and aggregate scores.

    Args:
        index: A ``RetrievalIndex`` (anything exposing ``search(query, k, mode)``).
        eval_set: Questions loaded from ``data/eval/qa_eval.json``.
        k: Number of chunks to retrieve per query.
        mode: Retrieval mode -- ``"dense"``, ``"bm25"``, or ``"hybrid"``.

    Returns:
        Dict with the mode, k, ``mean_precision_at_k``, ``mean_recall_at_k``,
        and a ``per_query`` list of per-question scores.
    """
    precisions, recalls, per_query = [], [], []
    for item in eval_set:
        retrieved = index.search(item["query"], k=k, mode=mode)
        precision, recall = score_retrieval(
            retrieved, item["gold_doc_id"], item["gold_section"]
        )
        precisions.append(precision)
        recalls.append(recall)
        per_query.append(
            {"id": item["id"], "precision_at_k": precision, "hit_at_k": recall}
        )

    return {
        "mode": mode,
        "k": k,
        "mean_precision_at_k": sum(precisions) / len(precisions),
        "mean_recall_at_k": sum(recalls) / len(recalls),
        "per_query": per_query,
    }
