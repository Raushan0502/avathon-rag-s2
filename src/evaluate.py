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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def is_relevant(chunk: dict, gold_doc_id: str, gold_section: str) -> bool:
    return chunk["doc_id"] == gold_doc_id and chunk["section"] == gold_section


def precision_at_k(retrieved: list[dict], gold_doc_id: str, gold_section: str) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for c in retrieved if is_relevant(c, gold_doc_id, gold_section))
    return hits / len(retrieved)


def hit_at_k(retrieved: list[dict], gold_doc_id: str, gold_section: str) -> float:
    """Recall@k under a single-known-relevant-item ground truth: 1.0 if the
    relevant section appears anywhere in the top-k, else 0.0."""
    return 1.0 if any(is_relevant(c, gold_doc_id, gold_section) for c in retrieved) else 0.0


def evaluate_retrieval(index, eval_set: list[dict], k: int, mode: str) -> dict:
    precisions, hits = [], []
    per_query = []
    for item in eval_set:
        retrieved = index.search(item["query"], k=k, mode=mode)
        p = precision_at_k(retrieved, item["gold_doc_id"], item["gold_section"])
        h = hit_at_k(retrieved, item["gold_doc_id"], item["gold_section"])
        precisions.append(p)
        hits.append(h)
        per_query.append({"id": item["id"], "precision_at_k": p, "hit_at_k": h})

    return {
        "mode": mode,
        "k": k,
        "mean_precision_at_k": sum(precisions) / len(precisions),
        "mean_recall_at_k": sum(hits) / len(hits),
        "per_query": per_query,
    }
