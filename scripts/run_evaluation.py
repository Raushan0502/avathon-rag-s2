"""
Step 6 — Full evaluation harness.

1. Retrieval quality: for each mode (dense, bm25, hybrid), compute mean
   Precision@k and Recall@k (see src/evaluate.py for what these mean given
   this eval set's ground-truth shape) over all 24 questions in
   data/eval/qa_eval.json. This is the quantitative dense-vs-hybrid
   comparison Step 4's write-up question asks for.
2. End-to-end faithfulness: using hybrid retrieval (the mode the system
   would actually run in), generate an answer for every question and
   record the faithfulness annotation (cited / refused / UNGROUNDED).

Saves results/retrieval_eval.json (metrics) and
results/qa_eval_results.json (full per-question trace: retrieved context,
generated answer, faithfulness flag, whether retrieval actually found the
gold section).

Usage:
    python scripts/run_evaluation.py
"""
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EVAL_SET_PATH, RESULTS_DIR
from src.evaluate import evaluate_retrieval, score_retrieval
from src.generation import generate_answer
from src.retrieval import RetrievalIndex

K = 5


def run_retrieval_comparison(index: RetrievalIndex, eval_set: list[dict]) -> dict:
    """Score every eval question under each retrieval mode.

    Args:
        index: Loaded retrieval index.
        eval_set: Questions from ``data/eval/qa_eval.json``.

    Returns:
        Metrics keyed by mode name ("dense", "bm25", "hybrid").
    """
    print(f"\n=== Retrieval comparison (k={K}, n={len(eval_set)} questions) ===")
    results = {}
    for mode in ["dense", "bm25", "hybrid"]:
        metrics = evaluate_retrieval(index, eval_set, k=K, mode=mode)
        results[mode] = metrics
        print(
            f"  {mode:7s}  mean P@{K} = {metrics['mean_precision_at_k']:.3f}   "
            f"mean R@{K} = {metrics['mean_recall_at_k']:.3f}"
        )
    return results


def run_generation_trace(index: RetrievalIndex, eval_set: list[dict]) -> list[dict]:
    """Answer every eval question with hybrid retrieval and record the trace.

    Args:
        index: Loaded retrieval index.
        eval_set: Questions from ``data/eval/qa_eval.json``.

    Returns:
        One record per question: the generated answer, which provider served
        it, its faithfulness annotation, whether retrieval found the gold
        section, and the retrieved context.
    """
    print(f"\n=== End-to-end generation + faithfulness (hybrid, k={K}) ===")
    records = []
    faithfulness_counts = {"cited": 0, "refused": 0, "UNGROUNDED": 0}

    for item in eval_set:
        retrieved = index.search(item["query"], k=K, mode="hybrid")
        _, recall = score_retrieval(retrieved, item["gold_doc_id"], item["gold_section"])
        retrieval_hit = recall == 1.0

        result = generate_answer(item["query"], retrieved)
        flag = result["faithfulness"]["faithfulness_flag"]
        faithfulness_counts[flag] = faithfulness_counts.get(flag, 0) + 1

        print(f"  {item['id']}: retrieval_hit={retrieval_hit}  faithfulness={flag}  provider={result['provider']}")

        records.append(
            {
                "id": item["id"],
                "query": item["query"],
                "reference_answer": item["reference_answer"],
                "gold_doc_id": item["gold_doc_id"],
                "gold_section": item["gold_section"],
                "retrieval_hit_at_k": retrieval_hit,
                "generated_answer": result["answer"],
                "provider": result["provider"],
                "faithfulness": result["faithfulness"],
                "retrieved_context": [
                    {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "section": c["section"], "score": c["score"]}
                    for c in retrieved
                ],
            }
        )

    print(f"\n  faithfulness distribution: {faithfulness_counts}")
    return records


def main() -> None:
    """Run the retrieval comparison and generation trace, saving both to results/."""
    t0 = time.time()
    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(eval_set)} eval questions from {EVAL_SET_PATH}")

    print("Loading retrieval index...")
    index = RetrievalIndex.load()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Retrieval needs no network and is the more expensive half to recompute,
    # so it is persisted before generation is attempted. An earlier run lost
    # a completed retrieval comparison because every LLM provider was down
    # and the crash took the finished work with it.
    retrieval_results = run_retrieval_comparison(index, eval_set)
    (RESULTS_DIR / "retrieval_eval.json").write_text(
        json.dumps(retrieval_results, indent=2), encoding="utf-8"
    )

    try:
        generation_records = run_generation_trace(index, eval_set)
    except Exception as exc:  # noqa: BLE001 -- provider outage must not discard the above
        print(f"\nGeneration stage failed: {exc}")
        print(f"Retrieval metrics were still saved to {RESULTS_DIR / 'retrieval_eval.json'}")
        return

    (RESULTS_DIR / "qa_eval_results.json").write_text(
        json.dumps(generation_records, indent=2), encoding="utf-8"
    )
    print(f"\nSaved results/retrieval_eval.json and results/qa_eval_results.json")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
