"""
Print every metric the README quotes, read from the results/ artifacts.

The README is prose and drifts; the JSON in results/ is generated. Three
figures in this project went stale or wrong between a pipeline change and the
paragraph describing it, so this script exists to make the documented numbers
checkable in one command rather than by memory.

Run it after any rebuild or re-evaluation and reconcile the README against
what it prints.

Usage:
    python scripts/report_metrics.py
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHUNKS_PATH, RESULTS_DIR


def load(name: str) -> dict | list | None:
    """Read a results artifact, returning None when it has not been generated."""
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def report_corpus() -> None:
    """Print corpus and chunk counts from the built index."""
    print("== Corpus ==")
    if not CHUNKS_PATH.exists():
        print("  chunks.jsonl missing -- run scripts/build_index.py")
        return
    docs, sections, chunks = set(), set(), 0
    for line in CHUNKS_PATH.open(encoding="utf-8"):
        c = json.loads(line)
        docs.add(c["doc_id"])
        sections.add((c["doc_id"], c["section"]))
        chunks += 1
    print(f"  documents {len(docs)}   sections {len(sections)}   chunks {chunks}")


def report_retrieval() -> None:
    """Print per-mode retrieval metrics and the k-tier breakdown."""
    print("\n== Retrieval (results/retrieval_eval.json) ==")
    data = load("retrieval_eval.json")
    if not data:
        print("  missing -- run scripts/run_evaluation.py")
        return
    print(f"  {'mode':8s} {'P@k':>6s} {'R@k':>6s} {'MRR':>6s} {'ceiling':>8s} {'attain':>7s}")
    for mode in ("dense", "bm25", "hybrid"):
        m = data[mode]
        print(
            f"  {mode:8s} {m['mean_precision_at_k']:6.3f} {m['mean_recall_at_k']:6.3f} "
            f"{m['mrr']:6.3f} {m['mean_max_precision_at_k']:8.3f} "
            f"{m['precision_attainment']:6.1%}"
        )
    per = data["dense"]["per_query"]
    print("  dense by k tier:")
    for tier in sorted({r["k"] for r in per}):
        rows = [r for r in per if r["k"] == tier]
        p = sum(r["precision_at_k"] for r in rows) / len(rows)
        r_ = sum(r["hit_at_k"] for r in rows) / len(rows)
        print(f"    k={tier:<3} n={len(rows):<3} P@k={p:.3f}  R@k={r_:.3f}")


def report_generation() -> None:
    """Print faithfulness distribution and retrieval hit rate."""
    print("\n== Generation (results/qa_eval_results.json) ==")
    recs = load("qa_eval_results.json")
    if not recs:
        print("  missing -- run scripts/run_evaluation.py")
        return
    counts: dict[str, int] = {}
    for r in recs:
        flag = r["faithfulness"]["faithfulness_flag"]
        counts[flag] = counts.get(flag, 0) + 1
    hits = sum(r["retrieval_hit_at_k"] for r in recs)
    print(f"  questions {len(recs)}   faithfulness {counts}")
    print(f"  retrieval hit@k {hits}/{len(recs)} = {hits / len(recs):.3f}")


def report_accuracy() -> None:
    """Print lexical and judge answer-accuracy figures."""
    print("\n== Answer accuracy (results/answer_accuracy.json) ==")
    s = load("answer_accuracy.json")
    if not s:
        print("  missing -- run scripts/score_answers.py --judge")
        return
    print(f"  answered {s['answered']}   refused {s['refused']}")
    print(f"  mean key-fact recall {s['mean_key_fact_recall']:.3f}")
    print(f"  lexical accuracy     {s['lexical_accuracy']:.1%}")
    verdicts = s.get("judge_verdicts") or {}
    if not verdicts:
        print("  judge: NOT RUN -- README must not quote judge figures")
        return
    total = sum(verdicts.values())
    c, p = verdicts.get("CORRECT", 0), verdicts.get("PARTIAL", 0)
    print(f"  judge verdicts       {verdicts}")
    print(f"  judge CORRECT        {c / total:.1%} ({c}/{total})")
    print(f"  judge CORRECT+PARTIAL {(c + p) / total:.1%} ({c + p}/{total})")


def report_embeddings() -> None:
    """Print the embedding-model comparison."""
    print("\n== Embedding comparison (results/embedding_comparison.json) ==")
    data = load("embedding_comparison.json")
    if not data:
        print("  missing -- run scripts/compare_embeddings.py")
        return
    for m in data:
        print(
            f"  {m['label']:22s} dim={m['dim']:<5} MRR={m['mrr']:.3f} "
            f"R@k={m['mean_recall_at_k']:.3f} P@k={m['mean_precision_at_k']:.3f} "
            f"attain={m['precision_attainment']:.1%} query={m['query_ms_per_question']:.0f}ms"
        )


def report_latency() -> None:
    """Print stage latencies."""
    print("\n== Latency (results/latency_benchmark.json) ==")
    data = load("latency_benchmark.json")
    if not data:
        print("  missing -- run scripts/benchmark_pipeline.py")
        return
    for stage, m in data.items():
        if isinstance(m, dict) and "p50_ms" in m:
            print(f"  {stage:24s} p50={m['p50_ms']:8.1f}ms  p95={m['p95_ms']:8.1f}ms")


def main() -> int:
    """Print every documented metric from its source artifact."""
    report_corpus()
    report_retrieval()
    report_generation()
    report_accuracy()
    report_embeddings()
    report_latency()
    print("\nReconcile README.md against the figures above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
