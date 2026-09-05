"""
Stage-by-stage latency benchmark for the query path.

Track D asks how the system would handle 1,000 concurrent queries and where
the bottlenecks are. That is unanswerable without knowing which stage costs
what, so this measures each stage separately over the real corpus and the
real evaluation questions.

Stages measured, in the order a query passes through them:

  1. query embedding  -- one forward pass through bge-small
  2. dense search     -- FAISS exact inner product over every vector
  3. bm25 search      -- sparse scoring over every document
  4. hybrid search    -- both of the above plus RRF
  5. generation       -- the external LLM call

Reported as p50/p95 rather than means: a mean hides the tail, and the tail
is what determines capacity under load. Generation is measured separately
and optionally, because it is a network call to a third party and dominates
everything else by two orders of magnitude -- which is itself the headline
result.

Usage:
    python scripts/benchmark_pipeline.py            # retrieval stages only
    python scripts/benchmark_pipeline.py --with-llm # include generation
"""
import json
import statistics
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EVAL_SET_PATH, RESULTS_DIR
from src.embed_index import embed_texts
from src.generation import generate_answer
from src.retrieval import RetrievalIndex

WARMUP = 3
TOP_K = 5
# Enough spacing to stay under the free tier's limit, so the paced run
# measures the model rather than our own backoff.
GENERATION_PAUSE_SECONDS = 20.0


def timed(fn, queries: list[str], warmup: int = WARMUP, pause_seconds: float = 0.0) -> dict:
    """Time a single-argument callable across queries, after warming up."""
    for query in queries[:warmup]:
        fn(query)

    samples = []
    for position, query in enumerate(queries):
        if pause_seconds and position:
            time.sleep(pause_seconds)
        started = time.perf_counter()
        fn(query)
        samples.append((time.perf_counter() - started) * 1000)

    samples.sort()
    p50 = statistics.median(samples)
    return {
        "n": len(samples),
        "p50_ms": p50,
        "p95_ms": samples[int(len(samples) * 0.95) - 1],
        "mean_ms": statistics.fmean(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "qps_single_thread": 1000 / p50 if p50 else 0.0,
    }


def main() -> None:
    """Benchmark each query stage and save the results."""
    with_llm = "--with-llm" in sys.argv
    queries = [q["query"] for q in json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))]

    print("Loading index...")
    index = RetrievalIndex.load()
    print(f"{len(index.chunk_dicts):,} chunks | {len(queries)} queries\n")

    stages = {
        "1_query_embedding": lambda q: embed_texts(index.embedder, [q], is_query=True),
        "2_dense_search": lambda q: index.search(q, k=TOP_K, mode="dense"),
        "3_bm25_search": lambda q: index.search(q, k=TOP_K, mode="bm25"),
        "4_hybrid_search": lambda q: index.search(q, k=TOP_K, mode="hybrid"),
    }
    results = {}
    for name, fn in stages.items():
        results[name] = timed(fn, queries)
        row = results[name]
        print(f"  {name:<20} p50={row['p50_ms']:8.2f} ms  p95={row['p95_ms']:8.2f} ms  "
              f"{row['qps_single_thread']:7.1f} q/s")

    if with_llm:
        # Two separate numbers, because conflating them overstates the model's
        # cost by ~4x. Fired back to back, the free tier throttles from about
        # the fourth call and call_llm's exponential backoff dominates the
        # measurement -- that is a rate-limit ceiling, not inference speed.
        print("\n  measuring generation, paced (true single-call latency)...")
        results["5_generation_paced"] = timed(
            lambda q: generate_answer(q, index.search(q, k=TOP_K, mode="hybrid")),
            queries[:6],
            warmup=1,
            pause_seconds=GENERATION_PAUSE_SECONDS,
        )
        print("  measuring generation, back to back (sustained throughput)...")
        results["6_generation_sustained"] = timed(
            lambda q: generate_answer(q, index.search(q, k=TOP_K, mode="hybrid")),
            queries[6:14],
            warmup=0,
        )
        for label in ("5_generation_paced", "6_generation_sustained"):
            row = results[label]
            print(f"  {label:<24} p50={row['p50_ms']:9.0f} ms  p95={row['p95_ms']:9.0f} ms  "
                  f"{row['qps_single_thread']:6.2f} q/s")

    results["corpus"] = {"chunks": len(index.chunk_dicts), "top_k": TOP_K}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "latency_benchmark.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
