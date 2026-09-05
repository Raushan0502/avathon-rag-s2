"""
Empirical embedding-model comparison on this corpus and these questions.

The embedding model was the one algorithm choice argued *analytically*
(MTEB standing, parameter count, CPU cost) rather than measured. Leaderboard
position is evidence about generic benchmarks, not about SEC filings, NIST
publications and business email -- so this script tests the claim directly.

Everything is held constant except the model: the same 8,146 chunks, the
same contextual prefixes, the same FAISS ``IndexFlatIP``, the same 45
questions with their per-question ``eval_k``. Only the vectors change.

Cost is reported alongside quality, because "which model is best" is not a
useful question for a CPU-only pipeline -- "which model is worth its
embedding time" is. bge-large is ~8.6x slower per chunk than bge-small,
so it has to earn that.

Every vector goes through the content-addressed cache, so the expensive
pass happens once per model and every later re-run is free.

Usage:
    python scripts/compare_embeddings.py
"""
import json
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import CHUNKS_PATH, EMBEDDING_MODEL_NAME, EVAL_SET_PATH, RESULTS_DIR
from src.embed_cache import cache_stats, embed_cached
from src.embed_index import build_index, embed_texts
from src.evaluate import evaluate_retrieval
from src.retrieval import TOKEN_RE, RetrievalIndex

# Candidates span a real trade-off axis rather than being arbitrary picks:
# the incumbent, and the same family at 3x the size to test whether capacity
# actually helps on this corpus.
MODELS = [
    {"name": EMBEDDING_MODEL_NAME, "label": "bge-small (incumbent)", "dim": 384},
    {"name": "BAAI/bge-large-en-v1.5", "label": "bge-large", "dim": 1024},
]
MODE = "dense"  # isolate the embedding model; BM25 is identical across runs
# Subset size, overridable with --chunks N. Every gold chunk is always
# included; the remainder are distractors sampled deterministically. Small
# by default because a full-corpus bge-large pass took over ten hours on
# CPU and had to be abandoned.
DEFAULT_CHUNKS = 200
EMBED_BATCH = 32
SAMPLE_SEED = 42


def select_chunks(chunks: list[dict], eval_set: list[dict], limit: int) -> list[dict]:
    """Take a subset that keeps every gold chunk, plus deterministic distractors.

    A comparison is only meaningful if each question's answer is still in the
    corpus, so gold chunks are never dropped. The remainder are sampled with a
    fixed seed so re-runs and the cache stay aligned.

    Args:
        chunks: All corpus chunks.
        eval_set: Evaluation questions.
        limit: Target subset size; gold chunks alone may exceed it.

    Returns:
        Chunks in their original corpus order.
    """
    gold_sections = {(q["gold_doc_id"], q["gold_section"]) for q in eval_set}
    keep_index = {
        i for i, c in enumerate(chunks) if (c["doc_id"], c["section"]) in gold_sections
    }
    distractors = [i for i in range(len(chunks)) if i not in keep_index]
    room = max(0, limit - len(keep_index))
    keep_index |= set(random.Random(SAMPLE_SEED).sample(distractors, min(room, len(distractors))))
    return [chunks[i] for i in sorted(keep_index)]


def evaluate_model(spec: dict, chunks: list[dict], eval_set: list[dict], sizes: dict) -> dict:
    """Embed the corpus with one model, index it, and score the eval set.

    Args:
        spec: Model entry from ``MODELS``.
        chunks: Corpus chunks, identical across models.
        eval_set: The 45 evaluation questions.
        sizes: Chunk count per ``(doc_id, section)``, for precision ceilings.

    Returns:
        Metrics plus embedding and query timings for this model.
    """
    name = spec["name"]
    texts = [c.get("embed_text") or c["text"] for c in chunks]
    before = cache_stats(name, texts)
    print(f"\n=== {spec['label']}  ({name})")
    print(f"    cache: {before['cached']:,}/{before['unique']:,} present, "
          f"{before['to_embed']:,} to embed")

    model = SentenceTransformer(name)
    started = time.perf_counter()

    def report(done: int, total: int) -> None:
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0
        remaining = (total - done) / rate if rate else 0
        print(f"      {done:>5}/{total} vectors  {elapsed:6.0f}s elapsed  "
              f"{rate:5.1f}/s  ~{remaining / 60:4.1f} min left", flush=True)

    vectors = embed_cached(
        model,
        name,
        texts,
        lambda m, batch: embed_texts(m, batch, is_query=False),
        batch_size=EMBED_BATCH,
        on_progress=report,
    )
    embed_seconds = time.perf_counter() - started
    print(f"    embedded in {embed_seconds / 60:.1f} min  -> {vectors.shape}")

    index = RetrievalIndex(
        embedder=model,
        faiss_index=build_index(vectors),
        bm25_index=BM25Okapi([TOKEN_RE.findall(c["text"].lower()) for c in chunks]),
        chunk_dicts=chunks,
    )

    started = time.perf_counter()
    metrics = evaluate_retrieval(index, eval_set, k=5, mode=MODE, gold_section_sizes=sizes)
    query_seconds = time.perf_counter() - started

    return {
        "model": name,
        "label": spec["label"],
        "dim": int(vectors.shape[1]),
        "mrr": metrics["mrr"],
        "mean_recall_at_k": metrics["mean_recall_at_k"],
        "mean_precision_at_k": metrics["mean_precision_at_k"],
        "precision_attainment": metrics["precision_attainment"],
        "embed_minutes": embed_seconds / 60,
        "chunks_newly_embedded": before["to_embed"],
        "query_ms_per_question": query_seconds / len(eval_set) * 1000,
        "per_query": metrics["per_query"],
    }


def main() -> None:
    """Compare every candidate model and save the results table."""
    limit = DEFAULT_CHUNKS
    if "--chunks" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--chunks") + 1])
    all_chunks = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    chunks = select_chunks(all_chunks, eval_set, limit)
    print(f"subset: {len(chunks):,} of {len(all_chunks):,} chunks "
          f"(all gold chunks retained; --chunks N to widen)")
    sizes: dict[tuple[str, str], int] = {}
    for chunk in chunks:
        key = (chunk["doc_id"], chunk["section"])
        sizes[key] = sizes.get(key, 0) + 1

    print(f"{len(eval_set)} questions | mode={MODE}\n")
    results = [evaluate_model(spec, chunks, eval_set, sizes) for spec in MODELS]

    print(f"\n{'model':<24}{'dim':>6}{'MRR':>8}{'R@k':>8}{'P@k':>8}{'attain':>8}"
          f"{'embed min':>11}{'ms/query':>10}")
    for r in results:
        print(f"{r['label']:<24}{r['dim']:>6}{r['mrr']:>8.3f}{r['mean_recall_at_k']:>8.3f}"
              f"{r['mean_precision_at_k']:>8.3f}{r['precision_attainment']:>8.0%}"
              f"{r['embed_minutes']:>11.1f}{r['query_ms_per_question']:>10.1f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "embedding_comparison.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
