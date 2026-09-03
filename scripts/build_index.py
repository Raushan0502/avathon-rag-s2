"""
Step 3 — Orchestrates the full ingestion pipeline: load raw filings ->
clean -> section-aware chunk -> embed -> build FAISS index -> persist.

Ends with a smoke test: a few hardcoded queries are embedded and searched
against the freshly built index, with top results printed, so a broken
pipeline (empty index, garbage chunks, mismatched embedding dims) fails
loudly here rather than silently in Step 4/5.

Usage:
    python scripts/build_index.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHUNKS_PATH, FAISS_INDEX_PATH
from src.embed_index import build_faiss_index, embed_texts, get_embedder, save_chunks, save_index
from src.ingest import build_all_chunks, chunks_to_dicts

SMOKE_TEST_QUERIES = [
    "What are the main risk factors related to competition?",
    "What did the company report about its quarterly revenue or financial results?",
    "What legal proceedings is the company involved in?",
]


def print_chunk_stats(chunk_dicts: list[dict]) -> None:
    word_counts = [c["word_count"] for c in chunk_dicts]
    docs = {c["doc_id"] for c in chunk_dicts}
    sections = {(c["doc_id"], c["section"]) for c in chunk_dicts}
    print(f"\nCorpus stats:")
    print(f"  documents: {len(docs)}")
    print(f"  sections:  {len(sections)}")
    print(f"  chunks:    {len(chunk_dicts)}")
    print(f"  words/chunk: min={min(word_counts)} max={max(word_counts)} "
          f"avg={sum(word_counts) / len(word_counts):.0f}")


def run_smoke_test(embedder, index, chunk_dicts: list[dict]) -> None:
    print("\nSmoke test -- sample retrievals:")
    query_vecs = embed_texts(embedder, SMOKE_TEST_QUERIES, is_query=True)
    scores, indices = index.search(query_vecs, k=3)
    for query, row_scores, row_indices in zip(SMOKE_TEST_QUERIES, scores, indices):
        print(f"\n  Q: {query}")
        for rank, (score, idx) in enumerate(zip(row_scores, row_indices), start=1):
            c = chunk_dicts[idx]
            snippet = c["text"][:140].replace("\n", " ")
            print(f"    {rank}. [{score:.3f}] {c['ticker']} {c['form']} / {c['section']}: {snippet}...")


def main() -> None:
    t0 = time.time()

    print("Loading + chunking documents...")
    chunks = build_all_chunks()
    chunk_dicts = chunks_to_dicts(chunks)
    print_chunk_stats(chunk_dicts)

    print("\nLoading embedding model (BAAI/bge-small-en-v1.5)...")
    embedder = get_embedder()

    print("Embedding chunks...")
    texts = [c["text"] for c in chunk_dicts]
    embeddings = embed_texts(embedder, texts, is_query=False)
    print(f"  embeddings shape: {embeddings.shape}")

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    save_index(index, FAISS_INDEX_PATH)
    save_chunks(chunk_dicts, CHUNKS_PATH)
    print(f"\nSaved index -> {FAISS_INDEX_PATH}")
    print(f"Saved chunks -> {CHUNKS_PATH}")

    run_smoke_test(embedder, index, chunk_dicts)

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
