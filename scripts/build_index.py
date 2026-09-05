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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer

from src.config import CHUNKS_PATH, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH
from src.embed_index import build_index, embed_corpus, embed_texts, save_artifacts
from src.ingest import build_all_chunks

SMOKE_TEST_QUERIES = [
    "What are the main risk factors related to competition?",
    "What did the company report about its quarterly revenue or financial results?",
    "What legal proceedings is the company involved in?",
]


def main() -> None:
    """Chunk the corpus, embed it, build and persist the FAISS index, then
    smoke-test the result with a few sample retrievals."""
    started = time.time()

    print("Loading + chunking documents...")
    chunk_dicts = build_all_chunks()
    word_counts = [c["word_count"] for c in chunk_dicts]
    print("\nCorpus stats:")
    print(f"  documents: {len({c['doc_id'] for c in chunk_dicts})}")
    print(f"  sections:  {len({(c['doc_id'], c['section']) for c in chunk_dicts})}")
    print(f"  chunks:    {len(chunk_dicts)}")
    print(
        f"  words/chunk: min={min(word_counts)} max={max(word_counts)} "
        f"avg={sum(word_counts) / len(word_counts):.0f}"
    )

    print(f"\nLoading embedding model ({EMBEDDING_MODEL_NAME})...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding chunks...")
    # Embed the context-prefixed form (see ingest.build_embed_text), falling
    # back to raw text for any chunk built before that field existed. The
    # clean `text` is what retrieval returns and what the model quotes.
    embeddings = embed_texts(
        embedder, [c.get("embed_text") or c["text"] for c in chunk_dicts], is_query=False
    )
    print(f"  embeddings shape: {embeddings.shape}")

    print("Building FAISS index...")
    index = build_index(embeddings)
    save_artifacts(index, chunk_dicts)
    print(f"\nSaved index -> {FAISS_INDEX_PATH}")
    print(f"Saved chunks -> {CHUNKS_PATH}")

    # Smoke test: a broken pipeline (empty index, garbage chunks, mismatched
    # dims) should fail loudly here rather than silently in Step 4/5.
    print("\nSmoke test -- sample retrievals:")
    query_vectors = embed_texts(embedder, SMOKE_TEST_QUERIES, is_query=True)
    scores, indices = index.search(query_vectors, k=3)
    for query, row_scores, row_indices in zip(SMOKE_TEST_QUERIES, scores, indices):
        print(f"\n  Q: {query}")
        for rank, (score, idx) in enumerate(zip(row_scores, row_indices), start=1):
            chunk = chunk_dicts[idx]
            snippet = chunk["text"][:140].replace("\n", " ")
            print(
                f"    {rank}. [{score:.3f}] {chunk['ticker']} {chunk['form']} "
                f"/ {chunk['section']}: {snippet}..."
            )

    print(f"\nDone in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
