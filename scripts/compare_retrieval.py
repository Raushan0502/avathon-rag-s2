"""
Step 4 — Qualitative smoke test for hybrid vs. dense-only retrieval.

This is not the rigorous comparison (that's Step 6: Precision@k/Recall@k
over the 20+ QA held-out set, for every mode). This script exists to
prove the hybrid path actually changes rankings before that harness is
built on top of it, and to surface at least one concrete example of BM25
rescuing an exact-term match dense search under-ranks.

Usage:
    python scripts/compare_retrieval.py
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval import RetrievalIndex

QUERIES = [
    "What are the main risk factors related to competition?",
    "What legal proceedings is the company involved in?",
    "What did the company say about its Copilot product?",
    "What did the company say about Cybertruck production?",
    "How does the company manage supply chain and manufacturing risk?",
]


def main() -> None:
    """Print dense-only vs hybrid top-3 results side by side for each query.

    The overlap count is the signal to watch: consistently 3/3 would mean
    BM25 contributes nothing beyond dense retrieval and fusion is dead weight.
    """
    print("Loading retrieval index (embedder + FAISS + BM25)...")
    index = RetrievalIndex.load()

    for query in QUERIES:
        dense = index.search(query, k=3, mode="dense")
        hybrid = index.search(query, k=3, mode="hybrid")
        overlap = len({c["chunk_id"] for c in dense} & {c["chunk_id"] for c in hybrid})

        print(f"\nQ: {query}")
        print(f"  top-3 overlap (dense vs hybrid): {overlap}/3")
        for label, results in [("dense-only", dense), ("hybrid (RRF)", hybrid)]:
            print(f"  {label}:")
            for c in results:
                snippet = c["text"][:120].replace("\n", " ")
                print(
                    f"    [{c['score']:.4f}] {c['ticker']} {c['form']} "
                    f"/ {c['section']}: {snippet}..."
                )


if __name__ == "__main__":
    main()
