"""
Step 5 — End-to-end Q&A demo.

Runs a small set of representative queries through the full pipeline
(hybrid retrieval -> grounded generation -> faithfulness annotation) and
saves the trace to results/qa_demo.json -- this is the artifact the
Track D requirement "demonstrate end-to-end Q&A with at least 5
representative queries including retrieved context, generated answer,
and a faithfulness annotation" points at.

Usage:
    python scripts/demo_qa.py
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS_DIR
from src.generation import generate_answer
from src.retrieval import RetrievalIndex

QUERIES = [
    "What are the main risk factors related to competition?",
    "What legal proceedings is the company involved in?",
    "What did Microsoft say about its Copilot product?",
    "What does Tesla say about its manufacturing and production capacity?",
    "What did the company announce in its most recent 8-K filing?",
    "What is the company's dividend or capital return policy?",  # deliberately out-of-strength probe
]

TOP_K = 5


def main() -> None:
    print("Loading retrieval index...")
    index = RetrievalIndex.load()

    records = []
    for query in QUERIES:
        print(f"\nQ: {query}")
        retrieved = index.search(query, k=TOP_K, mode="hybrid")
        result = generate_answer(query, retrieved)

        print(f"  provider: {result['provider']}")
        print(f"  faithfulness: {result['faithfulness']['faithfulness_flag']}")
        print(f"  answer: {result['answer'][:300]}")

        records.append(
            {
                "query": query,
                "provider": result["provider"],
                "answer": result["answer"],
                "faithfulness": result["faithfulness"],
                "retrieved_context": [
                    {
                        "chunk_id": c["chunk_id"],
                        "ticker": c["ticker"],
                        "form": c["form"],
                        "section": c["section"],
                        "score": c["score"],
                        "text": c["text"],
                    }
                    for c in retrieved
                ],
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "qa_demo.json"
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nSaved {len(records)} Q&A traces -> {out_path}")


if __name__ == "__main__":
    main()
