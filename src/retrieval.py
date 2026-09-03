"""
Step 4 — Hybrid retrieval: dense (FAISS) + sparse (BM25) + Reciprocal Rank
Fusion (RRF).

Why hybrid at all: dense embeddings retrieve on semantic similarity, which
under-ranks a chunk whose relevance hinges on an exact rare term (a
product name, a specific line-item, a dollar figure) that isn't well
separated in embedding space from similar-sounding but irrelevant text.
BM25 is the mirror image: strong on exact term overlap, blind to
paraphrase. Fusing both catches queries where either failure mode would
otherwise dominate.

Why Reciprocal Rank Fusion over a trained cross-encoder re-ranker: RRF
needs no extra model and no extra per-candidate inference pass -- it
combines two rank lists using only rank position (score = sum of
1/(60 + rank) across retrievers), a well-established, parameter-light
baseline for exactly this dense+sparse fusion problem. A cross-encoder
would likely score somewhat higher, at the cost of a second forward pass
per candidate per query and a second model to justify/maintain. Rejected
here because the corpus is small and single-genre (six financial
filings), so the marginal accuracy a cross-encoder buys is unlikely to be
worth doubling inference cost; revisited in the write-up as a documented
trade-off, not an oversight.
"""
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.embed_index import embed_texts, get_embedder, load_chunks, load_index

RRF_K = 60
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_bm25_index(chunk_dicts: list[dict]) -> BM25Okapi:
    return BM25Okapi([tokenize(c["text"]) for c in chunk_dicts])


def _dense_search(query: str, embedder, faiss_index, k: int) -> list[tuple[int, float]]:
    query_vec = embed_texts(embedder, [query], is_query=True)
    scores, indices = faiss_index.search(query_vec, k)
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]


def _bm25_search(query: str, bm25_index: BM25Okapi, k: int) -> list[tuple[int, float]]:
    scores = bm25_index.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(i, float(scores[i])) for i in ranked]


def reciprocal_rank_fusion(rank_lists: list[list[int]], k_rrf: int = RRF_K) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for ranked_ids in rank_lists:
        for rank, chunk_idx in enumerate(ranked_ids):
            fused[chunk_idx] = fused.get(chunk_idx, 0.0) + 1.0 / (k_rrf + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class RetrievalIndex:
    """Bundles the embedder, FAISS dense index, and BM25 sparse index over
    the same chunk set so Step 5 (generation) and Step 6 (evaluation) share
    one loading path and one `search()` interface regardless of mode."""

    def __init__(self, embedder, faiss_index, bm25_index: BM25Okapi, chunk_dicts: list[dict]):
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.chunk_dicts = chunk_dicts

    @classmethod
    def load(cls) -> "RetrievalIndex":
        chunk_dicts = load_chunks()
        return cls(
            embedder=get_embedder(),
            faiss_index=load_index(),
            bm25_index=build_bm25_index(chunk_dicts),
            chunk_dicts=chunk_dicts,
        )

    def search(self, query: str, k: int = 5, mode: str = "hybrid", candidate_pool: int = 20) -> list[dict]:
        if mode == "dense":
            results = _dense_search(query, self.embedder, self.faiss_index, k)
        elif mode == "bm25":
            results = _bm25_search(query, self.bm25_index, k)
        elif mode == "hybrid":
            dense_ids = [idx for idx, _ in _dense_search(query, self.embedder, self.faiss_index, candidate_pool)]
            bm25_ids = [idx for idx, _ in _bm25_search(query, self.bm25_index, candidate_pool)]
            results = reciprocal_rank_fusion([dense_ids, bm25_ids])[:k]
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        return [
            {**self.chunk_dicts[idx], "score": score, "retrieval_mode": mode}
            for idx, score in results
        ]
