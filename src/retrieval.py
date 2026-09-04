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

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import EMBEDDING_MODEL_NAME
from src.embed_index import embed_texts, load_artifacts

RRF_K = 60
TOKEN_RE = re.compile(r"[a-z0-9]+")
VALID_MODES = ("dense", "bm25", "hybrid")


def reciprocal_rank_fusion(
    rank_lists: list[list[int]], k_rrf: int = RRF_K
) -> list[tuple[int, float]]:
    """Fuse several ranked ID lists into one, by rank position only.

    Each list contributes ``1 / (k_rrf + rank)`` to every ID it ranks, so an
    item ranked highly by either retriever scores well, and an item ranked
    by both scores best. Raw scores are deliberately ignored -- dense cosine
    similarities and BM25 scores are on incomparable scales, and rank is the
    common currency that avoids having to calibrate between them.

    Args:
        rank_lists: One list of chunk indices per retriever, best-first.
        k_rrf: Smoothing constant; the standard 60 keeps any single
            retriever's top hit from dominating the fused ranking.

    Returns:
        ``(chunk_index, fused_score)`` pairs sorted best-first.
    """
    fused: dict[int, float] = {}
    for ranked_ids in rank_lists:
        for rank, chunk_idx in enumerate(ranked_ids):
            fused[chunk_idx] = fused.get(chunk_idx, 0.0) + 1.0 / (k_rrf + rank + 1)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)


class RetrievalIndex:
    """Dense (FAISS), sparse (BM25), and fused retrieval over one chunk set.

    Bundles the embedder, both indexes, and the chunk metadata so Step 5
    (generation) and Step 6 (evaluation) share a single loading path and a
    single ``search()`` entry point regardless of retrieval mode.
    """

    def __init__(
        self,
        embedder: SentenceTransformer,
        faiss_index: faiss.Index,
        bm25_index: BM25Okapi,
        chunk_dicts: list[dict],
    ):
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.chunk_dicts = chunk_dicts

    @classmethod
    def load(cls) -> "RetrievalIndex":
        """Load persisted artifacts and rebuild the in-memory BM25 index.

        The FAISS index and chunk metadata are read from disk (written by
        ``scripts/build_index.py``); the BM25 index is cheap to rebuild from
        the chunk text, so it is not persisted separately.

        Returns:
            A ready-to-query ``RetrievalIndex``.
        """
        faiss_index, chunk_dicts = load_artifacts()
        return cls(
            embedder=SentenceTransformer(EMBEDDING_MODEL_NAME),
            faiss_index=faiss_index,
            bm25_index=BM25Okapi(
                [TOKEN_RE.findall(c["text"].lower()) for c in chunk_dicts]
            ),
            chunk_dicts=chunk_dicts,
        )

    def search(
        self, query: str, k: int = 5, mode: str = "hybrid", candidate_pool: int = 20
    ) -> list[dict]:
        """Retrieve the top-k chunks for a query under the given mode.

        Args:
            query: Natural-language question.
            k: Number of chunks to return.
            mode: ``"dense"`` (FAISS cosine similarity), ``"bm25"`` (sparse
                keyword match), or ``"hybrid"`` (both, fused with RRF).
            candidate_pool: How many candidates each retriever contributes
                to fusion in hybrid mode. Larger than ``k`` on purpose, so
                fusion can promote a chunk that one retriever ranked outside
                its own top-k.

        Returns:
            Chunk dicts (copies, with ``score`` and ``retrieval_mode`` added)
            ordered best-first. ``score`` is a cosine similarity for dense, a
            BM25 score for bm25, and a fused RRF score for hybrid -- the
            scales are not comparable across modes.

        Raises:
            ValueError: If ``mode`` is not one of ``VALID_MODES``.
        """
        if mode not in VALID_MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {VALID_MODES}")

        def dense_hits(limit: int) -> list[tuple[int, float]]:
            query_vector = embed_texts(self.embedder, [query], is_query=True)
            scores, indices = self.faiss_index.search(query_vector, limit)
            return [
                (int(idx), float(score))
                for idx, score in zip(indices[0], scores[0])
                if idx != -1  # FAISS pads with -1 when fewer than `limit` vectors exist
            ]

        def bm25_hits(limit: int) -> list[tuple[int, float]]:
            scores = self.bm25_index.get_scores(TOKEN_RE.findall(query.lower()))
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return [(i, float(scores[i])) for i in ranked[:limit]]

        if mode == "dense":
            results = dense_hits(k)
        elif mode == "bm25":
            results = bm25_hits(k)
        else:
            results = reciprocal_rank_fusion(
                [
                    [idx for idx, _ in dense_hits(candidate_pool)],
                    [idx for idx, _ in bm25_hits(candidate_pool)],
                ]
            )[:k]

        return [
            {**self.chunk_dicts[idx], "score": score, "retrieval_mode": mode}
            for idx, score in results
        ]
