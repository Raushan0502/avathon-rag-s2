"""
Step 3 — Embedding & FAISS indexing.

Embedding model: BAAI/bge-small-en-v1.5 (33M params, 384-dim, CPU-friendly,
512-token context). Chosen over larger BGE/E5 variants because this corpus
is small (a handful of filings) and runs entirely on CPU with no GPU
budget available for this track -- bge-small consistently ranks near the
top of the MTEB retrieval leaderboard for its size class, giving most of
the retrieval quality of larger models at a fraction of the embedding
latency. The alternative considered and rejected was OpenAI's
text-embedding-3-small: better quality, but it would make every
ingestion run depend on a paid API call, which is unnecessary for a
retrieval component that isn't the generation step this track's cost
lives in.

BGE models are trained asymmetrically: queries should be prefixed with an
instruction, passages should not. That prefix is applied here for queries
only (see `embed_texts(..., is_query=True)`).

Vector store: FAISS, `IndexFlatIP` (exact inner-product search over
L2-normalized vectors, i.e. exact cosine similarity) -- chosen over an
approximate index (IVF/HNSW) because the corpus is on the order of
hundreds of chunks, where exact search is already sub-millisecond and
approximate indexing would only add tuning surface (nlist/nprobe, M/efSearch)
for no measurable benefit. This choice does not scale to a
production-sized corpus; see the write-up for where the switch to an
approximate index would become necessary.
"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CHUNKS_PATH, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(model: SentenceTransformer, texts: list[str], is_query: bool = False) -> np.ndarray:
    inputs = [QUERY_INSTRUCTION + t for t in texts] if is_query else texts
    embeddings = model.encode(
        inputs,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: Path = FAISS_INDEX_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path = FAISS_INDEX_PATH) -> faiss.Index:
    return faiss.read_index(str(path))


def save_chunks(chunk_dicts: list[dict], path: Path = CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunk_dicts:
            f.write(json.dumps(c) + "\n")


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
