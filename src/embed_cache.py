"""
Content-addressed embedding cache, so text is never embedded twice.

Embedding is the expensive, irreversible stage of this pipeline: ~19
minutes for the corpus with bge-small and ~165 with bge-large, all of it
CPU. Everything downstream -- building a FAISS index, switching Flat to
HNSW, re-running the evaluation, comparing retrieval modes -- is seconds.
Coupling them meant any experiment paid the full embedding cost again,
which is what made a second embedding model look unaffordable.

The cache is keyed by ``(model, sha256(text))``, which gives exactly the
invalidation behaviour wanted:

- **Re-running anything** with the same model and text is free.
- **Adding documents** embeds only the new chunks.
- **Changing chunking** re-embeds only the chunks whose text actually
  changed; untouched sections are reused.
- **Changing model** is a clean miss -- vectors from different models are
  not comparable, and keying on the model name makes that impossible to
  get wrong by accident.

Storage is a matrix plus a hash->row map per model, rather than one file
per vector: appending is a concatenate, and lookup is a dict hit. The
cache is a derived artifact and is gitignored -- deleting it costs time,
never correctness.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_PROCESSED_DIR

CACHE_DIR = DATA_PROCESSED_DIR / "embed_cache"


def _slug(model_name: str) -> str:
    """Filesystem-safe name for a model id (``BAAI/bge-small`` -> ``BAAI__bge-small``)."""
    return model_name.replace("/", "__")


def text_key(text: str) -> str:
    """Content hash used as the cache key for one piece of text.

    Args:
        text: The exact string that will be embedded, including any
            contextual prefix -- the prefix changes the vector, so it must
            change the key too.

    Returns:
        Hex SHA-256 digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache(model_name: str) -> tuple[dict[str, int], np.ndarray | None]:
    """Load one model's cached vectors.

    Args:
        model_name: Embedding model id.

    Returns:
        ``(key_to_row, matrix)``. Both are empty/None when no cache exists
        yet, so a first run behaves like a cold start rather than failing.
    """
    slug = _slug(model_name)
    keys_path, matrix_path = CACHE_DIR / f"{slug}.json", CACHE_DIR / f"{slug}.npy"
    if not keys_path.exists() or not matrix_path.exists():
        return {}, None
    return json.loads(keys_path.read_text(encoding="utf-8")), np.load(matrix_path)


def save_cache(model_name: str, key_to_row: dict[str, int], matrix: np.ndarray) -> None:
    """Persist one model's cache, replacing any previous version.

    Args:
        model_name: Embedding model id.
        key_to_row: Content hash -> row index in ``matrix``.
        matrix: Embedding matrix, one row per cached text.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug(model_name)
    np.save(CACHE_DIR / f"{slug}.npy", matrix)
    (CACHE_DIR / f"{slug}.json").write_text(json.dumps(key_to_row), encoding="utf-8")


def embed_cached(model, model_name: str, texts: list[str], encode_fn) -> np.ndarray:
    """Embed texts, reusing cached vectors and embedding only the misses.

    Args:
        model: Loaded embedding model, passed through to ``encode_fn``.
        model_name: Model id, used as part of the cache key.
        texts: Texts to embed, in the order the result should follow.
        encode_fn: ``(model, texts) -> np.ndarray`` doing the real work for
            cache misses. Injected so this module stays independent of how
            encoding is configured.

    Returns:
        Array of shape ``(len(texts), dim)`` aligned with ``texts``.
        Duplicate texts are embedded once and reused.
    """
    key_to_row, matrix = load_cache(model_name)
    keys = [text_key(text) for text in texts]

    # Deduplicate within this batch as well as against the cache: a repeated
    # boilerplate paragraph should cost one forward pass, not many.
    missing = list(dict.fromkeys(k for k in keys if k not in key_to_row))
    if missing:
        by_key = dict(zip(keys, texts))
        fresh = encode_fn(model, [by_key[k] for k in missing])
        matrix = fresh if matrix is None else np.vstack([matrix, fresh])
        for offset, key in enumerate(missing):
            key_to_row[key] = len(matrix) - len(missing) + offset
        save_cache(model_name, key_to_row, matrix)

    return np.vstack([matrix[key_to_row[k]] for k in keys]).astype("float32")


def cache_stats(model_name: str, texts: list[str]) -> dict:
    """Report how much of a batch is already cached, without embedding.

    Args:
        model_name: Embedding model id.
        texts: Texts that would be embedded.

    Returns:
        Dict with ``total``, ``cached``, ``to_embed`` and ``hit_rate``.
    """
    key_to_row, _ = load_cache(model_name)
    keys = {text_key(text) for text in texts}
    cached = sum(1 for key in keys if key in key_to_row)
    return {
        "total": len(texts),
        "unique": len(keys),
        "cached": cached,
        "to_embed": len(keys) - cached,
        "hit_rate": cached / len(keys) if keys else 0.0,
    }
