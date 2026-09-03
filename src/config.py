import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
FAISS_INDEX_PATH = DATA_PROCESSED_DIR / "index.faiss"
CHUNKS_PATH = DATA_PROCESSED_DIR / "chunks.jsonl"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def require_llm_key() -> str:
    key = ANTHROPIC_API_KEY if LLM_PROVIDER == "anthropic" else OPENAI_API_KEY
    if not key:
        raise RuntimeError(
            f"LLM_PROVIDER is '{LLM_PROVIDER}' but no matching API key is set in .env"
        )
    return key
