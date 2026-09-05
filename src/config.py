import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

MANIFEST_PATH = DATA_RAW_DIR / "manifest.json"
EVAL_SET_PATH = ROOT_DIR / "data" / "eval" / "qa_eval.json"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
FAISS_INDEX_PATH = DATA_PROCESSED_DIR / "index.faiss"
CHUNKS_PATH = DATA_PROCESSED_DIR / "chunks.jsonl"

# Generation provider keys, tried in this order (see src/generation.py for
# why: free-tier, zero-cost, and a fallback chain so one provider's outage
# doesn't stall the demo). Any subset may be set -- unset ones are skipped.
# Generation model ids live here with every other tunable. All three broke
# at least once during development (deprecated Gemini aliases, a retired
# Groq model), so they are the values most likely to need changing.
GROQ_MODEL = "openai/gpt-oss-120b"
MISTRAL_MODEL = "mistral-small-latest"
GEMINI_MODEL = "gemini-flash-latest"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
