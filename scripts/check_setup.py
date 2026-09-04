"""Sanity check for Step 1: confirms the environment and folder layout are in place."""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    MISTRAL_API_KEY,
    RESULTS_DIR,
    ROOT_DIR,
)


def main() -> None:
    """Report the Python version, configured LLM providers, and data dirs.

    Raises:
        SystemExit: If any required data directory is missing.
    """
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project root: {ROOT_DIR}")
    configured = [
        name
        for name, key in [("groq", GROQ_API_KEY), ("mistral", MISTRAL_API_KEY), ("gemini", GEMINI_API_KEY)]
        if key
    ]
    print(f"LLM providers configured: {configured or 'NONE'}")

    required_dirs = [DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_DIR]
    for d in required_dirs:
        status = "OK" if d.exists() else "MISSING"
        print(f"  [{status}] {d}")

    missing = [d for d in required_dirs if not d.exists()]
    if missing:
        raise SystemExit(f"Setup incomplete: missing {missing}")

    print("Step 1 setup check passed.")


if __name__ == "__main__":
    main()
