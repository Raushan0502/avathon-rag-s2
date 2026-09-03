"""Sanity check for Step 1: confirms the environment and folder layout are in place."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_DIR, ROOT_DIR, LLM_PROVIDER


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project root: {ROOT_DIR}")
    print(f"LLM provider configured: {LLM_PROVIDER}")

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
