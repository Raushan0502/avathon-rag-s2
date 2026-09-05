"""
Extraction-quality gate, run after fetching and before building the index.

Extracts and preprocesses every document in the manifest, scores it, and
writes a per-document report to results/extraction_validation.json. Exits
non-zero if any document fails, so this can gate a rebuild rather than
being advisory only.

Usage:
    python scripts/validate_corpus.py
"""
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MANIFEST_PATH, RESULTS_DIR, ROOT_DIR
from src.extract import load_document_text
from src.preprocess import normalise_text
from src.validate import check_document, measure_document, summarise


def main() -> int:
    """Validate extraction for every manifest document."""
    started = time.time()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    print(f"Validating extraction for {len(manifest)} documents...\n")

    reports = []
    for entry in manifest:
        path = ROOT_DIR / entry["local_path"]
        doc_id = path.stem
        try:
            text = normalise_text(load_document_text(path))
            metrics = measure_document(text, path.stat().st_size)
            status, issues = check_document(metrics, entry["doc_type"])
        except Exception as exc:  # noqa: BLE001 -- a parser blowing up is a result
            text, metrics, status = "", measure_document("", 0), "fail"
            issues = [f"extraction raised {type(exc).__name__}: {exc}"]

        reports.append(
            {
                "doc_id": doc_id,
                "doc_type": entry["doc_type"],
                "format": entry["format"],
                "status": status,
                "issues": issues,
                "metrics": metrics,
            }
        )
        if status != "ok":
            print(f"  [{status.upper()}] {doc_id}")
            for issue in issues:
                print(f"          {issue}")

    summary = summarise(reports)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "extraction_validation.json").write_text(
        json.dumps({"summary": summary, "documents": reports}, indent=2), encoding="utf-8"
    )

    print(f"\nStatus: {summary['by_status']}")
    print("Mean metrics:")
    for name, value in summary["mean_metrics"].items():
        print(f"  {name:20s} {value:.3f}")
    print("\nBy document type:")
    for doc_type, counts in sorted(summary["by_doc_type"].items()):
        print(f"  {doc_type:10s} {counts}")

    print(f"\nReport -> {RESULTS_DIR / 'extraction_validation.json'}")
    print(f"Done in {time.time() - started:.1f}s")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
