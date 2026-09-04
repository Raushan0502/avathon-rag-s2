"""
Extraction-quality validation, run between preprocessing and chunking.

Extraction fails *silently*. A scanned PDF has no text layer, so it yields
an empty string rather than an error; a filing whose markup defeats the
parser yields a handful of characters. Either way the document is indexed
as though it were legitimately short, and the only symptom is an answer
that cannot be found later. At six documents a human could eyeball every
one; at a hundred that stops being true, so the checks below stand in for
reading them.

Each check targets a failure this corpus can actually produce:

- ``yield_ratio`` -- extracted characters per source byte. Near-zero means
  no text layer (a scan needing OCR, which this pipeline does not do) or a
  parser that gave up. This is the check that catches the silent-empty case.
- ``alpha_ratio`` -- share of characters that are letters or digits. A low
  value means the "text" is mostly punctuation: dot-leader tables of
  contents, ASCII rules, or navigation furniture.
- ``boilerplate_ratio`` -- share of lines that repeat within the document.
  High values mean page headers/footers survived preprocessing.
- ``table_row_ratio`` -- share of lines that are rendered table rows, which
  tells you whether a financial filing's tables actually survived
  extraction rather than collapsing into prose.
- ``mean_words_per_line`` -- very low values indicate text fragmented into
  single words per line, a common symptom of multi-column PDF extraction.

Thresholds are deliberately loose: the goal is to surface documents worth
looking at, not to assert a quality score. Every document is reported;
``status`` is ``"fail"`` only for defects that make a document useless
(no usable text), and ``"warn"`` for everything else.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Minimum usable text, per document type. A single flat floor does not work:
# a 200-character email is a perfectly normal email, while a 200-character
# 10-K or NIST publication means extraction failed. Calibrated after the
# first run over the full corpus flagged 16 healthy emails as failures.
MIN_CHARS_BY_TYPE = {"email": 40}
MIN_CHARS_DEFAULT = 500
MIN_YIELD_RATIO = 0.005  # chars per source byte; PDFs are ~0.05-0.15
MIN_ALPHA_RATIO = 0.60
MAX_BOILERPLATE_RATIO = 0.30
MIN_WORDS_PER_LINE = 3.0

WORD_RE = re.compile(r"[A-Za-z0-9]")


def measure_document(text: str, source_bytes: int) -> dict:
    """Compute extraction-quality metrics for one preprocessed document.

    Args:
        text: The document's text after extraction and normalisation.
        source_bytes: Size of the original file on disk, for the yield ratio.

    Returns:
        Dict of metrics: ``chars``, ``lines``, ``yield_ratio``,
        ``alpha_ratio``, ``boilerplate_ratio``, ``table_row_ratio`` and
        ``mean_words_per_line``. A document with no text scores zero on
        every ratio rather than raising.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not text or not lines:
        return {
            "chars": len(text),
            "lines": 0,
            "yield_ratio": 0.0,
            "alpha_ratio": 0.0,
            "boilerplate_ratio": 0.0,
            "table_row_ratio": 0.0,
            "mean_words_per_line": 0.0,
        }

    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    repeated = sum(n for n in counts.values() if n > 1)

    return {
        "chars": len(text),
        "lines": len(lines),
        "yield_ratio": len(text) / source_bytes if source_bytes else 0.0,
        "alpha_ratio": sum(bool(WORD_RE.match(c)) for c in text) / len(text),
        "boilerplate_ratio": repeated / len(lines),
        "table_row_ratio": sum(line.startswith("|") for line in lines) / len(lines),
        "mean_words_per_line": sum(len(line.split()) for line in lines) / len(lines),
    }


def check_document(metrics: dict, doc_type: str = "") -> tuple[str, list[str]]:
    """Grade one document's metrics against the extraction thresholds.

    Args:
        metrics: Output of ``measure_document``.
        doc_type: Document type from the manifest, used to pick the minimum
            usable text length -- emails are legitimately far shorter than
            filings, so a single flat floor misgrades them.

    Returns:
        ``(status, issues)`` where status is ``"ok"``, ``"warn"`` or
        ``"fail"``. Only an unusable document -- too little text, or
        effectively no text for its file size -- fails; everything else is
        a warning worth a human glance.
    """
    issues = []
    status = "ok"
    min_chars = MIN_CHARS_BY_TYPE.get(doc_type, MIN_CHARS_DEFAULT)

    if metrics["chars"] < min_chars:
        issues.append(
            f"only {metrics['chars']} chars extracted, below the {min_chars} "
            f"floor for '{doc_type or 'default'}' "
            "(no text layer? a scanned document would need OCR)"
        )
        status = "fail"
    if metrics["yield_ratio"] < MIN_YIELD_RATIO:
        issues.append(f"yield_ratio {metrics['yield_ratio']:.4f} below {MIN_YIELD_RATIO}")
        status = "fail"

    if metrics["alpha_ratio"] < MIN_ALPHA_RATIO:
        issues.append(
            f"alpha_ratio {metrics['alpha_ratio']:.2f} below {MIN_ALPHA_RATIO} "
            "(mostly punctuation -- leftover contents pages or rules?)"
        )
        status = "warn" if status == "ok" else status
    if metrics["boilerplate_ratio"] > MAX_BOILERPLATE_RATIO:
        issues.append(
            f"boilerplate_ratio {metrics['boilerplate_ratio']:.2f} above "
            f"{MAX_BOILERPLATE_RATIO} (repeated headers/footers survived cleaning)"
        )
        status = "warn" if status == "ok" else status
    if metrics["mean_words_per_line"] < MIN_WORDS_PER_LINE:
        issues.append(
            f"mean_words_per_line {metrics['mean_words_per_line']:.1f} below "
            f"{MIN_WORDS_PER_LINE} (text fragmented -- multi-column extraction?)"
        )
        status = "warn" if status == "ok" else status

    return status, issues


def summarise(reports: list[dict]) -> dict:
    """Aggregate per-document reports into a corpus-level summary.

    Args:
        reports: Per-document dicts carrying ``status``, ``doc_type`` and
            ``metrics``.

    Returns:
        Counts by status, counts by document type, the failing and warning
        document ids, and mean metrics across the corpus.
    """
    by_status: dict[str, int] = {}
    by_type: dict[str, dict[str, int]] = {}
    for report in reports:
        by_status[report["status"]] = by_status.get(report["status"], 0) + 1
        per_type = by_type.setdefault(report["doc_type"], {})
        per_type[report["status"]] = per_type.get(report["status"], 0) + 1

    metric_names = ["yield_ratio", "alpha_ratio", "boilerplate_ratio", "table_row_ratio"]
    means = {
        name: sum(r["metrics"][name] for r in reports) / len(reports) for name in metric_names
    } if reports else {}

    return {
        "documents": len(reports),
        "by_status": by_status,
        "by_doc_type": by_type,
        "mean_metrics": means,
        "failed": [r["doc_id"] for r in reports if r["status"] == "fail"],
        "warned": [r["doc_id"] for r in reports if r["status"] == "warn"],
    }
