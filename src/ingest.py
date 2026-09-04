"""
Step 3 — Ingestion: multi-format document loading, cleaning, and chunking.

The corpus spans three file formats (HTML filings and exhibits, PDF
standards, plain-text emails), so loading dispatches on file type -- see
``load_document_text`` for what each path extracts and what it loses.

Chunking strategy (documented here so the "why" travels with the code):

SEC filings are internally structured around numbered "Item" sections
(e.g. "Item 1A. Risk Factors" in a 10-K, "Item 2.02" in an 8-K). A query
about risk factors should not retrieve a chunk that blends risk-factor
text with unrelated legal-proceedings text just because a fixed-size
window happened to straddle both. So chunking here is two-stage:

  1. Section-aware split: find Item-header lines with a regex and split
     the document on them, so each section's text stays together and
     carries its own section title as metadata.
  2. Fixed-size sliding window *within* each section: sections themselves
     range from a paragraph to tens of pages, which is far too long for a
     single embedding, so each section is further split into ~220-word
     windows with a 40-word overlap (roughly 300 tokens for the chosen
     embedding model, well under its 512-token limit, with overlap sized
     to preserve any sentence or clause that would otherwise be cut at a
     window boundary).

Section detection depends on the filer's HTML formatting being
regular enough for the Item-header regex to find genuine headers (and
not, say, a table-of-contents entry). This is a heuristic, not a parser
against the filing's actual document schema, so if fewer than 3
section headers are found, we fall back to chunking the whole document
as a single "Full Document" section rather than risk over-splitting on
false-positive header matches.

That fallback is also what the non-SEC documents take: NIST publications,
contract exhibits and emails have no "Item N" headers, so each becomes a
single section and is chunked by the sliding window alone. For short
documents (emails, exhibits) that is the right answer. For an 80-page
NIST PDF it is genuinely coarse -- the document's own numbered headings
could drive a finer split -- and that limitation is documented in the
README rather than papered over.
"""
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MANIFEST_PATH

CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
MIN_SECTIONS_TO_TRUST_SPLIT = 3

# Matches lines like "Item 1A. Risk Factors" or "Item 2.02 Results of
# Operations..." on their own line, case-insensitive.
ITEM_HEADER_RE = re.compile(
    r"(?m)^\s*(Item\s+\d+[A-Za-z]?(?:\.\d+)?\.?\s+[A-Z][A-Za-z0-9 ,.'&/\-]{2,90})\s*$",
    re.IGNORECASE,
)
# Just the "Item <number>" prefix of a matched header, used to group repeats
# of the same section (see split_into_sections).
ITEM_NUMBER_RE = re.compile(r"item\s+\d+[a-z]?(?:\.\d+)?", re.IGNORECASE)


def load_document_text(local_path: Path) -> str:
    """Extract plain text from a corpus document, dispatching on file type.

    The corpus spans three formats, each needing a different extraction
    path, and each losing something different in the process:

    - **HTML** (filings, exhibits): parsed with BeautifulSoup, dropping
      script/style. Blank lines are collapsed but single newlines are kept,
      because ``ITEM_HEADER_RE`` anchors on line boundaries.
    - **PDF** (NIST publications): text layer extracted per page with
      pdfplumber. These PDFs carry a real text layer, so no OCR is needed;
      a scanned document would come back empty here and would need an OCR
      stage, which this pipeline does not have.
    - **Plain text** (emails): read as-is, minus the AESLC corpus's
      ``@subject``/``@ann*`` trailer. The ``@subject`` line is promoted to
      the top of the text so the email's subject is embedded alongside its
      body; the ``@ann*`` lines are alternative subject lines belonging to
      that dataset's summarization task, not to the email, so they are
      dropped rather than indexed as if the sender wrote them.

    Args:
        local_path: Path to the document under ``data/raw``.

    Returns:
        Extracted text, or an empty string if the file yields none.

    Raises:
        ValueError: If the file extension is not a supported format.
    """
    suffix = local_path.suffix.lower()

    if suffix in {".htm", ".html"}:
        soup = BeautifulSoup(
            local_path.read_bytes().decode("utf-8", errors="ignore"), "lxml"
        )
        for tag in soup(["script", "style"]):
            tag.decompose()
        lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
        return "\n".join(line for line in lines if line)

    if suffix == ".pdf":
        import pdfplumber

        with pdfplumber.open(local_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        lines = [line.strip() for line in "\n".join(pages).splitlines()]
        return "\n".join(line for line in lines if line)

    if suffix == ".txt":
        raw = local_path.read_text(encoding="utf-8", errors="ignore")
        body, _, trailer = raw.partition("@subject")
        subject = trailer.split("@ann")[0].strip() if trailer else ""
        text = f"Subject: {subject}\n{body.strip()}" if subject else body.strip()
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)

    raise ValueError(f"unsupported document format {suffix!r} for {local_path.name}")


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split filing text on its numbered "Item" headers.

    Two kinds of false positives inflate a naive header match list, and both
    are corrected here:

    1. The table of contents lists every Item once near the top of the
       document, before the real sections.
    2. Printed-page running headers (e.g. "Item 7" reprinted at the top of
       every page of a long MD&A section) get flattened by HTML->text
       extraction into what looks like a fresh "Item 7 <next-page's-first-
       words>" header once per page -- so one real section can produce a
       dozen+ spurious matches sharing its item number.

    The fix runs in two passes over the raw matches: collapse consecutive
    matches sharing an item number down to the *first* of that run (the
    section's real heading, since the run only starts once the heading has
    been passed), then keep the *last* surviving candidate per item number
    (the real section, since it always follows the TOC entry).

    Args:
        text: Cleaned filing text from ``html_to_text``.

    Returns:
        ``(section_title, section_body)`` pairs in document order. If fewer
        than ``MIN_SECTIONS_TO_TRUST_SPLIT`` headers survive, returns a
        single ``("Full Document", text)`` pair rather than risk
        over-splitting on false-positive matches.
    """
    def item_key(title: str) -> str:
        match = ITEM_NUMBER_RE.match(title.strip())
        return re.sub(r"\s+", " ", match.group(0).lower()) if match else title.strip().lower()

    collapsed: list[re.Match] = []
    previous_key = None
    for match in ITEM_HEADER_RE.finditer(text):
        key = item_key(match.group(1))
        if key != previous_key:
            collapsed.append(match)
        previous_key = key

    last_by_key: dict[str, re.Match] = {item_key(m.group(1)): m for m in collapsed}
    headers = sorted(last_by_key.values(), key=lambda m: m.start())

    if len(headers) < MIN_SECTIONS_TO_TRUST_SPLIT:
        return [("Full Document", text)]

    sections = []
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[header.end() : end].strip()
        if body:
            sections.append((re.sub(r"\s+", " ", header.group(1).strip()), body))
    return sections


def chunk_words(
    text: str, size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS
) -> list[str]:
    """Slide a fixed-size word window over text, with overlap between windows.

    Args:
        text: Section body to split.
        size: Words per window.
        overlap: Words each window repeats from the previous one, so a
            sentence cut at a window boundary still appears intact in one
            of the two windows.

    Returns:
        Window strings in order; empty list for empty/whitespace-only text.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    for start in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def build_all_chunks(verbose: bool = True) -> list[dict]:
    """Build the full chunk set for every document listed in the manifest.

    Reads each filing named by ``data/raw/manifest.json``, cleans its HTML,
    splits it into Item sections, and slides a word window within each
    section, attaching the filing's provenance metadata to every chunk.

    Args:
        verbose: Print a per-document section/chunk count while running.

    Returns:
        Chunk dicts with keys: ``chunk_id``, ``doc_id``, ``ticker``,
        ``company``, ``form``, ``filing_date``, ``source_url``, ``section``,
        ``chunk_index``, ``word_count``, ``text``.
    """
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    all_chunks: list[dict] = []

    for doc_meta in manifest:
        local_path = Path(doc_meta["local_path"])
        # doc_id comes from the filename stem, which is unique per document.
        # Composing it from ticker/form/date instead would collide -- every
        # sampled email shares the same ticker, form and (empty) date.
        doc_id = local_path.stem
        sections = split_into_sections(load_document_text(local_path))

        doc_chunks: list[dict] = []
        for section_title, section_text in sections:
            for window in chunk_words(section_text):
                doc_chunks.append(
                    {
                        "chunk_id": f"{doc_id}::{len(doc_chunks)}",
                        "doc_id": doc_id,
                        "ticker": doc_meta["ticker"],
                        "company": doc_meta["company"],
                        "form": doc_meta["form"],
                        "doc_type": doc_meta["doc_type"],
                        "format": doc_meta["format"],
                        "filing_date": doc_meta["filing_date"],
                        "source_url": doc_meta["source_url"],
                        "section": section_title,
                        "chunk_index": len(doc_chunks),
                        "word_count": len(window.split()),
                        "text": window,
                    }
                )

        if verbose:
            n_sections = len({c["section"] for c in doc_chunks})
            print(
                f"  [{doc_meta['doc_type']:8s}/{doc_meta['format']:4s}] {doc_id}: "
                f"{n_sections} section(s) -> {len(doc_chunks)} chunks"
            )
        all_chunks.extend(doc_chunks)

    return all_chunks
