"""
Step 3 — Ingestion: HTML loading, cleaning, and chunking.

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
false-positive header matches. This trade-off (and where it does/doesn't
hold across the 6 sourced filings) is discussed in the write-up.
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


def html_to_text(html: str) -> str:
    """Strip an SEC filing's HTML down to plain text, one line per block.

    Script/style content is dropped, and blank lines are collapsed while
    single newlines are preserved -- ``ITEM_HEADER_RE`` anchors on line
    boundaries, so the line structure has to survive this step.

    Args:
        html: Raw filing HTML as fetched from EDGAR.

    Returns:
        Newline-joined text with empty lines removed.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
    return "\n".join(line for line in lines if line)


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
        html = Path(doc_meta["local_path"]).read_bytes().decode("utf-8", errors="ignore")
        sections = split_into_sections(html_to_text(html))
        doc_id = f"{doc_meta['ticker']}_{doc_meta['form']}_{doc_meta['filing_date']}"

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
                f"  {doc_meta['ticker']} {doc_meta['form']} {doc_meta['filing_date']}: "
                f"{n_sections} section(s) -> {len(doc_chunks)} chunks"
            )
        all_chunks.extend(doc_chunks)

    return all_chunks
