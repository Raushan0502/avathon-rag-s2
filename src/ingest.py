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
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_RAW_DIR

CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
MIN_SECTIONS_TO_TRUST_SPLIT = 3

# Matches lines like "Item 1A. Risk Factors" or "Item 2.02 Results of
# Operations..." on their own line, case-insensitive.
ITEM_HEADER_RE = re.compile(
    r"(?m)^\s*(Item\s+\d+[A-Za-z]?(?:\.\d+)?\.?\s+[A-Z][A-Za-z0-9 ,.'&/\-]{2,90})\s*$",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ticker: str
    company: str
    form: str
    filing_date: str
    source_url: str
    section: str
    chunk_index: int
    word_count: int
    text: str


def load_manifest() -> list[dict]:
    manifest_path = DATA_RAW_DIR / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse runs of blank lines/whitespace left over from dense HTML,
    # but keep single newlines so the Item-header regex (which anchors on
    # line boundaries) still works.
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


ITEM_NUMBER_RE = re.compile(r"item\s+\d+[a-z]?(?:\.\d+)?", re.IGNORECASE)


def _item_key(title: str) -> str:
    m = ITEM_NUMBER_RE.match(title.strip())
    return re.sub(r"\s+", " ", m.group(0).lower()) if m else title.strip().lower()


def _dedupe_header_matches(matches: list[re.Match]) -> list[re.Match]:
    """Two kinds of false positives inflate a naive header match list:

    1. The table of contents lists every Item once near the top of the
       document, before the real sections.
    2. Printed-page running headers (e.g. "Item 7" repeated at the top of
       every page of a long MD&A section) get flattened by HTML->text
       extraction into what looks like a fresh "Item 7 <next-page's-first-
       words>" header, once per page -- so a single real section can
       produce a dozen+ spurious matches sharing its item number.

    Fix, in two passes:
      a) Collapse consecutive matches that share the same item number
         (e.g. the ~15 "Item 7 ..." running-header repeats within one
         section) down to the *first* match of that run -- which is the
         section's real heading, since the run only starts once the real
         heading has been passed.
      b) Across the whole document, a given item number can still produce
         two candidates this way: one isolated match from the TOC, one
         collapsed-run match from the real section. Keep the *last* one,
         since the real section always appears after the TOC.
    """
    collapsed: list[re.Match] = []
    prev_key = None
    for m in matches:
        key = _item_key(m.group(1))
        if key != prev_key:
            collapsed.append(m)
        prev_key = key

    last_by_key: dict[str, re.Match] = {}
    for m in collapsed:
        last_by_key[_item_key(m.group(1))] = m
    return sorted(last_by_key.values(), key=lambda m: m.start())


def split_into_sections(text: str) -> list[tuple[str, str]]:
    raw_matches = list(ITEM_HEADER_RE.finditer(text))
    matches = _dedupe_header_matches(raw_matches)

    if len(matches) < MIN_SECTIONS_TO_TRUST_SPLIT:
        return [("Full Document", text)]

    sections = []
    for i, m in enumerate(matches):
        title = re.sub(r"\s+", " ", m.group(1).strip())
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def chunk_words(text: str, size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def build_chunks_for_document(doc_meta: dict) -> list[Chunk]:
    local_path = Path(doc_meta["local_path"])
    html = local_path.read_bytes().decode("utf-8", errors="ignore")
    text = html_to_text(html)
    sections = split_into_sections(text)

    doc_id = f"{doc_meta['ticker']}_{doc_meta['form']}_{doc_meta['filing_date']}"
    chunks: list[Chunk] = []
    chunk_index = 0
    for section_title, section_text in sections:
        for window in chunk_words(section_text):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{chunk_index}",
                    doc_id=doc_id,
                    ticker=doc_meta["ticker"],
                    company=doc_meta["company"],
                    form=doc_meta["form"],
                    filing_date=doc_meta["filing_date"],
                    source_url=doc_meta["source_url"],
                    section=section_title,
                    chunk_index=chunk_index,
                    word_count=len(window.split()),
                    text=window,
                )
            )
            chunk_index += 1
    return chunks


def build_all_chunks() -> list[Chunk]:
    manifest = load_manifest()
    all_chunks: list[Chunk] = []
    for doc_meta in manifest:
        doc_chunks = build_chunks_for_document(doc_meta)
        n_sections = len({c.section for c in doc_chunks})
        print(
            f"  {doc_meta['ticker']} {doc_meta['form']} {doc_meta['filing_date']}: "
            f"{n_sections} section(s) -> {len(doc_chunks)} chunks"
        )
        all_chunks.extend(doc_chunks)
    return all_chunks


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [asdict(c) for c in chunks]
