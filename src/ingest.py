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
  2. Structure-preserving, token-budgeted split *within* each section
     (``chunk_section``). Sections range from a paragraph to tens of
     pages, far too long for one embedding, so each is packed into chunks
     of at most ``CHUNK_MAX_TOKENS``.

Stage 2 replaced a fixed 220-word sliding window, which had four defects
that only showed up once tables and PDFs entered the corpus:

  * It joined windows with spaces, **discarding newlines** -- so the
    Markdown tables recovered during extraction were flattened straight
    back into one long line.
  * A table longer than one window was split with **no header row** in the
    later parts, putting the figures back in exactly the unlabelled state
    that rendering them was meant to fix.
  * Windows began **mid-sentence**, which weakens the embedding and reads
    badly when the chunk is shown as cited context.
  * Sizing in words is a proxy for the limit that is actually enforced.
    The model truncates at 512 *tokens*, silently, and the words-to-tokens
    ratio is far from constant between prose and numeric tables.

So the current splitter measures with the model's own tokenizer
(``count_tokens``), separates tables from prose (``split_blocks``), keeps
each table whole and repeats its header in every part when one must be
split (``chunk_table``), and packs prose on sentence boundaries with
whole-sentence overlap (``chunk_prose``).

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

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import EMBEDDING_MODEL_NAME, MANIFEST_PATH
from src.extract import load_document_text
from src.preprocess import normalise_text

# Chunk budget in *tokens*, measured with the embedding model's own
# tokenizer. bge-small truncates at 512 and does so silently, so the budget
# leaves headroom for the query instruction prefix and any contextual
# header prepended before embedding.
# Budget for the chunk body. build_embed_text prepends a provenance header
# ("Apple Inc. | 10-K | Item 1A. Risk Factors") *after* chunking, so the
# embedded string is longer than the chunk. CONTEXT_HEADER_RESERVE holds
# back room for it; without that reserve, glossary sections tokenising at
# ~3 tokens/word overshot the model's hard 512 limit and were silently
# truncated.
CHUNK_MAX_TOKENS = 360
CONTEXT_HEADER_RESERVE = 60
CHUNK_OVERLAP_TOKENS = 60
FALLBACK_TOKENS_PER_WORD = 1.6  # only used if the tokenizer cannot be loaded
MIN_SECTIONS_TO_TRUST_SPLIT = 3

_UNSET = object()
_TOKENIZER = _UNSET

# Sentence boundary: terminator, closing quote/bracket, then whitespace and
# a capital or digit. Keeps "Inc." and "U.S." from splitting a sentence.
# Each look-behind branch is fixed width, which Python's `re` requires; the
# look-ahead for a capital or digit is what keeps "Apple Inc. reported" and
# "the U.S. market" from being treated as sentence ends.
SENTENCE_SPLIT_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[A-Z0-9])")

# Matches lines like "Item 1A. Risk Factors" or "Item 2.02 Results of
# Operations..." on their own line, case-insensitive.
ITEM_HEADER_RE = re.compile(
    r"(?m)^\s*(Item\s+\d+[A-Za-z]?(?:\.\d+)?\.?\s+[A-Z][A-Za-z0-9 ,.'&/\-]{2,90})\s*$",
    re.IGNORECASE,
)
# Just the "Item <number>" prefix of a matched header, used to group repeats
# of the same section (see split_into_sections).
ITEM_NUMBER_RE = re.compile(r"item\s+\d+[a-z]?(?:\.\d+)?", re.IGNORECASE)

# Headings in non-SEC documents: "2.1 Events and Incidents" (NIST),
# "1. PURPOSE" (policies), "Section 5. Termination" and "Appendix A. Scope"
# (contracts). Anchored to a whole line and capped in length so a numbered
# list item or a sentence beginning with a figure is not mistaken for a
# heading.
# A bare capital must carry its period ("A. Scope"), or a cover page set in
# spaced capitals matches as heading "C" + "O M P U T E R ...". The title
# must also end on an alphanumeric, which rejects wrapped address lines
# such as "100 Bureau Drive (Mail Stop 8930),".
NUMBERED_HEADING_RE = re.compile(
    r"(?m)^\s*((?:Section\s+|Appendix\s+|Annex\s+)?"
    r"(?:\d+(?:\.\d+){0,2}\.?|[A-Z]\.)\s+"
    r"[A-Z][A-Za-z0-9 ,'&/()-]{2,68}[A-Za-z0-9)])\s*$"
)

def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split filing text on its numbered "Item" headers."""
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
        return split_on_numbered_headings(text)

    sections = []
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[header.end() : end].strip()
        if body:
            sections.append((re.sub(r"\s+", " ", header.group(1).strip()), body))
    return sections


def split_on_numbered_headings(text: str) -> list[tuple[str, str]]:
    """Split a non-SEC document on its own numbered or titled headings."""
    headers = [
        match
        for match in NUMBERED_HEADING_RE.finditer(text)
        # A heading line should not itself be a table row.
        if not match.group(1).startswith("|")
    ]
    if len(headers) < MIN_SECTIONS_TO_TRUST_SPLIT:
        return [("Full Document", text)]

    sections = []
    preamble = text[: headers[0].start()].strip()
    if preamble:
        sections.append(("Front Matter", preamble))

    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[header.end() : end].strip()
        if body:
            sections.append((re.sub(r"\s+", " ", header.group(1).strip()), body))
    return sections


def count_tokens(text: str) -> int:
    """Count tokens the way the embedding model will actually count them."""
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return int(len(text.split()) * FALLBACK_TOKENS_PER_WORD) + 1
    return len(tokenizer.encode(text, add_special_tokens=False))


def get_tokenizer() -> object | None:
    """Load and cache the embedding model's tokenizer, or None if unavailable."""
    global _TOKENIZER
    if _TOKENIZER is _UNSET:
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
            # Used purely as a counter here, never as model input. Without
            # this it warns on every over-length string -- but measuring an
            # over-length section is exactly the point: that is how we learn
            # it needs splitting.
            _TOKENIZER.model_max_length = int(1e9)
        except Exception:  # noqa: BLE001 -- offline or no cache; estimate instead
            _TOKENIZER = None
    return _TOKENIZER


def split_blocks(text: str) -> list[tuple[str, str]]:
    """Split text into table and prose blocks, preserving order."""
    blocks: list[tuple[str, str]] = []
    current: list[str] = []
    current_kind = "prose"

    for line in text.splitlines():
        kind = "table" if line.startswith("|") else "prose"
        if kind != current_kind and current:
            blocks.append((current_kind, "\n".join(current)))
            current = []
        current_kind = kind
        current.append(line)

    if current:
        blocks.append((current_kind, "\n".join(current)))
    return [(kind, body) for kind, body in blocks if body.strip()]


def chunk_table(table: str, max_tokens: int = CHUNK_MAX_TOKENS) -> list[str]:
    """Split a rendered table, repeating the header row in every part."""
    if count_tokens(table) <= max_tokens:
        return [table]

    lines = table.splitlines()
    header = lines[:2] if len(lines) > 1 and set(lines[1].replace("|", "").strip()) <= {"-", " "} else lines[:1]
    body = lines[len(header) :]
    header_text = "\n".join(header)
    header_cost = count_tokens(header_text)

    body = [part for row in body for part in split_oversized(row, max_tokens - header_cost)]

    parts, current, current_tokens = [], [], 0
    for row in body:
        row_tokens = count_tokens(row)
        if current and header_cost + current_tokens + row_tokens > max_tokens:
            parts.append("\n".join(header + current))
            current, current_tokens = [], 0
        current.append(row)
        current_tokens += row_tokens
    if current:
        parts.append("\n".join(header + current))
    return parts


def split_oversized(unit: str, max_tokens: int) -> list[str]:
    """Hard-split a single unit that no structural boundary can break up."""
    if count_tokens(unit) <= max_tokens:
        return [unit]

    words = unit.split()
    if len(words) <= 1:
        return [unit]  # a single unsplittable token; nothing more to do

    # Convert the budget into words via this unit's own measured ratio,
    # rather than a global constant -- numeric table text tokenises far
    # more heavily than prose.
    per_word = count_tokens(unit) / len(words)
    step = max(1, int(max_tokens / per_word))
    return [" ".join(words[i : i + step]) for i in range(0, len(words), step)]


def chunk_prose(
    text: str, max_tokens: int = CHUNK_MAX_TOKENS, overlap_tokens: int = CHUNK_OVERLAP_TOKENS
) -> list[str]:
    """Split prose on sentence boundaries, with token-sized overlap."""
    sentences = [s for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    if not sentences:
        return []
    # A sentence larger than the whole budget cannot be packed; break it
    # first so no chunk can exceed the limit and be silently truncated.
    sentences = [part for s in sentences for part in split_oversized(s, max_tokens)]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current))
            # Carry whole trailing sentences as overlap.
            tail, tail_tokens = [], 0
            for previous in reversed(current):
                previous_tokens = count_tokens(previous)
                if tail_tokens + previous_tokens > overlap_tokens:
                    break
                tail.insert(0, previous)
                tail_tokens += previous_tokens
            current, current_tokens = tail, tail_tokens
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def build_embed_text(doc_meta: dict, section_title: str, chunk_text: str) -> str:
    """Prefix a chunk with its document and section context, for embedding."""
    parts = [
        doc_meta.get("company", ""),
        doc_meta.get("form", ""),
        section_title if section_title != "Full Document" else "",
    ]
    header = " | ".join(part for part in parts if part)
    return f"{header}\n{chunk_text}" if header else chunk_text


def chunk_section(text: str, max_tokens: int = CHUNK_MAX_TOKENS) -> list[str]:
    """Chunk one section, keeping tables intact and prose sentence-aligned."""
    chunks: list[str] = []
    for kind, body in split_blocks(text):
        if kind == "table":
            chunks.extend(chunk_table(body, max_tokens))
        else:
            chunks.extend(chunk_prose(body, max_tokens))
    return chunks


def build_all_chunks(verbose: bool = True) -> list[dict]:
    """Build the full chunk set for every document listed in the manifest."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    all_chunks: list[dict] = []

    for doc_meta in manifest:
        local_path = Path(doc_meta["local_path"])
        # doc_id comes from the filename stem, which is unique per document.
        # Composing it from ticker/form/date instead would collide -- every
        # sampled email shares the same ticker, form and (empty) date.
        doc_id = local_path.stem
        # extract -> normalise -> section-split -> window. Normalisation sits
        # between extraction and splitting on purpose: section detection
        # matches on line shape, so it must run on cleaned lines.
        sections = split_into_sections(normalise_text(load_document_text(local_path)))

        doc_chunks: list[dict] = []
        for section_title, section_text in sections:
            for window in chunk_section(section_text):
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
                        "embed_text": build_embed_text(doc_meta, section_title, window),
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
