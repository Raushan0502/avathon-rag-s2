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
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import EMBEDDING_MODEL_NAME, MANIFEST_PATH

# Chunk budget in *tokens*, measured with the embedding model's own
# tokenizer. bge-small truncates at 512 and does so silently, so the budget
# leaves headroom for the query instruction prefix and any contextual
# header prepended before embedding.
CHUNK_MAX_TOKENS = 400
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

# --- Preprocessing (see normalise_text) ---------------------------------
# Characters that differ across filers but mean the same thing. NFKC alone
# leaves curly quotes and dashes untouched, so they are mapped explicitly.
PUNCTUATION_MAP = str.maketrans(
    {
        # Quotes
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        '“': '"', '”': '"', '„': '"', '‟': '"',
        # Dashes. NFKC runs first and folds some of these into each other
        # (U+2011 becomes U+2010), so the whole range is mapped rather than
        # only the characters seen in the raw source.
        "‐": "-", "‑": "-", "‒": "-", "–": "-",
        "—": "-", "―": "-", "−": "-",
        # Spaces and invisibles
        " ": " ", " ": " ", " ": " ",
        "​": "", "﻿": "", "­": "",
    }
)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# A word split across a line break by a PDF text layer: "informa-\ntion".
DEHYPHENATE_RE = re.compile(r"([A-Za-z]{2,})-\n([a-z]{2,})")
# Table-of-contents filler: "Events and Incidents ......... 6".
DOT_LEADER_RE = re.compile(r"\.{4,}\s*\d*\s*$")
# A line that is nothing but a page number.
PAGE_NUMBER_RE = re.compile(r"(?:page\s*)?\d{1,4}", re.IGNORECASE)
# Page furniture must be short and frequent; a long repeated line is more
# likely to be genuine repeated prose (e.g. a standard risk disclaimer).
# Furniture also has to be long enough to *be* furniture. Without a floor,
# "None." -- which is the real content of 10-K Items 1B, 4 and 9B, and so
# repeats several times per filing -- matches the short-and-frequent
# signature and gets deleted, taking whole sections with it.
MIN_BOILERPLATE_LEN = 15
MAX_BOILERPLATE_LEN = 80
# Digit masking (below) makes any two sentences differing only by a number
# look identical, so a length limit alone would delete real prose. Page
# furniture is label-shaped -- few words -- while body text is not.
MAX_BOILERPLATE_WORDS = 10
MIN_BOILERPLATE_REPEATS = 5
DIGIT_RE = re.compile(r"\d+")


def normalise_text(text: str) -> str:
    """Clean raw extracted text before it is split into sections and chunks.

    Extraction output is not yet fit to embed: it carries page furniture,
    table-of-contents filler, words broken across line ends, and characters
    that vary by source encoding. Each pass below targets one defect
    observed in this corpus, in an order that matters -- unicode is
    normalised first so later pattern matching sees consistent characters,
    and de-hyphenation runs before repeated-line detection so a rejoined
    word is not mistaken for boilerplate.

    Passes:
      1. **Unicode/NFKC + control characters** -- smart quotes, non-breaking
         hyphens and NBSP become their ASCII equivalents, so the same word
         embeds identically regardless of which filer produced it.
      2. **De-hyphenation** -- PDF text layers break words at line ends
         ("informa-\\ninformation"); rejoining them stops one word being
         embedded as two fragments.
      3. **Dot-leader / page-number stripping** -- table-of-contents lines
         ("Events and Incidents ....... 6") carry no meaning but occupy
         whole chunks.
      4. **Repeated-line removal** -- headers and footers reprinted on every
         page ("Apple Inc. | 2025 Form 10-K | 17"). A line is treated as
         furniture only when it recurs often *and* is short, so a genuinely
         repeated sentence of prose survives.
      5. **Whitespace collapse** -- trailing spaces and blank runs.

    Table rows (lines starting with "|") are passed through untouched by the
    line-level passes, so the structure recovered in ``render_table`` is not
    dismantled here.

    Args:
        text: Raw text from ``load_document_text``.

    Returns:
        Cleaned text, one logical line per line, with no blank lines.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(PUNCTUATION_MAP)
    text = CONTROL_CHAR_RE.sub("", text)
    text = DEHYPHENATE_RE.sub(r"\1\2", text)

    lines = [line.strip() for line in text.splitlines()]
    lines = [re.sub(r"[ \t]{2,}", " ", line) for line in lines]

    # Count short lines to find page furniture. Long lines are excluded up
    # front: a repeated long sentence is far more likely to be real content.
    # Digits are masked before counting, because a running footer carries the
    # page number ("Apple Inc. | 2025 Form 10-K | 17") and so is never twice
    # the same string -- exact matching misses it entirely.
    def is_furniture_candidate(line: str) -> bool:
        return (
            bool(line)
            and not line.startswith("|")
            and MIN_BOILERPLATE_LEN <= len(line) <= MAX_BOILERPLATE_LEN
            and len(line.split()) <= MAX_BOILERPLATE_WORDS
        )

    counts: dict[str, int] = {}
    for line in lines:
        if is_furniture_candidate(line):
            template = DIGIT_RE.sub("#", line)
            counts[template] = counts.get(template, 0) + 1
    boilerplate = {line for line, n in counts.items() if n >= MIN_BOILERPLATE_REPEATS}

    kept = []
    for line in lines:
        if not line:
            continue
        if line.startswith("|"):  # rendered table row -- leave alone
            kept.append(line)
            continue
        if is_furniture_candidate(line) and DIGIT_RE.sub("#", line) in boilerplate:
            continue
        if DOT_LEADER_RE.search(line):
            continue
        if PAGE_NUMBER_RE.fullmatch(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def render_table(rows: list[list[str]]) -> str:
    """Render a table as a Markdown pipe table, or plain text if it isn't one.

    Financial filings answer questions like "what were net sales in FY2024?"
    only if a figure stays attached to its row label and column header.
    Flattening a table to running text destroys that: three fiscal years
    collapse into an unlabelled number sequence. Markdown pipe rows keep the
    association, are compact, and are a format LLMs read reliably.

    HTML filings also use ``<table>`` heavily for pure layout, so anything
    without at least 2 rows and 2 populated columns is emitted as plain
    lines instead of being dressed up as a data table.

    Args:
        rows: Table cells as a list of rows; cells may be None or blank.

    Returns:
        A Markdown table (header row, separator, body rows), or newline-joined
        text for layout tables, or an empty string if there is no content.
    """
    cleaned = []
    for row in rows:
        cells = [re.sub(r"\s+", " ", (cell or "").strip()) for cell in row]
        # Merge symbol-only cells into their neighbour, per row. Filings put
        # the currency mark and percent sign in their own cells, and only on
        # some rows -- so this has to be row-local, not column-wide. Doing it
        # here also de-skews the table: a row carrying a lone "$" has one
        # more cell than its neighbours, and collapsing it restores alignment.
        merged: list[str] = []
        for cell in cells:
            if cell == "%" and merged:
                merged[-1] = f"{merged[-1]}%"
            elif cell == "$":
                merged.append("$")  # attached to the next value below
            elif merged and merged[-1] == "$":
                merged[-1] = f"${cell}"
            else:
                merged.append(cell)
        if any(merged):
            cleaned.append(merged)
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    # Drop columns that are empty throughout -- filings are full of spacer
    # columns, and the merge above empties more of them.
    keep = [i for i in range(width) if any(row[i] for row in cleaned)]
    cleaned = [[row[i] for i in keep] for row in cleaned]
    if not cleaned or not keep:
        return ""

    if len(cleaned) < 2 or len(keep) < 2:
        return "\n".join(" ".join(cell for cell in row if cell) for row in cleaned)

    header, *body = cleaned
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def load_document_text(local_path: Path) -> str:
    """Extract text from a corpus document, dispatching on file type.

    The corpus spans three formats, each needing a different extraction
    path, and each losing something different in the process:

    - **HTML** (filings, exhibits): parsed with BeautifulSoup, dropping
      script/style. ``<table>`` elements are rendered to Markdown *in place*
      before the surrounding text is flattened, so tabular figures keep
      their row/column association (see ``render_table``). Blank lines are
      collapsed but single newlines are kept, because ``ITEM_HEADER_RE``
      anchors on line boundaries.
    - **PDF** (NIST publications): tables are located first and extracted
      structurally, then the page's remaining prose is read with those table
      regions filtered out, so table content is not duplicated between the
      two. These PDFs carry a real text layer, so no OCR is needed.
    - **Plain text** (emails): read as-is, minus the AESLC corpus's
      ``@subject``/``@ann*`` trailer. The ``@subject`` line is promoted to
      the top of the text so the email's subject is embedded alongside its
      body; the ``@ann*`` lines are alternative subject lines belonging to
      that dataset's summarization task, not to the email, so they are
      dropped rather than indexed as if the sender wrote them.

    **Images are not extracted.** HTML ``<img>`` elements are discarded by
    ``get_text`` (including any alt text), and PDF extraction reads only the
    text layer, so charts, figures and any scanned page contribute nothing.
    A scanned document therefore yields little or no text *without raising* --
    supporting it would require an OCR stage (e.g. Tesseract or a document-AI
    service) that this pipeline deliberately does not have. The validation
    step is what surfaces such documents rather than letting them index
    silently.

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
        # Replace each table with its rendered form in document order, so the
        # table stays where it appeared relative to the surrounding prose.
        for table in soup.find_all("table"):
            rows = []
            for row in table.find_all("tr"):
                cells: list[str] = []
                for cell in row.find_all(["td", "th"]):
                    # Expand colspan so a cell lands in the column it spans
                    # from; without this, rows are ragged and figures drift
                    # out from under their header.
                    try:
                        span = max(1, int(cell.get("colspan", 1)))
                    except (TypeError, ValueError):
                        span = 1
                    cells.append(cell.get_text(" ", strip=True))
                    cells.extend([""] * (span - 1))
                rows.append(cells)
            table.replace_with(NavigableString(f"\n{render_table(rows)}\n"))
        lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
        return "\n".join(line for line in lines if line)

    if suffix == ".pdf":
        import pdfplumber

        blocks = []
        with pdfplumber.open(local_path) as pdf:
            for page in pdf.pages:
                tables = page.find_tables()
                # Read prose with the table regions masked out, so figures
                # inside tables are not emitted twice in two different shapes.
                boxes = [table.bbox for table in tables]
                page_text = page.filter(
                    lambda obj: not any(
                        box[0] <= (obj["x0"] + obj["x1"]) / 2 <= box[2]
                        and box[1] <= (obj["top"] + obj["bottom"]) / 2 <= box[3]
                        for box in boxes
                    )
                ).extract_text() or ""
                blocks.append(page_text)
                blocks.extend(render_table(table.extract()) for table in tables)

        lines = [line.strip() for line in "\n".join(blocks).splitlines()]
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
        return split_on_numbered_headings(text)

    sections = []
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[header.end() : end].strip()
        if body:
            sections.append((re.sub(r"\s+", " ", header.group(1).strip()), body))
    return sections


def split_on_numbered_headings(text: str) -> list[tuple[str, str]]:
    """Split a non-SEC document on its own numbered or titled headings.

    Only SEC filings have "Item N" headers. Everything else -- NIST
    publications, contract exhibits, policies -- previously fell through to
    a single ``"Full Document"`` section, which had a measurable cost: an
    80-page NIST PDF became one 177-chunk section, and because relevance is
    judged at ``(doc_id, section)`` granularity, *any* chunk retrieved from
    that file counted as relevant. Those questions scored a perfect
    Precision@5 of 1.000 purely because the section was the whole document,
    inflating the corpus-wide average and making it incomparable across
    document types.

    These documents do carry structure, just not SEC structure: NIST uses
    "2.1 Events and Incidents" and "Appendix A.", contracts and policies use
    "1. PURPOSE" or "Section 5.". A conservative pattern picks those up and
    falls back to a single section when it finds too few, on the same
    principle as the Item split: over-splitting on false positives is worse
    than under-splitting.

    Args:
        text: Cleaned document text.

    Returns:
        ``(section_title, section_body)`` pairs, or a single
        ``("Full Document", text)`` pair when no reliable structure is found.
    """
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
    """Count tokens the way the embedding model will actually count them.

    Sizing chunks in words is a proxy for the limit that is really enforced.
    The model truncates at 512 *tokens* and does so silently -- no error,
    the tail is simply discarded -- and the words-to-tokens ratio is not
    constant: prose runs near 1.3, while a financial table where "416,161"
    costs several tokens runs far higher. Measuring directly removes the
    guesswork.

    The tokenizer is loaded once and cached; if it is unavailable (offline,
    no model cache) the function falls back to a conservative word-based
    estimate rather than failing the whole ingest.

    Args:
        text: Text to measure.

    Returns:
        Token count under the embedding model's tokenizer, or an estimate.
    """
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return int(len(text.split()) * FALLBACK_TOKENS_PER_WORD) + 1
    return len(tokenizer.encode(text, add_special_tokens=False))


def _get_tokenizer():
    """Load and cache the embedding model's tokenizer, or None if unavailable."""
    global _TOKENIZER
    if _TOKENIZER is _UNSET:
        try:
            from transformers import AutoTokenizer

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
    """Split text into table and prose blocks, preserving order.

    Chunking has to know where tables are: a table split across a boundary
    loses its header row, which puts the figures back in exactly the
    unlabelled state that rendering them was meant to fix.

    Args:
        text: Normalised document or section text.

    Returns:
        ``(kind, block_text)`` pairs where kind is ``"table"`` or ``"prose"``.
    """
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
    """Split a rendered table, repeating the header row in every part.

    A table that fits stays whole. One that does not is split on row
    boundaries -- never mid-row -- and each part is re-prefixed with the
    header and separator rows, so every chunk remains self-describing:
    "| Line item 15 | 1500 |" is meaningless without the "| 2025 |" header
    that gives the column its year.

    Args:
        table: Markdown table text, header row first.
        max_tokens: Token budget per chunk.

    Returns:
        One or more table strings, each carrying the header.
    """
    if count_tokens(table) <= max_tokens:
        return [table]

    lines = table.splitlines()
    header = lines[:2] if len(lines) > 1 and set(lines[1].replace("|", "").strip()) <= {"-", " "} else lines[:1]
    body = lines[len(header) :]
    header_text = "\n".join(header)
    header_cost = count_tokens(header_text)

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


def chunk_prose(
    text: str, max_tokens: int = CHUNK_MAX_TOKENS, overlap_tokens: int = CHUNK_OVERLAP_TOKENS
) -> list[str]:
    """Split prose on sentence boundaries, with token-sized overlap.

    Packing whole sentences means a chunk never opens mid-clause, which the
    previous fixed word-window did routinely. Overlap is carried as whole
    trailing sentences rather than a fixed word count, so the repeated span
    is always readable.

    Args:
        text: Prose block.
        max_tokens: Token budget per chunk.
        overlap_tokens: Approximate tokens of trailing context to repeat.

    Returns:
        Chunk strings in order; empty list for blank input.
    """
    sentences = [s for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    if not sentences:
        return []

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
    """Prefix a chunk with its document and section context, for embedding.

    A chunk is embedded in isolation, having lost every clue about where it
    came from. "Competition is intense and margins are under pressure"
    could be any issuer in any year, so it embeds no closer to "Apple
    competition risk" than to any other company's near-identical wording --
    and the corpus is full of near-identical wording, because filings are
    written from common templates. Prefixing the provenance puts the
    issuer, form and section into the vector itself.

    Only the embedding input carries this prefix. The stored ``text`` stays
    clean, because that is what gets shown to the model as quotable context
    and to a reader as a citation; the prompt already states provenance
    separately, so duplicating it there would waste context.

    Args:
        doc_meta: Manifest entry for the document.
        section_title: Section the chunk came from.
        chunk_text: The chunk itself.

    Returns:
        ``"<company> | <form> | <section>\\n<chunk>"``, with empty fields
        omitted so non-SEC documents do not gain blank separators.
    """
    parts = [
        doc_meta.get("company", ""),
        doc_meta.get("form", ""),
        section_title if section_title != "Full Document" else "",
    ]
    header = " | ".join(part for part in parts if part)
    return f"{header}\n{chunk_text}" if header else chunk_text


def chunk_section(text: str, max_tokens: int = CHUNK_MAX_TOKENS) -> list[str]:
    """Chunk one section, keeping tables intact and prose sentence-aligned.

    Blocks are chunked by kind and never merged across kinds, so a table is
    not glued onto the end of a paragraph and split by a word counter that
    cannot see it.

    Args:
        text: Section body, post-normalisation.
        max_tokens: Token budget per chunk.

    Returns:
        Chunk strings in document order.
    """
    chunks: list[str] = []
    for kind, body in split_blocks(text):
        if kind == "table":
            chunks.extend(chunk_table(body, max_tokens))
        else:
            chunks.extend(chunk_prose(body, max_tokens))
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
