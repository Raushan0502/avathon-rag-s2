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

from bs4 import BeautifulSoup, NavigableString

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
