"""Document loading: raw files in, plain text out, dispatching on file type.

Each format loses something different. HTML tables are rendered to Markdown
in place so figures keep their row/column association; PDF tables are
extracted structurally with their regions masked out of the prose pass so
nothing is emitted twice; plain-text emails have the AESLC annotation
trailer stripped.

Images are not extracted at all -- HTML ``<img>`` (including alt text) and
anything in a PDF that is not in the text layer. A scanned document
therefore yields little or no text *without raising*, which is what the
validation gate exists to catch. Supporting them needs an OCR stage this
pipeline deliberately does not have.
"""
import re
import sys
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup, NavigableString

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def render_table(rows: list[list[str]]) -> str:
    """Render a table as a Markdown pipe table, or plain text if it isn't
    one."""
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
    """Extract text from a corpus document, dispatching on file type."""
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


