"""Text normalisation, run between extraction and section splitting.

Extraction output is not yet fit to embed: it carries page furniture,
table-of-contents filler, words broken across line ends, and characters that
vary by source encoding. Five ordered passes fix those, and the order
matters -- unicode is normalised before any pattern matching, and
de-hyphenation runs before repeated-line detection so a rejoined word is not
mistaken for boilerplate.

Rendered table rows are passed through untouched, so this does not dismantle
the structure extraction just recovered.
"""
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    """Clean raw extracted text before it is split into sections and chunks."""
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


