"""Unit tests for the chunking stage of src/ingest.py.

These cover the four defects that replaced the original fixed word-window
splitter: discarded newlines, tables split without their header, chunks
starting mid-sentence, and sizing in words rather than the tokens the
embedding model actually enforces.

Nothing here needs the network. ``count_tokens`` uses the real tokenizer
when it is cached locally and a word-based estimate otherwise, so the
assertions are written to hold either way.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import (
    build_embed_text,
    chunk_prose,
    chunk_section,
    chunk_table,
    count_tokens,
    split_blocks,
)

HEADER = "|  | 2025 | 2024 |"
SEPARATOR = "| --- | --- | --- |"
DATA_ROWS = [f"| Line item {i} | {i}00 | {i}10 |" for i in range(120)]
BIG_TABLE = "\n".join([HEADER, SEPARATOR] + DATA_ROWS)
LONG_PROSE = " ".join(f"This is sentence number {i} of the section." for i in range(200))


class TestCountTokens(unittest.TestCase):
    def test_longer_text_costs_more_tokens(self) -> None:
        self.assertGreater(count_tokens("a much longer piece of text here"), count_tokens("short"))

    def test_empty_text_costs_nothing_meaningful(self) -> None:
        self.assertLessEqual(count_tokens(""), 1)

    def test_numeric_table_row_is_not_one_token_per_word(self) -> None:
        # Precisely why word-based sizing was unsafe: figures such as
        # "$416,161" tokenise far more heavily than prose of equal word count.
        row = "| Total net sales | $416,161 | $391,035 | $383,285 |"
        self.assertGreaterEqual(count_tokens(row), len(row.split()))


class TestSplitBlocks(unittest.TestCase):
    def test_separates_tables_from_prose_preserving_order(self) -> None:
        text = "Intro line.\n| a | b |\n| 1 | 2 |\nClosing line."
        self.assertEqual([kind for kind, _ in split_blocks(text)], ["prose", "table", "prose"])

    def test_contiguous_table_rows_form_a_single_block(self) -> None:
        blocks = split_blocks("| a | b |\n| --- | --- |\n| 1 | 2 |")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][0], "table")

    def test_blank_input_yields_no_blocks(self) -> None:
        self.assertEqual(split_blocks("   \n  \n"), [])


class TestChunkTable(unittest.TestCase):
    def test_a_table_that_fits_is_never_split(self) -> None:
        small = "\n".join([HEADER, SEPARATOR] + DATA_ROWS[:3])
        self.assertEqual(chunk_table(small), [small])

    def test_every_part_of_a_split_table_repeats_the_header(self) -> None:
        # The defect being fixed: without the header, "| Line item 15 | 1500 |"
        # gives no clue which column is which year.
        parts = chunk_table(BIG_TABLE, max_tokens=200)
        self.assertGreater(len(parts), 1, "table should have needed splitting")
        for part in parts:
            self.assertTrue(part.startswith(HEADER), f"missing header: {part[:60]!r}")

    def test_splits_only_on_row_boundaries(self) -> None:
        for part in chunk_table(BIG_TABLE, max_tokens=200):
            for line in part.splitlines():
                self.assertTrue(line.startswith("|") and line.endswith("|"), line)

    def test_no_data_row_is_lost_or_duplicated_across_the_split(self) -> None:
        parts = chunk_table(BIG_TABLE, max_tokens=200)
        seen = [ln for part in parts for ln in part.splitlines() if "Line item" in ln]
        self.assertEqual(sorted(seen), sorted(DATA_ROWS))

    def test_row_structure_survives_as_newlines(self) -> None:
        # The old splitter joined windows with spaces and flattened rows.
        self.assertIn("\n", chunk_table(BIG_TABLE, max_tokens=200)[0])


class TestChunkProse(unittest.TestCase):
    def test_returns_empty_list_for_blank_text(self) -> None:
        self.assertEqual(chunk_prose(""), [])
        self.assertEqual(chunk_prose("   \n  "), [])

    def test_short_text_becomes_a_single_chunk(self) -> None:
        self.assertEqual(chunk_prose("One sentence only."), ["One sentence only."])

    def test_chunks_begin_at_a_sentence_boundary(self) -> None:
        for chunk in chunk_prose(LONG_PROSE, max_tokens=120, overlap_tokens=20):
            self.assertTrue(chunk.startswith("This is"), f"mid-sentence start: {chunk[:40]!r}")

    def test_splitting_actually_happens_for_long_input(self) -> None:
        self.assertGreater(len(chunk_prose(LONG_PROSE, max_tokens=120, overlap_tokens=20)), 1)

    def test_consecutive_chunks_share_overlapping_text(self) -> None:
        chunks = chunk_prose(LONG_PROSE, max_tokens=120, overlap_tokens=40)
        first_sentences = set(chunks[0].split(". "))
        second_sentences = set(chunks[1].split(". "))
        self.assertTrue(first_sentences & second_sentences, "no overlap carried between chunks")

    def test_no_sentence_is_dropped(self) -> None:
        chunks = chunk_prose(LONG_PROSE, max_tokens=120, overlap_tokens=20)
        joined = " ".join(chunks)
        for i in (0, 99, 199):
            self.assertIn(f"sentence number {i} ", joined + " ")

    def test_does_not_split_on_common_abbreviations(self) -> None:
        text = "Apple Inc. reported growth. The U.S. market expanded."
        self.assertEqual(len(chunk_prose(text)), 1)


class TestChunkSection(unittest.TestCase):
    def test_tables_are_chunked_apart_from_surrounding_prose(self) -> None:
        # A table must never be glued onto a paragraph and split by a
        # counter that cannot see its row structure.
        text = "Some intro prose.\n| a | b |\n| --- | --- |\n| 1 | 2 |\nClosing prose."
        table_chunks = [c for c in chunk_section(text) if c.startswith("|")]
        self.assertEqual(len(table_chunks), 1)
        self.assertNotIn("intro prose", table_chunks[0])

    def test_keeps_prose_and_table_content_both_present(self) -> None:
        text = "Some intro prose.\n| a | b |\n| --- | --- |\n| 1 | 2 |\nClosing prose."
        joined = " ".join(chunk_section(text))
        self.assertIn("intro prose", joined)
        self.assertIn("| 1 | 2 |", joined)

    def test_returns_nothing_for_empty_input(self) -> None:
        self.assertEqual(chunk_section(""), [])


class TestOversizedUnits(unittest.TestCase):
    """A sentence or table row larger than the whole budget has no structural
    boundary to split on. Emitted whole it is silently truncated at the
    model's 512-token limit -- 80 of 8,146 corpus chunks hit this, the worst
    at 3,374 tokens."""

    LIMIT = 512  # the embedding model's hard max_seq_length

    def test_a_single_giant_sentence_is_split_within_budget(self) -> None:
        giant = " ".join(f"word{i}" for i in range(3000))
        for chunk in chunk_prose(giant, max_tokens=400):
            self.assertLess(count_tokens(chunk), self.LIMIT, "would be silently truncated")

    def test_a_single_giant_table_row_is_split_within_budget(self) -> None:
        row = "| row | " + " ".join(f"x{i}" for i in range(3000)) + " |"
        table = f"{HEADER}\n{SEPARATOR}\n{row}"
        for chunk in chunk_table(table, max_tokens=400):
            self.assertLess(count_tokens(chunk), self.LIMIT)

    def test_no_content_is_lost_when_hard_splitting(self) -> None:
        giant = " ".join(f"word{i}" for i in range(1500))
        joined = " ".join(chunk_prose(giant, max_tokens=400))
        for probe in ("word0", "word749", "word1499"):
            self.assertIn(probe, joined)

    def test_normal_text_is_untouched_by_the_guard(self) -> None:
        # The hard split loses structural alignment, so it must apply only
        # when a unit already exceeds the budget on its own.
        text = "One sentence. Two sentences. Three sentences."
        self.assertEqual(chunk_prose(text, max_tokens=400), [text])

    def test_a_single_unsplittable_token_is_returned_rather_than_looping(self) -> None:
        self.assertEqual(len(chunk_prose("x" * 5000, max_tokens=10)), 1)


class TestBuildEmbedText(unittest.TestCase):
    META = {"company": "Apple Inc.", "form": "10-K", "ticker": "AAPL"}

    def test_prefixes_company_form_and_section(self) -> None:
        out = build_embed_text(self.META, "Item 1A. Risk Factors", "Competition is intense.")
        self.assertTrue(out.startswith("Apple Inc. | 10-K | Item 1A. Risk Factors"))

    def test_original_chunk_text_is_preserved_after_the_header(self) -> None:
        out = build_embed_text(self.META, "Item 2. Properties", "Cupertino, California.")
        self.assertTrue(out.endswith("Cupertino, California."))

    def test_placeholder_section_is_omitted(self) -> None:
        # Non-SEC documents use "Full Document", which carries no meaning
        # and would only dilute the embedding.
        out = build_embed_text(self.META, "Full Document", "Body text.")
        self.assertNotIn("Full Document", out)
        self.assertTrue(out.startswith("Apple Inc. | 10-K"))

    def test_missing_metadata_does_not_leave_empty_separators(self) -> None:
        out = build_embed_text({}, "Full Document", "Body text.")
        self.assertEqual(out, "Body text.")
        self.assertNotIn("|", out)


if __name__ == "__main__":
    unittest.main()
