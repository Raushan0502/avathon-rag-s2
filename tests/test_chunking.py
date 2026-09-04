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

from src.ingest import chunk_prose, chunk_section, chunk_table, count_tokens, split_blocks

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


if __name__ == "__main__":
    unittest.main()
