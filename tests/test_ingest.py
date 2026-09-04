"""Unit tests for src/ingest.py -- HTML cleaning, section splitting, chunking.

All tests run on small synthetic documents; nothing here touches the real
corpus, the network, or the embedding model.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import chunk_words, html_to_text, split_into_sections


class TestHtmlToText(unittest.TestCase):
    def test_strips_tags_and_keeps_line_structure(self) -> None:
        html = "<html><body><p>Item 1. Business</p><p>We sell things.</p></body></html>"
        self.assertEqual(html_to_text(html), "Item 1. Business\nWe sell things.")

    def test_drops_script_and_style_content(self) -> None:
        html = "<body><script>var x = 1;</script><style>p{color:red}</style><p>Real text</p></body>"
        self.assertEqual(html_to_text(html), "Real text")

    def test_collapses_blank_lines_but_not_line_breaks(self) -> None:
        # Section detection anchors on line boundaries, so real newlines must
        # survive while empty filler lines are removed.
        html = "<div>Alpha</div><div>   </div><div></div><div>Beta</div>"
        self.assertEqual(html_to_text(html), "Alpha\nBeta")


class TestSplitIntoSections(unittest.TestCase):
    def test_splits_on_item_headers(self) -> None:
        text = (
            "Item 1. Business\nWe design products.\n"
            "Item 1A. Risk Factors\nCompetition is intense.\n"
            "Item 2. Properties\nHeadquarters in Cupertino."
        )
        sections = split_into_sections(text)
        self.assertEqual(
            [title for title, _ in sections],
            ["Item 1. Business", "Item 1A. Risk Factors", "Item 2. Properties"],
        )
        self.assertEqual(sections[1][1], "Competition is intense.")

    def test_falls_back_to_full_document_when_too_few_headers(self) -> None:
        text = "Item 1. Business\nOnly one real header here."
        self.assertEqual(split_into_sections(text), [("Full Document", text)])

    def test_table_of_contents_entries_do_not_create_duplicate_sections(self) -> None:
        # Each Item appears twice: once in the TOC, once as the real heading.
        # The real (later) one must win, so bodies come from the document body.
        text = (
            "Item 1. Business\nItem 1A. Risk Factors\nItem 2. Properties\n"
            "Item 1. Business\nReal business content.\n"
            "Item 1A. Risk Factors\nReal risk content.\n"
            "Item 2. Properties\nReal property content."
        )
        sections = split_into_sections(text)
        titles = [title for title, _ in sections]
        self.assertEqual(len(titles), len(set(titles)), "no duplicated section titles")
        bodies = dict(sections)
        self.assertEqual(bodies["Item 1A. Risk Factors"], "Real risk content.")

    def test_repeated_running_page_headers_collapse_into_one_section(self) -> None:
        # The MSFT-10-K failure mode: "Item 7" reprinted atop every page gets
        # flattened into many pseudo-headers sharing one item number. They must
        # collapse to a single section, not one section per page.
        text = (
            "Item 1. Business\nBusiness content.\n"
            "Item 7. Managements Discussion\nPage one content.\n"
            "Item 7 Economic Conditions\nPage two content.\n"
            "Item 7 Dividends And Other\nPage three content.\n"
            "Item 8. Financial Statements\nFinancial content."
        )
        sections = split_into_sections(text)
        item7_sections = [t for t, _ in sections if t.lower().startswith("item 7")]
        self.assertEqual(len(item7_sections), 1, f"expected 1 Item 7 section, got {item7_sections}")
        self.assertIn("Page three content.", dict(sections)[item7_sections[0]])


class TestChunkWords(unittest.TestCase):
    def test_returns_empty_list_for_blank_text(self) -> None:
        self.assertEqual(chunk_words(""), [])
        self.assertEqual(chunk_words("   \n  "), [])

    def test_short_text_becomes_a_single_chunk(self) -> None:
        self.assertEqual(chunk_words("one two three", size=10, overlap=2), ["one two three"])

    def test_windows_overlap_by_the_requested_word_count(self) -> None:
        words = [str(i) for i in range(10)]
        chunks = chunk_words(" ".join(words), size=4, overlap=2)
        self.assertEqual(chunks[0], "0 1 2 3")
        self.assertEqual(chunks[1], "2 3 4 5", "second window repeats the last 2 words")

    def test_every_word_appears_in_at_least_one_chunk(self) -> None:
        words = [str(i) for i in range(25)]
        chunks = chunk_words(" ".join(words), size=7, overlap=3)
        covered = {w for chunk in chunks for w in chunk.split()}
        self.assertEqual(covered, set(words), "no words dropped at the tail")

    def test_no_chunk_exceeds_the_window_size(self) -> None:
        chunks = chunk_words(" ".join(str(i) for i in range(50)), size=8, overlap=2)
        self.assertTrue(all(len(c.split()) <= 8 for c in chunks))


if __name__ == "__main__":
    unittest.main()
