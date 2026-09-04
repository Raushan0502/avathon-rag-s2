"""Unit tests for src/ingest.py -- HTML cleaning, section splitting, chunking.

All tests run on small synthetic documents; nothing here touches the real
corpus, the network, or the embedding model.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import (
    load_document_text,
    normalise_text,
    render_table,
    split_into_sections,
)


class TestNormaliseText(unittest.TestCase):
    def test_normalises_smart_quotes_and_dashes_to_ascii(self) -> None:
        # The same word must embed identically whichever filer produced it.
        self.assertEqual(normalise_text("the Company’s “plan”"), "the Company's \"plan\"")
        self.assertEqual(normalise_text("non‑GAAP"), "non-GAAP")

    def test_strips_control_characters_and_zero_width_marks(self) -> None:
        self.assertEqual(normalise_text("clean\x00te​xt"), "cleantext")

    def test_rejoins_words_hyphenated_across_a_line_break(self) -> None:
        # PDF text layers break words at line ends; left alone, one word
        # embeds as two meaningless fragments.
        self.assertEqual(normalise_text("informa-\ntion security"), "information security")

    def test_does_not_rejoin_a_genuine_compound_at_line_end(self) -> None:
        # "Well-Known" keeps its hyphen: the continuation is capitalised,
        # so it is a real compound, not a broken word.
        self.assertIn("Well-", normalise_text("Well-\nKnown Sources"))

    def test_removes_table_of_contents_dot_leaders(self) -> None:
        text = "Real content here\n2.1 Events and Incidents ................ 6\nMore content"
        self.assertEqual(normalise_text(text), "Real content here\nMore content")

    def test_removes_standalone_page_numbers(self) -> None:
        self.assertEqual(normalise_text("Content\n17\nMore"), "Content\nMore")

    def test_removes_repeated_page_footers_that_differ_only_by_page_number(self) -> None:
        # The real case: a running footer is never twice the same string,
        # because it carries the page number. Exact matching misses it.
        text = "\n".join(
            f"In fiscal {2020 + i} the segment reported growth driven by higher unit "
            f"volumes across every major geography.\n"
            f"Apple Inc. | 2025 Form 10-K | {i + 10}"
            for i in range(6)
        )
        cleaned = normalise_text(text)
        self.assertNotIn("Form 10-K |", cleaned)

    def test_digit_masking_does_not_delete_prose_differing_only_by_a_number(self) -> None:
        # Masking digits makes such sentences look identical; only the
        # word-count guard stops them being deleted as page furniture.
        text = "\n".join(
            f"In fiscal {2020 + i} the segment reported growth driven by higher unit "
            f"volumes across every major geography."
            for i in range(6)
        )
        cleaned = normalise_text(text)
        self.assertEqual(len(cleaned.splitlines()), 6, "real prose must survive")
        self.assertIn("In fiscal 2023", cleaned)

    def test_keeps_a_long_repeated_line_which_is_probably_real_prose(self) -> None:
        sentence = (
            "The Company is subject to legal proceedings and claims that have not been "
            "fully resolved and that have arisen in the ordinary course of business."
        )
        text = "\n".join([sentence] * 6)
        self.assertIn(sentence, normalise_text(text))

    def test_leaves_rendered_table_rows_untouched(self) -> None:
        # Table rows repeat structurally and contain digits; the line-level
        # cleaners must not dismantle what render_table just recovered.
        row = "| Total net sales | $416,161 | $391,035 |"
        text = "\n".join([row] * 6)
        self.assertIn(row, normalise_text(text))

    def test_collapses_runs_of_spaces_and_blank_lines(self) -> None:
        self.assertEqual(normalise_text("a    b\n\n\n   \nc"), "a b\nc")


class TestRenderTable(unittest.TestCase):
    def test_renders_markdown_pipe_table_with_header_separator(self) -> None:
        rendered = render_table([["Year", "Sales"], ["2025", "416,161"], ["2024", "391,035"]])
        self.assertEqual(
            rendered.splitlines(),
            ["| Year | Sales |", "| --- | --- |", "| 2025 | 416,161 |", "| 2024 | 391,035 |"],
        )

    def test_keeps_figures_attached_to_row_label_and_column_header(self) -> None:
        # The whole point: a value must stay recoverable as
        # (row label, column header) rather than becoming a loose number.
        rendered = render_table([["", "2025", "2024"], ["Total net sales", "416,161", "391,035"]])
        row = [line for line in rendered.splitlines() if "Total net sales" in line][0]
        self.assertEqual(row, "| Total net sales | 416,161 | 391,035 |")

    def test_merges_currency_cell_into_the_following_value(self) -> None:
        # Filings put the currency mark in its own cell.
        rendered = render_table([["Region", "Amount", ""], ["Americas", "$", "178,353"]])
        self.assertIn("| Americas | $178,353 |", rendered)

    def test_merges_percent_cell_into_the_preceding_value(self) -> None:
        rendered = render_table([["Region", "Change", ""], ["Europe", "10", "%"]])
        self.assertIn("| Europe | 10% |", rendered)

    def test_symbol_merge_realigns_rows_of_differing_cell_counts(self) -> None:
        # Only some rows carry a "$" cell, which skews the grid; merging it
        # away has to bring those rows back into alignment.
        rendered = render_table(
            [["Region", "2025"], ["Americas", "$", "178,353"], ["Europe", "111,032"]]
        )
        body = [line for line in rendered.splitlines() if "|" in line][2:]
        self.assertEqual(body[0].count("|"), body[1].count("|"), f"misaligned: {body}")

    def test_drops_columns_that_are_empty_throughout(self) -> None:
        rendered = render_table([["A", "", "B"], ["1", "", "2"]])
        self.assertEqual(rendered.splitlines()[0], "| A | B |")

    def test_single_column_layout_table_is_not_dressed_up_as_a_table(self) -> None:
        # HTML filings use <table> for page layout; those must not gain a
        # fake header row implying tabular meaning.
        rendered = render_table([["Just a heading"], ["and a paragraph"]])
        self.assertNotIn("|", rendered)
        self.assertEqual(rendered, "Just a heading\nand a paragraph")

    def test_empty_table_yields_empty_string(self) -> None:
        self.assertEqual(render_table([]), "")
        self.assertEqual(render_table([["", ""], ["", ""]]), "")


class TestLoadDocumentText(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def write(self, name: str, content: str) -> Path:
        path = self.dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_html_strips_tags_and_keeps_line_structure(self) -> None:
        path = self.write("f.htm", "<html><body><p>Item 1. Business</p><p>We sell things.</p></body></html>")
        self.assertEqual(load_document_text(path), "Item 1. Business\nWe sell things.")

    def test_html_drops_script_and_style_content(self) -> None:
        path = self.write("f.html", "<body><script>var x=1;</script><style>p{color:red}</style><p>Real</p></body>")
        self.assertEqual(load_document_text(path), "Real")

    def test_html_collapses_blank_lines_but_not_line_breaks(self) -> None:
        # Section detection anchors on line boundaries, so real newlines must
        # survive while empty filler lines are removed.
        path = self.write("f.htm", "<div>Alpha</div><div>   </div><div></div><div>Beta</div>")
        self.assertEqual(load_document_text(path), "Alpha\nBeta")

    def test_email_promotes_subject_and_drops_annotation_trailer(self) -> None:
        # The @ann* lines are the AESLC dataset's alternative subject lines,
        # not part of the email -- indexing them would attribute text to the
        # sender that they never wrote.
        path = self.write(
            "e.txt",
            "Please confirm the meeting.\n\n@subject\nMeeting Confirmation\n\n@ann0\nmtg confirm\n",
        )
        text = load_document_text(path)
        self.assertEqual(text, "Subject: Meeting Confirmation\nPlease confirm the meeting.")
        self.assertNotIn("@ann0", text)
        self.assertNotIn("mtg confirm", text)

    def test_email_without_a_subject_trailer_is_read_as_is(self) -> None:
        path = self.write("e.txt", "Just a plain body.\n")
        self.assertEqual(load_document_text(path), "Just a plain body.")

    def test_html_tables_are_rendered_in_place_not_flattened(self) -> None:
        path = self.write(
            "f.htm",
            "<body><p>Segment results:</p><table>"
            "<tr><td></td><td>2025</td><td>2024</td></tr>"
            "<tr><td>Total net sales</td><td>416,161</td><td>391,035</td></tr>"
            "</table><p>After the table.</p></body>",
        )
        text = load_document_text(path)
        self.assertIn("| Total net sales | 416,161 | 391,035 |", text)
        # Rendered in document order, between the surrounding prose.
        self.assertLess(text.index("Segment results:"), text.index("Total net sales"))
        self.assertLess(text.index("Total net sales"), text.index("After the table."))

    def test_html_table_colspan_is_expanded_so_columns_stay_aligned(self) -> None:
        path = self.write(
            "f.htm",
            "<body><table>"
            "<tr><td></td><td colspan='2'>FY2025</td><td>FY2024</td></tr>"
            "<tr><td>Revenue</td><td>1</td><td>2</td><td>3</td></tr>"
            "</table></body>",
        )
        header = [ln for ln in load_document_text(path).splitlines() if "FY2025" in ln][0]
        self.assertEqual(header, "| | FY2025 | | FY2024 |".replace("| |", "|  |"))

    def test_html_images_are_dropped_including_alt_text(self) -> None:
        # Documented limitation: image content needs an OCR stage we do not have.
        path = self.write("f.htm", "<body><p>Chart:</p><img alt='Revenue grew 12%'></body>")
        self.assertNotIn("Revenue grew", load_document_text(path))

    def test_unsupported_format_raises_with_the_filename(self) -> None:
        path = self.write("data.csv", "a,b,c")
        with self.assertRaises(ValueError) as ctx:
            load_document_text(path)
        self.assertIn("data.csv", str(ctx.exception))


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


class TestNumberedHeadingSections(unittest.TestCase):
    """Non-SEC documents (NIST, contracts, policies) carry their own
    structure; before this they all became one "Full Document" section."""

    def test_splits_a_policy_on_its_numbered_clauses(self) -> None:
        text = (
            "Insider Trading Policy\n"
            "1. PURPOSE\nTo prevent misuse of material nonpublic information.\n"
            "2. SUMMARY\nThis policy prohibits trading on such information.\n"
            "3. REQUIREMENTS\nYou must preclear all trades."
        )
        titles = [title for title, _ in split_into_sections(text)]
        self.assertIn("1. PURPOSE", titles)
        self.assertIn("3. REQUIREMENTS", titles)

    def test_splits_a_standard_on_dotted_subsection_numbers(self) -> None:
        text = (
            "1.1 Authority\nThis publication was developed by NIST.\n"
            "1.2 Purpose and Scope\nThe guide assists organizations.\n"
            "1.3 Audience\nThis document is for incident responders."
        )
        self.assertEqual(
            [t for t, _ in split_into_sections(text)],
            ["1.1 Authority", "1.2 Purpose and Scope", "1.3 Audience"],
        )

    def test_text_before_the_first_heading_is_kept_as_front_matter(self) -> None:
        text = (
            "Some cover page text.\n"
            "1. First\nBody one.\n2. Second\nBody two.\n3. Third\nBody three."
        )
        sections = dict(split_into_sections(text))
        self.assertIn("Front Matter", sections)
        self.assertIn("Some cover page text.", sections["Front Matter"])

    def test_falls_back_to_one_section_when_structure_is_absent(self) -> None:
        # An email has no headings and should stay whole.
        text = "Subject: Historical flows\nHoward, per our conversation, these are the meters."
        self.assertEqual(split_into_sections(text), [("Full Document", text)])

    def test_spaced_capital_cover_titles_are_not_treated_as_headings(self) -> None:
        # A NIST cover page sets the title in spaced capitals, which matched
        # as heading "C" followed by "O M P U T E R ..." before the pattern
        # required a bare capital to carry its period.
        text = (
            "C O M P U T E R S E C U R I T Y\n"
            "1. First\nBody one.\n2. Second\nBody two.\n3. Third\nBody three."
        )
        titles = [title for title, _ in split_into_sections(text)]
        self.assertNotIn("C O M P U T E R S E C U R I T Y", titles)

    def test_table_rows_are_never_treated_as_headings(self) -> None:
        text = (
            "| 1. Something | 2 |\n| 2. Another | 3 |\n| 3. Third | 4 |\n"
            "| 4. Fourth | 5 |\n| 5. Fifth | 6 |"
        )
        self.assertEqual(split_into_sections(text), [("Full Document", text)])


if __name__ == "__main__":
    unittest.main()
