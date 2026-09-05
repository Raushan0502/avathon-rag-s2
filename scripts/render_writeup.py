"""
Render the README's "Technical write-up" section to the submission PDF.

The brief wants a 1-2 page PDF, but a PDF maintained by hand drifts from the
README that describes the same system -- which is exactly the failure this
project kept hitting with stale metrics. So the README section is the source
and this script is the only way the PDF gets made.

Markdown is rendered with markdown-it-py (already a dependency) and printed
by headless Chrome, so there is no new toolchain to install.

Usage:
    python scripts/render_writeup.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from markdown_it import MarkdownIt

from src.config import ROOT_DIR

SECTION_HEADING = "## Technical write-up"
OUT_DIR = ROOT_DIR / "write-up"

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]

CSS = """
@page { size: A4; margin: 10mm 11mm; }
body { font-family: Georgia, "Times New Roman", serif; font-size: 11pt;
       line-height: 1.24; color: #111; margin: 0; }
h1 { font-size: 16pt; margin: 0 0 2pt; letter-spacing: -0.2pt; }
.sub { font-size: 10pt; color: #333; margin: 0 0 8pt;
       border-bottom: 1.2pt solid #222; padding-bottom: 5pt; }
h2 { display: none; }
h3 { font-size: 11.5pt; margin: 7pt 0 3pt; padding-bottom: 2pt;
     border-bottom: 0.5pt solid #ccc; }
p { margin: 0 0 4pt; text-align: justify; }
strong { color: #000; }
code { font-family: "Consolas", monospace; font-size: 10pt;
       background: #f2f2f2; padding: 0 2px; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 5pt;
        font-size: 9.5pt; font-family: Helvetica, Arial, sans-serif; }
th { background: #ececec; text-align: left; }
th, td { border: 0.6pt solid #bbb; padding: 1.8pt 5pt; }
em { color: #222; }
"""


def find_chrome() -> Path | None:
    """Locate a Chromium-family browser able to print to PDF."""
    for path in CHROME_CANDIDATES:
        if path.exists():
            return path
    found = shutil.which("chrome") or shutil.which("msedge")
    return Path(found) if found else None


def extract_section(readme: str) -> str:
    """Return the write-up section's markdown, without its own heading."""
    start = readme.index(SECTION_HEADING) + len(SECTION_HEADING)
    end = readme.index("\n## ", start)
    return readme[start:end].strip()


def build_html(markdown_body: str) -> str:
    """Render the section markdown into a print-styled standalone page."""
    body = MarkdownIt("commonmark", {"html": False}).enable("table").render(markdown_body)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>"
        "<h1>Grounded Q&amp;A over Enterprise Documents</h1>"
        "<p class='sub'>Avathon AI/ML Hiring Challenge &mdash; Track D (RAG / LLM "
        "Knowledge Systems) &times; Scenario S2 &nbsp;&middot;&nbsp; Raushan Kumar "
        "&nbsp;&middot;&nbsp; github.com/Raushan0502/avathon-rag-s2</p>"
        f"{body}</body></html>"
    )


def main() -> int:
    """Render the README write-up section to write-up/technical-writeup.pdf."""
    readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    try:
        section = extract_section(readme)
    except ValueError:
        print(f"Could not find '{SECTION_HEADING}' section in README.md")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUT_DIR / "technical-writeup.html"
    pdf_path = OUT_DIR / "technical-writeup.pdf"
    html_path.write_text(build_html(section), encoding="utf-8")
    print(f"Rendered HTML -> {html_path}  ({len(section.split())} words)")

    chrome = find_chrome()
    if chrome is None:
        print("No Chrome/Edge found; open the HTML and print to PDF manually.")
        return 1

    subprocess.run(
        [
            str(chrome), "--headless", "--disable-gpu", "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}", html_path.as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    if not pdf_path.exists():
        print("Chrome ran but produced no PDF.")
        return 1
    print(f"Rendered PDF  -> {pdf_path}  ({pdf_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
