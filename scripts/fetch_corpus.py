"""
Step 2 — Data acquisition.

Sources the S2 (Gen AI for Enterprise Documents) knowledge corpus. S2
describes a company drowning in "contracts, reports, SOPs, emails", so the
corpus deliberately spans all four document types and three file formats
rather than one homogeneous source:

  reports   (HTML) -- 10-K annual reports and 8-K material-event filings
                      from SEC EDGAR, per company.
  contracts (HTML) -- EX-10 material-contract exhibits filed alongside the
                      10-Ks (equity award and RSU agreements).
  policies  (HTML) -- EX-19/EX-97 corporate policy exhibits (insider
                      trading, compensation recovery) -- real internal
                      governance documents, the closest public analogue to
                      an enterprise SOP.
  standards (PDF)  -- NIST security publications: an incident-handling
                      procedure guide and the Cybersecurity Framework.
                      These pair directly with the filings' Item 1C
                      Cybersecurity disclosures, so cross-document
                      questions ("how does this company's incident
                      response compare to the NIST stages?") are possible.
  emails    (TXT)  -- a sample of the public AESLC/Enron business email
                      corpus. No public email archive exists for the
                      issuers above, so this is a documented domain gap:
                      the emails are real business correspondence, but
                      from a different company and era than the filings.

Everything is free, keyless, and re-fetchable; each document is recorded
in the manifest with its source URL and a SHA-256 checksum.

Usage:
    python scripts/fetch_corpus.py

Writes documents to data/raw/ and the manifest to data/raw/manifest.json.
"""
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_RAW_DIR, MANIFEST_PATH, ROOT_DIR

# SEC requires a descriptive User-Agent identifying the requester; it does
# not need to be a real personal address, just a contactable-looking one.
SEC_USER_AGENT = "avathon-rag-s2-project research-contact@avathon-challenge.local"
REQUEST_DELAY_SECONDS = 0.3  # stay well under SEC's 10 req/sec fair-use limit

COMPANIES = [
    {"name": "Apple Inc.", "ticker": "AAPL", "cik": "0000320193"},
    {"name": "Microsoft Corporation", "ticker": "MSFT", "cik": "0000789019"},
    {"name": "Tesla, Inc.", "ticker": "TSLA", "cik": "0001318605"},
]

FORM_TYPES = ["10-K", "8-K"]

# Exhibit families worth indexing, keyed by the EX-<n> prefix in the
# filename. EX-10 are material contracts (Reg S-K item 601(b)(10)); EX-19
# and EX-97 are corporate policy documents. Certifications (EX-31/32),
# subsidiary lists (EX-21) and auditor consents (EX-23) are boilerplate and
# deliberately excluded -- they would add noise, not document diversity.
EXHIBIT_TYPES = {"ex10": "contract", "ex19": "policy", "ex97": "policy"}
EXHIBIT_RE = re.compile(r"ex[_-]?(10|19|97)[_.x-]?\d*\.htm$", re.IGNORECASE)

# Public-domain NIST publications, used as the standards/SOP portion of the
# corpus and as the PDF-format documents.
NIST_PDFS = [
    {
        "title": "NIST SP 800-61r2: Computer Security Incident Handling Guide",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r2.pdf",
        "slug": "NIST_SP-800-61r2",
    },
    {
        "title": "NIST Cybersecurity Framework 2.0",
        "url": "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf",
        "slug": "NIST_CSF-2.0",
    },
]

# AESLC (Annotated Enron Subject Line Corpus): individual plain-text business
# emails. Each file holds the body, then "@subject", then human-written
# alternative subject lines ("@ann0"...) that belong to that dataset's
# summarization task rather than the email itself -- those are stripped at
# ingestion time.
AESLC_LISTING = "https://api.github.com/repos/ryanzhumich/AESLC/contents/enron_subject_line/dev"
EMAIL_SAMPLE_SIZE = 25
EMAIL_SAMPLE_SEED = 42


def sec_get(url: str) -> requests.Response:
    """GET an EDGAR URL with the required User-Agent, then pause for fair use.

    Args:
        url: Full EDGAR API or Archives URL.

    Returns:
        The successful response.

    Raises:
        requests.HTTPError: On any non-2xx response.
    """
    response = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return response


def find_latest_filing(submissions: dict, form_type: str) -> dict | None:
    """Find the most recent filing of a given form type in a submissions index.

    EDGAR returns the recent-filings index as parallel arrays in reverse
    chronological order, so the first match is the newest.

    Args:
        submissions: Parsed ``data.sec.gov/submissions/CIK*.json`` payload.
        form_type: Form to look for, e.g. ``"10-K"`` or ``"8-K"``.

    Returns:
        Dict with ``filingDate``, ``accessionNumber`` and ``primaryDocument``,
        or None if the company has no filing of that type on the index.
    """
    recent = submissions["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == form_type:
            return {
                "filingDate": recent["filingDate"][i],
                "accessionNumber": recent["accessionNumber"][i],
                "primaryDocument": recent["primaryDocument"][i],
            }
    return None


def save_document(content: bytes, local_path: Path, **metadata) -> dict:
    """Write one fetched document to data/raw/ and build its manifest entry.

    Args:
        content: Raw bytes as fetched.
        local_path: Destination inside ``data/raw``.
        **metadata: Provenance fields (company, ticker, form, doc_type,
            source_url, ...) merged into the manifest entry.

    Returns:
        The manifest entry, including the file's SHA-256 and fetch timestamp.
    """
    local_path.write_bytes(content)
    return {
        **metadata,
        "local_path": str(local_path.relative_to(ROOT_DIR)),
        "format": local_path.suffix.lstrip(".").lower(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_sec_documents() -> list[dict]:
    """Fetch each company's latest 10-K and 8-K, plus their contract and
    policy exhibits.

    Exhibits are discovered from the 10-K's own filing index rather than
    hardcoded, since exhibit numbering changes year to year; only the
    families in ``EXHIBIT_TYPES`` are kept. A company with no matching
    exhibits (Apple incorporates most of its by reference) simply
    contributes none.

    Returns:
        Manifest entries for every SEC document fetched.
    """
    entries = []
    for company in COMPANIES:
        print(f"Fetching submissions index for {company['name']} ({company['ticker']})...")
        submissions = sec_get(
            f"https://data.sec.gov/submissions/CIK{company['cik']}.json"
        ).json()
        base = f"https://www.sec.gov/Archives/edgar/data/{int(company['cik'])}"
        common = {"company": company["name"], "ticker": company["ticker"], "cik": company["cik"]}

        for form_type in FORM_TYPES:
            filing = find_latest_filing(submissions, form_type)
            if filing is None:
                print(f"  [SKIP] no {form_type} found for {company['ticker']}")
                continue

            accession = filing["accessionNumber"].replace("-", "")
            source_url = f"{base}/{accession}/{filing['primaryDocument']}"
            extension = Path(filing["primaryDocument"]).suffix or ".html"
            local_path = (
                DATA_RAW_DIR / f"{company['ticker']}_{form_type}_{filing['filingDate']}{extension}"
            )
            print(f"  Downloading {form_type} filed {filing['filingDate']} -> {local_path.name}")
            entries.append(
                save_document(
                    sec_get(source_url).content,
                    local_path,
                    **common,
                    form=form_type,
                    doc_type="report",
                    filing_date=filing["filingDate"],
                    source_url=source_url,
                )
            )

            if form_type != "10-K":
                continue

            # Contract/policy exhibits live in the same filing directory.
            listing = sec_get(f"{base}/{accession}/index.json").json()
            for item in listing["directory"]["item"]:
                match = EXHIBIT_RE.search(item["name"])
                if not match:
                    continue
                doc_type = EXHIBIT_TYPES[f"ex{match.group(1)}"]
                exhibit_url = f"{base}/{accession}/{item['name']}"
                exhibit_path = DATA_RAW_DIR / f"{company['ticker']}_{item['name']}"
                print(f"    exhibit ({doc_type}): {item['name']}")
                entries.append(
                    save_document(
                        sec_get(exhibit_url).content,
                        exhibit_path,
                        **common,
                        form=f"EX-{match.group(1)}",
                        doc_type=doc_type,
                        filing_date=filing["filingDate"],
                        source_url=exhibit_url,
                    )
                )
    return entries


def fetch_nist_pdfs() -> list[dict]:
    """Fetch the public-domain NIST security publications (PDF format).

    Returns:
        Manifest entries for each NIST PDF.
    """
    entries = []
    for pub in NIST_PDFS:
        print(f"Downloading {pub['slug']}.pdf ...")
        response = requests.get(pub["url"], headers={"User-Agent": SEC_USER_AGENT}, timeout=60)
        response.raise_for_status()
        entries.append(
            save_document(
                response.content,
                DATA_RAW_DIR / f"{pub['slug']}.pdf",
                company="NIST",
                ticker="NIST",
                form=pub["slug"],
                doc_type="standard",
                filing_date="",
                source_url=pub["url"],
                title=pub["title"],
            )
        )
        time.sleep(REQUEST_DELAY_SECONDS)
    return entries


def fetch_emails() -> list[dict]:
    """Fetch a deterministic sample of AESLC business emails (plain text).

    The sample is drawn with a fixed seed so re-running reproduces the same
    emails, keeping the corpus (and therefore the evaluation) stable.

    Returns:
        Manifest entries for each sampled email.
    """
    listing = requests.get(AESLC_LISTING, timeout=60)
    listing.raise_for_status()
    files = sorted(item["name"] for item in listing.json())
    sample = random.Random(EMAIL_SAMPLE_SEED).sample(files, EMAIL_SAMPLE_SIZE)
    print(f"Downloading {len(sample)} emails from AESLC (of {len(files)} available)...")

    entries = []
    for name in sample:
        url = (
            "https://raw.githubusercontent.com/ryanzhumich/AESLC/master/"
            f"enron_subject_line/dev/{name}"
        )
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        entries.append(
            save_document(
                response.content,
                DATA_RAW_DIR / f"EMAIL_{name.replace('.subject', '')}.txt",
                company="Enron Corp. (AESLC corpus)",
                ticker="EMAIL",
                form="email",
                doc_type="email",
                filing_date="",
                source_url=url,
            )
        )
    return entries


def main() -> None:
    """Fetch every document family and write the combined manifest."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = fetch_sec_documents() + fetch_nist_pdfs() + fetch_emails()

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    by_type: dict[str, int] = {}
    for entry in manifest:
        by_type[entry["doc_type"]] = by_type.get(entry["doc_type"], 0) + 1
    print(f"\nWrote {len(manifest)} documents by type: {by_type}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
