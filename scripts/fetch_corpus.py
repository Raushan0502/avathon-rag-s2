"""
Step 2 — Data acquisition.

Sources the S2 (Gen AI for Enterprise Documents) knowledge corpus: real,
public SEC filings for a small set of companies, pulled directly from the
SEC EDGAR submissions API (no API key required). For each company we grab
the most recent 10-K (long, structured annual report) and the most recent
8-K (short, event-driven material-change filing), giving the corpus both
document-length and document-style diversity within a single, verifiably
real, freely-reproducible source.

Usage:
    python scripts/fetch_corpus.py

Writes documents to data/raw/ and a manifest (source URLs, filing dates,
checksums) to data/raw/manifest.json.
"""
import hashlib
import json
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


def main() -> None:
    """Download each company's latest 10-K and 8-K, then write the manifest."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for company in COMPANIES:
        print(f"Fetching submissions index for {company['name']} ({company['ticker']})...")
        submissions = sec_get(
            f"https://data.sec.gov/submissions/CIK{company['cik']}.json"
        ).json()

        for form_type in FORM_TYPES:
            filing = find_latest_filing(submissions, form_type)
            if filing is None:
                print(f"  [SKIP] no {form_type} found for {company['ticker']}")
                continue

            # Archives paths use the CIK without leading zeros and the
            # accession number without dashes.
            source_url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(company['cik'])}/"
                f"{filing['accessionNumber'].replace('-', '')}/{filing['primaryDocument']}"
            )
            extension = Path(filing["primaryDocument"]).suffix or ".html"
            local_path = (
                DATA_RAW_DIR
                / f"{company['ticker']}_{form_type}_{filing['filingDate']}{extension}"
            )

            print(f"  Downloading {form_type} filed {filing['filingDate']} -> {local_path.name}")
            content = sec_get(source_url).content
            local_path.write_bytes(content)

            manifest.append(
                {
                    "company": company["name"],
                    "ticker": company["ticker"],
                    "cik": company["cik"],
                    "form": form_type,
                    "filing_date": filing["filingDate"],
                    "source_url": source_url,
                    "local_path": str(local_path.relative_to(ROOT_DIR)),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(manifest)} documents. Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
