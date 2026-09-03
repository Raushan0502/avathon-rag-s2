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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_RAW_DIR

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
    resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return resp


def get_submissions(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    return sec_get(url).json()


def find_latest_filing(submissions: dict, form_type: str) -> dict | None:
    recent = submissions["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == form_type:
            return {
                "form": form,
                "filingDate": recent["filingDate"][i],
                "accessionNumber": recent["accessionNumber"][i],
                "primaryDocument": recent["primaryDocument"][i],
            }
    return None


def build_doc_url(cik: str, accession_number: str, primary_document: str) -> str:
    cik_int = str(int(cik))  # strip leading zeros
    accession_nodash = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
        f"{accession_nodash}/{primary_document}"
    )


def download(url: str, dest_path: Path) -> str:
    resp = sec_get(url)
    dest_path.write_bytes(resp.content)
    return hashlib.sha256(resp.content).hexdigest()


def main() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for company in COMPANIES:
        print(f"Fetching submissions index for {company['name']} ({company['ticker']})...")
        submissions = get_submissions(company["cik"])

        for form_type in FORM_TYPES:
            filing = find_latest_filing(submissions, form_type)
            if filing is None:
                print(f"  [SKIP] no {form_type} found for {company['ticker']}")
                continue

            source_url = build_doc_url(
                company["cik"], filing["accessionNumber"], filing["primaryDocument"]
            )
            ext = Path(filing["primaryDocument"]).suffix or ".html"
            local_name = f"{company['ticker']}_{form_type}_{filing['filingDate']}{ext}"
            local_path = DATA_RAW_DIR / local_name

            print(f"  Downloading {form_type} filed {filing['filingDate']} -> {local_name}")
            checksum = download(source_url, local_path)

            manifest.append(
                {
                    "company": company["name"],
                    "ticker": company["ticker"],
                    "cik": company["cik"],
                    "form": form_type,
                    "filing_date": filing["filingDate"],
                    "source_url": source_url,
                    "local_path": str(local_path.relative_to(DATA_RAW_DIR.parent.parent)),
                    "sha256": checksum,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    manifest_path = DATA_RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(manifest)} documents. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
