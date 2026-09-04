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
from bs4 import BeautifulSoup

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_RAW_DIR, MANIFEST_PATH, ROOT_DIR

# SEC requires a descriptive User-Agent identifying the requester; it does
# not need to be a real personal address, just a contactable-looking one.
SEC_USER_AGENT = "avathon-rag-s2-project research-contact@avathon-challenge.local"
REQUEST_DELAY_SECONDS = 0.4  # stay well under SEC's 10 req/sec fair-use limit
MAX_ATTEMPTS = 4  # EDGAR returns transient 503s under sustained crawling
BACKOFF_SECONDS = 1.5

# The first three are listed first deliberately: the evaluation set is
# authored against their filings and exhibits, and the per-type caps below
# keep documents in fetch order, so these are always retained.
COMPANIES = [
    {"name": "Apple Inc.", "ticker": "AAPL", "cik": "0000320193"},
    {"name": "Microsoft Corporation", "ticker": "MSFT", "cik": "0000789019"},
    {"name": "Tesla, Inc.", "ticker": "TSLA", "cik": "0001318605"},
    {"name": "Amazon.com, Inc.", "ticker": "AMZN", "cik": "0001018724"},
    {"name": "Alphabet Inc.", "ticker": "GOOGL", "cik": "0001652044"},
    {"name": "Meta Platforms, Inc.", "ticker": "META", "cik": "0001326801"},
    {"name": "NVIDIA Corporation", "ticker": "NVDA", "cik": "0001045810"},
    {"name": "JPMorgan Chase & Co.", "ticker": "JPM", "cik": "0000019617"},
    {"name": "Walmart Inc.", "ticker": "WMT", "cik": "0000104169"},
    {"name": "Johnson & Johnson", "ticker": "JNJ", "cik": "0000200406"},
    {"name": "The Coca-Cola Company", "ticker": "KO", "cik": "0000021344"},
    {"name": "Cisco Systems, Inc.", "ticker": "CSCO", "cik": "0000858877"},
]

FORM_TYPES = ["10-K", "8-K"]

# Target roughly this many documents per type. Balance is by *document
# count*, which is not the same as balance by retrieval weight: a 10-K is
# ~250 chunks and an email is 1, so reports still dominate chunk share.
# That trade-off is stated explicitly in the README rather than implied
# away by the even document counts.
TARGET_PER_TYPE = 20

# Exhibit families worth indexing, keyed by the EX-<n> prefix in the
# filename. EX-10 are material contracts (Reg S-K item 601(b)(10)); EX-19
# and EX-97 are corporate policy documents. Certifications (EX-31/32),
# subsidiary lists (EX-21) and auditor consents (EX-23) are boilerplate and
# deliberately excluded -- they would add noise, not document diversity.
EXHIBIT_TYPES = {"10": "contract", "19": "policy", "97": "policy"}
# Matched against EDGAR's authoritative Type column (e.g. "EX-10.9"), not
# the filename: filers name exhibit files inconsistently (Apple writes
# "a10-kexhibit4109272025.htm"), so filename matching silently misses them.
# The optional sub-number is anchored to the end so "EX-101.SCH" -- an XBRL
# taxonomy file, not a contract -- cannot match "EX-10".
EXHIBIT_TYPE_RE = re.compile(r"^EX-(10|19|97)(\.\d+)?$", re.IGNORECASE)

# Exhibits are harvested from a deeper filing history than the reports:
# material contracts (EX-10) are filed only occasionally, so scanning just
# the latest filing per company finds almost none.
EXHIBIT_SCAN_DEPTH = {"10-K": 3, "8-K": 8}

# Public-domain NIST publications, used as the standards/SOP portion of the
# corpus and as the PDF-format documents.
# Publications are listed smallest-first after the two the evaluation set
# depends on, so raising or lowering TARGET_PER_TYPE trims the largest PDFs
# first and keeps CPU embedding time predictable. Two very large volumes
# (SP 800-53r5 at 5.9 MB, SP 800-82r2 at 4.2 MB) were deliberately left out
# for the same reason.
NIST_PDFS = [
    ("NIST_SP-800-61r2", "SpecialPublications/NIST.SP.800-61r2", "Computer Security Incident Handling Guide"),
    ("NIST_CSF-2.0", "CSWP/NIST.CSWP.29", "Cybersecurity Framework 2.0"),
    ("NIST_SP-800-150", "SpecialPublications/NIST.SP.800-150", "Guide to Cyber Threat Information Sharing"),
    ("NIST_SP-800-181r1", "SpecialPublications/NIST.SP.800-181r1", "Workforce Framework for Cybersecurity"),
    ("NIST_SP-800-40r4", "SpecialPublications/NIST.SP.800-40r4", "Guide to Enterprise Patch Management"),
    ("NIST_SP-800-46r2", "SpecialPublications/NIST.SP.800-46r2", "Guide to Enterprise Telework Security"),
    ("NIST_SP-800-124r1", "SpecialPublications/NIST.SP.800-124r1", "Managing the Security of Mobile Devices"),
    ("NIST_SP-800-88r1", "SpecialPublications/NIST.SP.800-88r1", "Guidelines for Media Sanitization"),
    ("NIST_SP-800-218", "SpecialPublications/NIST.SP.800-218", "Secure Software Development Framework"),
    ("NIST_SP-800-184", "SpecialPublications/NIST.SP.800-184", "Guide for Cybersecurity Event Recovery"),
    ("NIST_SP-800-209", "SpecialPublications/NIST.SP.800-209", "Security Guidelines for Storage Infrastructure"),
    ("NIST_CSWP-04162018", "CSWP/NIST.CSWP.04162018", "Framework for Improving Critical Infrastructure Cybersecurity"),
    ("NIST_SP-800-210", "SpecialPublications/NIST.SP.800-210", "General Access Control Guidance for Cloud Systems"),
    ("NIST_SP-800-172", "SpecialPublications/NIST.SP.800-172", "Enhanced Security Requirements for CUI"),
    ("NIST_SP-800-128", "SpecialPublications/NIST.SP.800-128", "Security-Focused Configuration Management"),
    ("NIST_SP-800-171r2", "SpecialPublications/NIST.SP.800-171r2", "Protecting Controlled Unclassified Information"),
    ("NIST_SP-800-63-3", "SpecialPublications/NIST.SP.800-63-3", "Digital Identity Guidelines"),
    ("NIST_SP-800-66r2", "SpecialPublications/NIST.SP.800-66r2", "Implementing the HIPAA Security Rule"),
    ("NIST_SP-800-37r2", "SpecialPublications/NIST.SP.800-37r2", "Risk Management Framework for Information Systems"),
    ("NIST_SP-800-213", "SpecialPublications/NIST.SP.800-213", "IoT Device Cybersecurity Guidance"),
]

# AESLC (Annotated Enron Subject Line Corpus): individual plain-text business
# emails. Each file holds the body, then "@subject", then human-written
# alternative subject lines ("@ann0"...) that belong to that dataset's
# summarization task rather than the email itself -- those are stripped at
# ingestion time.
AESLC_LISTING = "https://api.github.com/repos/ryanzhumich/AESLC/contents/enron_subject_line/dev"
EMAIL_SAMPLE_SIZE = 25
EMAIL_SAMPLE_SEED = 42


def sec_get(url: str, attempts: int = MAX_ATTEMPTS) -> requests.Response:
    """GET an EDGAR URL with the required User-Agent, retrying transient errors.

    EDGAR intermittently returns 503 under sustained crawling. Fetching a
    ~100-document corpus makes hundreds of requests, so a single transient
    failure must not abort the run: this retries with exponential backoff
    and only gives up after ``attempts`` tries.

    Args:
        url: Full EDGAR API or Archives URL.
        attempts: Maximum tries before giving up.

    Returns:
        The successful response.

    Raises:
        requests.RequestException: If every attempt failed.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=45)
            response.raise_for_status()
            time.sleep(REQUEST_DELAY_SECONDS)
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(BACKOFF_SECONDS * (2**attempt))
    raise last_error  # type: ignore[misc]


def find_recent_filings(submissions: dict, form_type: str, limit: int = 1) -> list[dict]:
    """Find the most recent filings of a given form type, newest first.

    EDGAR returns the recent-filings index as parallel arrays in reverse
    chronological order, so matches come out newest-first naturally.

    Args:
        submissions: Parsed ``data.sec.gov/submissions/CIK*.json`` payload.
        form_type: Form to look for, e.g. ``"10-K"`` or ``"8-K"``.
        limit: Maximum filings to return.

    Returns:
        Up to ``limit`` dicts with ``filingDate``, ``accessionNumber`` and
        ``primaryDocument``; empty if the company filed none of that type.
    """
    recent = submissions["filings"]["recent"]
    matches = []
    for i, form in enumerate(recent["form"]):
        if form == form_type:
            matches.append(
                {
                    "filingDate": recent["filingDate"][i],
                    "accessionNumber": recent["accessionNumber"][i],
                    "primaryDocument": recent["primaryDocument"][i],
                }
            )
            if len(matches) >= limit:
                break
    return matches


def find_exhibits(base_url: str, accession: str) -> list[tuple[str, str, str]]:
    """List a filing's contract/policy exhibits using EDGAR's Type column.

    The filing's index page carries an authoritative document-type table.
    Reading that is far more reliable than pattern-matching filenames,
    which differ per filer.

    Args:
        base_url: ``https://www.sec.gov/Archives/edgar/data/<cik>``.
        accession: Accession number with dashes removed.

    Returns:
        ``(exhibit_type, doc_type, filename)`` triples for exhibits whose
        type is in ``EXHIBIT_TYPES`` -- e.g. ``("EX-10.9", "contract", ...)``.
    """
    dashed = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
    page = sec_get(f"{base_url}/{accession}/{dashed}-index.htm")
    soup = BeautifulSoup(page.text, "lxml")

    exhibits = []
    for row in soup.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in row.find_all("td")]
        if len(cells) < 4:
            continue
        document, exhibit_type = cells[2], cells[3]
        match = EXHIBIT_TYPE_RE.match(exhibit_type)
        if not match:
            continue
        # The Document cell can carry a trailing "iXBRL" marker.
        filename = document.replace("iXBRL", "").strip()
        if filename.lower().endswith((".htm", ".html", ".txt")):
            exhibits.append((exhibit_type, EXHIBIT_TYPES[match.group(1)], filename))
    return exhibits


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
    entries: list[dict] = []
    seen_documents: set[str] = set()
    for company in COMPANIES:
        print(f"Fetching submissions index for {company['name']} ({company['ticker']})...")
        try:
            submissions = sec_get(
                f"https://data.sec.gov/submissions/CIK{company['cik']}.json"
            ).json()
        except requests.RequestException as exc:
            print(f"  [SKIP] {company['ticker']} submissions index unavailable: {exc}")
            continue

        base = f"https://www.sec.gov/Archives/edgar/data/{int(company['cik'])}"
        common = {"company": company["name"], "ticker": company["ticker"], "cik": company["cik"]}

        for form_type in FORM_TYPES:
            filings = find_recent_filings(
                submissions, form_type, limit=EXHIBIT_SCAN_DEPTH[form_type]
            )
            if not filings:
                print(f"  [SKIP] no {form_type} found for {company['ticker']}")
                continue

            for position, filing in enumerate(filings):
                accession = filing["accessionNumber"].replace("-", "")

                # Only the newest filing of each form becomes a "report" --
                # older ones are scanned purely to harvest exhibits, so
                # report coverage stays spread across companies rather than
                # being dominated by one company's filing history.
                if position == 0:
                    source_url = f"{base}/{accession}/{filing['primaryDocument']}"
                    extension = Path(filing["primaryDocument"]).suffix or ".html"
                    local_path = (
                        DATA_RAW_DIR
                        / f"{company['ticker']}_{form_type}_{filing['filingDate']}{extension}"
                    )
                    try:
                        print(f"  {form_type} filed {filing['filingDate']} -> {local_path.name}")
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
                    except requests.RequestException as exc:
                        print(f"  [SKIP] {company['ticker']} {form_type}: {exc}")

                try:
                    exhibits = find_exhibits(base, accession)
                except requests.RequestException as exc:
                    print(f"    [SKIP] index for {accession}: {exc}")
                    continue

                for exhibit_type, doc_type, filename in exhibits:
                    # Filings are scanned newest-first, and an exhibit name
                    # can repeat across years (a policy re-filed annually).
                    # Keeping the first occurrence therefore keeps the most
                    # recent version, and keeps document ids stable at
                    # "<TICKER>_<filename>" so the evaluation set's gold
                    # references survive a corpus refresh.
                    local_path = DATA_RAW_DIR / f"{company['ticker']}_{filename}"
                    if local_path.stem in seen_documents:
                        continue
                    seen_documents.add(local_path.stem)

                    exhibit_url = f"{base}/{accession}/{filename}"
                    try:
                        content = sec_get(exhibit_url).content
                    except requests.RequestException as exc:
                        print(f"    [SKIP] {exhibit_type} {filename}: {exc}")
                        continue
                    print(f"    {exhibit_type} ({doc_type}) {filing['filingDate']}: {filename}")
                    entries.append(
                        save_document(
                            content,
                            local_path,
                            **common,
                            form=exhibit_type,
                            doc_type=doc_type,
                            filing_date=filing["filingDate"],
                            source_url=exhibit_url,
                        )
                    )
    return entries


def fetch_nist_pdfs(limit: int = TARGET_PER_TYPE) -> list[dict]:
    """Fetch public-domain NIST security publications (the PDF documents).

    Args:
        limit: How many publications to take from ``NIST_PDFS``, in listed
            order (smallest-first after the two the eval set depends on).

    Returns:
        Manifest entries for each NIST PDF that downloaded successfully.
        A publication that 404s is skipped with a warning rather than
        aborting the whole corpus fetch.
    """
    entries = []
    for slug, path, title in NIST_PDFS[:limit]:
        url = f"https://nvlpubs.nist.gov/nistpubs/{path}.pdf"
        try:
            response = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=90)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"  [SKIP] {slug}: {exc}")
            continue

        print(f"  {slug}.pdf ({len(response.content) // 1024} KB)")
        entries.append(
            save_document(
                response.content,
                DATA_RAW_DIR / f"{slug}.pdf",
                company="NIST",
                ticker="NIST",
                form=slug,
                doc_type="standard",
                filing_date="",
                source_url=url,
                title=title,
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


def cap_per_type(entries: list[dict], limit: int = TARGET_PER_TYPE) -> list[dict]:
    """Keep at most ``limit`` documents of each doc_type, in fetch order.

    Fetch order matters: the companies and publications the evaluation set
    is authored against are listed first, so capping never drops a document
    a gold answer depends on.

    Args:
        entries: All fetched manifest entries.
        limit: Maximum documents to retain per doc_type.

    Returns:
        The retained entries, original order preserved.
    """
    kept: list[dict] = []
    counts: dict[str, int] = {}
    for entry in entries:
        doc_type = entry["doc_type"]
        if counts.get(doc_type, 0) >= limit:
            continue
        counts[doc_type] = counts.get(doc_type, 0) + 1
        kept.append(entry)
    return kept


def main() -> None:
    """Fetch every document family and write the combined manifest."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetched = fetch_sec_documents() + fetch_nist_pdfs() + fetch_emails()
    manifest = cap_per_type(fetched)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    by_type: dict[str, int] = {}
    by_format: dict[str, int] = {}
    for entry in manifest:
        by_type[entry["doc_type"]] = by_type.get(entry["doc_type"], 0) + 1
        by_format[entry["format"]] = by_format.get(entry["format"], 0) + 1
    print(f"\nFetched {len(fetched)} documents, retained {len(manifest)} after per-type cap")
    print(f"  by type:   {by_type}")
    print(f"  by format: {by_format}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
