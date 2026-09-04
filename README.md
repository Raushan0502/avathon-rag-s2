# Avathon AI Intelligence Challenge — Track D (RAG) × Scenario S2

## Assignment brief

- **Event:** Avathon AI/ML Hiring Challenge — Technical Assessment
- **Chosen track:** D — RAG / LLM Knowledge Systems
- **Chosen scenario:** S2 — Gen AI for Enterprise Documents
- **Problem to solve:** A company drowning in unstructured content (reports,
  policies, filings) needs AI to extract, classify, summarize, or answer
  questions over that content.
- **What this repo must deliver** (per the assignment's Track D requirements):
  1. A curated knowledge corpus of real enterprise documents, sourced and
     documented (not provided by the assignment — sourced here).
  2. A full ingestion pipeline: document loading → chunking → embedding →
     indexing, with the chunking strategy explicitly justified.
  3. A vector store (choice justified) plus **hybrid retrieval**: dense
     vector similarity + sparse keyword search (BM25), combined with a
     fusion/re-ranking step, compared against dense-only retrieval.
  4. A held-out evaluation set of **≥ 20 question-answer pairs**, scored on
     Precision@k, Recall@k, and faithfulness (is the answer supported by the
     retrieved context?).
  5. End-to-end Q&A demonstrated on ≥ 5 representative queries, each showing
     the retrieved context, the generated answer, and a faithfulness
     annotation.
  6. A written justification addressing, explicitly: chunking strategy
     choice and what's lost at chunk boundaries; embedding model choice;
     how re-ranking/fusion improves over a single retriever, shown
     quantitatively; and the hallucination-prevention/detection strategy.
- **Data policy:** no dataset is provided by Avathon. Data must be sourced
  or synthesized by the candidate, with the source documented and the
  choice justified (see [Step 2](#step-2--data-acquisition) once complete).
- **Evaluation emphasis:** reasoning and alternative-analysis are weighted
  higher than raw accuracy (Algorithm Selection & Alternative Analysis is
  30% of the score — the single highest-weighted rubric dimension).
- **Deliverables required for submission:** this GitHub repo, a 5-minute
  walkthrough video, and a 1–2 page technical write-up (PDF). The write-up
  and video are tracked outside this repo and linked here once produced.

## Data source (Step 2)

The corpus is real, public SEC EDGAR filings — not synthetic — pulled
directly from the SEC EDGAR submissions API
(`https://data.sec.gov/submissions/CIK##########.json`), which requires no
API key, only a descriptive `User-Agent` header (SEC's fair-use policy).

**Companies:** 12 large issuers spanning consumer hardware, enterprise
software, autos, retail, banking, pharma and networking (AAPL, MSFT, TSLA,
AMZN, GOOGL, META, NVDA, JPM, WMT, JNJ, KO, CSCO), so evaluation questions
span genuinely different business content rather than near-duplicate
filings. Apple, Microsoft and Tesla are listed first deliberately — the
evaluation set is authored against their filings, and the per-type caps
retain documents in fetch order, so capping can never drop a document a
gold answer depends on.

S2 describes a company drowning in *"contracts, reports, SOPs, emails"*,
so the corpus deliberately spans all four types and **three file formats**
rather than one homogeneous source. **100 documents, balanced 20 per type:**

| Type | Documents | Format | Source |
|---|---|---|---|
| Reports | 20 — latest 10-K + 8-K across 10 issuers | HTML | SEC EDGAR |
| Contracts | 20 — EX-10 material-contract exhibits | HTML | SEC EDGAR exhibits |
| Policies | 20 — EX-19 insider trading, EX-97 clawback | HTML | SEC EDGAR exhibits |
| Standards / SOP | 20 — NIST SP 800 series + CSF 2.0 | **PDF** | NIST (public domain) |
| Emails | 20 — AESLC business emails | **plain text** | AESLC (Enron) corpus |

**Balance here is by document count, not retrieval weight.** A 10-K is
~250 chunks and an email is 1, so reports still dominate chunk share
even at 20 documents each. That is stated rather than implied away by the
even counts: corpus *balance* matters far less in RAG than it does in
supervised learning, because nothing is trained on this distribution — the
embedding model is frozen and retrieval is query-conditional, so distant
chunks never compete. The one place composition genuinely feeds the
algorithm is **BM25**, whose IDF is computed corpus-wide: adding 20 NIST
security publications measurably reduces how discriminative terms like
"incident" and "controls" are. The reason for the diversity is therefore
evidence, not statistics — demonstrating that multi-format loading and all
four S2 document types actually work.

Exhibits are discovered from **EDGAR's authoritative document Type column**
(`EX-10.9`, `EX-19.1`), not from filenames. Filename matching was tried
first and silently under-matched: Apple names exhibits
`a10-kexhibit4109272025.htm`, which no `ex10`-style pattern catches, and
that alone was why an early run found 2 contracts instead of 20. The type
regex is anchored so `EX-101.SCH` — an XBRL taxonomy file — cannot
masquerade as `EX-10`. Because material contracts are filed only
occasionally, exhibits are harvested from a deeper filing history (last 3
10-Ks, 8 8-Ks per issuer) while only the newest filing of each form
becomes a *report*, keeping report coverage spread across issuers.

Fetching is resilient by necessity: EDGAR returns transient 503s under
sustained crawling, and an early run died mid-corpus because one flaky
request aborted everything. `sec_get` now retries with exponential backoff,
and a document that still fails is skipped with a warning rather than
taking the run down.

- **Reports** — 10-Ks are long (2–8 MB) and highly structured; 8-Ks are
  short and announcement-like, so chunking and retrieval behave
  differently across the two.
- **Contracts** — real Reg S-K item 601(b)(10) material-contract exhibits
  (equity award, RSU and similar agreements), discovered dynamically as
  described above. Issuers that file none — Apple incorporates most of its
  by reference — simply contribute none rather than failing the run.
- **Policies** — insider-trading and compensation-recovery policies. These
  are genuine internal governance documents, the closest public analogue
  to an enterprise SOP.
- **Standards** — NIST security publications, chosen because they pair
  directly with the filings' Item 1C Cybersecurity disclosures, making
  genuine cross-document questions possible ("how does this company's
  incident response compare to the NIST phases?"). They are also the
  corpus's PDF-format documents.
- **Emails** — real business correspondence in plain text, a register and
  format completely unlike the filings.

Every document is fetched by
[`scripts/fetch_corpus.py`](scripts/fetch_corpus.py) and logged in
[`data/raw/manifest.json`](data/raw/manifest.json) with its source URL,
document type, format, and SHA-256 checksum, so the corpus is fully
reproducible and auditable. Everything is keyless and free. The email
sample is drawn with a fixed seed, so re-running reproduces the same
emails and the evaluation stays stable. Raw files are gitignored (large,
trivially re-fetchable); the manifest is committed as the record.

**Limitations acknowledged:**
- The **emails are a documented domain gap.** No public email archive
  exists for Apple, Microsoft or Tesla, so the corpus uses the canonical
  public business-email corpus (Enron, via AESLC). The emails are real
  correspondence, but from a different company and era than the filings —
  they add format and register diversity, not thematic continuity.
- AESLC files carry `@subject` / `@ann*` trailers belonging to that
  dataset's *subject-line summarization* task. The `@subject` line is
  promoted into the text; the `@ann*` lines are human-written alternative
  subject lines and are **dropped**, since indexing them would attribute
  text to the sender that they never wrote.

## Ingestion & chunking (Step 3)

Pipeline: **extract → normalise → validate → section-split → chunk → embed → index.**

```
fetch_corpus.py ─→ data/raw/ ─→ validate_corpus.py ──(gate, exits non-zero)
                                        │
                     build_index.py ─→ ingest.py ─→ embed_index.py ─→ data/processed/
                        load_document_text → normalise_text → split_into_sections → chunk_words
```

### Extraction — dispatches on file type

- **HTML** — BeautifulSoup, dropping script/style. `<table>` elements are
  rendered to Markdown **in place** before surrounding text is flattened.
- **PDF** — tables are located first and extracted structurally, then the
  page's prose is read with those regions *masked out*, so table content is
  never emitted twice in two shapes. The NIST PDFs carry a real text layer
  so no OCR is needed.
- **Plain text** — read as-is, minus the AESLC trailer handling above.

**Tables keep their structure.** Flattening destroyed the thing that makes
a financial figure answerable: three fiscal years collapsed into an
unlabelled number sequence. Tables now render as Markdown pipe rows, which
took three fixes to work on real filings — `colspan` expansion (without it
rows are ragged and figures drift out from under their headers), row-local
`$`/`%` merging (filings put currency marks in their own cells, and only on
*some* rows, so a column-wide rule fails and leaves the grid skewed), and
dropping all-empty spacer columns. Layout tables — anything under 2×2 — are
emitted as plain lines rather than dressed up as data.

```
before:  Net sales: Products $ 307,003 $ 294,866 $ 298,085 Total net sales 416,161 391,035 383,285
after:   | Total net sales | $416,161 | 6% | $391,035 | 2% | $383,285 |
```

**Images are not extracted, and this is a known limitation.** HTML `<img>`
elements are discarded including their `alt` text, and PDF extraction reads
only the text layer, so charts, figures and scanned pages contribute
nothing. **Supporting them requires an OCR stage** (Tesseract, or a
document-AI service) that this pipeline deliberately does not have. The
consequence is that a scanned document yields little or no text *without
raising* — which is precisely what the validation gate below exists to
catch.

### Preprocessing (`normalise_text`)

Extraction output is not yet fit to embed. Five ordered passes, each
targeting a defect measured in this corpus — order matters, since unicode
is normalised before any pattern matching and de-hyphenation runs before
repeated-line detection:

1. **Unicode NFKC + punctuation mapping** — smart quotes, the full dash
   range, NBSP and zero-width marks, so the same word embeds identically
   whichever filer produced it.
2. **De-hyphenation** — PDF text layers break words at line ends
   (`informa-\ntion`); left alone one word embeds as two fragments.
3. **Dot-leader / page-number stripping** — contents-page filler.
4. **Repeated page-furniture removal** — running headers and footers.
5. **Whitespace collapse.**

Rendered table rows are passed through untouched, so preprocessing does not
dismantle what extraction just recovered.

Measured on real documents: dot-leaders **80 → 0**, page footers
**58 → 0**, bare page numbers **101 → 0**, with 620 table rows preserved.

Two subtleties worth naming, both found by tests rather than reasoning:
NFKC rewrites U+2011 into U+2010 *before* the punctuation map runs, so
mapping only the characters visible in the source silently missed
non-breaking hyphens. And running footers are never twice the same string
because they carry the page number — they need **digit-masked** comparison
to detect. That fix initially deleted *everything*, because body sentences
differing only by a year also collapse to one template; a word-count guard
(furniture is label-shaped, prose is not) is what separates them.

### Validation gate (`scripts/validate_corpus.py`)

Extraction fails silently: a scanned PDF returns an empty string, not an
error. At 6 documents a human can read every one; at 100 that stops being
true, so five metrics stand in for reading them — `yield_ratio` (chars per
source byte, the check that catches a missing text layer), `alpha_ratio`
(punctuation-heavy leftovers), `boilerplate_ratio`, `table_row_ratio`
(did tables actually survive), and `mean_words_per_line` (fragmented
multi-column extraction). It writes a per-document report and **exits
non-zero**, so it can gate a rebuild rather than being advisory.

Current status: **100/100 documents pass** — mean yield 0.374, alpha 0.799,
boilerplate 0.088, table rows 0.115.

Its first run over the full corpus reported 16 failures that were **all
false positives of the gate's own making** — healthy 178–472 character
emails failing a 500-character floor calibrated for filings. Thresholds are
now per document type. A gate that cries wolf on 16% of a corpus gets
ignored, so that calibration mattered more than the gate itself.

### Chunking strategy

Two stages: split each filing on its numbered "Item" section headers (so a
chunk never blends, say, Risk Factors with Legal Proceedings), then split
**within** each section using a token budget that preserves structure.

Stage 2 originally slid a fixed 220-word / 40-word-overlap window. That was
replaced once tables and PDFs entered the corpus, because it had four
defects — each demonstrated on real data before being fixed:

| Defect | Consequence | Fix |
|---|---|---|
| Windows joined with spaces, **discarding newlines** | The Markdown tables recovered during extraction were flattened straight back into one line | Chunk on block boundaries, preserving newlines |
| A split table's later parts had **no header row** | `\| Line item 15 \| 1500 \|` gives no clue which column is which year — the exact problem rendering tables was meant to solve | `chunk_table` repeats the header in every part and splits only on row boundaries |
| Windows began **mid-sentence** | Weaker embedding, and poor reading as cited context | `chunk_prose` packs whole sentences, carrying whole-sentence overlap |
| Sizing in **words**, not tokens | The model truncates at 512 tokens *silently*; the words→tokens ratio is ~1.3 for prose but far higher for numeric tables | `count_tokens` measures with the model's own tokenizer |

Measured on Apple's 10-K after the change: **0 chunks exceed the 512-token
limit** (max 400, the configured budget), **all 61 table chunks carry a
header row**, and only 3 of 123 prose chunks start mid-sentence. The
tokenizer is loaded once and cached, and falls back to a conservative
word-based estimate if unavailable, so ingest still runs offline.

**Contextual enrichment.** A chunk is embedded in isolation, having lost
every clue about where it came from — and filings are written from common
templates, so "Competition is intense and margins are under pressure"
reads identically across a dozen issuers. Each chunk is therefore embedded
with a provenance prefix, `Apple Inc. | 10-K | Item 1A. Risk Factors`,
which puts the issuer, form and section into the vector itself
(`build_embed_text`). Only the *embedding input* carries the prefix: the
stored `text` stays clean, because that is what the model quotes and what
a reader sees as a citation, and the prompt already states provenance
separately. The placeholder `"Full Document"` section is omitted rather
than embedded, since it carries no meaning and would only dilute the
vector.

Section detection is a regex heuristic, not a
schema-aware parser, with two known failure modes that are corrected for
explicitly in code (see `_dedupe_header_matches`): table-of-contents
entries matching the same pattern as real headers, and long sections
whose printed-page running headers (e.g. "Item 7" reprinted at the top of
every MD&A page) get flattened by HTML→text conversion into what look
like dozens of spurious new headers. Both are collapsed back to one
boundary per real section. Residual limitation, left as-is and
acknowledged rather than further chased: a couple of MD&A subsection
labels end up named after a mid-page running-header artifact rather than
the section's true title — the chunk *boundaries* are still correct,
only the section *label* metadata is occasionally imprecise.

**Non-SEC documents are split on their own structure.** Only filings have
"Item" headers, so NIST publications, contracts and policies used to fall
through to a single `"Full Document"` section — and that had a measurable
cost, not just an aesthetic one. An 80-page NIST PDF became one 177-chunk
section, and since relevance is judged at `(doc_id, section)` granularity,
*any* chunk retrieved from that file counted as relevant: those questions
scored a perfect Precision@5 of 1.000 purely because the section was the
whole document.

`split_on_numbered_headings` now picks up the structure these documents do
have — `1.1 Authority` (NIST), `1. PURPOSE` (policies), `Section 5.` /
`Appendix A.` (contracts) — with the same conservative fallback as the Item
split: too few headings means one section, because over-splitting on false
positives is worse than under-splitting.

Result: the NIST incident-handling guide goes from **1 section to 56**, and
its largest section from **177 chunks to 30**. Microsoft's insider-trading
policy goes from 1 to 13, correctly named (`1. PURPOSE`, `2. SUMMARY`,
`3. REQUIREMENTS`). Emails correctly stay whole.

Two rounds of false positives were needed to get there: a cover page set in
spaced capitals matched as heading `C` followed by `O M P U T E R ...`
(fixed by requiring a bare capital to carry its period), and one wrapped
address line still matches. That last one is left as-is and noted rather
than chased — one artifact in 56 sections is not worth another regex
epicycle.

Result on the current corpus: **8,146 chunks across 100 documents.**

A built-in smoke test in `build_index.py` embeds 3 hand-written queries
and prints top-3 FAISS results after every rebuild — verified on this
corpus to return correctly on-topic chunks (competition risk → Risk
Factors sections; legal question → Legal Proceedings section, etc.)
before Step 4 (retrieval quality evaluation proper) begins.

**Embedding model:** `BAAI/bge-small-en-v1.5` (384-dim, CPU, 512-token
context) — chosen over a hosted embeddings API so ingestion has no cost
or external dependency; over larger local models for CPU-only latency on
this corpus size. Full trade-off discussion in `src/embed_index.py`.

**Vector store:** FAISS `IndexFlatIP` (exact cosine similarity, via
L2-normalized vectors) — exact search is sub-millisecond at this corpus
scale (hundreds of chunks), so an approximate index (IVF/HNSW) would only
add tuning surface with no measurable benefit here; where that trade-off
flips is discussed in the write-up.

`data/processed/index.faiss` and `data/processed/chunks.jsonl` are
gitignored and regenerated by `python scripts/build_index.py` — this
keeps the repo free of large re-derivable binaries while staying fully
reproducible from the pinned `requirements.txt`.

## Hybrid retrieval (Step 4)

`src/retrieval.py` adds BM25 sparse search alongside the FAISS dense
search from Step 3, combined via Reciprocal Rank Fusion (RRF, k=60) — a
parameter-light, rank-based fusion that needs no extra model, chosen over
a cross-encoder re-ranker because this corpus is small and single-genre,
so a second per-candidate inference pass is unlikely to be worth its
cost (full trade-off in the module docstring). `RetrievalIndex.search(query,
mode="dense"|"bm25"|"hybrid")` is the shared interface Step 5 (generation)
and Step 6 (evaluation) both call.

`scripts/compare_retrieval.py` is a qualitative smoke test showing that
BM25 genuinely reorders results rather than duplicating dense retrieval.
The quantitative comparison is in Step 6 below — and it does **not**
favour hybrid.

**Hybrid does not beat dense on this corpus, and that is the finding.**
On an earlier 39-document corpus hybrid won on recall (0.917 vs 0.833),
which is what this section originally claimed. Growing the corpus to 100
documents reversed it: dense now leads on every metric, and hybrid sits
between dense and BM25 — it is being *dragged down* by its weaker
component rather than lifted by a complementary one.

The mechanism is BM25's own statistics. **IDF is computed corpus-wide**,
so adding 20 NIST security publications and nine more issuers made terms
like "incident", "controls", "risk" and "net sales" far less
discriminative. BM25's MRR fell from competitive to 0.554 while dense
held at 0.762, and RRF — which weights the two retrievers equally by rank
position — has no way to discount the degraded one.

That is a genuinely more useful result than the original: it shows the
fusion win was **corpus-dependent, not a property of hybrid retrieval**.
A weighted fusion (or a learned re-ranker) that could down-weight BM25 as
its IDF quality degrades is the obvious next step, and is named in the
write-up as identified-but-not-implemented rather than quietly dropped.

## Generation (Step 5)

`src/generation.py` builds a grounded-answer prompt from the top-k hybrid
retrieval results (each source numbered, e.g. `[1]`, with its
ticker/form/section shown), calls an LLM, and annotates the answer for
faithfulness.

**Provider fallback chain: Groq → Mistral → Gemini.** All three are
external LLM APIs with a free tier — explicitly allowed for Track D, and
kept at zero inference cost. Groq (`openai/gpt-oss-120b`) is primary for
latency; Mistral (`mistral-small-latest`) and Gemini (`gemini-flash-latest`) are
tested, working fallbacks — though Groq answered every call in this
session's testing, so neither fallback was actually triggered end-to-end.
(Finding Gemini's working model ID took some trial and error: the
originally-planned `gemini-2.0-flash` and `gemini-2.5-flash` both now
404 as deprecated; `gemini-flash-latest` is the current alias. A live
test also hit a transient 503 "high demand" error from Google's side —
a real example, caught live, of exactly the kind of single-provider
outage this fallback chain exists to route around.) Which provider actually served
an answer is recorded on every response (`result["provider"]`), so the
fallback is observable rather than silent. Full rationale in the module
docstring.

**Hallucination mitigation:** the system prompt requires every claim to
carry a source citation and instructs the model to say it cannot answer
rather than guess when context is insufficient.
**Hallucination detection:** `annotate_faithfulness()` checks each answer
for citation markers and for the explicit "cannot answer" refusal phrase,
flagging any answer with zero citations as `UNGROUNDED`. This caught a
real bug during testing, not a hypothetical one: two answers were
initially flagged `UNGROUNDED` because Groq's model emitted full-width
citation brackets (`【4】`) instead of the ASCII `[4]` the prompt
requested — the citation *was* there, the detector's regex just didn't
recognize that bracket style. Fixed by widening the regex to accept both
forms; the original false-positive-flagged answers are what surfaced the
gap. This is exactly the kind of silent-failure mode Track D's write-up
questions ask about, and it's called out here rather than quietly patched
without a record.

`scripts/demo_qa.py` runs 6 representative queries (one deliberately
probing a topic — dividend policy — not obviously covered by the risk/
legal/competition-heavy sections retrieved earlier, to see the pipeline
handle a different retrieval path) end-to-end and saves the full trace
(retrieved context + generated answer + faithfulness annotation per
query) to `results/qa_demo.json`. All 6 came back correctly grounded and
cited on the corpus.

## Evaluation (Step 6)

**Held-out eval set:** [`data/eval/qa_eval.json`](data/eval/qa_eval.json),
**45 hand-authored question/answer pairs** (the assignment asks for 20),
covering all five document types. Every question was authored *against a
specific, read, verified chunk* — not generated then checked afterwards —
so each `reference_answer` and `gold_chunk_id` is traceable to real source
text. Relevance is judged at `(gold_doc_id, gold_section)` rather than
exact `chunk_id`, since overlapping chunks from the same section are
equally valid evidence.

### Questions are tiered by k, because Precision@k measures answer-set size

The first 33 questions were all scored at k=5, which turned out to be
close to meaningless for many of them. **Precision@5 cannot exceed
`min(gold_chunks, 5) / 5`.** "Where is Apple headquartered?" is answered
by exactly one chunk in the corpus, so a *perfect* retriever scores
0.20 — the other four slots have nothing correct to hold. Eleven of the
original 33 questions had one-chunk answers, dragging mean attainable
precision to 0.65 rather than 1.0.

The fix is not to invent extra gold chunks — those passages genuinely do
not answer the question, and labelling them would inflate the metric
while measuring less. Instead each question now carries its own
**`eval_k`, matched to how many chunks really answer it**, and 12 new
questions were added spanning four tiers, each grounded in a real section
of a real document:

| Tier | Question type | Example gold section | Gold chunks |
|---|---|---|---|
| **k=1** | known-item / pinpoint | Apple `Item 9. Changes in and Disagreements with Accountants` (`"None."`) | 1 |
| **k=5** | narrow topical | Meta `Item 1C. Cybersecurity` | 4 |
| **k=10** | broader topical | Amazon `Item 1. Business` | 10 |
| **k=20** | wide topical | Meta `Item 3. Legal Proceedings` | 23 |

Results report the raw figure, the **attainable ceiling**, and
**attainment** against it, plus **MRR** — because Precision@1 is
all-or-nothing and cannot distinguish a gold at rank 2 from one at rank
500.

### Retrieval results (n=45, per-question k)

| Mode | P@k | ceiling | attainment | R@k | **MRR** |
|--------|------:|------:|------:|------:|------:|
| **Dense** | **0.349** | 0.738 | **47%** | **0.867** | **0.762** |
| BM25 | 0.218 | 0.738 | 30% | 0.733 | 0.554 |
| Hybrid (RRF) | 0.290 | 0.738 | 39% | 0.867 | 0.727 |

**Dense wins on every metric.** See Step 4 above for why hybrid lost once
the corpus grew — BM25's corpus-wide IDF degraded, and equal-weight RRF
cannot compensate.

### By tier (hybrid)

| Tier | n | P@k | ceiling | attainment | R@k | MRR |
|---|---:|---:|---:|---:|---:|---:|
| k=1 | 3 | 0.000 | 1.000 | 0% | 0.000 | 0.278 |
| k=5 | 36 | 0.328 | 0.678 | 48% | 0.917 | 0.756 |
| k=10 | 3 | 0.267 | 0.967 | 28% | 1.000 | 0.556 |
| k=20 | 3 | 0.150 | 0.967 | 16% | 1.000 | 1.000 |

Two things this exposes that a single blended number hid:

- **k=1 scores 0.000 but MRR is 0.278** — the golds sit at ranks 2–3, not
  missing. For `k1c` the rank-1 result was a *different* Apple RSU exhibit
  whose "1. General" clause is **textually identical** to the gold.
  Retrieval cannot separate them because nothing distinguishes them; that
  is a corpus property, not a retriever fault.
- **As k grows, recall reaches 1.000 while precision attainment falls to
  16%.** The system reliably *locates* the right section and then fills
  the remaining slots with neighbours — a ranking-depth weakness, not a
  findability one.

**Caveat stated plainly:** the k=1/10/20 tiers have only 3 questions each,
so those figures move in 33% steps. Only the k=5 tier (n=36) is
statistically meaningful. Widening the thin tiers is the obvious next step.

### End-to-end faithfulness (n=45)

**41 cited, 4 refused, 0 ungrounded**, every answer served by Groq.

The four refusals — q07, q21, q27, q31 — line up almost exactly with the
five retrieval misses. **When retrieval fails, the system declines rather
than fabricating**, which is the hallucination mitigation working as
designed. `results/qa_eval_results.json` holds the full per-question
trace; `results/qa_demo.json` holds six representative end-to-end Q&A
traces with retrieved context, generated answer and faithfulness
annotation.

Reaching 0 ungrounded took two real fixes to the *detector*, not the
model. Answers were initially misflagged `UNGROUNDED` because the model
cited using full-width brackets (`【4】`) and a browsing-style format
(`【1†L1-L4】`) that the regex did not recognise. Both were confirmed by
reading the raw answers — every one was correctly grounded — before
widening the pattern. Misreading the model's own citation format is
exactly the kind of silent detector gap the "how do you detect
hallucination" question is really asking about.

### "Cited" does not mean "correct"

The most important limitation. `annotate_faithfulness` verifies an answer
**is grounded in retrieved context** — it catches uncited assertions but
is blind to **mis-grounded** ones: answers that faithfully cite context
which does not actually address the question.

This was caught live on an earlier corpus. Asked where Tesla's 10-K
points readers for legal proceedings (correct answer: "Note 13,
Commitments and Contingencies"), the system returned a fluent, confidently
**cited, wrong** answer about SEC filings and investor-relations websites —
because Tesla's one-chunk Item 3 had been crowded out of the top 5.

Two consequences, stated so the numbers are not over-read:

1. **41/45 "cited" is a grounding rate, not an accuracy rate.** It should
   not be read as "91% correct".
2. **Growing a corpus can regress specific queries.** Small, terse,
   high-value sections are the first casualties as a corpus scales.

Catching mis-grounding needs answer-vs-reference comparison — an LLM judge
or entailment scoring against the gold answer — which this pipeline does
not have. Identified, not implemented.

## Tests

```bash
python -m unittest discover -s tests
```

139 unit tests across the six `src/` modules, using the standard library's
`unittest` (no extra dependency to install). They are deliberately
**offline and fast** (~0.02s total): no network calls, no API keys, and no
embedding-model download — `embed_texts` is exercised against a stub
encoder, and retrieval tests assemble a `RetrievalIndex` by hand from a
3-vector FAISS index and a real BM25 index over three short strings.

What they pin down, beyond the happy path:
- **Multi-format loading** — HTML tag stripping, the email `@subject` /
  `@ann*` trailer handling (annotation lines must never be indexed as if
  the sender wrote them), and an unsupported extension raising rather
  than silently yielding empty text.
- **Chunking** — every word survives windowing, no window exceeds the size
  limit, and overlap actually repeats the right words.
- **Section splitting** — the two real failure modes found while building
  this: table-of-contents entries must not create duplicate sections, and
  repeated running page-headers (`Item 7` reprinted on every MD&A page)
  must collapse into one section rather than one per page.
- **Faithfulness detection** — all three citation formats seen live
  (`[1]`, `【1】`, `【1†L1-L4】`) count as grounded; a citation to a source
  number that was never supplied does *not*, since that is itself a
  hallucination.
- **Retrieval** — RRF ranks a chunk found by *both* retrievers above one
  that merely tops a single list; an unknown mode raises rather than
  silently defaulting; `search()` returns copies so results can't mutate
  the loaded corpus.
- **Metrics** — the same section label in a *different* filing is not
  relevant (every 10-K has an "Item 2. Properties"), and empty retrieval
  scores 0.0 rather than dividing by zero.

## Repository structure

```
avathon-rag-s2/
├── data/
│   ├── raw/            # sourced source documents (gitignored; see Step 2)
│   └── processed/      # cleaned/chunked intermediates (gitignored)
├── src/                # pipeline modules (ingestion, retrieval, generation, evaluation)
├── scripts/            # runnable pipeline stages (see Setup)
├── tests/              # unit tests, one module per src/ module
├── results/            # evaluation outputs, metrics, comparison tables
├── write-up/           # technical write-up source (final PDF submitted separately)
├── requirements.txt
├── .env.example        # copy to .env and fill in your own LLM API key (never committed)
└── README.md
```

## Setup (Step 1)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install --index-url https://download.pytorch.org/whl/cpu torch --no-deps
pip install -r requirements.txt
copy .env.example .env          # then fill in ANTHROPIC_API_KEY or OPENAI_API_KEY
python scripts/check_setup.py   # sanity check: env + folder layout
python scripts/fetch_corpus.py  # sources 100 documents into data/raw/ (see Data source below)
python scripts/validate_corpus.py    # extraction-quality gate; exits non-zero on failure
python scripts/build_index.py   # ingest -> chunk -> embed -> FAISS index (see Ingestion below)
python scripts/compare_retrieval.py  # dense vs hybrid qualitative smoke test (see Hybrid retrieval below)
python scripts/demo_qa.py       # end-to-end Q&A demo -> results/qa_demo.json (see Generation below)
python scripts/run_evaluation.py  # full eval: retrieval P@k/R@k + faithfulness (see Evaluation below)
python -m unittest discover -s tests   # unit tests (offline, no API keys needed)
```

`torch` is installed separately first from PyTorch's CPU-only wheel index —
plain PyPI doesn't host the `+cpu` build pinned in `requirements.txt`, and
this keeps the install free of an unused multi-GB CUDA download since no
GPU is required or used by this track.

`requirements.txt` grows incrementally as each pipeline step is built and
verified — see the roadmap below.

## Build roadmap

Each step is implemented, run/tested locally, committed, and pushed to
`dev` before the next step starts. `main` only receives a merge once the
full pipeline is verified working end-to-end.

- [x] **Step 1 — Repo scaffold & environment setup**
- [x] **Step 2 — Data acquisition & corpus documentation**
- [x] **Step 3 — Ingestion pipeline** (load → chunk → embed → index)
- [x] **Step 4 — Hybrid retrieval** (dense + BM25 + fusion/re-ranking, vs. dense-only)
- [x] **Step 5 — Generation** (grounded answers + faithfulness annotation)
- [x] **Step 6 — Evaluation harness** (45 QA pairs, Precision@k/Recall@k/MRR/faithfulness)
- [x] **Step 7 — Corpus expansion to 100 docs**, multi-format extraction, preprocessing, validation gate
- [x] **Step 8 — Production chunking** (token budgets, table preservation, contextual enrichment, structure-aware sections)
- [ ] **Step 9 — Empirical embedding-model comparison** (the one algorithm choice still argued analytically rather than measured)
- [ ] **Step 10 — Write-up, video, merge to `main`**

## Notes on reproducibility

- No GPU required — embeddings run on CPU, generation calls an external
  LLM API.
- All dependency versions are pinned in `requirements.txt` as they're
  introduced.
- Random seeds are fixed wherever sampling occurs (documented per script).
