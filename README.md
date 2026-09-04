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

**Only the SEC filings have "Item" headers**, so the non-SEC documents
(NIST PDFs, contract/policy exhibits, emails) all take the
`"Full Document"` fallback and are chunked by the sliding window alone.
For emails and short exhibits that is the right answer. For an 80-page
NIST PDF it is genuinely coarse — the document's own numbered headings
could drive a finer split — and, as the Evaluation section shows, this
coarseness **distorts Precision@k** for those documents rather than being
a merely cosmetic limitation.

> **Rebuild pending.** The corpus was expanded to 100 documents and the
> extraction/preprocessing stages rewritten *after* the index was last
> built. Every chunk count, retrieval metric and evaluation figure below
> this point still describes the earlier 39-document / 1,148-chunk build.
> They are left in place rather than deleted so the before/after comparison
> survives, but they are **not** current. The index rebuild and
> re-evaluation happen together once the chunking work lands, since chunk
> ids shift and the eval set's `gold_chunk_id` references need remapping.

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

`scripts/compare_retrieval.py` is a qualitative smoke test (not the
rigorous metric) run across 5 queries: hybrid changed the top-3 result
set on every one of them (1–2 of 3 results reordered/replaced vs.
dense-only), confirming BM25 is pulling in genuinely different
candidates rather than being redundant with dense search. The actual
quantitative dense-vs-hybrid comparison (Precision@k, Recall@k) against
the held-out QA set is Step 6's job, using this same module.

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
33 hand-authored question/answer pairs (above the assignment's 20-pair
minimum), covering all five document types: 24 on reports, 3 on policies,
2 each on contracts, standards and emails. Each question was authored
*against a specific, read, verified chunk* — not generated then checked
after the fact — so every `reference_answer` and `gold_chunk_id` is
traceable to real source text. Ground truth is recorded at
`(gold_doc_id, gold_section)` granularity rather than exact `chunk_id`,
since adjacent overlapping chunks from the same section are equally
valid evidence for a question (see `src/evaluate.py` docstring for the
full reasoning, including why Recall@k reduces to a binary hit-rate here:
each question has exactly one known-relevant section, not an exhaustive
relevance-judged set).

**Retrieval comparison, k=5, n=33** (`scripts/run_evaluation.py` →
`results/retrieval_eval.json`):

| Mode   | Mean Precision@5 | Mean Recall@5 |
|--------|------------------:|---------------:|
| Dense  | 0.455             | 0.879           |
| BM25   | 0.412             | 0.879           |
| **Hybrid (RRF)** | **0.455** | **0.909** |

**Hybrid wins on recall (0.909) over both single retrievers (0.879 each)
at equal precision to dense.** That is the quantitative answer to the
Step 4 question ("show hybrid vs. dense-only quantitatively"): fusion
surfaces relevant sections that *either retriever alone missed*, without
diluting precision. Recall is also the metric to trust here — see below.

### Precision@k is not comparable across document types here

Aggregate precision rose from 0.400 (24 questions, filings only) to 0.455
after the corpus was widened. **That is a metric artifact, not an
improvement**, and it is worth stating plainly rather than banking as a
win. Breaking hybrid precision down by document type against the size of
each question's gold section:

| doc_type | n | mean P@5 | median gold-section size |
|---|---:|---:|---:|
| standard (NIST PDF) | 2 | **1.000** | **177 chunks** |
| contract | 2 | 0.600 | 20 |
| policy | 3 | 0.600 | 7 |
| report | 24 | 0.400 | 13 |
| email | 2 | **0.200** | **1 chunk** |

The correlation is the whole story. The NIST PDFs score a perfect 1.000
because each is a single 177-chunk `"Full Document"` section, so *any*
chunk retrieved from that file counts as relevant — precision measures
"did we land in the right file", not "did we find the right passage".
Emails sit at the opposite extreme: their gold section is one chunk, so
Precision@5 is **capped at 0.200 by construction**, and scoring exactly
0.200 actually means the right email was retrieved every single time.

So Precision@k here is a function of section granularity as much as
retrieval quality, and the aggregate is only meaningful when compared
*within* a document type. Recall@k, being a binary hit-rate, is immune to
this and is the sounder basis for the dense/BM25/hybrid comparison above.
Fixing this properly means finer section splitting for non-SEC documents
(the NIST publications have their own numbered headings) — identified,
not implemented, within the time budget.

### "Cited" does not mean "correct" — a concrete failure

The most important finding of the whole build, surfaced only because the
corpus grew. **q14** ("Where does Tesla's 10-K direct readers for details
on its material pending legal proceedings?") is answered, confidently and
*with a citation*, from the wrong place:

> Tesla's 10-K tells readers that details … are provided in its periodic
> SEC filings — Form 10-K, 10-Q, 8-K and proxy statements — accessible
> through the SEC's website and Tesla's investor-relations site **[2]**

The correct answer is "Note 13, Commitments and Contingencies". Tesla's
Item 3 is a single terse chunk that merely cross-references that note;
when the corpus grew 42%, that tiny section was crowded out of the top 5,
and retrieval returned Item 8 / Item 1 chunks instead. The model then did
exactly what it was told — grounded its answer in the retrieved context
and cited it — and produced a fluent, cited, **wrong** answer.

This exposes the real limit of the hallucination-detection strategy:
`annotate_faithfulness` verifies that an answer *is grounded in retrieved
context*, which catches uncited assertions but is blind to
**mis-grounded** ones — answers faithfully citing context that does not
actually address the question. Catching those needs answer-vs-reference
comparison (an LLM judge, or entailment scoring against the gold answer),
which this pipeline does not have. Two honest consequences:
1. The 30/33 "cited" figure below is a **grounding** rate, not an accuracy
   rate. It should not be read as "91% correct".
2. Growing a corpus can *regress* specific queries. q14 passed before the
   expansion and fails after it. Small, terse, high-value sections are
   the first casualties as a corpus scales — an argument for section-aware
   boosting or a higher k, both untested here.

**End-to-end grounding** (hybrid retrieval → generation, all 33
questions, `results/qa_eval_results.json`): **30 cited, 3 refused, 0
ungrounded** — read as a grounding rate, with the q14 caveat above.
Getting to 0 ungrounded took a second real fix, not just
the Step 5 one: 4 of the answers were initially misflagged
`UNGROUNDED` because the model cited using a browsing-style format
(`【1†L1-L4】` — source number + dagger + line range) that the Step 5
regex didn't recognize, on top of the plain `[1]` and `【1】` forms it
already handled. Confirmed by inspecting the raw answer text (all 4 were
in fact correctly grounded and cited), then widened the regex in
`src/generation.py` to accept any non-bracket suffix between the source
number and the closing bracket. Documented in the module rather than
quietly patched, since misreading the model's own citation format is
exactly the kind of silent detector gap the "how do you detect
hallucination" write-up question is really asking about.

The **3 refusals are legitimate, not bugs** — q08 / q16 ("Does
\[company\] report unresolved staff comments?"), whose gold answer is the
single word "None", plus q21 below. Such a terse, low-signal section is
hard for both dense and sparse retrieval to rank highly against a
full-sentence question; it didn't make the top 5, and the model correctly
refused rather than guess. Note this is the *same* root cause as q14 —
tiny sections lose to larger ones — but with the safe outcome (refusal)
rather than the dangerous one (a confident wrong answer).

**One borderline case worth naming: q21 (MSFT total revenue).** Across
repeated runs this question flips between answering-with-citation and
refusing, even though retrieval is deterministic and hits the right
section (`Item 8`) every time. The cause is that retrieval hits the right
*section* but not reliably the right *sub-table*: MSFT's Item 8 is one
giant section (a consequence of the Step 3 chunking limitation — running
header artifacts collapse many distinct financial statements under one
label) holding dozens of unrelated tables, so at k=5 the specific
revenue-breakdown chunk sits right at the edge of the retrieved window.
Whether the model can answer therefore depends on borderline context, and
its refusal threshold is not perfectly stable run to run. Both outcomes
are acceptable behaviour — it either cites correctly or declines, and in
no run does it fabricate a revenue figure — but it is an honest caveat
that the headline faithfulness split moves by one question between runs,
and a concrete illustration of how a coarse section boundary upstream
propagates into answer-level instability downstream.

## Tests

```bash
python -m unittest discover -s tests
```

115 unit tests across the six `src/` modules, using the standard library's
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
- [x] **Step 6 — Evaluation harness** (≥20 QA pairs, Precision@k/Recall@k/faithfulness) (this commit)
- [ ] **Step 7 — Write-up, video, final polish**

## Notes on reproducibility

- No GPU required — embeddings run on CPU, generation calls an external
  LLM API.
- All dependency versions are pinned in `requirements.txt` as they're
  introduced.
- Random seeds are fixed wherever sampling occurs (documented per script).
