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

**Companies:** Apple (AAPL), Microsoft (MSFT), Tesla (TSLA) — three large,
well-known issuers from different sectors (consumer hardware, enterprise
software, auto/energy), chosen so evaluation questions can span genuinely
different business content rather than near-duplicate filings.

**Document types, per company:**
- Latest **10-K** (annual report) — long (2–8 MB HTML), highly structured:
  business overview, risk factors, MD&A, financial statements, legal
  proceedings. Representative of the "reports" document type in S2.
- Latest **8-K** (current report on a material event) — short (25–40 KB
  HTML), narrower and more announcement-like in style. Included
  specifically to give the corpus document-length and document-style
  diversity, since chunking and retrieval behave differently on long
  structured filings vs. short single-topic announcements.

6 documents total. Every document is fetched by
[`scripts/fetch_corpus.py`](scripts/fetch_corpus.py) and logged in
[`data/raw/manifest.json`](data/raw/manifest.json) with its source URL,
filing date, and a SHA-256 checksum, so the corpus is fully reproducible
and auditable — re-running the script re-derives the same (or, past the
next filing, an updated) corpus from the primary source. Raw HTML files
themselves are gitignored (large, and trivially re-fetchable); the
manifest is committed as the record of what was sourced.

**Scope/limitation acknowledged:** S2's prompt also mentions contracts,
SOPs, and emails as document types. This corpus uses only SEC filings —
still genuinely unstructured/semi-structured enterprise text requiring
retrieval over long documents, but narrower in document *type* than a
real enterprise content repository. This trade-off was made to keep the
corpus verifiably real, free, and instantly reproducible without
depending on any single company's website structure staying stable.

## Ingestion & chunking (Step 3)

Pipeline: `scripts/build_index.py` → `src/ingest.py` (HTML → clean text →
section-aware chunks) → `src/embed_index.py` (embed with
`BAAI/bge-small-en-v1.5` → build a FAISS `IndexFlatIP` index).

**Chunking strategy** (full rationale in the `src/ingest.py` module
docstring): two-stage — split each filing on its numbered "Item" section
headers first (so a chunk never blends, say, Risk Factors with Legal
Proceedings), then slide a 220-word / 40-word-overlap window within each
section (keeps each chunk well under the embedding model's 512-token
limit while the overlap preserves sentences that would otherwise be cut
at a window boundary). Section detection is a regex heuristic, not a
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

Result on the 6 sourced filings: 809 chunks across 64 detected sections.
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
24 hand-authored question/answer pairs (above the assignment's 20-pair
minimum), spread across all 6 sourced documents. Each question was
authored *against a specific, read, verified chunk* — not generated then
checked after the fact — so every `reference_answer` and `gold_chunk_id`
is traceable to real filing text. Ground truth is recorded at
`(gold_doc_id, gold_section)` granularity rather than exact `chunk_id`,
since adjacent overlapping chunks from the same section are equally
valid evidence for a question (see `src/evaluate.py` docstring for the
full reasoning, including why Recall@k reduces to a binary hit-rate here:
each question has exactly one known-relevant section, not an exhaustive
relevance-judged set).

**Retrieval comparison, k=5, n=24** (`scripts/run_evaluation.py` →
`results/retrieval_eval.json`):

| Mode   | Mean Precision@5 | Mean Recall@5 |
|--------|------------------:|---------------:|
| Dense  | 0.400             | 0.833           |
| BM25   | 0.375             | 0.875           |
| **Hybrid (RRF)** | **0.400** | **0.917** |

Hybrid retrieval matches dense on precision and beats both single
retrievers on recall — 91.7% of questions found their gold section
somewhere in the top 5 under hybrid, vs. 83.3% dense-only and 87.5%
BM25-only. This is the quantitative answer to the Step 4 write-up
question ("show hybrid vs. dense-only quantitatively"): fusion is
finding relevant sections that *either single retriever alone missed*,
without diluting precision.

**End-to-end faithfulness** (hybrid retrieval → generation, all 24
questions, `results/qa_eval_results.json`): **22 cited, 2 refused, 0
ungrounded.** Getting to 0 ungrounded took a second real fix, not just
the Step 5 one: 4 of the 24 answers were initially misflagged
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

The remaining **2 refusals are legitimate, not bugs** — q08 / q16 ("Does
\[company\] report unresolved staff comments?"), whose gold answer is the
single word "None." Such a terse, low-signal section is hard for both
dense and sparse retrieval to rank highly against a full-sentence
question; it didn't make the top 5, and the model correctly refused
rather than guess.

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

57 unit tests across the five `src/` modules, using the standard library's
`unittest` (no extra dependency to install). They are deliberately
**offline and fast** (~0.02s total): no network calls, no API keys, and no
embedding-model download — `embed_texts` is exercised against a stub
encoder, and retrieval tests assemble a `RetrievalIndex` by hand from a
3-vector FAISS index and a real BM25 index over three short strings.

What they pin down, beyond the happy path:
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
python scripts/fetch_corpus.py  # sources the corpus into data/raw/ (see Data source below)
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
