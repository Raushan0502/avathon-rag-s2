# Grounded Q&A over Enterprise Documents
### Track D (RAG / LLM Knowledge Systems) × Scenario S2 — Technical Write-Up

**Repository:** https://github.com/Raushan0502/avathon-rag-s2 · **Author:** Raushan Kumar

---

## 1. Problem and track selection

A compliance officer needs a specific fact out of a corpus that is large,
constantly changing, and legally consequential — and needs to see *where it
came from*. The corpus here is **100 real SEC EDGAR documents** (10-K/8-K
reports, EX-10 contracts, EX-19/EX-97 policies, plus NIST publications and
Enron emails) across **three file formats** (HTML, PDF, plain text).

**Why Track D over the others** was driven by fit *and* by what could be
built to depth inside a **48-hour, CPU-only, free-tier** budget:

- **Not Track B (fine-tuning).** The knowledge changes every filing season;
  weights bake in a snapshot and cannot cite a source. This corpus grew
  6 → 100 documents mid-project, which a re-index absorbed and a fine-tune
  would not have. Practically, fine-tuning needs a GPU I did not have, plus
  dataset curation and training iterations measured in hours — inside 48
  hours that is one or two shots at a configuration, with no room to be wrong.
- **Not Track C (optimisation).** "Find the passage and answer from it" has
  no decision variable, objective, or constraint set to formulate.
- **Not Track A (agents).** Multi-agent orchestration adds latency, cost and
  non-deterministic control flow — the hardest failure mode to reproduce
  under time pressure — and the capability it would coordinate is still
  retrieval.

The hours that would have gone into a training loop went into **measured**
comparisons instead of asserted ones. That is the trade this submission makes.

## 2. Pipeline

`extract → normalise → validate → section-split → chunk → embed → index → retrieve → generate`

Format-aware **extraction** renders HTML/PDF tables to Markdown (expanding
`colspan`, merging row-local currency/percent cells) so numeric rows survive
as rows; PDF table regions are masked from the prose pass to stop
double-counting. Images are out of scope and documented as requiring OCR.
**Normalisation** runs five ordered passes (NFKC + punctuation map,
de-hyphenation, dot-leader stripping, digit-masked page-furniture removal,
whitespace collapse). A **validation gate** then scores every document on
type-aware thresholds and exits non-zero on failure, so a bad parse cannot
silently reach the index — **100/100 documents pass**.

## 3. Chunking: choice, and what is lost at the boundary

**Chosen:** token-budgeted, structure-preserving chunking with contextual
enrichment — split on the document's own section headings first, then pack
blocks to a **360-token budget with 60-token overlap**, never splitting a
table row, and prepend a `document › section` header to each chunk's embed
text.

**Alternatives rejected:** fixed-size character windows (cut mid-sentence and
mid-table, and the header context needed to disambiguate "Item 7" across 20
issuers is lost); one-chunk-per-section (10-K Item 7 runs far past any
context window); pure semantic/embedding-based splitting (cost scales with
corpus, and the gain is unproven against structured filings that already
carry explicit headings).

**What is genuinely lost at boundaries.** Two things, stated honestly:
(a) *long-range coreference* — a chunk beginning "These amounts exclude…"
loses its antecedent; the 60-token overlap and the section header mitigate
but do not eliminate this. (b) *cross-section reasoning* — a question needing
Item 7 **and** Item 8 requires two chunks to be retrieved together, and
nothing in the chunker guarantees that. This is the dominant residual error
mode in §6.

The token budget is enforced, not assumed: an earlier build silently
truncated **80 chunks** past the encoder's 512-token limit. Adding an
explicit oversize split plus a 60-token reserve for the context header
brought that to **0** (max observed 511).

## 4. Embedding model

**Chosen:** `BAAI/bge-small-en-v1.5` (384-dim). **Head-to-head against
`bge-large-en-v1.5` (1024-dim)** on the same evaluation set:

| Model | Dim | MRR | Recall@k | P@k | Attainment |
|---|---|---|---|---|---|
| **bge-small** | 384 | **0.948** | 0.978 | 0.554 | 0.752 |
| bge-large | 1024 | 0.940 | 0.978 | 0.554 | 0.752 |

bge-large is roughly **8.3× more expensive to embed on CPU and did not win** —
it lost by 0.008 MRR. On this corpus the retrieval bottleneck is not encoder
capacity. That is a measured result, not an assumption, and it is why the
small model ships. Both are used asymmetrically (query-instruction prefix on
queries, none on passages) per the model card. Embeddings are cached
content-addressed on `(model, sha256(text))` with batch checkpointing — added
after a 10.7-hour uncheckpointed run produced nothing.

## 5. Fusion vs. a single retriever — the quantitative result

Dense (FAISS `IndexFlatIP`, exact cosine), BM25, and **Reciprocal Rank
Fusion** (k=60), scored on the same 45 questions:

| Mode | P@k | Recall@k | MRR | Attainment |
|---|---|---|---|---|
| **dense** | **0.349** | **0.867** | **0.762** | **47.3%** |
| bm25 | 0.218 | 0.733 | 0.554 | 29.5% |
| hybrid (RRF) | 0.290 | 0.867 | 0.727 | 39.3% |

**Hybrid did not beat dense.** The expected result is the opposite, and an
earlier draft of this project claimed it before the numbers were in. The
cause is **corpus-wide IDF dilution**: as the corpus grew 6 → 100 documents,
the terms that discriminate in these filings ("revenue", "Item 7",
"Company") appear everywhere, BM25's IDF flattens, and fusing a weaker ranker
into a stronger one drags the stronger one down. Reporting this rather than
the expected result is the point — RRF is retained behind a flag because it
is the right default on a lexically diverse corpus, but **dense is the
shipped mode here** because that is what the measurement supports.

Precision is also **structurally capped**: a third of the questions have gold
sections containing a single chunk, so P@5 cannot exceed 0.2 for them. Mean
ceiling is 0.738, so **attainment** (P@k ÷ ceiling) and **MRR** are the
honest headline metrics, and questions are tiered by k = 1/5/10/20 to match
real answer-set size.

## 6. Hallucination prevention and detection

Three layers, because prevention alone is not verifiable:

1. **Prevention** — the prompt constrains the model to the retrieved context
   and requires a bracketed citation index per claim; refusal is explicitly
   permitted when context is insufficient.
2. **Detection** — every answer is parsed for citation markers and annotated
   `cited` / `refused` / `UNGROUNDED`. Result on 45 questions: **41 cited,
   4 refused, 0 UNGROUNDED**. The 4 refusals are correct behaviour, not
   failures — retrieval genuinely missed (hit@k = 40/45).
3. **Correctness, measured separately** — **cited ≠ correct.** Grounding says
   the answer pointed at context; it does not say the answer is right. Answer
   accuracy is therefore scored independently, by a lexical key-fact recall
   scorer (deterministic, free, punishes paraphrase) *and* an LLM judge
   (handles paraphrase, costs a call, carries its own bias). Lexical accuracy
   is **39.0%**, mean key-fact recall **0.534**; where the two scorers
   disagree is exactly where a human should look. Question q14 is a concrete
   case: a properly-cited answer that is still wrong.

**Residual failure modes:** cross-section questions (§3), and the gap between
grounded and correct — the system's honest weakness is not fabrication but
under-specification.

## 7. Engineering and reproducibility

Provider fallback chain (Groq → Mistral → Gemini) with retry and backoff.
**170 unit tests pass.** Every number above is regenerable:
`validate_corpus.py → build_index.py → run_evaluation.py → score_answers.py`.
Latency, measured: retrieval **168 ms** p50, generation **1,557 ms** paced —
a naive first measurement reported 14,482 ms before I noticed it was my own
rate-limit backoff, not inference. Free-tier limits, not the pipeline, set
sustained throughput.
