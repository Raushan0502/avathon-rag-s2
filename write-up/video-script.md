# 5-Minute Walkthrough — Recording Script

Target: **5:00**. Timings are cumulative. Lead with the honest results — the
rubric weights reasoning and alternative analysis at 30%, more than raw
accuracy, so the measured-and-reversed findings are the strongest material.

**Before recording**
- `python scripts/report_metrics.py` — have the output on screen, it's your
  cue card and proves the numbers are live.
- Open in tabs: README (Technical write-up section), `results/qa_eval_results.json`,
  `src/ingest.py`.
- Pre-warm the model: run `python scripts/demo_qa.py` once so the recorded run
  isn't waiting on a cold tokenizer load.

---

### 0:00–0:35 — Problem and track choice

> "Track D, Scenario S2. The problem: a compliance officer needs a specific
> fact out of a corpus that's large, constantly changing, and legally
> consequential — and needs to see where it came from.
>
> I chose RAG over fine-tuning for two reasons. Technically, this knowledge
> changes every filing season, and weights can't cite a source. Practically —
> fine-tuning needs a GPU I don't have, and inside 48 hours that's one or two
> training runs with no room to be wrong. So I spent the time on measurement
> instead."

### 0:35–1:20 — Corpus and ingestion

*Show: `results/extraction_validation.json` summary, then `src/ingest.py`.*

> "100 real documents — SEC EDGAR 10-Ks and 8-Ks, material contracts, insider-
> trading policies, NIST publications, business email. Three formats: HTML, PDF,
> plain text. Not synthetic.
>
> Extraction renders tables to Markdown so numeric rows survive as rows, and a
> validation gate scores every document and **exits non-zero** if one fails — so
> a bad parse can't silently reach the index. 100 out of 100 pass.
>
> Chunking is token-budgeted and structure-preserving: split on the document's
> own headings, pack to 360 tokens with 60-token overlap, never split a table
> row. That gives 9,409 chunks."

### 1:20–2:00 — The chunking bug worth showing

> "One thing I'd point at specifically. An earlier build silently truncated 80
> chunks past the encoder's 512-token limit — no error, just missing text at
> the end of every long chunk. I only caught it by measuring token lengths
> directly instead of trusting the config. Adding an explicit oversize split
> plus a reserve for the context header brought it to zero.
>
> Fixing it moved every headline number: precision went from 0.349 to 0.378,
> recall 0.867 to 0.889."

### 2:00–3:05 — Retrieval, and the result that reversed

*Show: the retrieval table.*

> "Dense, BM25, and Reciprocal Rank Fusion, same 45 questions.
>
> **Hybrid lost to dense.** Dense MRR 0.781, hybrid 0.735. That's the opposite
> of what I expected, and an earlier draft of my own README asserted hybrid won
> before I had the numbers.
>
> The cause is IDF dilution. When the corpus grew from 6 to 100 documents, the
> terms that discriminate in filings — 'revenue', 'Item 7', 'Company' — started
> appearing everywhere. BM25's IDF flattened, and RRF weights both retrievers
> equally by rank, so fusing a weaker ranker into a stronger one dragged it
> down. I kept RRF behind a flag because it's the right default on a lexically
> diverse corpus, but dense is what ships here, because that's what the
> measurement supports.
>
> One more thing about precision: it's structurally capped. A third of my
> questions have gold sections containing a single chunk, so P@5 can't exceed
> 0.2 for them. That's why I report attainment against the ceiling, and MRR,
> rather than a raw precision number that looks bad for the wrong reason."

### 3:05–3:45 — Embedding model: the trade

*Show: the bge comparison table.*

> "I tested bge-small against bge-large rather than arguing from leaderboard
> position. bge-large is genuinely better on precision — 0.604 against 0.554.
> But it costs 8.3× more per query, 531 milliseconds against 64, and about 3.3
> hours to index this corpus on CPU.
>
> So bge-large is the better retriever; bge-small is the better engineering
> choice given CPU-only inference. On a GPU deployment I'd take the large one.
> That's a cost decision, and I want to be clear it's a cost decision rather
> than pretending the small model won on quality."

### 3:45–4:35 — Live demo and hallucination handling

*Run `python scripts/demo_qa.py`. Let one answer render with its citation.*

> "End to end: retrieve, generate with a citation per claim, and annotate.
> Across 45 questions: 41 cited, 4 refused, **zero ungrounded**. The refusals
> are correct behaviour — retrieval genuinely missed, and the model declined
> instead of inventing.
>
> But grounding is not correctness, and I think that's the most important slide
> here. **Cited does not mean correct.** So I score accuracy separately, with a
> lexical scorer and an LLM judge: 58.5% fully correct, 85.4% correct or
> partial. Six answers are **wrong while properly cited** — they point at real
> retrieved context and still get it wrong. That's why faithfulness alone is an
> insufficient safety metric."

### 4:35–5:00 — Limits and next step

> "What I'd do next isn't a bigger embedding model — I measured that and it
> doesn't pay. Recall is 0.889 while precision is 0.378, which means the right
> chunk is usually retrieved but not ranked first. That's exactly what a
> cross-encoder reranker fixes, and it's the one experiment I'd run with
> another day.
>
> Everything I've quoted regenerates with one command — `report_metrics.py`
> prints every number from the artifacts in `results/`. Thanks."

---

## If you overrun

Cut in this order: the 1:20–2:00 chunking-bug segment first (it's the most
expendable), then trim the corpus detail at 0:35. **Never cut** the hybrid
reversal or the cited-≠-correct point — those two are the submission's
strongest evidence of engineering judgement.
