# 5-Minute Walkthrough — Recording Script

Structured to the brief's mandated segments and timings:
**(1:00) Problem and approach choice → (2:00) Key algorithm decisions and what
you ruled out → (1:30) Results and what didn't work → (0:30) What you'd do with
more time.**

Two rules from the brief that shape this script:
- **Do NOT narrate code line by line.** Show *outputs and tables*, not source
  files. There is no point in this script where you open a `.py` file.
- The video is graded on whether it *communicates insight*, and whether a
  technical stakeholder who is not the reviewer could follow it.

**Before recording**
- Run `python scripts/report_metrics.py` and leave the output on screen — it is
  your cue card and it shows the numbers are live, not typed into a slide.
- Have open: the README dataset table, the retrieval table, `results/qa_eval_results.json`.
- Run `python scripts/demo_qa.py` once first so the recorded run isn't waiting
  on a cold model load.
- Upload to Drive / Dropbox / YouTube-unlisted and **put the link in the README** —
  the brief requires it there.

---

## Segment 1 — Problem and approach choice · 0:00–1:00

*Screen: the README dataset table.*

Open plainly. Assume the listener knows nothing about this project.

> "Companies sit on thousands of documents — contracts, filings, policies,
> email — and the answer to a question is usually in there somewhere. Finding
> it means knowing which document, opening it, and reading. That's the problem
> I picked: **ask a question in plain English, get an answer back with a
> citation you can check.**
>
> The citation is the whole point. This is compliance and legal content, so an
> answer nobody can verify is worse than no answer at all.
>
> **The data is real, not synthetic.** 100 public documents I pulled myself:
> SEC EDGAR 10-K and 8-K filings, material contracts, insider-trading policies,
> NIST security publications, and business email from the public Enron corpus.
> Twenty of each type, across twelve companies in seven industries — and
> deliberately in **three different file formats**: HTML, PDF and plain text.
> That mix is intentional. It forces the pipeline to handle real extraction
> problems — financial tables in HTML, PDF page layouts, email headers —
> instead of one clean tidy source.
>
> **Why RAG and not fine-tuning?** This knowledge changes every filing season,
> and a fine-tuned model can't cite a source — it just produces text. On top of
> that, fine-tuning needs a GPU I don't have, and inside a 48-hour window it
> would have meant one or two training runs with no room to be wrong. So I
> spent the time on measurement instead. That trade is the theme of the rest of
> this video."

## Segment 2 — Key algorithm decisions and what I ruled out · 1:00–3:00

*Screen: the embedding comparison table, then the retrieval table.*

This is the highest-weighted part of the rubric. Four decisions, each with the
rejected alternative named.

> "Four decisions worth explaining.
>
> **Chunking.** I split on each document's own headings and pack to a 360-token
> budget with 60-token overlap, never splitting a table row. I ruled out
> fixed-size character windows — they cut straight through financial tables,
> and they lose the context you need to tell twenty different 'Item 7' sections
> apart when you have twenty companies. What I lose at the boundary is
> long-range reference: a chunk starting 'these amounts exclude' has lost what
> 'these' refers to.
>
> **Vector store — FAISS, exact search.** I benchmarked the approximate indexes:
> HNSW was 204 times faster but recall dropped to 0.55 untuned. At 9,400 chunks
> exact search takes 8 milliseconds, so approximation buys speed I don't need
> and costs recall I do. At ten million chunks I'd choose differently, and I say
> so.
>
> **Embedding model.** I tested two rather than arguing from a leaderboard.
> bge-large is genuinely *better* — precision 0.604 against 0.554. But it costs
> 8.3 times more per query, 531 milliseconds against 64. So I shipped the small
> one **on cost, not on quality.** On a GPU deployment I'd take the large one.
> I want to be precise about that: it's a budget decision, not a claim that the
> small model won.
>
> **Retrieval — and this is the one that surprised me.** Dense, BM25, and
> Reciprocal Rank Fusion. **Hybrid lost to dense.** Dense MRR 0.781, hybrid
> 0.735. An earlier draft of my own README claimed hybrid won, before I had the
> numbers.
>
> The cause is IDF dilution. When my corpus grew from 6 documents to 100, the
> words that discriminate in filings — 'revenue', 'Item 7', 'Company' — started
> appearing everywhere. BM25's scoring flattened out, and fusion weights both
> retrievers equally, so blending a weaker ranker into a stronger one dragged it
> down. I kept fusion behind a flag because it's the right default on a more
> varied corpus, but dense is what ships here, because that's what the
> measurement says."

## Segment 3 — Results and what didn't work · 3:00–4:30

*Screen: `report_metrics.py` output, then run `demo_qa.py` and let one cited
answer render.*

> "45 held-out questions. Retrieval recall 0.889, MRR 0.781.
>
> On precision I want to be careful, because the raw number looks weak at 0.378.
> A third of my questions have a gold section containing exactly one chunk — so
> Precision-at-5 *mathematically cannot exceed 0.2* for those. Reporting 0.378
> without that context would be misleading in the other direction. That's why I
> report attainment against the achievable ceiling, and MRR.
>
> On grounding: 41 answers cited their source, 4 refused, **zero ungrounded.**
> The refusals are the system working — retrieval genuinely missed, and it
> declined instead of inventing something.
>
> **But here's what didn't work, and it's the most important thing in this
> video: cited does not mean correct.** Grounding only tells you the answer
> pointed at a document. It doesn't tell you the answer is right. So I scored
> correctness separately — a lexical scorer and an LLM judge: 58.5% fully
> correct, 85.4% correct or partial.
>
> **Six answers are wrong while properly cited.** They point at real retrieved
> text and still get it wrong. That's the finding I'd flag to anyone deploying
> this: faithfulness on its own is not a safety metric, and if I'd only measured
> grounding I would have reported zero failures and been wrong.
>
> One more honest failure: my hardest tier scores zero on precision. Looking at
> why — one question's top result is a *different* Apple stock-award exhibit
> whose clause is word-for-word identical to the right one. No retriever can
> separate those. That's a property of the corpus, not a bug in the ranking."

## Segment 4 — What I'd do with more time · 4:30–5:00

> "Three things.
>
> **First — a cross-encoder reranker.** Recall is 0.889 but precision is 0.378,
> which means the right chunk is usually retrieved, just not ranked first. That's
> exactly what reranking fixes, and I'd expect it to clear up most of those six
> grounded-but-wrong answers. I didn't build it because on CPU it would add
> several hundred milliseconds to a 168-millisecond retrieval path.
>
> **Second — a bigger embedding model is *not* on that list**, and I only know
> that because I measured it.
>
> **Third — production.** The bottleneck isn't the vector search, it's 8
> milliseconds. It's query embedding on CPU and the LLM call. For a thousand
> concurrent queries the encoder goes behind a GPU batching service and the
> index replicates read-only — it's stateless, so that's horizontal.
>
> Everything I've quoted regenerates from one command against the saved
> artifacts. Thank you."

---

## If you overrun

Cut in this order:
1. The vector-store paragraph in Segment 2 (shortest to lose, least distinctive).
2. The `k=1` identical-exhibit example in Segment 3.
3. The production point in Segment 4.

**Never cut:** the dataset description in Segment 1 (the brief weights data
sourcing), **hybrid losing to dense**, or **cited ≠ correct**. Those three are
the submission's strongest evidence of judgement, and two of them are findings
that went against what I originally expected.
