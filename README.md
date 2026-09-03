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

## Repository structure

```
avathon-rag-s2/
├── data/
│   ├── raw/            # sourced source documents (gitignored; see Step 2)
│   └── processed/      # cleaned/chunked intermediates (gitignored)
├── src/                # pipeline modules (ingestion, retrieval, generation, evaluation)
├── scripts/            # one-off / setup / data-sourcing scripts
├── tests/              # test scripts run at each step checkpoint
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
pip install -r requirements.txt
copy .env.example .env          # then fill in ANTHROPIC_API_KEY or OPENAI_API_KEY
python scripts/check_setup.py   # sanity check: env + folder layout
```

`requirements.txt` grows incrementally as each pipeline step is built and
verified — see the roadmap below.

## Build roadmap

Each step is implemented, run/tested locally, committed, and pushed to
`dev` before the next step starts. `main` only receives a merge once the
full pipeline is verified working end-to-end.

- [x] **Step 1 — Repo scaffold & environment setup** (this commit)
- [ ] **Step 2 — Data acquisition & corpus documentation**
- [ ] **Step 3 — Ingestion pipeline** (load → chunk → embed → index)
- [ ] **Step 4 — Hybrid retrieval** (dense + BM25 + fusion/re-ranking, vs. dense-only)
- [ ] **Step 5 — Generation** (grounded answers + faithfulness annotation)
- [ ] **Step 6 — Evaluation harness** (≥20 QA pairs, Precision@k/Recall@k/faithfulness)
- [ ] **Step 7 — Write-up, video, final polish**

## Notes on reproducibility

- No GPU required — embeddings run on CPU, generation calls an external
  LLM API.
- All dependency versions are pinned in `requirements.txt` as they're
  introduced.
- Random seeds are fixed wherever sampling occurs (documented per script).
