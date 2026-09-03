"""
Step 5 — Generation: grounded answer synthesis over retrieved context.

Provider fallback chain: Groq -> Mistral -> Gemini. All three are
external LLM APIs (explicitly allowed for Track D), chosen because each
offers a free tier with no committed spend, keeping inference cost at
zero to match this track's "no compute budget required" framing. Groq is
tried first for its very low latency (LPU-hosted open-weight models);
Mistral and Gemini are automatic fallbacks so a single provider's outage,
rate limit, or exhausted free-tier quota mid-demo doesn't stall the whole
pipeline. Which provider actually served a given answer is recorded on
every response — the fallback is observable, not silent, addressing the
Track D question about failure modes (agent silently returning nothing,
or an unannounced degraded answer, is exactly what this guards against).

Hallucination mitigation strategy (the other mandatory write-up
question): the prompt requires every claim to be attributed to a
numbered source chunk (e.g. "[2]"), and instructs the model to say it
cannot answer rather than guess when the retrieved context doesn't cover
the question. `annotate_faithfulness` then checks, per answer, whether
at least one citation marker is present and whether the model invoked
the explicit "cannot answer" fallback -- a lightweight, fast heuristic
appropriate for Step 5's end-to-end demo. The rigorous version (does the
answer's content actually appear in the cited context, over the full
held-out QA set) is Step 6's job.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import GEMINI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY

GROQ_MODEL = "openai/gpt-oss-120b"
MISTRAL_MODEL = "mistral-small-latest"
GEMINI_MODEL = "gemini-2.0-flash"

CANNOT_ANSWER_PHRASE = "I cannot answer this from the provided documents."

SYSTEM_PROMPT = f"""You are a financial-filings assistant. Answer the user's \
question using ONLY the numbered source excerpts provided below -- never use \
outside knowledge, even if you are confident it is correct.

Rules:
- Every factual claim in your answer must end with a citation to the source \
number it came from, like [1] or [2][3].
- If the sources do not contain enough information to answer, respond with \
exactly: "{CANNOT_ANSWER_PHRASE}" -- do not guess or fill gaps from general \
knowledge.
- Be concise. Do not repeat the question."""


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    sources = []
    for i, c in enumerate(retrieved_chunks, start=1):
        sources.append(
            f"[{i}] ({c['ticker']} {c['form']}, {c['filing_date']}, section: {c['section']})\n{c['text']}"
        )
    sources_block = "\n\n".join(sources)
    return f"SOURCES:\n{sources_block}\n\nQUESTION: {query}\n\nANSWER:"


def _call_groq(prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def _call_mistral(prompt: str) -> str:
    from mistralai.client import Mistral

    client = Mistral(api_key=MISTRAL_API_KEY)
    resp = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def _call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1024,
            temperature=0.0,
        ),
    )
    return resp.text.strip()


_PROVIDERS = [
    ("groq", GROQ_API_KEY, _call_groq),
    ("mistral", MISTRAL_API_KEY, _call_mistral),
    ("gemini", GEMINI_API_KEY, _call_gemini),
]


def call_llm(prompt: str) -> tuple[str, str]:
    """Tries each configured provider in order, returns (provider_name, text).
    Raises RuntimeError only if every available provider fails."""
    errors = []
    for name, api_key, fn in _PROVIDERS:
        if not api_key:
            continue
        try:
            return name, fn(prompt)
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError(f"All LLM providers failed or unconfigured: {errors}")


def annotate_faithfulness(answer: str, num_sources: int) -> dict:
    # Models occasionally emit full-width CJK brackets (e.g. "【4】") instead of
    # the ASCII "[4]" requested in the prompt -- observed live from the Groq
    # provider during Step 5 testing. Both forms count as a citation; only
    # a genuinely uncited claim should be flagged.
    citation_markers = re.findall(r"[\[【](\d+)[\]】]", answer)
    cited = sorted({int(n) for n in citation_markers})
    cited = [n for n in cited if 1 <= n <= num_sources]
    refused = CANNOT_ANSWER_PHRASE.lower() in answer.lower()
    return {
        "citations": cited,
        "has_citation": len(cited) > 0,
        "refused_unsupported": refused,
        "faithfulness_flag": "refused" if refused else ("cited" if cited else "UNGROUNDED"),
    }


def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    prompt = build_prompt(query, retrieved_chunks)
    provider, answer = call_llm(prompt)
    faithfulness = annotate_faithfulness(answer, len(retrieved_chunks))
    return {
        "query": query,
        "answer": answer,
        "provider": provider,
        "retrieved_chunks": retrieved_chunks,
        "faithfulness": faithfulness,
    }
