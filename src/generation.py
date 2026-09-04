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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import GEMINI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY

GROQ_MODEL = "openai/gpt-oss-120b"
MISTRAL_MODEL = "mistral-small-latest"
GEMINI_MODEL = "gemini-flash-latest"

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


MAX_ANSWER_TOKENS = 1024
# Free-tier providers rate-limit and return transient 503s under load, so
# the whole chain is retried rather than each provider being tried once.
PROVIDER_ROUNDS = 4
PROVIDER_BACKOFF_SECONDS = 4.0

_CLIENTS: dict[str, object] = {}


def _client(provider: str, api_key: str):
    """Return a cached SDK client, constructing it on first use.

    Clients are cached rather than built per call for a concrete reason,
    not just tidiness: constructing a fresh ``google-genai`` client on every
    request left earlier instances to be garbage-collected, and their
    cleanup closes an HTTP transport shared with the live client. Over an
    evaluation run this surfaced as "Cannot send a request, as the client
    has been closed" -- Gemini failing while working perfectly in a fresh
    process. Reusing one client per provider also avoids reopening a
    connection pool on every question.

    Args:
        provider: ``"groq"``, ``"mistral"`` or ``"gemini"``.
        api_key: Key for that provider.

    Returns:
        The cached client instance.
    """
    if provider not in _CLIENTS:
        if provider == "groq":
            from groq import Groq

            _CLIENTS[provider] = Groq(api_key=api_key)
        elif provider == "mistral":
            from mistralai.client import Mistral

            _CLIENTS[provider] = Mistral(api_key=api_key)
        else:
            from google import genai

            _CLIENTS[provider] = genai.Client(api_key=api_key)
    return _CLIENTS[provider]


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Render retrieved chunks and the question into the grounding prompt.

    Sources are numbered from 1 and labelled with their filing provenance,
    so the model can cite "[2]" and a reader can trace that citation back to
    a specific section of a specific filing.

    Args:
        query: The user's natural-language question.
        retrieved_chunks: Top-k chunks from ``RetrievalIndex.search``.

    Returns:
        The user-turn prompt string (the instructions themselves live in
        ``SYSTEM_PROMPT``).
    """
    sources = [
        f"[{i}] ({c['ticker']} {c['form']}, {c['filing_date']}, section: {c['section']})\n{c['text']}"
        for i, c in enumerate(retrieved_chunks, start=1)
    ]
    return "SOURCES:\n" + "\n\n".join(sources) + f"\n\nQUESTION: {query}\n\nANSWER:"


def call_llm(prompt: str) -> tuple[str, str]:
    """Send the prompt to the first provider that answers, in fallback order.

    Providers are tried Groq -> Mistral -> Gemini; ones without a configured
    key are skipped, and any provider that raises (rate limit, outage,
    exhausted quota) falls through to the next. The three SDKs are called
    inline here rather than through per-provider wrappers so the whole
    fallback chain reads in one place. SDK imports are deferred into each
    branch so a missing optional SDK only breaks that one provider.

    Args:
        prompt: The user-turn prompt from ``build_prompt``.

    Returns:
        ``(provider_name, answer_text)`` -- the name is recorded on every
        response so a fallback is observable rather than silent.

    Raises:
        RuntimeError: If no provider is configured, or all of them failed.
            The message carries each provider's error for diagnosis.
    """
    errors: list[str] = []
    for attempt in range(PROVIDER_ROUNDS):
        if attempt:
            # Every provider failed this round. Rate limits and "high demand"
            # 503s are transient, so back off and try the chain again rather
            # than abandoning a 33-question evaluation to a momentary outage.
            time.sleep(PROVIDER_BACKOFF_SECONDS * (2 ** (attempt - 1)))
            errors = []
        result = _try_providers(prompt, errors)
        if result is not None:
            return result
    raise RuntimeError(f"All LLM providers failed or unconfigured: {errors}")


def _try_providers(prompt: str, errors: list[str]) -> tuple[str, str] | None:
    """One pass down the provider chain; returns None if all of them failed."""
    for name, api_key in [
        ("groq", GROQ_API_KEY),
        ("mistral", MISTRAL_API_KEY),
        ("gemini", GEMINI_API_KEY),
    ]:
        if not api_key:
            continue
        try:
            if name == "groq":
                response = _client("groq", api_key).chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=MAX_ANSWER_TOKENS,
                    temperature=0.0,
                )
                return name, response.choices[0].message.content.strip()

            if name == "mistral":
                response = _client("mistral", api_key).chat.complete(
                    model=MISTRAL_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=MAX_ANSWER_TOKENS,
                    temperature=0.0,
                )
                return name, response.choices[0].message.content.strip()

            from google.genai import types

            response = _client("gemini", api_key).models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=MAX_ANSWER_TOKENS,
                    temperature=0.0,
                ),
            )
            return name, response.text.strip()
        except Exception as exc:  # noqa: BLE001 -- any SDK failure should fall through
            errors.append(f"{name}: {exc}")

    return None


def annotate_faithfulness(answer: str, num_sources: int) -> dict:
    """Check whether an answer is grounded in its numbered sources.

    Args:
        answer: The model's generated answer text.
        num_sources: How many sources were supplied, so citations pointing
            outside that range (a hallucinated source number) are discarded.

    Returns:
        Dict with ``citations`` (valid source numbers found),
        ``has_citation``, ``refused_unsupported`` (the model used the
        explicit "cannot answer" phrase), and ``faithfulness_flag`` --
        ``"refused"``, ``"cited"``, or ``"UNGROUNDED"`` for an answer that
        asserts something while citing nothing.
    """
    # Models emit citations in more than one bracket style despite the
    # prompt requesting plain "[4]": full-width CJK brackets ("【4】",
    # observed in Step 5 testing), and a browsing-style format with a
    # dagger and a line range ("【1†L1-L4】", observed in Step 6's broader
    # 24-question eval -- 4 answers were initially misflagged UNGROUNDED
    # purely because this second format wasn't recognized, even though
    # every one of them was in fact correctly grounded). The regex below
    # accepts a source number immediately after an opening ASCII or
    # full-width bracket, followed by an optional non-bracket suffix
    # (the dagger/line-range part) before the closing bracket, so both
    # citation styles -- and plain "[4]" -- all count.
    citation_markers = re.findall(r"[\[【](\d+)[^\]】\[【]{0,20}[\]】]", answer)
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
    """Answer a question from retrieved context, with a faithfulness check.

    Args:
        query: The user's natural-language question.
        retrieved_chunks: Top-k chunks from ``RetrievalIndex.search``.

    Returns:
        Dict with ``query``, ``answer``, ``provider`` (which LLM served it),
        ``retrieved_chunks`` (the context used), and ``faithfulness``.
    """
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
