"""
Re-point the evaluation set's gold references at the current chunk set.

Gold references are stored as ``(gold_doc_id, gold_section, gold_chunk_id)``.
Chunk ids embed a positional index, and section labels come from the
splitter, so **both move whenever the ingestion pipeline changes** -- a
retuned chunk size, a new section splitter, or a corpus refresh. Left
stale, they silently corrupt every metric downstream: a question whose
gold chunk no longer exists scores zero regardless of how well retrieval
actually performed.

Rather than hand-editing 33 references, each question is re-located by
content. The reference answer's distinctive terms are matched against the
chunks of its gold document, the best-scoring chunk wins, and the
question's gold pointers are rewritten to it.

This is deliberately conservative: it only ever searches **within the
question's existing ``gold_doc_id``**, so a bad match can move a reference
to the wrong passage of the right document, never to a different document.
Anything it cannot match confidently is reported for a human to look at
rather than being silently rewritten.

Usage:
    python scripts/remap_eval_set.py            # report only, writes nothing
    python scripts/remap_eval_set.py --write    # apply the remapping
"""
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHUNKS_PATH, EVAL_SET_PATH

STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "have", "has", "are", "was",
    "were", "its", "his", "her", "their", "which", "when", "what", "does", "did", "not",
    "will", "would", "can", "could", "any", "all", "may", "must", "such", "than", "then",
    "there", "these", "those", "been", "being", "into", "over", "under", "about",
}
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9,.$%-]*")
MIN_OVERLAP = 0.30
# Moving a reference to a *different* section is the change most likely to
# be wrong, so it needs stronger evidence than merely re-pointing a chunk
# id within the section the question was authored against.
STRONG_OVERLAP = 0.75


def salient_terms(text: str) -> set[str]:
    """Extract the terms worth matching on from a reference answer.

    Args:
        text: A gold reference answer.

    Returns:
        Lowercased tokens with stopwords and very short tokens removed, so
        scoring is driven by distinctive words and figures ("416,161",
        "cupertino") rather than filler shared by every chunk.
    """
    return {
        token
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def best_matching_chunk(question: dict, chunks: list[dict]) -> tuple[dict | None, float]:
    """Find the chunk in the question's gold document that best matches it.

    Scoring is the fraction of the reference answer's salient terms present
    in the chunk. The question text is included at a lower weight, since a
    question often shares wording with the passage that answers it.

    Args:
        question: An eval-set entry.
        chunks: All chunks for that question's ``gold_doc_id``.

    Returns:
        ``(chunk, score)`` for the best candidate, or ``(None, 0.0)``.
    """
    answer_terms = salient_terms(question["reference_answer"])
    question_terms = salient_terms(question["query"])
    if not answer_terms:
        return None, 0.0

    best, best_score = None, 0.0
    for chunk in chunks:
        chunk_terms = salient_terms(chunk["text"])
        answer_hit = len(answer_terms & chunk_terms) / len(answer_terms)
        question_hit = (
            len(question_terms & chunk_terms) / len(question_terms) if question_terms else 0.0
        )
        score = answer_hit + 0.25 * question_hit
        if score > best_score:
            best, best_score = chunk, score
    return best, best_score


def main() -> int:
    """Report, and optionally apply, the gold-reference remapping.

    Returns:
        Exit code: 1 if any question could not be confidently remapped.
    """
    write = "--write" in sys.argv
    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]

    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["doc_id"], []).append(chunk)

    print(f"{len(eval_set)} questions against {len(chunks)} chunks "
          f"in {len(by_doc)} documents\n")

    unresolved, changed = [], 0
    for question in eval_set:
        candidates = by_doc.get(question["gold_doc_id"], [])
        if not candidates:
            unresolved.append((question["id"], "gold_doc_id absent from corpus"))
            continue

        # Prefer to stay inside the original section when it still exists.
        # A weak content match can otherwise drag a reference onto an
        # unrelated passage that merely echoes the question's wording --
        # observed moving Tesla's "unresolved staff comments" answer into a
        # paragraph about workplace conduct.
        same_section = [c for c in candidates if c["section"] == question["gold_section"]]
        if same_section:
            match, score = best_matching_chunk(question, same_section)
            if match is not None and score >= MIN_OVERLAP:
                question["gold_chunk_id"] = match["chunk_id"]
                if match["chunk_id"] != question["gold_chunk_id"]:
                    changed += 1
                continue

        match, score = best_matching_chunk(question, candidates)
        if match is None or score < MIN_OVERLAP:
            unresolved.append((question["id"], f"best overlap only {score:.2f}"))
            continue
        if match["section"] != question["gold_section"] and score < STRONG_OVERLAP:
            unresolved.append(
                (
                    question["id"],
                    f"would move section {question['gold_section']!r} -> "
                    f"{match['section']!r} on a weak score of {score:.2f}",
                )
            )
            continue

        was = (question["gold_chunk_id"], question["gold_section"])
        now = (match["chunk_id"], match["section"])
        if was != now:
            changed += 1
            print(f"  {question['id']}  score {score:.2f}")
            print(f"      chunk   {was[0]}  ->  {now[0]}")
            if was[1] != now[1]:
                print(f"      section {was[1]!r}  ->  {now[1]!r}")
        question["gold_chunk_id"], question["gold_section"] = now

    print(f"\nremapped: {changed}   unchanged: {len(eval_set) - changed - len(unresolved)}"
          f"   unresolved: {len(unresolved)}")
    for question_id, reason in unresolved:
        print(f"  [UNRESOLVED] {question_id}: {reason}")

    if write and not unresolved:
        EVAL_SET_PATH.write_text(json.dumps(eval_set, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {EVAL_SET_PATH}")
    elif write:
        print("\nNot written: resolve the questions above first.")
    else:
        print("\nDry run: pass --write to apply.")

    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
