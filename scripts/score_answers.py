"""
Score generated answers for correctness against the gold reference answers.

Faithfulness says an answer cited its context; it does not say the answer is
right. This runs the two scorers in ``src/answer_scoring`` over the saved
evaluation results and writes ``results/answer_accuracy.json``.

Both scorers are reported because they disagree in an informative way. The
lexical one is free and deterministic but punishes paraphrase -- an answer
saying "repeatedly slashing prices" against a reference of "aggressively cut
prices" is correct and scores poorly. The judge handles paraphrase but costs
an API call per question and carries its own biases. Where they disagree is
where a human should look.

Usage:
    python scripts/score_answers.py             # lexical only, no API calls
    python scripts/score_answers.py --judge     # adds the LLM judge
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.answer_scoring import KEY_FACT_PASS, score_answers
from src.config import RESULTS_DIR


def main() -> int:
    """Score the saved evaluation answers and write the accuracy report.

    Returns:
        Process exit code: 1 if the evaluation results are missing.
    """
    source = RESULTS_DIR / "qa_eval_results.json"
    if not source.exists():
        print(f"Missing {source}. Run: python scripts/run_evaluation.py")
        return 1

    use_judge = "--judge" in sys.argv
    records = json.loads(source.read_text(encoding="utf-8"))
    print(f"Scoring {len(records)} answers"
          f"{' with LLM judge' if use_judge else ' (lexical only)'}...\n")

    scores = score_answers(records, use_judge=use_judge)

    print(f"  questions            : {scores['questions']}")
    print(f"  refused              : {scores['refused']}  "
          "(excluded -- declining without context is correct behaviour)")
    print(f"  answered             : {scores['answered']}")
    print(f"  mean key-fact recall : {scores['mean_key_fact_recall']:.3f}")
    print(f"  lexical accuracy     : {scores['lexical_accuracy']:.1%}  "
          f"(>= {KEY_FACT_PASS:.0%} of reference key facts present)")

    if scores["judge_verdicts"]:
        verdicts = scores["judge_verdicts"]
        total = sum(verdicts.values())
        correct = verdicts.get("CORRECT", 0)
        partial = verdicts.get("PARTIAL", 0)
        print(f"  judge verdicts       : {verdicts}")
        print(f"  judge CORRECT        : {correct / total:.1%}")
        print(f"  judge CORRECT+PARTIAL: {(correct + partial) / total:.1%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "answer_accuracy.json"
    out.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
