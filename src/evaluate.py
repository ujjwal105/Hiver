"""
evaluate.py — The accuracy / evaluation system (the heart of this project).

WHAT "ACCURATE" MEANS FOR A SUGGESTED REPLY
-------------------------------------------
Exact match is the wrong target: two replies can be equally good while sharing
almost no words. So surface-overlap metrics (BLEU/ROUGE/exact-match) are too
strict — they punish valid paraphrases and reward copying. We instead score the
things that actually make a support reply good, using several *complementary*
signals so no single failure mode dominates:

  1. quality      — LLM-as-judge rubric, 5 dimensions (relevance, correctness,
                    completeness, tone, actionability), each 1–5, WITH a written
                    rationale. The judge sees the human reference as the gold
                    standard. This captures semantic correctness a string metric
                    cannot.  weight 0.50
  2. similarity   — TF-IDF cosine to the human reference. A reference-based anchor:
                    "how close to what a human actually sent". Cheap, deterministic,
                    and a useful independent cross-check on the judge.  weight 0.20
  3. facts        — fact-carryover: fraction of concrete facts in the reference
                    (order #s, amounts, emails, IDs) correctly reproduced. Support
                    replies live or die on getting these right.  weight 0.20
  4. format       — deterministic guardrails: greeting + sign-off present, sensible
                    length. Cheap sanity that the output is a sendable email. 0.10

We ALSO flag *hallucinated specifics*: concrete facts in the candidate that appear
in neither the incoming email nor the reference. Those are a safety red-flag and
apply a penalty even if the rest scores well.

Composite = weighted sum of the available signals, rescaled to 0–100. If a ticket
has no concrete facts, the 'facts' weight is redistributed rather than faked.

WHY THIS IS THE RIGHT METRIC: see validate_metric.py — we don't just assert it,
we test that the metric recovers a known quality ordering and that its independent
signals agree.

CLI:
  python -m src.evaluate --in outputs/generated.jsonl --out outputs/scored.jsonl
  python -m src.evaluate --provider mock ...
"""
from __future__ import annotations
import argparse, json, os, re
from .llm import LLM
from .retrieval import cosine

FACT_RE = re.compile(r"#\d{3,}|\$\d[\d,]*|[\w.+-]+@[\w.-]+\.\w+|GB\d+|\b\d{4,}\b")

WEIGHTS = {"quality": 0.50, "similarity": 0.20, "facts": 0.20, "format": 0.10}

JUDGE_SYSTEM = (
    "You are a strict but fair QA reviewer for customer-support replies. You are "
    "given the customer's incoming email, the reply a human agent actually sent "
    "(the reference / gold standard), and a candidate reply to score. Judge the "
    "CANDIDATE on how well it would serve the customer — not on whether it matches "
    "the reference word-for-word (a different-but-equally-good reply should score "
    "high). Score each dimension 1-5 (5=excellent). Return JSON with integer keys "
    "relevance, correctness, completeness, tone, actionability, and a short "
    "'rationale' string explaining the main strengths/weaknesses."
)


def facts_in(text: str) -> set[str]:
    return set(f.strip() for f in FACT_RE.findall(text or ""))


def _judge_prompt(incoming, reference, candidate):
    return (
        f"INCOMING EMAIL:\n{incoming}\n\n"
        f"REFERENCE REPLY:\n{reference}\n\n"
        f"CANDIDATE REPLY:\n{candidate}\n"
    )


def _format_ok(candidate: str) -> float:
    c = candidate.lower()
    greeting = bool(re.search(r"\b(hi|hello|hey|dear)\b", c))
    signoff = bool(re.search(r"\b(best|regards|thanks|thank you|cheers|sincerely)\b", c))
    n = len(candidate.split())
    length_ok = 15 <= n <= 300
    return round((greeting + signoff + length_ok) / 3.0, 3)


def score_response(incoming, reference, candidate, llm: LLM) -> dict:
    # 1. LLM judge (or mock)
    judge = llm.complete_json(JUDGE_SYSTEM, _judge_prompt(incoming, reference, candidate))
    dims = ["relevance", "correctness", "completeness", "tone", "actionability"]
    dim_scores = {}
    for d in dims:
        try:
            dim_scores[d] = max(1, min(5, int(round(float(judge.get(d, 3))))))
        except (TypeError, ValueError):
            dim_scores[d] = 3
    quality = sum((dim_scores[d] - 1) / 4.0 for d in dims) / len(dims)  # ->0..1

    # 2. similarity to reference
    similarity = cosine(reference, candidate)

    # 3. fact carryover + hallucination flag
    ref_facts = facts_in(reference)
    cand_facts = facts_in(candidate)
    in_facts = facts_in(incoming)
    if ref_facts:
        facts_score = len(ref_facts & cand_facts) / len(ref_facts)
    else:
        facts_score = None  # no facts to carry -> exclude from weighting
    hallucinated = sorted(cand_facts - ref_facts - in_facts)
    hallucination_penalty = min(0.15, 0.05 * len(hallucinated))  # up to -15 pts

    # 4. format
    fmt = _format_ok(candidate)

    # ---- composite with weight redistribution when facts absent ----
    signals = {"quality": quality, "similarity": similarity,
               "facts": facts_score, "format": fmt}
    active = {k: v for k, v in signals.items() if v is not None}
    total_w = sum(WEIGHTS[k] for k in active)
    composite = sum(WEIGHTS[k] * active[k] for k in active) / total_w
    composite = max(0.0, composite - hallucination_penalty)

    return {
        "composite": round(100 * composite, 1),
        "dimensions": dim_scores,
        "signals": {
            "quality": round(quality, 3),
            "similarity": round(similarity, 3),
            "fact_carryover": None if facts_score is None else round(facts_score, 3),
            "format": fmt,
        },
        "hallucinated_facts": hallucinated,
        "hallucination_penalty": round(hallucination_penalty, 3),
        "rationale": judge.get("rationale", ""),
    }


def summarize(scored: list[dict]) -> dict:
    comps = [s["result"]["composite"] for s in scored]
    n = len(comps)
    mean = sum(comps) / n if n else 0.0
    # per-dimension means
    dims = ["relevance", "correctness", "completeness", "tone", "actionability"]
    dim_means = {}
    for d in dims:
        vals = [s["result"]["dimensions"][d] for s in scored]
        dim_means[d] = round(sum(vals) / len(vals), 2) if vals else 0.0
    # per-category means
    cats = {}
    for s in scored:
        c = s.get("category", "?")
        cats.setdefault(c, []).append(s["result"]["composite"])
    cat_means = {c: round(sum(v) / len(v), 1) for c, v in cats.items()}
    halluc = sum(1 for s in scored if s["result"]["hallucinated_facts"])
    return {
        "n": n,
        "overall_score": round(mean, 1),
        "min": round(min(comps), 1) if comps else 0,
        "max": round(max(comps), 1) if comps else 0,
        "dimension_means_1to5": dim_means,
        "category_means": cat_means,
        "responses_with_hallucinated_facts": halluc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/generated.jsonl")
    ap.add_argument("--out", default="outputs/scored.jsonl")
    ap.add_argument("--report", default="outputs/report.json")
    ap.add_argument("--provider", default=None, choices=["anthropic", "mock"])
    args = ap.parse_args()

    with open(args.inp) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    llm = LLM(provider=args.provider)
    print(f"[evaluate] provider={llm.provider}  scoring {len(rows)} responses")

    scored = []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for i, r in enumerate(rows, 1):
            res = score_response(r["incoming"], r["reference"], r["generated"], llm)
            rec = {"id": r.get("id"), "category": r.get("category"),
                   "subject": r.get("subject"), "result": res}
            scored.append(rec)
            f.write(json.dumps(rec) + "\n")
            print(f"  [{i}/{len(rows)}] {r.get('category','?'):<15} "
                  f"score={res['composite']:>5}  "
                  f"({'/'.join(str(res['dimensions'][d]) for d in ['relevance','correctness','completeness','tone','actionability'])})")

    report = summarize(scored)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== OVERALL ===")
    print(json.dumps(report, indent=2))
    print(f"\n[evaluate] per-response -> {args.out}   report -> {args.report}")


if __name__ == "__main__":
    main()
