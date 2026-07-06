"""
evaluate.py — The accuracy / evaluation system (the heart of this project).

ARCHITECTURE OF THE EVALUATION ENGINE
-------------------------------------
  Generated Reply ──▶ Evaluation Engine
                       ├── LLM-as-a-Judge (5-dim rubric + rationale)
                       ├── Intent match   (does it address the request?)
                       ├── Tone match     (is the register right?)
                       ├── Cosine similarity to reference (TF-IDF)
                       ├── BERTScore      (optional: auto-activates if installed)
                       ├── Fact carryover (order #s, $, emails, IDs)
                       ├── Format guardrails (greeting/sign-off/length)
                       ├── ROUGE-L        (diagnostic, unweighted)
                       └── BLEU           (diagnostic, unweighted)
                              │
                              ▼
              Final weighted score (0–100) + human-readable explanation

WHAT "ACCURATE" MEANS FOR A SUGGESTED REPLY
-------------------------------------------
Exact match is the wrong target: two replies can be equally good while sharing
almost no words. So surface-overlap metrics (BLEU/ROUGE/exact-match) are too
strict — they punish valid paraphrases and reward copying. We still COMPUTE
BLEU and ROUGE-L for every response, but as *diagnostics with zero weight*;
the validation harness (src/validate_metric.py) runs an ablation showing the
weighted composite discriminates good from bad replies better than any surface
metric alone — that ablation is the evidence for this design choice.

Weighted signals:
  quality       0.40  LLM-judge rubric mean (relevance, correctness,
                      completeness, tone, actionability; 1–5 each + rationale)
  intent_match  0.10  judge's relevance dim — does the reply address the actual
                      request? (surfaced separately: it's the first thing a
                      support lead checks)
  tone_match    0.05  judge's tone dim — right register for the situation
  similarity    0.15  TF-IDF cosine to the human reference (deterministic anchor)
  semantic      0.00* BERTScore F1 if installed (*takes 0.10 from similarity+quality)
  facts         0.15  fraction of reference facts correctly carried over
  format        0.15  greeting + sign-off + sensible length
Plus a hallucinated-facts penalty (facts present in neither the incoming email
nor the reference) of up to -15 points.

If a ticket has no concrete facts, the 'facts' weight is redistributed rather
than faked. Every per-response record includes all signal values, the judge's
rationale, and the composite — so a reviewer can always answer "why this score?".

CLI:
  python3 -m src.evaluate --in outputs/generated.jsonl --out outputs/scored.jsonl
  python3 -m src.evaluate --provider mock ...
"""
from __future__ import annotations
import argparse, json, os, re
from .llm import LLM
from .retrieval import cosine
from .metrics import rouge_l, bleu, bertscore, bertscore_available

FACT_RE = re.compile(r"#\d{3,}|\$\d[\d,]*|[\w.+-]+@[\w.-]+\.\w+|GB\d+|\b\d{4,}\b")

# Weights for the composite. BERTScore only enters when the package is present.
WEIGHTS = {"quality": 0.40, "intent_match": 0.10, "tone_match": 0.05,
           "similarity": 0.15, "facts": 0.15, "format": 0.15}
WEIGHTS_WITH_BERT = {"quality": 0.35, "intent_match": 0.10, "tone_match": 0.05,
                     "similarity": 0.10, "semantic": 0.10, "facts": 0.15,
                     "format": 0.15}

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


def score_response(incoming, reference, candidate, llm: LLM,
                   use_bertscore: bool | None = None) -> dict:
    if use_bertscore is None:
        use_bertscore = bertscore_available()

    # 1. LLM judge (or deterministic mock)
    judge = llm.complete_json(JUDGE_SYSTEM, _judge_prompt(incoming, reference, candidate))
    dims = ["relevance", "correctness", "completeness", "tone", "actionability"]
    dim_scores = {}
    for d in dims:
        try:
            dim_scores[d] = max(1, min(5, int(round(float(judge.get(d, 3))))))
        except (TypeError, ValueError):
            dim_scores[d] = 3
    to01 = lambda v: (v - 1) / 4.0
    quality = sum(to01(dim_scores[d]) for d in dims) / len(dims)
    intent_match = to01(dim_scores["relevance"])   # does it address the request?
    tone_match = to01(dim_scores["tone"])          # right register?

    # 2. reference-similarity signals
    similarity = cosine(reference, candidate)
    semantic = bertscore(reference, candidate) if use_bertscore else None
    if semantic is not None:
        semantic = max(0.0, min(1.0, semantic))    # rescaled BERTScore can dip <0

    # 3. fact carryover + hallucination flag
    ref_facts = facts_in(reference)
    cand_facts = facts_in(candidate)
    in_facts = facts_in(incoming)
    facts_score = (len(ref_facts & cand_facts) / len(ref_facts)) if ref_facts else None
    hallucinated = sorted(cand_facts - ref_facts - in_facts)
    hallucination_penalty = min(0.15, 0.05 * len(hallucinated))

    # 4. format guardrails
    fmt = _format_ok(candidate)

    # 5. diagnostics (computed + reported, ZERO weight — see module docstring)
    diag_rouge = rouge_l(reference, candidate)
    diag_bleu = bleu(reference, candidate)

    # ---- composite with weight redistribution for missing signals ----
    weights = WEIGHTS_WITH_BERT if semantic is not None else WEIGHTS
    signals = {"quality": quality, "intent_match": intent_match,
               "tone_match": tone_match, "similarity": similarity,
               "facts": facts_score, "format": fmt}
    if semantic is not None:
        signals["semantic"] = semantic
    active = {k: v for k, v in signals.items() if v is not None}
    total_w = sum(weights[k] for k in active)
    composite = sum(weights[k] * active[k] for k in active) / total_w
    composite = max(0.0, composite - hallucination_penalty)

    explanation = _explain(dim_scores, signals, hallucinated, judge.get("rationale", ""))

    return {
        "composite": round(100 * composite, 1),
        "dimensions": dim_scores,
        "signals": {
            "quality": round(quality, 3),
            "intent_match": round(intent_match, 3),
            "tone_match": round(tone_match, 3),
            "similarity": round(similarity, 3),
            "semantic_bertscore": semantic,
            "fact_carryover": None if facts_score is None else round(facts_score, 3),
            "format": fmt,
        },
        "diagnostics": {"rouge_l": diag_rouge, "bleu": diag_bleu},
        "hallucinated_facts": hallucinated,
        "hallucination_penalty": round(hallucination_penalty, 3),
        "rationale": judge.get("rationale", ""),
        "explanation": explanation,
    }


def _explain(dims, signals, hallucinated, judge_rationale) -> str:
    """One human-readable sentence a support lead could act on."""
    parts = []
    weak = [d for d, v in dims.items() if v <= 2]
    strong = [d for d, v in dims.items() if v >= 4]
    if strong:
        parts.append("strong " + ", ".join(strong))
    if weak:
        parts.append("weak " + ", ".join(weak))
    fc = signals.get("facts")
    if fc is not None and fc < 1.0:
        parts.append(f"carries only {int(round(fc*100))}% of the reference's concrete facts")
    if hallucinated:
        parts.append(f"HALLUCINATED specifics: {', '.join(hallucinated[:3])}")
    summary = "; ".join(parts) if parts else "solid across all dimensions"
    return (summary + (f" — judge: {judge_rationale}" if judge_rationale else "")).strip()


def summarize(scored: list[dict]) -> dict:
    comps = [s["result"]["composite"] for s in scored]
    n = len(comps)
    mean = sum(comps) / n if n else 0.0
    dims = ["relevance", "correctness", "completeness", "tone", "actionability"]
    dim_means = {d: round(sum(s["result"]["dimensions"][d] for s in scored) / n, 2)
                 for d in dims} if n else {}
    cats = {}
    for s in scored:
        cats.setdefault(s.get("category", "?"), []).append(s["result"]["composite"])
    cat_means = {c: round(sum(v) / len(v), 1) for c, v in cats.items()}
    halluc = sum(1 for s in scored if s["result"]["hallucinated_facts"])
    diag = {}
    if n:
        diag = {"rouge_l_mean": round(sum(s["result"]["diagnostics"]["rouge_l"] for s in scored) / n, 3),
                "bleu_mean": round(sum(s["result"]["diagnostics"]["bleu"] for s in scored) / n, 3)}
    return {
        "n": n,
        "overall_score": round(mean, 1),
        "min": round(min(comps), 1) if comps else 0,
        "max": round(max(comps), 1) if comps else 0,
        "dimension_means_1to5": dim_means,
        "category_means": cat_means,
        "surface_metric_diagnostics": diag,
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
    use_bert = bertscore_available()
    print(f"[evaluate] provider={llm.provider}  bertscore={'ON' if use_bert else 'off (optional)'}  "
          f"scoring {len(rows)} responses")

    scored = []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for i, r in enumerate(rows, 1):
            res = score_response(r["incoming"], r["reference"], r["generated"], llm,
                                 use_bertscore=use_bert)
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
