"""
validate_metric.py — Does our accuracy metric actually measure quality?

A score is only trustworthy if we can show it tracks real quality rather than
being an arbitrary number. We validate it three independent ways, all runnable
offline (no human panel required), because each uses a *known* ground truth
about relative quality:

1) DISCRIMINATION (known-quality-ordering test)  --- the main one
   For each held-out ticket we take the human reference reply (known GOOD) and
   build controlled degradations whose quality we know is WORSE:
     - truncated : first sentence only            -> incomplete
     - generic   : boilerplate, no specifics      -> low correctness/actionability
     - off_topic : another ticket's reference     -> irrelevant
     - fact_error: reference with its facts swapped-> wrong concrete details
   A valid metric MUST score reference > each degradation. We measure the
   PAIRWISE RANKING ACCURACY: over every (good, degraded) pair, how often does
   the metric put them in the correct order? This is equivalent to the metric's
   AUC as a good/bad classifier. 1.0 = perfect discrimination; 0.5 = no better
   than chance. We report it overall and per degradation type (which tells us
   *what* the metric is good/blind at catching).

2) CONVERGENT VALIDITY
   Our composite blends signals; two of them — the LLM judge and TF-IDF
   similarity — are *independent* (different mechanisms). If they correlate
   across responses, the underlying construct ("quality") is real, not an
   artifact of one signal. We report Pearson & Spearman correlation.

3) SENSITIVITY
   Progressively corrupt a reply (drop an increasing fraction of its words) and
   confirm the score declines monotonically. A metric that ignores growing
   damage isn't measuring quality.

Run:  python -m src.validate_metric --provider mock
"""
from __future__ import annotations
import argparse, json, re
from .llm import LLM
from .evaluate import score_response, facts_in

FACT_RE = re.compile(r"#\d{3,}|\$\d[\d,]*|[\w.+-]+@[\w.-]+\.\w+|GB\d+")


def load(path="data/emails.jsonl", split="test"):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return [r for r in rows if r["split"] == split], rows


# ---------- degradation constructors (known to be WORSE than reference) ------
def deg_truncated(ref, other, incoming):
    first = re.split(r"(?<=[.!?])\s", ref.strip())[0]
    return first if len(first) < len(ref) else ref[: max(20, len(ref)//4)]


def deg_generic(ref, other, incoming):
    return ("Hi there,\n\nThank you for contacting us. We have received your "
            "message and will get back to you soon.\n\nBest,\nSupport")


def deg_off_topic(ref, other, incoming):
    return other  # a different ticket's reference reply


def deg_fact_error(ref, other, incoming):
    """Swap concrete facts in the reference for wrong ones."""
    def repl(m):
        s = m.group(0)
        if s.startswith("#"):
            return "#00000"
        if s.startswith("$"):
            return "$999"
        if "@" in s:
            return "wrong@example.com"
        if s.startswith("GB"):
            return "GB000000000"
        return s
    out = FACT_RE.sub(repl, ref)
    return out if out != ref else ref + "\n\nP.S. your order #00000 for $999 is noted."


DEGRADERS = {
    "truncated": deg_truncated,
    "generic": deg_generic,
    "off_topic": deg_off_topic,
    "fact_error": deg_fact_error,
}


# ---------- lightweight stats (no scipy) ------------------------------------
def pearson(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x)/n, sum(y)/n
    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    dx = sum((a-mx)**2 for a in x) ** 0.5
    dy = sum((b-my)**2 for b in y) ** 0.5
    return num/(dx*dy) if dx > 0 and dy > 0 else 0.0


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0]*len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    return pearson(ranks(x), ranks(y))


def run(provider=None, data="data/emails.jsonl"):
    llm = LLM(provider=provider)
    test, allrows = load(data)
    if len(test) < 2:
        # tiny datasets: fall back to using all rows so validation is meaningful
        test = allrows
    print(f"[validate] provider={llm.provider}  tickets={len(test)}")

    # score reference + each degradation for every ticket
    per_ticket = []
    q_signal, sim_signal = [], []  # for convergent validity (all scored items)

    def _signal_values(sc):
        """Flatten a score_response result into {signal_name: value} for ablation."""
        vals = {"composite": sc["composite"],
                "quality_judge": sc["signals"]["quality"],
                "similarity_tfidf": sc["signals"]["similarity"],
                "format": sc["signals"]["format"],
                "rouge_l": sc["diagnostics"]["rouge_l"],
                "bleu": sc["diagnostics"]["bleu"]}
        if sc["signals"]["fact_carryover"] is not None:
            vals["fact_carryover"] = sc["signals"]["fact_carryover"]
        return vals

    for i, r in enumerate(test):
        other = test[(i + 1) % len(test)]["reference"]  # neighbour = off-topic
        ref_score = score_response(r["incoming"], r["reference"], r["reference"], llm)
        entry = {"id": r["id"], "category": r["category"],
                 "reference": ref_score["composite"],
                 "reference_signals": _signal_values(ref_score), "degraded": {},
                 "degraded_signals": {}}
        q_signal.append(ref_score["signals"]["quality"])
        sim_signal.append(ref_score["signals"]["similarity"])
        for name, fn in DEGRADERS.items():
            cand = fn(r["reference"], other, r["incoming"])
            sc = score_response(r["incoming"], r["reference"], cand, llm)
            entry["degraded"][name] = sc["composite"]
            entry["degraded_signals"][name] = _signal_values(sc)
            q_signal.append(sc["signals"]["quality"])
            sim_signal.append(sc["signals"]["similarity"])
        per_ticket.append(entry)

    # 1) discrimination: pairwise ranking accuracy (reference vs each degradation)
    pair_total, pair_correct = 0, 0
    by_type = {k: {"n": 0, "correct": 0, "ref_mean": 0.0, "deg_mean": 0.0} for k in DEGRADERS}
    for e in per_ticket:
        for name in DEGRADERS:
            good, bad = e["reference"], e["degraded"][name]
            pair_total += 1
            by_type[name]["n"] += 1
            by_type[name]["ref_mean"] += good
            by_type[name]["deg_mean"] += bad
            if good > bad:
                pair_correct += 1
                by_type[name]["correct"] += 1
    for k in by_type:
        b = by_type[k]
        b["accuracy"] = round(b["correct"]/b["n"], 3) if b["n"] else 0.0
        b["ref_mean"] = round(b["ref_mean"]/b["n"], 1) if b["n"] else 0.0
        b["deg_mean"] = round(b["deg_mean"]/b["n"], 1) if b["n"] else 0.0
        b["separation"] = round(b["ref_mean"] - b["deg_mean"], 1)

    discrimination = round(pair_correct / pair_total, 3) if pair_total else 0.0

    # 1b) ABLATION: pairwise ranking accuracy of each signal ALONE vs composite.
    # This is the evidence for the composite's design: surface metrics
    # (BLEU/ROUGE-L) should underperform, justifying their zero weight.
    all_signal_names = set()
    for e in per_ticket:
        all_signal_names.update(e["reference_signals"].keys())
    ablation = {}
    for sig in sorted(all_signal_names):
        tot, corr = 0, 0
        for e in per_ticket:
            good = e["reference_signals"].get(sig)
            if good is None:
                continue
            for name in DEGRADERS:
                bad = e["degraded_signals"][name].get(sig)
                if bad is None:
                    continue
                tot += 1
                if good > bad:
                    corr += 1
        if tot:
            ablation[sig] = {"pairwise_ranking_accuracy": round(corr / tot, 3),
                             "pairs": tot}

    # 2) convergent validity
    conv = {"pearson": round(pearson(q_signal, sim_signal), 3),
            "spearman": round(spearman(q_signal, sim_signal), 3),
            "n_points": len(q_signal)}

    # 4) PARAPHRASE ROBUSTNESS — the case surface metrics fail by construction.
    # Some dataset records carry a hand-authored `paraphrase`: an equally-good
    # reply in completely different words (same facts, same resolution). A valid
    # metric must score it HIGH (close to the reference's own score) and above
    # every degradation; BLEU/ROUGE-L must drop sharply because word overlap is
    # gone. This is the direct evidence that "exact match is too strict" and
    # that the composite fixes it.
    para_rows = [r for r in allrows if r.get("paraphrase")]
    paraphrase = {"n": len(para_rows), "per_ticket": [], }
    tol_by_signal: dict[str, list[float]] = {}
    para_above_degraded = 0
    para_pairs = 0
    for i, r in enumerate(para_rows):
        other = para_rows[(i + 1) % len(para_rows)]["reference"]
        ref_sc = score_response(r["incoming"], r["reference"], r["reference"], llm)
        par_sc = score_response(r["incoming"], r["reference"], r["paraphrase"], llm)
        rv, pv = _signal_values(ref_sc), _signal_values(par_sc)
        # tolerance ratio per signal: paraphrase score / reference score (1.0 = robust)
        for sig in rv:
            if rv[sig] and rv[sig] > 0:
                tol_by_signal.setdefault(sig, []).append(pv.get(sig, 0.0) / rv[sig])
        # paraphrase must beat every degradation
        for name, fn in DEGRADERS.items():
            cand = fn(r["reference"], other, r["incoming"])
            deg_sc = score_response(r["incoming"], r["reference"], cand, llm)
            para_pairs += 1
            if par_sc["composite"] > deg_sc["composite"]:
                para_above_degraded += 1
        paraphrase["per_ticket"].append({
            "id": r["id"], "reference_score": ref_sc["composite"],
            "paraphrase_score": par_sc["composite"],
            "paraphrase_rouge_l": par_sc["diagnostics"]["rouge_l"],
            "paraphrase_bleu": par_sc["diagnostics"]["bleu"]})
    paraphrase["tolerance_by_signal"] = {
        s: round(sum(v)/len(v), 3) for s, v in sorted(tol_by_signal.items())}
    paraphrase["paraphrase_ranked_above_degradations"] = (
        round(para_above_degraded / para_pairs, 3) if para_pairs else None)
    paraphrase["interpretation"] = (
        "tolerance = signal(good paraphrase)/signal(reference); 1.0 means the "
        "signal treats an equally-good rewording as equally good. Surface "
        "metrics (bleu, rouge_l) should crater here — that is WHY they get zero "
        "weight in the composite.")

    # 3) sensitivity: drop increasing fraction of words from a sample reference
    sample = max(test, key=lambda r: len(r["reference"]))
    words = sample["reference"].split()
    curve = []
    for frac in [0.0, 0.25, 0.5, 0.75, 0.9]:
        keep = max(1, int(len(words) * (1 - frac)))
        cand = " ".join(words[:keep])
        sc = score_response(sample["incoming"], sample["reference"], cand, llm)
        curve.append({"dropped_frac": frac, "score": sc["composite"]})
    monotonic = all(curve[i]["score"] >= curve[i+1]["score"] - 1e-9
                    for i in range(len(curve)-1))

    report = {
        "provider": llm.provider,
        "tickets_evaluated": len(test),
        "1_discrimination": {
            "overall_pairwise_ranking_accuracy": discrimination,
            "interpretation": "1.0 = always ranks the good reply above the degraded one; 0.5 = chance.",
            "by_degradation_type": by_type,
        },
        "1b_signal_ablation": {
            "interpretation": ("pairwise ranking accuracy of each signal used ALONE. "
                               "Justifies the composite: surface metrics (bleu, rouge_l) "
                               "should rank near-duplicates of the reference above "
                               "genuinely-good paraphrases and miss fact errors."),
            "by_signal": ablation,
        },
        "2_convergent_validity": {
            **conv,
            "interpretation": "correlation between two independent signals (LLM judge vs TF-IDF similarity); higher = the score reflects a real shared 'quality' construct.",
        },
        "3_sensitivity": {
            "curve": curve,
            "monotonic_decline": monotonic,
            "interpretation": "score should fall as more of the reply is removed.",
        },
        "4_paraphrase_robustness": paraphrase,
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, choices=["anthropic", "mock"])
    ap.add_argument("--data", default="data/emails.jsonl")
    ap.add_argument("--out", default="outputs/metric_validation.json")
    args = ap.parse_args()
    rep = run(provider=args.provider, data=args.data)
    import os
    os.makedirs("outputs", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    print(f"\n[validate] -> {args.out}")
    da = rep["1_discrimination"]["overall_pairwise_ranking_accuracy"]
    print(f"\nHEADLINE: metric ranks good>bad {da*100:.0f}% of the time; "
          f"judge~similarity r={rep['2_convergent_validity']['pearson']}; "
          f"sensitivity monotonic={rep['3_sensitivity']['monotonic_decline']}")


if __name__ == "__main__":
    main()
