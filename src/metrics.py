"""
metrics.py — Deterministic text-similarity metrics (pure Python/numpy).

ROUGE-L and BLEU are included as *diagnostic* signals: they're the classic
reference-overlap metrics, but they punish valid paraphrase and reward copying,
so they get low/zero weight in the composite. We keep them because (a) they're
cheap and transparent, and (b) the validation harness uses them in an ABLATION:
we show the composite discriminates good/bad replies better than any surface
metric alone — which is the argument for the composite's design.

BERTScore is supported as an optional semantic signal: if the `bert-score`
package is installed it activates automatically; otherwise its weight is
redistributed. We don't hard-require it because it downloads a ~1.4GB model,
which would break the "runs anywhere, offline" property of this repo.
"""
from __future__ import annotations
import math, re
from collections import Counter

_WORD = re.compile(r"[a-z0-9#$@.]+")


def _toks(s: str) -> list[str]:
    return _WORD.findall((s or "").lower())


# ---------------- ROUGE-L (LCS-based F1) ------------------------------------
def rouge_l(reference: str, candidate: str) -> float:
    r, c = _toks(reference), _toks(candidate)
    if not r or not c:
        return 0.0
    # LCS length via DP (O(len(r)*len(c)); emails are short)
    dp = [0] * (len(c) + 1)
    for i in range(1, len(r) + 1):
        prev = 0
        for j in range(1, len(c) + 1):
            cur = dp[j]
            if r[i - 1] == c[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    lcs = dp[len(c)]
    prec = lcs / len(c)
    rec = lcs / len(r)
    return round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0


# ---------------- BLEU (smoothed, up to 4-grams, brevity penalty) ------------
def bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    r, c = _toks(reference), _toks(candidate)
    if not r or not c:
        return 0.0
    log_p_sum = 0.0
    for n in range(1, max_n + 1):
        r_ngrams = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        c_ngrams = Counter(tuple(c[i:i + n]) for i in range(len(c) - n + 1))
        overlap = sum(min(cnt, r_ngrams[g]) for g, cnt in c_ngrams.items())
        total = max(1, sum(c_ngrams.values()))
        p = (overlap + 1) / (total + 1)  # add-1 smoothing
        log_p_sum += math.log(p)
    geo_mean = math.exp(log_p_sum / max_n)
    bp = 1.0 if len(c) >= len(r) else math.exp(1 - len(r) / max(1, len(c)))
    return round(bp * geo_mean, 4)


# ---------------- BERTScore (optional) ---------------------------------------
_BERTSCORE = None  # cached scorer


def bertscore_available() -> bool:
    try:
        import bert_score  # noqa: F401
        return True
    except ImportError:
        return False


def bertscore(reference: str, candidate: str) -> float | None:
    """Semantic similarity via BERTScore F1, or None if package not installed."""
    global _BERTSCORE
    if not bertscore_available():
        return None
    from bert_score import BERTScorer
    if _BERTSCORE is None:
        _BERTSCORE = BERTScorer(lang="en", rescale_with_baseline=True)
    _, _, f1 = _BERTSCORE.score([candidate], [reference])
    return round(float(f1[0]), 4)
