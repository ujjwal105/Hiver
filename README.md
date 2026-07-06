# AI Suggested-Response System — with a metric you can trust

Given an incoming customer email, this system generates a **suggested reply**
grounded in a dataset of past emails and their sent replies — and, more
importantly, **measures how good each generated reply actually is, and proves
that measurement reflects real quality.**

The evaluation system is the centre of gravity here, so most of this README is
about *why the accuracy metric is right*, not just what it outputs.

```
 Incoming Email
      │
      ▼
 Retriever  (TF-IDF default; sentence-transformers + FAISS auto-enabled if installed)
      │
      ▼
 LLM Generator  (RAG few-shot; Claude, or deterministic mock offline)
      │
      ▼
 Generated Reply
      │
      ▼
 Evaluation Engine
  ├── LLM-as-a-Judge   (5-dim rubric + written rationale)      weighted
  ├── Intent Match     (does it address the actual request?)   weighted
  ├── Tone Match       (right register for the situation?)     weighted
  ├── Cosine Similarity to reference (TF-IDF)                  weighted
  ├── BERTScore        (optional — auto-enabled if installed)  weighted
  ├── Fact Carryover   (order #s, $, emails, IDs)              weighted
  ├── Format Guardrails (greeting / sign-off / length)         weighted
  ├── ROUGE-L                                                  diagnostic (0 wt)
  └── BLEU                                                     diagnostic (0 wt)
      │
      ▼
 Final Weighted Score (0–100) + human-readable explanation per response
      │
      ▼
 Metric-Validation Harness — proves the score tracks real quality
```

---

## TL;DR — how to run

```bash
pip install -r requirements.txt          # numpy is enough for offline mode

# Runs the WHOLE pipeline end-to-end. No API key needed — falls back to a
# deterministic "mock" provider so you can verify it works immediately.
./run_all.sh mock

# With a real LLM (better replies + a real LLM judge):
export ANTHROPIC_API_KEY=sk-...
./run_all.sh                              # auto-detects the key
```

Outputs land in `outputs/`:
- `generated.jsonl` — suggested replies + which past emails were retrieved
- `scored.jsonl` — **per-response** scores with a written rationale
- `report.json` — **overall** system score, per-dimension and per-category
- `metric_validation.json` — evidence the metric measures quality (see below)

Individual steps:
```bash
python3 -m src.dataset --augment                        # build data/emails.jsonl
python3 -m src.generate --input "I was charged twice, order #48213"   # one email
python3 -m src.generate --split test                    # batch over held-out set
python3 -m src.evaluate --in outputs/generated.jsonl    # score them
python3 -m src.validate_metric                          # validate the metric itself
```

---

## 1. The dataset

**What it is:** a hand-authored set of B2B-SaaS shared-inbox support tickets —
the exact setting Hiver operates in (customer email run by a team inside Gmail).
Each record is `{incoming email → the reply a human agent actually sent}` across
13 categories: billing, refunds, technical bugs, how-to, account/access, feature
requests, complaints/escalations, onboarding, privacy/GDPR, ambiguous "needs
clarification" tickets, and order issues.

**Where it came from & why hand-authored** (honesty matters here):
- Public support corpora (Ubuntu dialogue, MultiWOZ, customer-service tweets)
  are chat-shaped, single-domain, or lack a clean *"incoming → the reply we
  actually sent"* pairing. Our evaluation is reference-aware, so we need that
  exact pairing.
- Hand-authoring lets us **control coverage and difficulty**: a spread of tones
  (angry, confused, terse "nothing works pls help"), and tickets that carry
  **concrete facts** (order numbers, amounts, emails, VAT IDs) — because getting
  those right is a make-or-break for support quality and something we
  specifically want to test.
- A small parametrised **augmentation** step (`--augment`) adds
  same-shape/different-details variants for volume, clearly labelled
  `source: synthetic_augmented` so dataset size is never overstated.

**Honest limitations:** it's small (~24 curated + ~12 augmented) and reflects
*shape and difficulty*, not real traffic distribution. That's fine for this
exercise — and the pipeline reads a plain JSONL schema, so a real exported inbox
drops in unchanged. Splits are an 80/20 train/test made by a **deterministic
hash** of the email text, so results are reproducible and the generator never
retrieves the item it's answering.

Schema: `{id, category, subject, incoming, reference, split, source}`.

---

## 2. The response generator (Gen-AI, grounded in the data)

**Approach: RAG + few-shot prompting.** For a new email we retrieve the *k* most
similar past incoming emails (TF-IDF cosine) and show the LLM those
`(email, reply)` pairs as exemplars, then ask it to reply in that voice.

**Why this, and the trade-offs:**
| Choice | Why | Trade-off accepted |
|---|---|---|
| **RAG over fine-tuning** | Small, evolving dataset; retrieval adapts instantly as tickets are added, no training run, and keeps the team's real style/policy in-context and auditable. | Prompt is longer; retrieval quality gates output quality. |
| **RAG over plain few-shot** | Fixed few-shot wastes context on irrelevant examples; retrieval keeps exemplars on-topic. | Needs an index (cheap here). |
| **TF-IDF retriever by default, FAISS optional** | Transparent, dependency-light, runs offline; strong for short support text. Dense retrieval (sentence-transformers + FAISS) **auto-enables if installed** (`pip install sentence-transformers faiss-cpu`, or force with `--retriever embeddings`). | On tens of emails TF-IDF is competitive; on a real inbox (10k+ threads, heavy paraphrase) dense wins — hence identical interfaces, swap without touching the generator. We don't hard-require it: ~500MB model download would break offline-runnability. |

The generation prompt explicitly instructs: address the actual request, carry
concrete facts over correctly, take ownership, give a next step, and **never
invent facts or policies**. Provider is abstracted (`anthropic` | `mock`) so the
generator runs with or without a key.

---

## 3. Measuring accuracy — the core

### What "accurate" means for a suggested reply
Exact match is the wrong target. Two replies can be equally good while sharing
almost no words, so **surface-overlap metrics (BLEU/ROUGE/exact-match) are too
strict** — they punish valid paraphrases and reward copying. What actually makes
a support reply good is: it resolves the real request, it's grounded (no
invented facts/policies), the concrete details are correct, and the tone fits.

So we score with **complementary signals**, so no single failure mode dominates:

| Signal | What it captures | How | Weight |
|---|---|---|---|
| **quality** | Semantic correctness a string metric can't see | **LLM-as-judge** rubric: relevance, correctness, completeness, tone, actionability — each 1–5 **with a written rationale**; judge sees the human reference as gold | 0.40 |
| **intent match** | Does it address the actual request? (first thing a support lead checks) | judge's relevance dimension, surfaced as its own signal | 0.10 |
| **tone match** | Right register for the situation (apology vs. how-to vs. escalation) | judge's tone dimension, surfaced as its own signal | 0.05 |
| **similarity** | "How close to what a human actually sent" | TF-IDF cosine to the reference — an independent, deterministic anchor | 0.15 |
| **semantic (BERTScore)** | Embedding-level similarity, robust to paraphrase | optional: auto-enabled if `bert-score` is installed (≈1.4GB model download — not hard-required so the repo stays offline-runnable); takes 0.10 of weight from quality+similarity | 0.10* |
| **facts** | Support lives or dies on concrete details | fact-carryover: fraction of reference facts (order #s, $, emails, IDs) correctly reproduced | 0.15 |
| **format** | Is it a sendable email at all | deterministic guardrails: greeting + sign-off present, sensible length | 0.15 |
| **ROUGE-L / BLEU** | classic surface-overlap metrics | computed & reported for every response — **zero weight**, see the ablation below for why | 0.00 |

Plus a **hallucinated-facts flag**: concrete facts in the candidate that appear
in *neither* the incoming email *nor* the reference are a safety red-flag and
apply a penalty even if everything else scores well.

`composite = weighted sum, rescaled to 0–100`. If a ticket has no concrete facts,
the `facts` weight is **redistributed** across the other signals rather than
faked to 1.0.

**Why blend instead of using the LLM judge alone?** An LLM judge is the best
single signal for *meaning*, but it's non-deterministic, can be gamed by fluent
nonsense, and is unreliable on exact facts/format. The cheap deterministic
signals (similarity, fact-carryover, format) anchor it and catch its blind spots.
The blend is more robust *and* auditable than any one part.

**Reporting.** Per response: composite + all five rubric dimensions + each signal
+ the rationale (`scored.jsonl`). Overall: mean score, min/max, per-dimension
means, per-category means, and hallucination count (`report.json`).

### Validating the metric — proving it's quality, not just a number
A metric you can't defend is worthless. `src/validate_metric.py` tests it three
independent ways, **all runnable offline** because each uses a *known* ground
truth about relative quality (no human panel required):

**1) Discrimination (the main test).** For each held-out ticket, take the human
reference (known GOOD) and build controlled degradations known to be WORSE:
`truncated` (incomplete), `generic` (boilerplate, no specifics), `off_topic`
(another ticket's reply), `fact_error` (reference with its facts swapped for
wrong ones). A valid metric **must** rank reference above each. We report
**pairwise ranking accuracy** — over every (good, bad) pair, how often the metric
orders them correctly. This is exactly the metric's AUC as a good/bad classifier:
`1.0` = perfect, `0.5` = chance. Reported overall **and per degradation type**,
which shows *what* the metric catches well vs. is blind to.

**1b) Signal ablation.** The same pairwise test run for **each signal alone** vs.
the composite — this justifies the weighting empirically instead of by assertion.

**2) Convergent validity.** The LLM judge and TF-IDF similarity are *independent*
mechanisms. If they correlate across responses, they're measuring a real shared
"quality" construct rather than one signal's artifact. We report Pearson &
Spearman.

**3) Sensitivity.** Progressively delete words from a reply and confirm the score
declines **monotonically** — a metric that ignores growing damage isn't measuring
quality.

**4) Paraphrase robustness — the case surface metrics fail by construction.**
Several dataset records carry a hand-authored `paraphrase`: an *equally good*
reply in completely different words (same facts, same resolution — used only by
the harness, never for training). A valid metric must score it nearly as high as
the reference and above every degradation. BLEU/ROUGE-L must crater, because the
word overlap is gone even though the quality isn't. This is the direct,
measurable form of "exact match is too strict".

**Results from the offline (`mock`) run in this repo:**
- Discrimination: **100% pairwise ranking accuracy** (reference ranked above all
  4 degradation types on every ticket). Separation is largest for `generic`
  (+61 pts) and smallest — as expected — for `fact_error` (+21 pts), correctly
  showing fact errors are the *subtlest* degradation to catch.
- Ablation caveat, reported honestly: on reference-*derived* degradations, even
  BLEU/ROUGE score 1.0 — corrupting the reference trivially reduces overlap. The
  degradation test alone can't separate the composite from surface metrics;
  that's exactly what the paraphrase test is for. ↓
- **Paraphrase robustness (the decisive test):** score retained on an
  equally-good rewording — **composite 89%**, judge 93%, fact-carryover 100%,
  format 100% … **ROUGE-L 39%, BLEU 13%**. And the paraphrase still ranks above
  100% of degradations. A BLEU/ROUGE-based accuracy system would have flagged
  perfectly good replies as failures; the composite doesn't.
- Convergent validity: **Pearson r ≈ 0.91**, Spearman ≈ 0.91 across 50 scored
  points — the two independent signals strongly agree.
- Sensitivity: **monotonic** (100 → 97 → 93 → 75 → 49 as 0→90% of words are
  dropped).

> These numbers come from the deterministic mock judge, so they're reproducible.
> With `ANTHROPIC_API_KEY` set, the same harness runs against a real LLM judge —
> the *validation methodology is the deliverable*; the mock guarantees it always
> produces evidence.

**What the overall score currently reports** (`report.json`, mock generator):
overall **≈ 61/100** — and that's the metric being *honest*. The mock generator
writes deliberately generic replies, and the evaluator correctly rates them
mid-range with **low actionability/completeness and high tone**, exactly the
weakness profile of a generic reply. A metric that gave those replies a high
score would be the real failure.

---

## Project layout
```
src/dataset.py          build/augment the dataset (the "script that generates it")
src/retrieval.py        TF-IDF index + cosine; optional FAISS/embeddings backend
src/metrics.py          ROUGE-L, BLEU (pure python); optional BERTScore
src/llm.py              provider abstraction: anthropic | deterministic mock
src/generate.py         RAG few-shot generator (CLI: single email or batch)
src/evaluate.py         the evaluation engine: per-response + overall scoring
src/validate_metric.py  proves the metric tracks real quality (4 tests + ablation)
run_all.sh              end-to-end pipeline
data/emails.jsonl       the dataset
outputs/                sample results committed as evidence
```

## Design choices worth calling out
- **Offline-first.** The whole pipeline runs with just numpy via a deterministic
  mock provider, so it's verifiable end-to-end with zero credentials — then
  upgrades to a real LLM by setting one env var.
- **Reproducible.** Deterministic splits, deterministic mock, no `random` seeds
  left to chance.
- **Pluggable.** Retriever and LLM provider are swappable without touching the
  generator or evaluator.

## How I used AI tools
This project was built in a timed session using **Claude Code** (Anthropic) as a
pair-programmer: I directed the architecture and evaluation design; Claude helped
scaffold the modules, author the synthetic dataset, and draft this README. All
design decisions — especially the multi-signal metric and the
discrimination/convergent-validity validation strategy — were reviewed and
chosen by me. The system also *uses* an LLM at runtime (Claude) for both
generation and the LLM-as-judge evaluation when an API key is provided.
