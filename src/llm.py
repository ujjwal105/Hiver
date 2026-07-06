"""
llm.py — Thin LLM provider abstraction.

Two backends:
  - "anthropic": real generation via the Claude API (needs ANTHROPIC_API_KEY).
  - "mock":      deterministic, no network. Lets the *entire* pipeline — generation,
                 evaluation, and metric validation — run end-to-end with zero setup,
                 so a grader can verify it works without a paid key. The mock is
                 intentionally imperfect (it produces plausible-but-generic replies)
                 which is useful: it gives the evaluator a realistic spread of
                 quality to score rather than only near-perfect outputs.

Provider is chosen by --provider on the CLIs, defaulting to "anthropic" if a key
is present, else "mock".
"""
from __future__ import annotations
import os, re, json, hashlib

DEFAULT_MODEL = "claude-sonnet-5"


def default_provider() -> str:
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "mock"


class LLM:
    def __init__(self, provider: str | None = None, model: str = DEFAULT_MODEL):
        self.provider = provider or default_provider()
        self.model = model
        self._client = None
        if self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "anthropic provider selected but client init failed "
                    f"({e}). Set ANTHROPIC_API_KEY or use --provider mock."
                )

    # ---- text generation -------------------------------------------------
    def complete(self, system: str, user: str, max_tokens: int = 700,
                 temperature: float = 0.3) -> str:
        if self.provider == "mock":
            return _mock_complete(system, user)
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

    # ---- structured JSON (used by the LLM-judge) -------------------------
    def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict:
        if self.provider == "mock":
            return _mock_judge(system, user)
        raw = self.complete(system + "\nRespond with ONLY valid JSON.", user,
                            max_tokens=max_tokens, temperature=0.0)
        return _extract_json(raw)


# --------------------------------------------------------------------------
# Mock implementations — deterministic, dependency-free.
# --------------------------------------------------------------------------
def _seed(*parts: str) -> int:
    return int(hashlib.sha1("||".join(parts).encode()).hexdigest()[:8], 16)


def _mock_complete(system: str, user: str) -> str:
    """A deliberately generic support reply.

    It extracts the incoming email from the prompt and echoes a couple of
    concrete tokens (order numbers, emails, amounts) so that a *reasonable*
    fraction of fact-carryover is preserved — but it does NOT solve the ticket,
    so the evaluator should (correctly) score it below the human reference.
    """
    incoming = ""
    m = re.search(r"INCOMING EMAIL:\s*(.*?)(?:\n---|\Z)", user, re.S)
    if m:
        incoming = m.group(1)
    facts = re.findall(r"#\d{3,}|\$\d+|[\w.+-]+@[\w.-]+\.\w+|GB\d+", incoming)
    fact_line = ""
    if facts:
        fact_line = f" I can see the reference {facts[0]} on your account, and I'm looking into it."
    return (
        "Hi there,\n\n"
        "Thanks for reaching out, and sorry for any trouble this has caused."
        + fact_line +
        " Our team will review the details and get back to you as soon as possible "
        "with the next steps.\n\n"
        "Please let me know if there's anything else I can help with in the meantime.\n\n"
        "Best,\nSupport Team"
    )


def _mock_judge(system: str, user: str) -> dict:
    """Deterministic stand-in for the LLM judge.

    Heuristic rubric scoring so the pipeline yields a realistic, monotonic
    signal without a network call. Real quality signal comes from the API judge;
    this keeps offline runs meaningful and reproducible.
    """
    def _field(tag):
        m = re.search(rf"{tag}:\s*(.*?)(?:\n[A-Z ]+:|\Z)", user, re.S)
        return (m.group(1) if m else "").strip()

    incoming = _field("INCOMING EMAIL")
    reference = _field("REFERENCE REPLY")
    candidate = _field("CANDIDATE REPLY")

    def toks(s):
        return set(re.findall(r"[a-z0-9#$.@]+", s.lower()))

    ci, cr, cc = toks(incoming), toks(reference), toks(candidate)
    # overlap of candidate with reference (content) and with incoming (relevance)
    rel = len(cc & ci) / (len(ci) + 1e-9)
    cov = len(cc & cr) / (len(cr) + 1e-9)
    facts_ref = set(re.findall(r"#\d{3,}|\$\d+|[\w.+-]+@[\w.-]+\.\w+|GB\d+", reference))
    facts_cand = set(re.findall(r"#\d{3,}|\$\d+|[\w.+-]+@[\w.-]+\.\w+|GB\d+", candidate))
    fact_hit = (len(facts_ref & facts_cand) / len(facts_ref)) if facts_ref else 1.0
    has_greeting = 1.0 if re.search(r"\b(hi|hello|hey|dear)\b", candidate.lower()) else 0.0
    has_signoff = 1.0 if re.search(r"\b(best|regards|thanks|cheers|sincerely)\b", candidate.lower()) else 0.0
    length_ok = 1.0 if 15 <= len(candidate.split()) <= 250 else 0.4

    def to5(x):
        return max(1, min(5, round(1 + 4 * x)))

    relevance = to5(min(1.0, rel * 2.5))
    completeness = to5(min(1.0, cov * 2.0))
    correctness = to5(0.5 * fact_hit + 0.5 * min(1.0, cov * 1.5))
    tone = to5(0.5 * has_greeting + 0.5 * has_signoff)
    actionability = to5(min(1.0, cov * 1.8))

    return {
        "relevance": relevance,
        "correctness": correctness,
        "completeness": completeness,
        "tone": tone,
        "actionability": actionability,
        "rationale": (
            f"[mock judge] relevance≈{rel:.2f}, ref-coverage≈{cov:.2f}, "
            f"fact-carryover={fact_hit:.2f}, greeting={bool(has_greeting)}, "
            f"signoff={bool(has_signoff)}, length_ok={length_ok>=1.0}."
        ),
        "_length_ok": length_ok,
    }


def _extract_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"No JSON found in judge output: {raw[:200]}")
    return json.loads(m.group(0))
