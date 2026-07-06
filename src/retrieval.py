"""
retrieval.py — TF-IDF retrieval over past emails (pure numpy, no heavy deps).

We ground generation in the dataset with RAG: for a new incoming email we find
the most similar *past incoming emails* and feed those (email, reply) pairs to
the LLM as few-shot exemplars. TF-IDF cosine is a deliberate choice here:

  - It's transparent and dependency-light (no model download, runs offline).
  - For short support emails the vocabulary overlap signal is strong.
  - The retriever is pluggable — swap in dense embeddings for a larger corpus
    without touching the generator.

The same cosine similarity function is reused by the evaluator as one signal.
"""
from __future__ import annotations
import re, math
import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return _TOKEN.findall(text.lower())


class TfidfIndex:
    def __init__(self, docs: list[str]):
        self.docs = docs
        self.vocab: dict[str, int] = {}
        df: dict[str, int] = {}
        tokenized = [tokenize(d) for d in docs]
        for toks in tokenized:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        for t in df:
            self.vocab.setdefault(t, len(self.vocab))
        n = max(1, len(docs))
        self.idf = np.zeros(len(self.vocab))
        for t, i in self.vocab.items():
            self.idf[i] = math.log((1 + n) / (1 + df[t])) + 1.0
        self.matrix = np.vstack([self._vec(toks) for toks in tokenized]) if docs else np.zeros((0, len(self.vocab)))

    def _vec(self, toks: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        if not toks:
            return v
        for t in toks:
            j = self.vocab.get(t)
            if j is not None:
                v[j] += 1.0
        v = v / len(toks)          # term frequency (normalised)
        v = v * self.idf           # tf-idf
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def vector(self, text: str) -> np.ndarray:
        return self._vec(tokenize(text))

    def query(self, text: str, k: int = 3, exclude_idx: int | None = None):
        if self.matrix.shape[0] == 0:
            return []
        q = self.vector(text)
        sims = self.matrix @ q
        order = np.argsort(-sims)
        out = []
        for idx in order:
            if exclude_idx is not None and idx == exclude_idx:
                continue
            out.append((int(idx), float(sims[idx])))
            if len(out) >= k:
                break
        return out


def cosine(a_text: str, b_text: str) -> float:
    """Standalone TF-IDF cosine between two strings (fit on just the pair).

    Used by the evaluator as the reference-similarity signal. Fitting on the
    pair keeps it self-contained and comparable across responses.
    """
    idx = TfidfIndex([a_text, b_text])
    va = idx.matrix[0]
    vb = idx.matrix[1]
    return float(np.dot(va, vb))
