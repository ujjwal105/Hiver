"""
generate.py — Suggested-reply generator (Gen-AI, grounded in the dataset via RAG).

Approach & trade-offs (see README for the long version):
  - RAG + few-shot prompting over the dataset. We retrieve the k most similar
    *past* incoming emails and show the LLM how a human agent replied to them,
    then ask it to reply to the new email in that voice.
  - Why RAG over fine-tuning: with a small, evolving dataset, retrieval adapts
    instantly as new tickets are added, needs no training run, and keeps the
    agent's real style/policies in-context. Fine-tuning would be premature here
    and harder to audit. Few-shot alone (no retrieval) wastes context on
    irrelevant examples; retrieval keeps exemplars on-topic.
  - The retriever excludes the email's own record when generating for a training
    item, so we never leak the answer.

CLI:
  python -m src.generate --input "my incoming email text"      # single, prints reply
  python -m src.generate --split test --out outputs/generated.jsonl   # batch
  python -m src.generate --provider mock ...                    # force offline
"""
from __future__ import annotations
import argparse, json, os
from .llm import LLM
from .retrieval import make_index

SYSTEM = (
    "You are an experienced customer-support agent working from a shared team "
    "inbox. You write suggested replies that a human agent can send with minimal "
    "editing. Be warm, specific, and concise. Address the customer's actual "
    "request, carry over concrete details (order numbers, amounts, emails) "
    "correctly, take clear ownership, and give a concrete next step. Never invent "
    "facts, policies, or promises you can't see support for. Match the tone and "
    "structure of the example replies provided."
)


def load_dataset(path="data/emails.jsonl"):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class Generator:
    def __init__(self, dataset, provider=None, k=3, retriever="auto"):
        self.dataset = dataset
        self.train = [r for r in dataset if r["split"] == "train"]
        self.index, self.retriever_backend = make_index(
            [r["incoming"] for r in self.train], backend=retriever)
        self.k = k
        self.llm = LLM(provider=provider)

    def _build_user_prompt(self, incoming, exemplars):
        parts = ["Here are examples of past customer emails and the replies our "
                 "team sent. Use them as a guide for tone and structure.\n"]
        for i, ex in enumerate(exemplars, 1):
            parts.append(f"--- EXAMPLE {i} ---")
            parts.append(f"Past email: {ex['incoming']}")
            parts.append(f"Reply sent: {ex['reference']}\n")
        parts.append("--- NOW WRITE THE REPLY ---")
        parts.append("Write the suggested reply for this new email. Output ONLY "
                     "the reply text (greeting through sign-off), nothing else.\n")
        parts.append(f"INCOMING EMAIL:\n{incoming}")
        return "\n".join(parts)

    def generate(self, incoming, exclude_idx=None):
        hits = self.index.query(incoming, k=self.k, exclude_idx=exclude_idx)
        exemplars = [self.train[i] for i, _ in hits]
        user = self._build_user_prompt(incoming, exemplars)
        reply = self.llm.complete(SYSTEM, user)
        return {
            "reply": reply,
            "retrieved": [
                {"id": self.train[i]["id"], "category": self.train[i]["category"],
                 "score": round(s, 3)} for i, s in hits
            ],
            "provider": self.llm.provider,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="single incoming email text")
    ap.add_argument("--split", help="generate for all rows in this split (e.g. test)")
    ap.add_argument("--data", default="data/emails.jsonl")
    ap.add_argument("--out", default="outputs/generated.jsonl")
    ap.add_argument("--provider", default=None, choices=["anthropic", "mock"])
    ap.add_argument("--retriever", default="auto", choices=["auto", "tfidf", "embeddings"],
                    help="embeddings uses sentence-transformers+FAISS if installed")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    ds = load_dataset(args.data)
    gen = Generator(ds, provider=args.provider, k=args.k, retriever=args.retriever)
    print(f"[generate] provider={gen.llm.provider}  retriever={gen.retriever_backend}  "
          f"train={len(gen.train)}  k={args.k}")

    if args.input:
        res = gen.generate(args.input)
        print("\n=== SUGGESTED REPLY ===\n")
        print(res["reply"])
        print("\n[retrieved]", res["retrieved"])
        return

    split = args.split or "test"
    rows = [r for r in ds if r["split"] == split]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for i, r in enumerate(rows, 1):
            res = gen.generate(r["incoming"])
            out = {
                "id": r["id"], "category": r["category"], "subject": r["subject"],
                "incoming": r["incoming"], "reference": r["reference"],
                "generated": res["reply"], "retrieved": res["retrieved"],
                "provider": res["provider"],
            }
            f.write(json.dumps(out) + "\n")
            print(f"  [{i}/{len(rows)}] {r['category']:<15} id={r['id']}")
    print(f"[generate] wrote {len(rows)} suggested replies -> {args.out}")


if __name__ == "__main__":
    main()
