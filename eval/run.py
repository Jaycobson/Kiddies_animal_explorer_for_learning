"""Offline evaluation for the RAG layer.  Run:  python -m eval.run

1. Retrieval: recall@k and MRR on eval/gold.json for BM25-only, vector-only and hybrid (no animal filter,
   so retrieval has to find the right animal too).
2. Verifier probe: do true sentences from the corpus pass, and do corrupted ones (wrong number, or a sentence
   from a different animal's document) get caught?
"""
import json
import re
from pathlib import Path

from rag.ingest import split_sentences
from rag.retriever import HybridIndex, build_index
from rag.verifier import check_claim
from rag.writer import _swap_number

ROOT = Path(__file__).resolve().parent.parent


def retrieval_eval(index: HybridIndex, gold: list[dict], k: int = 3) -> dict:
    out = {}
    for mode in ("bm25", "vector", "hybrid"):
        hits_at_k, rr = 0, 0.0
        if not gold:
            continue
        for g in gold:
            res = index.search(g["q"], k=k, mode=mode)
            pos = next((i for i, h in enumerate(res) if g["answer_contains"].lower() in h.chunk.text.lower()), None)
            if pos is not None:
                hits_at_k += 1
                rr += 1 / (pos + 1)
        out[mode] = {f"recall@{k}": round(hits_at_k / len(gold), 3), "mrr": round(rr / len(gold), 3)}
    return out


def verifier_probe(index: HybridIndex) -> dict:
    true_ok = true_n = bad_caught = bad_n = cross_caught = cross_n = 0
    by_animal: dict[str, list] = {}
    for c in index.chunks:
        by_animal.setdefault(c.animal, []).append(c)
    animals = list(by_animal)
    for ai, animal in enumerate(animals):
        for c in by_animal[animal]:
            for s in split_sentences(c.text):
                if len(s.split()) < 5:
                    continue
                true_n += 1
                true_ok += check_claim(s, [c.text]).supported
                bad = _swap_number(s)
                if bad:
                    bad_n += 1
                    bad_caught += not check_claim(bad, [c.text]).supported
                other = by_animal[animals[(ai + 1) % len(animals)]][0]
                cross_n += 1
                cross_caught += not check_claim(s, [other.text]).supported
    return {"true_sentences_accepted": f"{true_ok}/{true_n}", "wrong_number_caught": f"{bad_caught}/{bad_n}",
            "wrong_source_caught": f"{cross_caught}/{cross_n}"}


def main():
    import os
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    key = os.getenv("OPENAI_API_KEY") if "--openai" in sys.argv else None
    index = build_index(ROOT / "corpus", api_key=key, cache_dir=None)
    gold = json.loads((ROOT / "eval" / "gold.json").read_text(encoding="utf-8"))
    print(f"{len(index.chunks)} chunks · {len(index.animals())} animals · embedder: {index.embedder.name}")
    print("(add --openai to use real semantic embeddings)\n" if key is None else "\n")
    for tier in ("direct", "paraphrase"):
        sub = [g for g in gold if g.get("tier", "direct") == tier]
        print(f"Retrieval, {tier} questions (n={len(sub)}, k=3):")
        for mode, m in retrieval_eval(index, sub).items():
            print(f"  {mode:<7} {m}")
    print("\nVerifier probe:", json.dumps(verifier_probe(index), indent=2))


if __name__ == "__main__":
    main()
