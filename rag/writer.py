"""Evidence gathering + script writers. Writers only ever see retrieved evidence, never the open web.

gather_evidence : aspect-driven retrieval (look / home / food / family / skill / fun) per animal
LLMWriter       : OpenAI writes kid-friendly claims, each citing chunk ids
ExtractiveWriter: offline fallback, builds scenes from the best sentences in the evidence (no API key needed)
Both return the same structure, which the verifier then checks:
  {"facts": [{"claims": [{"text", "cite": [ids]}], "visual": str}],
   "quiz":  {"question": str, "answer": {"text", "cite": [ids]}} | None}
"""
import json
import os
import re
from dataclasses import dataclass

from .ingest import Chunk, split_sentences
from .retriever import HybridIndex, tokenize

ASPECTS = [("look", "what it looks like: body, size, colour, skin, fur"),
           ("home", "where it lives: habitat, country, ocean, forest, savanna"),
           ("food", "what it eats and drinks"),
           ("family", "babies, young, born, mother, herd, group"),
           ("skill", "amazing ability, special body part, how it moves, behaviour"),
           ("fun", "surprising fun fact: how long it lives, how fast, how big")]
FACT_ORDER = ["skill", "food", "family", "home", "fun", "look"]
MODELS = [os.getenv("OPENAI_WRITER_MODEL"), os.getenv("OPENAI_PLANNER_MODEL"), "gpt-5.6", "gpt-4.1", "gpt-4o-mini"]


@dataclass
class Evidence:
    animal: str
    chunks: list[Chunk]
    by_aspect: dict[str, list[Chunk]]


def gather_evidence(index: HybridIndex, animal: str, k_per_aspect: int = 2, max_chunks: int = 10):
    by_aspect, trace, seen = {}, [], {}
    for key, desc in ASPECTS:
        q = f"{animal} {desc}"
        hits = index.search(q, k=k_per_aspect, animal=animal)
        by_aspect[key] = [h.chunk for h in hits]
        trace.append({"animal": animal, "aspect": key, "query": q,
                      "hits": [{"id": h.chunk.id, "score": round(h.score, 4), "bm25_rank": h.bm25_rank,
                                "vector_rank": h.vector_rank} for h in hits]})
        for h in hits:
            seen.setdefault(h.chunk.id, h.chunk)
    chunks = list(seen.values())[:max_chunks]
    ids = {c.id for c in chunks}
    by_aspect = {k: [c for c in v if c.id in ids] for k, v in by_aspect.items()}
    return Evidence(animal, chunks, by_aspect), trace


# ------------------------------------------------------------------ LLM writer
SYSTEM = """You write narration for a short educational video that teaches children about an animal.
You are given EVIDENCE chunks, each with an id. Use ONLY facts stated in the evidence. Never add facts from memory.
Reply with ONLY a JSON object:
{{"facts": [{{"claims": [{{"text": str, "cite": [chunk ids]}}], "visual": str}}],
  "quiz": {{"question": str, "answer": {{"text": str, "cite": [chunk ids]}}}}}}
Rules:
- Write exactly {n_facts} fact scenes, each with 1-2 claims. Each claim is ONE short sentence for children aged {age}.
- Every claim must cite the chunk id(s) that state it. Do not put a number in a claim unless the cited chunk has it.
- Keep a warm, friendly voice. Simple words. You may start with 'Did you know' or 'Look!'. No jokes that change facts.
- visual: one sentence describing the picture for that scene (what the animal is doing, the setting). No text or letters in the picture.
  You may include the young guide character, {mascot}, in some visuals.
- {quiz_rule}"""


class LLMWriter:
    def __init__(self, api_key=None, mascot_name="Ada", client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.client, self.mascot = client, mascot_name

    def write(self, animal: str, ev: Evidence, n_facts: int, quiz: bool, age_band: str) -> dict:
        quiz_rule = ("quiz: one simple question whose answer is a single fact in the evidence; answer.text restates it with a cite."
                     if quiz else 'quiz: set to null.')
        system = SYSTEM.format(n_facts=n_facts, age=age_band, mascot=self.mascot, quiz_rule=quiz_rule)
        evidence = "\n".join(f"[{c.id}] ({c.section}) {c.text}" for c in ev.chunks)
        last = None
        for model in [m for m in MODELS if m]:
            try:
                res = self.client.chat.completions.create(
                    model=model, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": f"Animal: {animal}\n\nEVIDENCE:\n{evidence}"}])
                return normalize_script(json.loads(res.choices[0].message.content), {c.id for c in ev.chunks}, n_facts)
            except Exception as e:
                last = e
        raise RuntimeError(f"Writer failed: {last}")


def normalize_script(data: dict, valid_ids: set[str], n_facts: int) -> dict:
    """Keep only well-formed claims with citations that point at chunks the writer was actually given."""
    def claim(c):
        text = str((c or {}).get("text", "")).strip()
        cite = [i for i in (c or {}).get("cite", []) if i in valid_ids]
        return {"text": text, "cite": cite} if text else None

    facts = []
    for f in (data.get("facts") or [])[:n_facts]:
        claims = [x for x in (claim(c) for c in f.get("claims") or []) if x]
        if claims:
            facts.append({"claims": claims, "visual": str(f.get("visual", "")).strip()})
    quiz = None
    q = data.get("quiz")
    if isinstance(q, dict) and str(q.get("question", "")).strip() and claim(q.get("answer")):
        quiz = {"question": str(q["question"]).strip(), "answer": claim(q["answer"])}
    return {"facts": facts, "quiz": quiz}


# ------------------------------------------------------------------ offline writer
LEADS = ["Did you know? ", "Look! ", "Here is something amazing. ", "Listen to this! "]
_SWAPS = {"two": "three", "three": "five", "four": "six", "five": "seven", "six": "nine", "seven": "ten",
          "eight": "six", "nine": "seven", "ten": "twelve", "twelve": "ten", "twenty": "thirty"}


def _swap_number(sentence: str) -> str | None:
    m = re.search(r"(?<![\d,.])(\d{1,2})(?![\d,.])", sentence)
    if m:
        return sentence[:m.start()] + str(int(m.group(1)) + 2) + sentence[m.end():]
    m = re.search(r"\b(" + "|".join(_SWAPS) + r")\b", sentence, re.I)
    if m:
        return sentence[:m.start()] + _SWAPS[m.group(1).lower()] + sentence[m.end():]
    return None


class ExtractiveWriter:
    """No LLM: pick the best sentence per aspect from the evidence. Plain, but guaranteed to be on-source."""

    def write(self, animal: str, ev: Evidence, n_facts: int, quiz: bool, age_band: str) -> dict:
        used, facts, picked = set(), [], []
        for aspect in FACT_ORDER:
            if len(facts) >= n_facts:
                break
            want = set(tokenize(dict(ASPECTS)[aspect]))
            best = None
            for c in ev.by_aspect.get(aspect, []):
                for s in split_sentences(c.text):
                    n = len(s.split())
                    if s in used or not 5 <= n <= 28 or re.match(r"(They|Their|These|Those|Both)\b", s):
                        continue   # skip sentences that would open with a pronoun the child cannot resolve
                    score = len(want & set(tokenize(s))) + (0.5 if re.search(r"\d", s) else 0)
                    score += 1.5 if animal.lower().split()[-1] in s.lower() else 0   # name the animal: no dangling "It"
                    if best is None or score > best[0]:
                        best = (score, s, c.id)
            if best:
                _, sentence, cid = best
                used.add(sentence)
                name = animal.lower()
                sentence = re.sub(r"^It\b(?!')", f"The {name}", sentence)   # "It has..." -> "The octopus has..."
                sentence = re.sub(r"^Its\b", f"The {name}'s", sentence)
                lead = LEADS[len(facts) % len(LEADS)]
                facts.append({"claims": [{"text": lead + sentence, "cite": [cid]}],
                              "visual": f"A friendly, colourful illustration of a {animal}. {sentence}"})
                picked.append((sentence, cid))
        quiz_out = None
        if quiz and picked:
            sentence, cid = max(picked, key=lambda p: bool(re.search(r"\d|\b(" + "|".join(_SWAPS) + r")\b", p[0])))
            variant = _swap_number(sentence) if len(animal) % 2 == 0 else None
            if variant:
                quiz_out = {"question": f"True or false: {variant}", "answer": {"text": f"False! {sentence}", "cite": [cid]}}
            else:
                quiz_out = {"question": f"True or false: {sentence}", "answer": {"text": f"True! {sentence}", "cite": [cid]}}
        return {"facts": facts, "quiz": quiz_out}
