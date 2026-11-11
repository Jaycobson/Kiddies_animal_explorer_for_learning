"""Claim verifier: every sentence in the script must be supported by the chunks it cites.

Checks (all offline and deterministic):
  1. it has at least one valid citation
  2. every number in the claim also appears in the cited text (digits and number words are normalised)
  3. enough of the claim's content words appear in the cited text (default 60%)
Optional 4: an LLM judge (OpenAIJudge) must also agree the evidence entails the claim.

Known limits: this is a conservative lexical check, so it can reject a correct claim that is heavily reworded
or does arithmetic ("22 months" -> "almost two years"). It is designed to err on the side of dropping claims.
"""
import os
import re
from dataclasses import dataclass

from .retriever import tokenize

NUM_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30, "hundred": 100, "thousand": 1000}
# words we ignore when checking support: friendly framing, quiz words, and the ambiguous word "one"
KID_FILLER = set("""did know amazing wow look listen here something let meet hello friend friends guess quiz time quick
true false answer correct yes super great cool fun fact imagine really big little""".split()) | {"one"}


@dataclass
class Verdict:
    supported: bool
    recall: float
    reason: str


def numbers(text: str) -> set[float]:
    out = {float(m.replace(",", "")) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
    out |= {float(NUM_WORDS[w]) for w in re.findall(r"[a-z]+", text.lower()) if w in NUM_WORDS}
    return out


def content_tokens(text: str) -> list[str]:
    toks = {t for t in tokenize(text) if t not in KID_FILLER and t not in NUM_WORDS and not t.isdigit() and len(t) > 1}
    return sorted(toks)


def check_claim(text: str, evidence_texts: list[str], min_recall: float = 0.6) -> Verdict:
    claim = content_tokens(text)
    if not claim:
        return Verdict(False, 0.0, "nothing checkable in this sentence")
    ev = " ".join(evidence_texts)
    ev_tokens = set(content_tokens(ev))
    recall = sum(t in ev_tokens for t in claim) / len(claim)
    missing = numbers(text) - numbers(ev)
    if missing:
        return Verdict(False, recall, f"number(s) not in the sources: {sorted(missing)}")
    if recall < min_recall:
        return Verdict(False, recall, f"only {recall:.0%} of its key words appear in the cited sources")
    return Verdict(True, recall, "ok")


class OpenAIJudge:
    """Second opinion from an LLM: does the evidence fully support the claim?"""

    def __init__(self, api_key=None, model=None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")

    def __call__(self, claim: str, evidence: str) -> bool:
        res = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": "You are a strict fact checker. Answer only YES or NO: does the "
                       "EVIDENCE fully support every factual statement in the CLAIM? Extra friendly wording is fine."},
                      {"role": "user", "content": f"EVIDENCE:\n{evidence}\n\nCLAIM:\n{claim}"}])
        return res.choices[0].message.content.strip().upper().startswith("YES")


def verify_claims(claims: list[dict], chunks_by_id: dict, judge=None, min_recall: float = 0.6) -> list[Verdict]:
    """claims: [{"text": str, "cite": [chunk ids]}] -> one Verdict per claim."""
    out = []
    for c in claims:
        cites = [i for i in c.get("cite", []) if i in chunks_by_id]
        if not cites:
            out.append(Verdict(False, 0.0, "no valid citation"))
            continue
        texts = [chunks_by_id[i].text for i in cites]
        v = check_claim(c.get("text", ""), texts, min_recall)
        if v.supported and judge is not None:
            try:
                if not judge(c["text"], "\n".join(texts)):
                    v = Verdict(False, v.recall, "AI fact-checker did not agree the sources support it")
            except Exception:
                pass  # judge unavailable: keep the deterministic verdict
        out.append(v)
    return out
