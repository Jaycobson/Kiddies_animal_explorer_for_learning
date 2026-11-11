"""Turn a free-text request into a list of animals using the corpus itself.

Three ways to pick animals, tried in this order and combined:
  1. names mentioned in the prompt ("elephants and octopuses")
  2. letters ("letter P", "starting with B", "A to Z")
  3. topic discovery: if nothing above matches, retrieve over the corpus ("animals that live in the sea")
"""
import re
from dataclasses import dataclass, field

from .retriever import HybridIndex

_LET = r"((?:\b[A-Za-z]\b[\s,&]*(?:and\s+)?)+)"


@dataclass
class Resolution:
    animals: list[str]
    notes: list[str] = field(default_factory=list)


def _name_pattern(name: str) -> str:
    words = name.lower().split()
    forms = [name.lower()] + ([words[-1]] if len(words) > 1 else [])
    return "|".join(re.escape(f) for f in forms)


def find_letters(prompt: str, available: list[str]) -> list[str]:
    if re.search(r"\ba\s*(?:to|-|–|through)\s*z\b", prompt, re.I):
        return list(available)
    m = (re.search(r"\bletters?\s+" + _LET, prompt, re.I)
         or re.search(r"\bstart(?:s|ing)?\s+with\s+(?:the\s+)?(?:letters?\s+)?([A-Z])\b", prompt))
    if not m:
        return []
    got = sorted({x.upper() for x in re.findall(r"\b[A-Za-z]\b", m.group(1))})
    return [x for x in got if x in available]


def resolve_animals(prompt: str, index: HybridIndex, max_animals: int = 3) -> Resolution:
    notes, animals = [], []
    low = prompt.lower()
    named = sorted(((m.start(), a) for a in index.animals()
                    if (m := re.search(rf"\b(?:{_name_pattern(a)})(?:s|es)?\b", low))), key=lambda t: t[0])
    animals += [a for _, a in named]
    if named:
        notes.append(f"Named in your request: {', '.join(a for _, a in named)}")

    letters = find_letters(prompt, index.letters())
    if letters:
        by_letter = [a for L in letters for a in index.animals() if a[:1].upper() == L]
        notes.append(f"Letters {', '.join(letters)} -> {', '.join(by_letter) or 'no animals in the library'}")
        animals += by_letter

    if not animals:
        seen: list[str] = []
        for h in index.search(prompt, k=12):
            if h.chunk.animal not in seen:
                seen.append(h.chunk.animal)
        animals = seen
        notes.append(f"No names or letters found, so I searched the library for: {', '.join(seen) or 'nothing relevant'}")

    animals = list(dict.fromkeys(animals))
    if len(animals) > max_animals:
        notes.append(f"Limited to {max_animals} animals (of {len(animals)} matched) to keep the video short.")
        animals = animals[:max_animals]
    return Resolution(animals, notes)
