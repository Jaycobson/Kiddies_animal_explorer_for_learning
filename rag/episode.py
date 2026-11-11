"""Plan an episode: resolve animals -> retrieve evidence -> write -> verify -> assemble scenes with sources."""
from .bible import build_image_prompt
from .ingest import split_sentences
from .request import resolve_animals
from .retriever import HybridIndex
from .verifier import verify_claims
from .writer import gather_evidence


def _sources(index: HybridIndex, cite_ids: list[str]) -> list[dict]:
    out, seen = [], set()
    for cid in cite_ids:
        c = index.by_id.get(cid)
        if c and (c.source, c.title) not in seen:
            seen.add((c.source, c.title))
            out.append({"id": cid, "title": c.title, "source": c.source, "url": c.url})
    return out


def _look(ev) -> str:
    for c in ev.chunks:
        if c.look:
            return c.look
    for c in ev.by_aspect.get("look", []):
        s = split_sentences(c.text)
        if s:
            return s[0]
    return ""


def plan_episode(prompt: str, index: HybridIndex, writer, bible: dict, judge=None, max_animals: int = 3,
                 facts_per_animal: int = 3, quiz: bool = True, age_band: str = "5-7") -> dict:
    res = resolve_animals(prompt, index, max_animals)
    trace, rejected, skipped, animals_meta, groups = [], [], [], {}, []
    total = ok = 0

    for animal in res.animals:
        ev, tr = gather_evidence(index, animal)
        trace += tr
        try:
            script = writer.write(animal, ev, facts_per_animal, quiz, age_band)
        except Exception as e:
            skipped.append({"animal": animal, "reason": f"writer failed: {e}"})
            continue
        look = _look(ev)
        letter = animal[:1].upper()
        facts = []
        for f in script["facts"]:
            verdicts = verify_claims(f["claims"], index.by_id, judge)
            kept = []
            for c, v in zip(f["claims"], verdicts):
                total += 1
                if v.supported:
                    ok += 1
                    kept.append({**c, "recall": round(v.recall, 2)})
                else:
                    rejected.append({"animal": animal, "text": c["text"], "cite": c["cite"], "reason": v.reason})
            if kept:
                cites = sorted({i for c in kept for i in c["cite"]})
                facts.append({"kind": "fact", "animal": animal, "letter": letter,
                              "text": " ".join(c["text"] for c in kept), "claims": kept, "cites": cites,
                              "sources": _sources(index, cites),
                              "prompt": build_image_prompt(f["visual"] or f"A friendly {animal}", animal, look, bible)})
        if not facts:
            skipped.append({"animal": animal, "reason": "no claim could be supported by the documents (abstained)"})
            continue
        quiz_scene = None
        q = script.get("quiz")
        if q:
            v = verify_claims([q["answer"]], index.by_id, judge)[0]
            total += 1
            if v.supported:
                ok += 1
                cites = sorted(set(q["answer"]["cite"]))
                quiz_scene = {"kind": "quiz", "animal": animal, "letter": letter, "question": q["question"],
                              "answer": q["answer"]["text"], "cites": cites, "sources": _sources(index, cites),
                              "text": f"Quick quiz time! {q['question']} ... Here is the answer. {q['answer']['text']}"}
            else:
                rejected.append({"animal": animal, "text": q["answer"]["text"], "cite": q["answer"]["cite"],
                                 "reason": "quiz answer: " + v.reason})
        card = {"kind": "card", "animal": animal, "letter": letter, "text": f"{letter} is for {animal}!",
                "cites": [], "sources": []}
        animals_meta[animal] = {"letter": letter, "look": look}
        groups.append([card, *facts] + ([quiz_scene] if quiz_scene else []))

    scenes = [s for g in groups for s in g]
    if scenes:
        scenes.append({"kind": "outro", "animal": "", "letter": "", "cites": [], "sources": [],
                       "text": "That is all for today, explorers! Which animal was your favourite?"})
    for i, s in enumerate(scenes, start=1):
        s["index"], s["include"] = i, True
    names = list(animals_meta)
    return {"title": "Animal Explorers: " + ", ".join(names) if names else "Animal Explorers",
            "request": prompt, "age_band": age_band, "resolution": res.notes, "style": bible["style"],
            "animals": animals_meta, "scenes": scenes, "skipped": skipped, "trace": trace,
            "report": {"claims": total, "supported": ok, "faithfulness": round(ok / total, 3) if total else 1.0,
                       "rejected": rejected}}
