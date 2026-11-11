import shutil
import subprocess
import tempfile
from pathlib import Path

from pipeline.cards import PlaceholderBackend
from pipeline.explorer import render_episode
from pipeline.narration import SilentNarrator
from rag.bible import load_bible
from rag.episode import plan_episode
from rag.ingest import chunk_text, load_corpus, parse_front_matter
from rag.request import resolve_animals
from rag.retriever import HashEmbedder, HybridIndex
from rag.verifier import check_claim, verify_claims
from rag.writer import ExtractiveWriter, LLMWriter, gather_evidence

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
BIBLE = load_bible(Path(__file__).resolve().parent.parent / "bible" / "bible.json")


def make_index():
    return HybridIndex.build(CORPUS, HashEmbedder(), cache_dir=None)


# ---------------------------------------------------------------- ingest
def test_front_matter_and_fallbacks():
    meta, body = parse_front_matter("---\nanimal: Owl\nlook: brown owl\n---\nText here.")
    assert meta == {"animal": "Owl", "look": "brown owl"} and body == "Text here."
    chunks = load_corpus(CORPUS)
    animals = {c.animal for c in chunks}
    assert {"Elephant", "Octopus", "Blue Whale"} <= animals
    assert "Penguin" in animals and "Zebra" in animals  # pdf and no-front-matter txt, animal from file name
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunking_keeps_sections_and_respects_size():
    text = "# A\nOne sentence here. " * 3 + "\n\n## B\n- bullet one\n- bullet two"
    out = chunk_text(text, max_chars=60)
    assert all(len(t) <= 80 for _, t in out)
    assert {s for s, _ in out} == {"B"} | ({"A"} if any(s == "A" for s, _ in out) else set())


# ---------------------------------------------------------------- retrieval
def test_retrieval_finds_right_animal_without_filter():
    idx = make_index()
    top = idx.search("how many hearts does it have and why is its blood blue", k=3)
    assert top[0].chunk.animal == "Octopus"
    top = idx.search("which animal can hop and has a baby in a pouch", k=3)
    assert top[0].chunk.animal == "Kangaroo"


def test_filters_and_modes():
    idx = make_index()
    assert all(h.chunk.animal == "Giraffe" for h in idx.search("food", k=5, animal="Giraffe"))
    assert all(h.chunk.letter == "E" for h in idx.search("big", k=5, letter="E"))
    for mode in ("bm25", "vector", "hybrid"):
        assert idx.search("zebra stripes", k=2, mode=mode)


def test_vector_cache_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        a = HybridIndex.build(CORPUS, HashEmbedder(), cache_dir=d)
        assert list(Path(d).glob("*.npy"))
        b = HybridIndex.build(CORPUS, HashEmbedder(), cache_dir=d)
        assert (a.vectors == b.vectors).all()


# ---------------------------------------------------------------- request resolution
def test_resolve_names_letters_and_discovery():
    idx = make_index()
    assert resolve_animals("a video about elephants and octopuses", idx).animals == ["Elephant", "Octopus"]
    assert resolve_animals("blue whales please", idx).animals == ["Blue Whale"]
    assert resolve_animals("animals that start with the letter K", idx).animals == ["Kangaroo"]
    assert resolve_animals("letters B and Z", idx).animals == ["Blue Whale", "Zebra"]
    assert len(resolve_animals("A to Z", idx, max_animals=4).animals) == 4
    found = resolve_animals("animals that live in the ocean", idx).animals
    assert {"Octopus", "Blue Whale"} & set(found)


# ---------------------------------------------------------------- verifier
def test_verifier_accepts_supported_and_rejects_hallucinations():
    idx = make_index()
    ev = [idx.by_id["octopus-02"].text]
    assert check_claim("An octopus has three hearts.", ev).supported
    assert not check_claim("An octopus has five hearts.", ev).supported          # wrong number
    assert not check_claim("An octopus can fly over mountains.", ev).supported   # invented fact
    cross = verify_claims([{"text": "Elephants have a very long trunk.", "cite": ["octopus-02"]}], idx.by_id)[0]
    assert not cross.supported                                                    # cited the wrong source
    assert not verify_claims([{"text": "Octopuses are clever.", "cite": []}], idx.by_id)[0].supported


# ---------------------------------------------------------------- planning
class FakeWriter:
    """Pretends to be an LLM that sneaks in two made-up claims."""

    def write(self, animal, ev, n_facts, quiz, age_band):
        c = ev.chunks[0]
        return {"facts": [
            {"claims": [{"text": "Look! " + c.text.split(".")[0] + ".", "cite": [c.id]}], "visual": f"{animal} outside"},
            {"claims": [{"text": f"A {animal} can live for 900 years.", "cite": [c.id]}], "visual": "x"},
            {"claims": [{"text": f"A {animal} loves pizza.", "cite": []}], "visual": "y"}], "quiz": None}


def test_plan_drops_unsupported_claims_and_reports_faithfulness():
    plan = plan_episode("elephants", make_index(), FakeWriter(), BIBLE, facts_per_animal=3)
    kinds = [s["kind"] for s in plan["scenes"]]
    assert kinds == ["card", "fact", "outro"]
    assert plan["report"]["claims"] == 3 and plan["report"]["supported"] == 1
    assert plan["report"]["faithfulness"] < 1 and len(plan["report"]["rejected"]) == 2
    assert plan["scenes"][1]["sources"] and plan["scenes"][1]["cites"]


def test_extractive_plan_is_fully_grounded_and_has_sources():
    plan = plan_episode("octopus and kangaroo", make_index(), ExtractiveWriter(), BIBLE, facts_per_animal=3)
    assert plan["report"]["faithfulness"] == 1.0
    assert [s["kind"] for s in plan["scenes"]].count("card") == 2
    for s in plan["scenes"]:
        if s["kind"] in ("fact", "quiz"):
            assert s["sources"], s
    assert "Octopus" in plan["animals"] and plan["animals"]["Octopus"]["look"]
    assert "Main subject: Octopus" in next(s["prompt"] for s in plan["scenes"] if s["kind"] == "fact")


def test_abstains_when_nothing_is_supported():
    class Liar:
        def write(self, *a, **k):
            return {"facts": [{"claims": [{"text": "It can teleport.", "cite": ["zebra-notes-01"]}], "visual": ""}], "quiz": None}

    plan = plan_episode("zebra", make_index(), Liar(), BIBLE)
    assert plan["scenes"] == [] and plan["skipped"][0]["animal"] == "Zebra"


def test_evidence_covers_aspects():
    ev, trace = gather_evidence(make_index(), "Elephant")
    assert len(trace) == 6 and ev.chunks and all(c.animal == "Elephant" for c in ev.chunks)


# ---------------------------------------------------------------- render (needs ffmpeg)
def test_render_end_to_end_offline():
    if not shutil.which("ffmpeg"):
        return
    plan = plan_episode("giraffe", make_index(), ExtractiveWriter(), BIBLE, facts_per_animal=2)
    with tempfile.TemporaryDirectory() as d:
        out = render_episode(plan, PlaceholderBackend(256), d, SilentNarrator(), video_size=(256, 256), card_size=256)
        assert Path(out["video"]).stat().st_size > 1000
        assert (Path(d) / "sources.md").read_text(encoding="utf-8").count("## Scene") == len(plan["scenes"])
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                    "default=nw=1:nk=1", out["video"]], capture_output=True, text=True).stdout)
        assert dur > 5


# ---------------------------------------------------------------- LLM writer with a fake OpenAI client
class _FakeClient:
    """Mimics client.chat.completions.create and records the prompt it was given."""

    def __init__(self, payload):
        import json
        self.payload, self.calls = json.dumps(payload), []
        outer = self

        class _C:
            @staticmethod
            def create(**kw):
                outer.calls.append(kw)
                msg = type("M", (), {"content": outer.payload})
                return type("R", (), {"choices": [type("Ch", (), {"message": msg})]})
        self.chat = type("Chat", (), {"completions": _C})


def test_llm_writer_prompt_and_citation_filtering():
    idx = make_index()
    ev, _ = gather_evidence(idx, "Octopus")
    good = ev.chunks[0].id
    fake = _FakeClient({"facts": [{"claims": [{"text": "An octopus has eight arms.", "cite": [good, "made-up-99"]}], "visual": "Ada meets an octopus"}],
                        "quiz": {"question": "How many arms?", "answer": {"text": "Eight!", "cite": ["made-up-99"]}}})
    script = LLMWriter(client=fake).write("Octopus", ev, 2, True, "5-7")
    assert script["facts"][0]["claims"][0]["cite"] == [good]   # invented chunk ids are stripped
    assert script["quiz"]["answer"]["cite"] == []              # kept here so the verifier can report it...
    assert not verify_claims([script["quiz"]["answer"]], idx.by_id)[0].supported  # ...and rejects it: no valid citation
    sent = fake.calls[0]["messages"]
    assert "ONLY facts stated in the evidence" in sent[0]["content"] and f"[{good}]" in sent[1]["content"]
    arms = next(c.id for c in ev.chunks if "eight arms" in c.text)
    ok = _FakeClient({"facts": [{"claims": [{"text": "Look! An octopus has eight arms covered with suckers.", "cite": [arms]}], "visual": "v"},
                                {"claims": [{"text": "Look! An octopus has eight arms covered with suckers.", "cite": [good]}], "visual": "w"}],
                      "quiz": None})
    plan = plan_episode("octopus", idx, LLMWriter(client=ok), BIBLE, facts_per_animal=2)
    assert [s["kind"] for s in plan["scenes"]] == ["card", "fact", "outro"]   # right citation kept, wrong citation dropped
    assert plan["report"]["rejected"][0]["cite"] == [good]
