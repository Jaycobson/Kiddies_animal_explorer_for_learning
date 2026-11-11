"""Render a verified episode plan into images, narration and an MP4 (with a sources.md citation file)."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cards import make_card, make_outro_card, source_label, stamp_sources
from .video import build_slideshow


def render_episode(plan, backend, out_dir, narrator, on_progress=None, video_size=(1024, 1024), card_size=1024):
    notify = on_progress or (lambda stage, done, total: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = [s for s in plan["scenes"] if s.get("include", True)]
    if not scenes:
        raise ValueError("No scenes selected.")
    raw: dict[int, bytes] = {}
    total = len(scenes)
    done = 0

    # 1) drawn cards (free, always spelled right)
    for s in scenes:
        if s["kind"] == "card":
            raw[s["index"]] = make_card(s["letter"], s["animal"], card_size)
        elif s["kind"] == "outro":
            raw[s["index"]] = make_outro_card(card_size)
    done = len(raw)
    notify("images", done, total)

    # 2) AI images: each animal's first scene becomes that animal's visual reference for its other scenes
    facts = [s for s in scenes if s["kind"] == "fact"]
    by_animal: dict[str, list[dict]] = {}
    for s in facts:
        by_animal.setdefault(s["animal"], []).append(s)

    def gen(scene, ref):
        raw[scene["index"]] = backend.generate(scene["prompt"], reference=ref)

    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in [pool.submit(gen, group[0], None) for group in by_animal.values()]:
            f.result()
            done += 1
            notify("images", done, total)
        rest = []
        for group in by_animal.values():
            ref = raw[group[0]["index"]] if backend.supports_reference else None
            rest += [pool.submit(gen, s, ref) for s in group[1:]]
        for f in rest:
            f.result()
            done += 1
            notify("images", done, total)

    # 3) quiz scenes reuse their animal's first picture (saves an image call)
    for s in scenes:
        if s["kind"] == "quiz":
            src = by_animal.get(s["animal"])
            raw[s["index"]] = raw[src[0]["index"]] if src else backend.generate(
                f"Main subject: {s['animal']}. Scene: a friendly {s['animal']}", reference=None)
            done += 1
            notify("images", done, total)

    # 4) burn sources onto fact/quiz frames and save
    images = []
    for s in scenes:
        png = raw[s["index"]]
        if s["kind"] in ("fact", "quiz"):
            png = stamp_sources(png, source_label(s["sources"]))
        p = out_dir / f"scene_{s['index']:02d}.png"
        p.write_bytes(png)
        images.append(p)
        s["image"] = str(p)

    # 5) narration
    audio = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(narrator, s["text"], out_dir / f"scene_{s['index']:02d}.mp3") for s in scenes]
        for i, f in enumerate(futs, start=1):
            audio.append(f.result())
            notify("narration", i, total)
    for s, a in zip(scenes, audio):
        s["audio"] = str(a)

    # 6) video
    notify("video", 0, 1)
    video = build_slideshow(images, out_dir / "video.mp4", audio_files=audio, size=video_size)
    notify("video", 1, 1)

    # 7) citations + plan
    lines = [f"# {plan['title']}", "", f"Request: {plan['request']}", "",
             f"Faithfulness (claims supported by sources): {plan['report']['faithfulness']:.0%}", ""]
    for s in scenes:
        src = "; ".join(f"{x['title']} ({x['source']})" for x in s["sources"]) or "generated card (no factual claims)"
        lines += [f"## Scene {s['index']} · {s['kind']} · {s['animal'] or '-'}", s["text"], f"_Sources: {src}_", ""]
    (out_dir / "sources.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "plan.json").write_text(json.dumps({**plan, "scenes": scenes}, indent=2, ensure_ascii=False))
    return {"video": str(video), "dir": str(out_dir), "scenes": scenes}
