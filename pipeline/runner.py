"""Orchestrates: story -> AI plan -> images -> narration -> video."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .narration import synthesize
from .planner import plan_story
from .script_parser import Scene, build_prompt
from .video import build_slideshow


def with_retry(fn, attempts=3, base=3):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(base * 2 ** i)


def run_pipeline(story, backend, out_dir, consistency=True, api_key=None, on_progress=None):
    notify = on_progress or (lambda stage, done, total: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = with_retry(lambda: plan_story(story, api_key))
    notify("planning", 1, 1)
    scenes = [Scene(i + 1, s["text"], visual=s["visual"]) for i, s in enumerate(plan["scenes"])]
    for s in scenes:
        s.prompt = build_prompt(s, plan["characters"], plan["style"])
    n = len(scenes)

    def make_image(scene, ref):
        png = with_retry(lambda: backend.generate(scene.prompt, reference=ref))
        (out_dir / f"scene_{scene.index:02d}.png").write_bytes(png)
        return png

    def make_audio(scene):
        path = out_dir / f"scene_{scene.index:02d}.mp3"
        return with_retry(lambda: synthesize(scene.text, path, api_key))

    # Scene 1 first: it becomes the visual reference that keeps characters consistent.
    first = make_image(scenes[0], None)
    notify("images", 1, n)
    ref = first if (consistency and backend.supports_reference) else None
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(make_image, s, ref) for s in scenes[1:]]
        for i, f in enumerate(futures, start=2):
            f.result()
            notify("images", i, n)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(make_audio, s) for s in scenes]
        audio = []
        for i, f in enumerate(futures, start=1):
            audio.append(f.result())
            notify("narration", i, n)

    notify("video", 0, 1)
    images = [out_dir / f"scene_{s.index:02d}.png" for s in scenes]
    video = build_slideshow(images, out_dir / "video.mp4", audio_files=audio, size=(1024, 1024))
    notify("video", 1, 1)

    results = [{"index": s.index, "text": s.text, "prompt": s.prompt,
                "image": str(images[i]), "audio": str(audio[i])} for i, s in enumerate(scenes)]
    (out_dir / "plan.json").write_text(json.dumps({**plan, "scenes": results}, indent=2))
    return {"plan": plan, "scenes": results, "video": str(video), "dir": str(out_dir)}