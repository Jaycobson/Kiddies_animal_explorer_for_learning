"""Visual bible: fixed style + recurring guide character + per-animal look, injected into every image prompt.
This is what keeps the same mascot and the same animal looking the same from scene to scene."""
import json
from pathlib import Path

DEFAULT = {
    "style": "Bright, friendly children's picture-book illustration, soft shapes, warm colours, clean background, no text",
    "mascot": {"name": "Ada", "description": "an 8-year-old Nigerian girl explorer with braids tied with a yellow ribbon, "
                                              "a green explorer vest, khaki shorts and a small brown backpack"},
}


def load_bible(path="bible/bible.json") -> dict:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {"style": data.get("style", DEFAULT["style"]), "mascot": {**DEFAULT["mascot"], **data.get("mascot", {})}}


def build_image_prompt(visual: str, animal: str, look: str, bible: dict) -> str:
    m = bible["mascot"]
    parts = [f"Style: {bible['style']}.",
             f"Main subject: {animal} ({look}). Keep the {animal} looking identical in every scene." if look
             else f"Main subject: {animal}."]
    if m["name"].lower() in visual.lower():
        parts.append(f"Guide character (keep identical): {m['name']}, {m['description']}.")
    parts.append(f"Scene: {visual}")
    return " ".join(parts)
