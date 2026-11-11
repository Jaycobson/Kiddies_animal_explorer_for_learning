"""Turn a plain-text script into scenes and build consistent image prompts."""
import re
from dataclasses import dataclass


@dataclass
class Scene:
    index: int
    text: str
    prompt: str = ""
    visual: str = ""


def parse_script(script: str) -> list[Scene]:
    """Split on 'Scene N:' markers if present, otherwise on blank lines."""
    script = script.strip()
    if not script:
        return []
    if re.search(r"^\s*scene\s*\d+", script, re.I | re.M):
        parts = re.split(r"^\s*scene\s*\d+\s*[:.\-]?\s*", script, flags=re.I | re.M)
    else:
        parts = re.split(r"\n\s*\n", script)
    chunks = [" ".join(p.split()) for p in parts if p.strip()]
    return [Scene(i + 1, c) for i, c in enumerate(chunks)]


def parse_characters(text: str) -> dict[str, str]:
    """'Name: description' per line -> {name: description}."""
    chars = {}
    for line in text.splitlines():
        if ":" in line:
            name, desc = line.split(":", 1)
            if name.strip() and desc.strip():
                chars[name.strip()] = desc.strip()
    return chars


def build_prompt(scene: Scene, characters: dict[str, str], style: str) -> str:
    """Inject the style and the description of every character named in the scene."""
    present = [
        f"{name} ({desc})"
        for name, desc in characters.items()
        if re.search(rf"\b{re.escape(name)}\b", scene.text + " " + scene.visual, re.I)
    ]
    parts = [
        f"Style: {style}." if style else "",
        f"Characters (keep their look identical in every scene): {'; '.join(present)}." if present else "",
        f"Scene: {scene.visual or scene.text}",
    ]
    return " ".join(p for p in parts if p)
