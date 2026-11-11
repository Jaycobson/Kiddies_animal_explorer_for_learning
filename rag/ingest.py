"""Load documents (.md / .txt / .pdf), read optional front matter, and split into citable chunks.

Front matter is optional:
    ---
    animal: Elephant
    source: Where this text came from (shown in the video)
    url: https://...
    look: One line describing how the animal looks (keeps images consistent)
    ---
Files without it still work: the animal is taken from the file name.
"""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf"}


@dataclass
class Chunk:
    id: str
    text: str
    animal: str
    source: str
    title: str
    url: str = ""
    section: str = ""
    look: str = ""

    @property
    def letter(self) -> str:
        return self.animal[:1].upper()


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text.strip()) if s.strip()]


def parse_front_matter(raw: str) -> tuple[dict, str]:
    m = re.match(r"^---[ \t]*\n(.*?)\n---[ \t]*\n?", raw, re.S)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip()
    return meta, raw[m.end():]


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _looks_like_title(line: str) -> bool:
    """Short line with no sentence punctuation at the end: 'Penguin fact card', 'Zebra notes (informal)'."""
    return len(line) <= 60 and len(line.split()) <= 8 and not re.search(r"[.!?:,;]$", line)


def _blocks(text: str) -> list[tuple[str, str]]:
    """Return (section, paragraph) pairs. Bullets become their own paragraphs; title-like first lines become sections."""
    section, blocks = "", []
    for para in re.split(r"\n\s*\n", text):
        body = []
        for n, ln in enumerate(para.splitlines()):
            s = ln.strip()
            if not s:
                continue
            if s.startswith("#"):
                section = s.lstrip("# ").strip()
            elif n == 0 and not body and _looks_like_title(s) and not s.startswith(("- ", "* ", "• ")):
                section = s
            elif s.startswith(("- ", "* ", "• ")):
                if body:
                    blocks.append((section, " ".join(body)))
                    body = []
                blocks.append((section, s[2:].strip()))
            else:
                body.append(s)
        if body:
            blocks.append((section, " ".join(body)))
    return [(sec, " ".join(b.split())) for sec, b in blocks if b.strip()]


def chunk_text(text: str, max_chars: int = 520) -> list[tuple[str, str]]:
    """Merge small paragraphs (same section) up to max_chars; split oversized ones by sentence."""
    pieces = []
    for sec, para in _blocks(text):
        if len(para) <= max_chars:
            pieces.append((sec, para))
            continue
        cur = ""
        for s in split_sentences(para):
            if cur and len(cur) + len(s) + 1 > max_chars:
                pieces.append((sec, cur))
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            pieces.append((sec, cur))
    merged: list[tuple[str, str]] = []
    for sec, t in pieces:
        if merged and merged[-1][0] == sec and len(merged[-1][1]) + len(t) + 1 <= max_chars:
            merged[-1] = (sec, f"{merged[-1][1]} {t}")
        else:
            merged.append((sec, t))
    return merged


def load_corpus(folder, max_chars: int = 520) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).rglob("*")):
        if (not path.is_file() or path.suffix.lower() not in SUPPORTED
                or path.name.startswith(".") or path.stem.lower().startswith("readme")):
            continue
        meta, body = parse_front_matter(read_file(path))
        animal = meta.get("animal") or re.sub(r"[_\-]+", " ", path.stem.replace("_notes", "")).strip().title()
        for n, (section, text) in enumerate(chunk_text(body, max_chars), start=1):
            chunks.append(Chunk(id=f"{slug(path.stem)}-{n:02d}", text=text, animal=animal,
                                source=meta.get("source") or path.name, title=path.name,
                                url=meta.get("url", ""), section=section, look=meta.get("look", "")))
    return chunks


def fingerprint(folder) -> str:
    h = hashlib.sha1()
    for path in sorted(Path(folder).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()
