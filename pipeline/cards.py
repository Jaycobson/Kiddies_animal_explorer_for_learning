"""Pillow-drawn assets: alphabet cards, outro card, source banners, and a free placeholder image backend.
Cards are drawn (not AI-generated) so letters and names are always spelled correctly."""
import colorsys
import io
import re
from PIL import Image, ImageDraw, ImageFont

from .backends import ImageBackend

_FONTS = ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf",
          "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "Arial Bold.ttf"]


def _font(size: int) -> ImageFont.ImageFont:
    for f in _FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def _colour(key: str, s=0.45, v=0.97) -> tuple[int, int, int]:
    i = (sum(map(ord, key)) * 7) % 26 if key else 0
    r, g, b = colorsys.hsv_to_rgb(i / 26, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _centered(d: ImageDraw.ImageDraw, text: str, cy: int, width: int, start_size: int, fill, max_w: float = 0.86):
    size = start_size
    while size > 14:
        f = _font(size)
        l, t, r, b = d.textbbox((0, 0), text, font=f)
        if r - l <= width * max_w:
            break
        size -= 4
    d.text(((width - (r - l)) / 2 - l, cy - (b - t) / 2 - t), text, font=f, fill=fill)


def make_card(letter: str, animal: str, size: int = 1024) -> bytes:
    img = Image.new("RGB", (size, size), _colour(letter))
    d = ImageDraw.Draw(img)
    m = size // 14
    d.rounded_rectangle([m, m, size - m, size - m], radius=size // 12, fill=(255, 255, 255))
    _centered(d, letter.upper(), int(size * 0.40), size, int(size * 0.62), _colour(letter, 0.8, 0.75))
    _centered(d, f"{letter.upper()} is for {animal}", int(size * 0.80), size, int(size * 0.09), (40, 40, 60))
    return _png(img)


def make_outro_card(size: int = 1024, text: str = "Great job, explorers!") -> bytes:
    img = Image.new("RGB", (size, size), _colour("outro"))
    d = ImageDraw.Draw(img)
    _centered(d, text, size // 2, size, int(size * 0.10), (255, 255, 255))
    return _png(img)


def stamp_sources(png: bytes, text: str) -> bytes:
    """Burn a 'Source: ...' banner along the bottom edge."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = img.size
    bar = int(h * 0.075)
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(over).rectangle([0, h - bar, w, h], fill=(0, 0, 0, 170))
    img = Image.alpha_composite(img.convert("RGBA"), over)
    d = ImageDraw.Draw(img)
    size = int(bar * 0.5)
    f = _font(size)
    while d.textlength(text, font=f) > w * 0.94 and len(text) > 8:
        text = text[:-4].rstrip() + "…"
    d.text((w * 0.03, h - bar + (bar - size) / 2.4), text, font=f, fill=(255, 255, 255, 255))
    return _png(img)


def source_label(sources: list[dict]) -> str:
    names = list(dict.fromkeys(s["title"] if not s.get("url") else s["url"] for s in sources))
    return "Source: " + "; ".join(names) if names else ""


class PlaceholderBackend(ImageBackend):
    """Free offline 'image model': coloured card with the animal name and a short scene description.
    Lets you run and demo the whole pipeline without a GPU or API key."""
    supports_reference = False

    def __init__(self, size: int = 1024):
        self.size = size

    def generate(self, prompt, reference=None):
        animal = (re.search(r"Main subject: ([^(.]+?)(?: \(|\.)", prompt) or [None, "Animal"])[1].strip()
        scene = prompt.split("Scene:", 1)[-1].strip()
        img = Image.new("RGB", (self.size, self.size), _colour(animal, 0.35, 0.98))
        d = ImageDraw.Draw(img)
        _centered(d, animal, int(self.size * 0.30), self.size, int(self.size * 0.14), (50, 50, 70))
        f = _font(int(self.size * 0.035))
        y, line = int(self.size * 0.45), ""
        for word in scene.split():
            if d.textlength(f"{line} {word}".strip(), font=f) > self.size * 0.8:
                d.text((self.size * 0.1, y), line, font=f, fill=(70, 70, 90))
                y, line = y + int(self.size * 0.05), word
            else:
                line = f"{line} {word}".strip()
            if y > self.size * 0.82:
                break
        d.text((self.size * 0.1, y), line, font=f, fill=(70, 70, 90))
        return _png(img)
