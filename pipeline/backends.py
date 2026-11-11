"""Image backends. Both expose generate(prompt, reference) -> PNG bytes."""
import base64
import json
import os
import random
import time
import uuid
from pathlib import Path

import httpx
import openai
import requests


class ImageBackend:
    supports_reference = False

    def generate(self, prompt: str, reference: bytes | None = None) -> bytes:
        raise NotImplementedError


class OpenAIBackend(ImageBackend):
    supports_reference = True

    def __init__(self, api_key=None, model="gpt-image-2", size="1024x1024", quality="low"):
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            timeout=httpx.Timeout(300.0, connect=30.0),
            max_retries=3,  # backoff on timeouts, 429s, 5xx
        )
        self.model, self.size, self.quality = model, size, quality

    def generate(self, prompt, reference=None):
        kw = {"model": self.model, "prompt": prompt, "size": self.size}
        if self.model.startswith("gpt-image"):
            kw["quality"] = self.quality
        else:
            kw["response_format"] = "b64_json"
        res = None
        if reference and self.model.startswith("gpt-image"):
            try:  # use the first scene as a visual reference for consistency
                res = self.client.images.edit(image=("ref.png", reference, "image/png"), **kw)
            except (openai.BadRequestError, openai.NotFoundError, openai.PermissionDeniedError):
                res = None  # fall back only if edit is unsupported, never on timeouts
        if res is None:
            res = self.client.images.generate(**kw)
        item = res.data[0]
        if getattr(item, "b64_json", None):
            return base64.b64decode(item.b64_json)
        return requests.get(item.url, timeout=120).content


class ComfyUIBackend(ImageBackend):
    """Drives a local ComfyUI server through its HTTP API using an API-format workflow."""

    def __init__(self, url=None, checkpoint=None, workflow="workflows/txt2img_api.json",
                 width=1024, height=1024, timeout=300,
                 negative="blurry, low quality, deformed hands, extra fingers, text, watermark"):
        self.url = (url or os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")).rstrip("/")
        self.checkpoint = checkpoint or os.getenv("COMFYUI_CHECKPOINT")
        self.workflow, self.width, self.height = workflow, width, height
        self.timeout, self.negative = timeout, negative

    def generate(self, prompt, reference=None):
        wf = json.loads(Path(self.workflow).read_text())
        wf["6"]["inputs"]["text"] = prompt            # positive prompt node
        wf["7"]["inputs"]["text"] = self.negative     # negative prompt node
        wf["5"]["inputs"].update(width=self.width, height=self.height)
        wf["3"]["inputs"]["seed"] = random.randint(0, 2**31)
        if self.checkpoint:
            wf["4"]["inputs"]["ckpt_name"] = self.checkpoint
        r = requests.post(f"{self.url}/prompt",
                          json={"prompt": wf, "client_id": str(uuid.uuid4())}, timeout=30)
        r.raise_for_status()
        pid = r.json()["prompt_id"]
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            hist = requests.get(f"{self.url}/history/{pid}", timeout=30).json()
            if pid in hist:
                for node in hist[pid]["outputs"].values():
                    for img in node.get("images", []):
                        return requests.get(f"{self.url}/view", params=img, timeout=60).content
                raise RuntimeError("ComfyUI finished but returned no image.")
            time.sleep(1)
        raise TimeoutError("ComfyUI did not finish in time.")