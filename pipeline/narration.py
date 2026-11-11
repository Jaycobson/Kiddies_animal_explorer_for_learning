"""AI narration with a Nigerian-English delivery (OpenAI text-to-speech)."""
import os
from pathlib import Path

import httpx
import openai

INSTRUCTIONS = (
    "Speak as a warm, cheerful narrator with a natural Nigerian English accent, talking to children aged 5 to 7. "
    "Clear and expressive, with a gentle pace and a little excitement for surprising facts."
)


def synthesize(text, out_path, api_key=None, voice="coral"):
    client = openai.OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        timeout=httpx.Timeout(120.0, connect=30.0),
        max_retries=3,  # backoff on timeouts, 429s, 5xx
    )
    try:  # gpt-4o-mini-tts accepts style/accent instructions
        resp = client.audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=text,
                                          instructions=INSTRUCTIONS, response_format="mp3")
    except (openai.BadRequestError, openai.NotFoundError, openai.PermissionDeniedError):
        # fallback only if the model or parameter is unsupported, never on timeouts
        resp = client.audio.speech.create(model="tts-1", voice="nova", input=text,
                                          response_format="mp3")
    Path(out_path).write_bytes(resp.content)
    return Path(out_path)


class OpenAINarrator:
    def __init__(self, api_key=None, voice="coral"):
        self.api_key, self.voice = api_key, voice

    def __call__(self, text, out_path):
        return synthesize(text, out_path, self.api_key, self.voice)


class SilentNarrator:
    """No API key? Generate silent audio with a duration based on the word count, so timing and video still work."""

    def __call__(self, text, out_path):
        import subprocess

        dur = max(2.0, len(text.split()) / 2.6)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                        "-t", f"{dur:.2f}", "-c:a", "libmp3lame", str(out_path)], check=True)
        return Path(out_path)


def get_narrator(api_key=None, voice="coral"):
    return OpenAINarrator(api_key, voice) if api_key else SilentNarrator()