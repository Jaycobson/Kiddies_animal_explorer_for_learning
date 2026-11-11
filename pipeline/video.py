"""Assemble scene images (and optional narration) into a slideshow video with FFmpeg."""
import shutil
import subprocess
import tempfile
from pathlib import Path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def audio_duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def build_slideshow(images, out_path, seconds=4, fps=25, size=(1280, 720),
                    audio_files=None, pad=0.6) -> Path:
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg not found. Install it and make sure it is on your PATH.")
    w, h = size
    with tempfile.TemporaryDirectory() as tmp:
        clips = []
        for i, img in enumerate(images):
            audio = audio_files[i] if audio_files else None
            dur = audio_duration(audio) + pad if audio else seconds
            frames = int(dur * fps)
            clip = Path(tmp) / f"clip_{i:03d}.mp4"
            vf = (f"scale={w*2}:{h*2}:force_original_aspect_ratio=increase,crop={w*2}:{h*2},"
                  f"zoompan=z='min(zoom+0.0012,1.2)':d={frames}:x='iw/2-(iw/zoom/2)':"
                  f"y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps},format=yuv420p")
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(img)]
            if audio:
                cmd += ["-i", str(audio)]
            cmd += ["-vf", vf]
            if audio:
                cmd += ["-af", "apad", "-ar", "44100", "-ac", "2", "-c:a", "aac"]
            cmd += ["-t", f"{dur:.2f}", "-c:v", "libx264", "-r", str(fps), str(clip)]
            subprocess.run(cmd, check=True)
            clips.append(clip)
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-c", "copy", str(out_path)], check=True)
    return out_path