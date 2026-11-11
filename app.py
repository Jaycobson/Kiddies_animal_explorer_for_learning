import io
import os
import time
import zipfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from pipeline.backends import OpenAIBackend
from pipeline.explorer import render_episode
from pipeline.narration import get_narrator
from pipeline.video import ffmpeg_available
from rag.bible import load_bible
from rag.episode import plan_episode
from rag.ingest import SUPPORTED, fingerprint
from rag.retriever import build_index
from rag.verifier import OpenAIJudge
from rag.writer import LLMWriter

load_dotenv()
CORPUS = Path("corpus")
CORPUS.mkdir(exist_ok=True)
API_KEY = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="Animal Explorers", page_icon="🦒", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1150px; padding-top: 2rem;}
.hero {padding: 2rem 2.4rem; border-radius: 18px; color: #fff; margin-bottom: 1.2rem;
       background: linear-gradient(120deg, #065f46 0%, #0d9488 55%, #f59e0b 100%);}
.hero h1 {color:#fff; margin:0 0 .3rem 0; font-size:2.2rem;}
.hero p {color:#ecfdf5; margin:0; font-size:1.05rem; max-width:70%;}
</style>
<div class="hero"><h1>🦒 Animal Explorers</h1>
<p>Drop your animal notes, PDFs and fact sheets. The app retrieves facts from them, writes a kids' script that can only use
those facts, checks every sentence against its source, and turns it into a narrated video with sources on screen.</p></div>
""", unsafe_allow_html=True)

if not API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to your .env file and restart.")
    st.stop()


@st.cache_resource(show_spinner="Indexing documents…")
def get_index(fp: str):
    return build_index(CORPUS, API_KEY)


with st.sidebar:
    st.header("Settings")
    max_animals = st.slider("Max animals", 1, 6, 3)
    facts_per_animal = st.slider("Facts per animal", 1, 5, 3)
    age_band = st.selectbox("Age group", ["0-3","3-5", "5-7", "7-9"], index=1)

index = get_index(fingerprint(CORPUS))
tab_make, tab_kb = st.tabs(["🎬 Create episode", "📚 Knowledge base"])

with tab_kb:
    up = st.file_uploader("Add documents (.md, .txt, .pdf)", type=[e.strip(".") for e in SUPPORTED],
                          accept_multiple_files=True)
    if up and st.button("Save to library"):
        for f in up:
            (CORPUS / f.name).write_bytes(f.getvalue())
        st.cache_resource.clear()
        st.rerun()
    st.caption(f"{len(index.chunks)} chunks · {len(index.animals())} animals · embeddings: {index.embedder.name}")
    for a in index.animals():
        docs = sorted({c.title for c in index.chunks if c.animal == a})
        st.markdown(f"**{a[0].upper()} · {a}** — {', '.join(docs)}")

with tab_make:
    letters = ", ".join(index.letters())
    prompt = st.text_area(
        "What should the video teach?", "Teach 6-year-olds about the octopus and the giraffe.", height=90,
        help=f"Name animals, ask for letters ('letter P', 'A to Z'), or describe a topic ('animals that live in the sea'). "
             f"Letters in your library: {letters}")

    if st.button("✨ Generate video", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a request first.")
        elif not ffmpeg_available():
            st.error("FFmpeg not found on PATH.")
        else:
            try:
                t0 = time.time()
                with st.status("Building your video…", expanded=True) as status:
                    bar = st.progress(0.0, text="Retrieving facts and writing the script…")
                    plan = plan_episode(prompt, index, LLMWriter(API_KEY), load_bible(), OpenAIJudge(API_KEY),
                                        max_animals, facts_per_animal, True, age_band)
                    if not plan["scenes"]:
                        status.update(label="Nothing to make", state="error")
                        st.warning("The documents don't contain enough supported facts for this request. "
                                   + " ".join(f"{s['animal']}: {s['reason']}" for s in plan["skipped"]))
                        st.stop()
                    weights = {"images": (0.1, 0.55, "Generating scene images"),
                               "narration": (0.65, 0.2, "Recording narration"),
                               "video": (0.85, 0.15, "Assembling video")}

                    def on_progress(stage, done, total):
                        a, b, label = weights[stage]
                        bar.progress(min(a + b * done / max(total, 1), 1.0), text=f"{label} ({done}/{total})")

                    out_dir = Path("outputs") / time.strftime("%Y%m%d_%H%M%S")
                    out = render_episode(plan, OpenAIBackend(API_KEY), out_dir, get_narrator(API_KEY), on_progress)
                    out["plan"], out["seconds"] = plan, time.time() - t0
                    status.update(label="Done", state="complete", expanded=False)
                st.session_state["out"] = out
            except Exception as e:
                st.error(f"Generation failed: {e}")

    out = st.session_state.get("out")
    if out:
        plan = out["plan"]
        st.divider()
        st.subheader(plan["title"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Scenes", len(out["scenes"]))
        m2.metric("Verified against sources", f"{plan['report']['faithfulness']:.0%}")
        m3.metric("Generated in", f"{out['seconds']:.0f}s")
        st.video(out["video"])
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for f in Path(out["dir"]).iterdir():
                z.write(f, f.name)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Video (MP4)", Path(out["video"]).read_bytes(), "animal_episode.mp4", "video/mp4",
                           use_container_width=True)
        d2.download_button("📦 Everything (images, audio, sources.md, plan.json)", buf.getvalue(),
                           "episode_package.zip", "application/zip", use_container_width=True)