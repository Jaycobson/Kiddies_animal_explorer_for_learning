"""Hybrid retrieval: BM25 (keywords) + dense vectors (meaning), fused with Reciprocal Rank Fusion.

Embedders:
  OpenAIEmbedder - real semantic vectors (needs OPENAI_API_KEY)
  HashEmbedder   - offline hashed word/char n-gram vectors; deterministic, free, good for tests.
                   It is lexical-ish, so quality gains come mainly from OpenAIEmbedder.
Vectors are cached on disk, keyed by corpus fingerprint + embedder, so unchanged corpora are never re-embedded.
"""
import hashlib
import math
import os
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ingest import Chunk, fingerprint, load_corpus

RRF_K = 60
STOP = set("""a an the and or but of to in on at by for with from as is are was were be been being it its this that these
those there their they them he she his her you your we our i me my not no so if then than too very can could will would
should do does did has have had which who whom what when where why how also about into over under up down out off more
most some any each other such only own same just""".split())


def stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) or 1.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for t in set(query):
            if t not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class HashEmbedder:
    name, dim = "hash-512", 512

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            feats += [f"#{w[j:j + 3]}" for w in toks for j in range(max(1, len(w) - 2))]
            for f in feats:
                h = zlib.crc32(f.encode())
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class OpenAIEmbedder:
    name = "openai-text-embedding-3-small"

    def __init__(self, api_key=None, model="text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), 96):
            res = self.client.embeddings.create(model=self.model, input=texts[i:i + 96])
            vecs += [d.embedding for d in res.data]
        arr = np.array(vecs, dtype=np.float32)
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder, vectors: np.ndarray):
        self.chunks, self.embedder, self.vectors = chunks, embedder, vectors
        self.by_id = {c.id: c for c in chunks}
        self.bm25 = BM25([tokenize(f"{c.animal} {c.section} {c.text}") for c in chunks])

    @classmethod
    def build(cls, folder, embedder=None, cache_dir: str | None = ".index") -> "HybridIndex":
        embedder = embedder or HashEmbedder()
        chunks = load_corpus(folder)
        if not chunks:
            raise ValueError(f"No .md/.txt/.pdf documents found in '{folder}'.")
        cache = None
        if cache_dir:
            key = hashlib.sha1((fingerprint(folder) + embedder.name).encode()).hexdigest()[:16]
            cache = Path(cache_dir) / f"{key}.npy"
        if cache and cache.exists():
            vectors = np.load(cache)
        else:
            vectors = embedder.embed([f"{c.animal}. {c.section}. {c.text}" for c in chunks])
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, vectors)
        return cls(chunks, embedder, vectors)

    # ---- catalog helpers
    def animals(self) -> list[str]:
        return sorted({c.animal for c in self.chunks}, key=str.lower)

    def letters(self) -> list[str]:
        return sorted({c.letter for c in self.chunks})

    # ---- search
    def search(self, query: str, k: int = 5, animal: str | None = None, letter: str | None = None,
               mode: str = "hybrid") -> list[Hit]:
        idx = [i for i, c in enumerate(self.chunks)
               if (not animal or c.animal.lower() == animal.lower()) and (not letter or c.letter == letter.upper())]
        if not idx:
            return []
        sub = np.array(idx)
        ranks: dict[str, dict[int, int]] = {}
        if mode in ("hybrid", "bm25"):
            s = self.bm25.scores(tokenize(query))[sub]
            order = [j for j in np.argsort(-s, kind="stable") if s[j] > 0]
            ranks["bm25"] = {int(sub[j]): r for r, j in enumerate(order)}
        if mode in ("hybrid", "vector"):
            s = self.vectors[sub] @ self.embedder.embed([query])[0]
            ranks["vector"] = {int(sub[j]): r for r, j in enumerate(np.argsort(-s, kind="stable"))}
        fused: dict[int, float] = {}
        for rk in ranks.values():
            for ci, r in rk.items():
                fused[ci] = fused.get(ci, 0.0) + 1.0 / (RRF_K + r + 1)
        top = sorted(fused, key=lambda ci: -fused[ci])[:k]
        return [Hit(self.chunks[ci], fused[ci], ranks.get("bm25", {}).get(ci), ranks.get("vector", {}).get(ci))
                for ci in top]


def build_index(folder, api_key=None, cache_dir: str | None = ".index") -> HybridIndex:
    """Use OpenAI embeddings when a key is available; fall back to the offline embedder on any failure."""
    if api_key:
        try:
            return HybridIndex.build(folder, OpenAIEmbedder(api_key), cache_dir)
        except Exception:
            pass
    return HybridIndex.build(folder, HashEmbedder(), cache_dir)
