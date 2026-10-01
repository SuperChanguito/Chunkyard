"""Embedding, retrieval, and the cross-strategy comparison."""

from __future__ import annotations

import os
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .chunking import STRATEGIES, Chunk, chunk_all

MODEL_NAME = os.environ.get("CHUNKYARD_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_DOCS = 5          # documents kept in memory
AGREE_RATIO = 0.25    # spans "agree" if they share >= 25% of the shorter one


class Embedder:
    """Loads the sentence-transformers model once, in the background."""

    def __init__(self, name: str = MODEL_NAME):
        self.name = name
        self.model = None
        self.error: str | None = None
        self._ready = threading.Event()

    def load_async(self) -> None:
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.name)
        except Exception as e:  # surfaced to the UI via /api/status
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self._ready.set()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def wait(self) -> None:
        self._ready.wait()
        if self.model is None:
            raise RuntimeError(f"Embedding model failed to load: {self.error}")

    @property
    def max_tokens(self) -> int:
        return int(self.model.max_seq_length)

    def encode(self, texts: list[str]) -> np.ndarray:
        self.wait()
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

    def token_counts(self, texts: list[str]) -> list[int]:
        self.wait()
        enc = self.model.tokenizer(texts, add_special_tokens=True, truncation=False)
        return [len(ids) for ids in enc["input_ids"]]


@dataclass
class IndexedDoc:
    id: str
    name: str
    text: str
    settings: dict
    chunks: dict[str, list[Chunk]]
    vectors: dict[str, np.ndarray]
    tokens: dict[str, list[int]]


class Store:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.docs: OrderedDict[str, IndexedDoc] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, name: str, text: str, size: int, overlap: int, max_chars: int) -> IndexedDoc:
        chunks = chunk_all(text, size, overlap, max_chars)
        vectors, tokens = {}, {}
        for s in STRATEGIES:
            texts = [c.text for c in chunks[s]] or [""]
            vectors[s] = self.embedder.encode(texts)[: len(chunks[s])]
            tokens[s] = self.embedder.token_counts(texts)[: len(chunks[s])]
        doc = IndexedDoc(uuid.uuid4().hex[:12], name, text,
                         {"size": size, "overlap": overlap, "max_chars": max_chars},
                         chunks, vectors, tokens)
        with self._lock:
            self.docs[doc.id] = doc
            while len(self.docs) > MAX_DOCS:
                self.docs.popitem(last=False)
        return doc

    def get(self, doc_id: str) -> IndexedDoc:
        with self._lock:
            if doc_id not in self.docs:
                raise KeyError(doc_id)
            return self.docs[doc_id]


# ---------------------------------------------------------------------------
# Querying and comparison
# ---------------------------------------------------------------------------

def overlap_ratio(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    return inter / max(1, min(a[1] - a[0], b[1] - b[0]))


def _coverage(spans: list[tuple[int, int]], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for a, b in spans:
        mask[a:b] = True
    return mask


def query(store: Store, doc_id: str, question: str, k: int = 5) -> dict:
    doc = store.get(doc_id)
    qv = store.embedder.encode([question])[0]
    max_tokens = store.embedder.max_tokens

    results: dict[str, list[dict]] = {}
    for s in STRATEGIES:
        if not doc.chunks[s]:
            results[s] = []
            continue
        scores = doc.vectors[s] @ qv
        top = np.argsort(-scores)[:k]
        results[s] = [{
            "rank": r + 1,
            "chunk_index": int(i),
            "score": round(float(scores[i]), 4),
            "start": doc.chunks[s][i].start,
            "end": doc.chunks[s][i].end,
            "text": doc.chunks[s][i].text,
            "headers": doc.chunks[s][i].headers,
            "tokens": doc.tokens[s][i],
            "truncated": doc.tokens[s][i] > max_tokens,
        } for r, i in enumerate(top)]

    # Which other strategies retrieved an overlapping passage, and at what rank?
    for s in STRATEGIES:
        for res in results[s]:
            found = []
            for t in STRATEGIES:
                if t == s:
                    continue
                ranks = [o["rank"] for o in results[t]
                         if overlap_ratio((res["start"], res["end"]), (o["start"], o["end"])) >= AGREE_RATIO]
                if ranks:
                    found.append({"strategy": t, "rank": min(ranks)})
            res["also_found_by"] = found
            res["unique"] = not found

    # Pairwise agreement: does the #1 hit point at the same passage, and how
    # much of the retrieved text overlaps across all top-k hits (Jaccard)?
    n = len(doc.text)
    cover = {s: _coverage([(r["start"], r["end"]) for r in results[s]], n) for s in STRATEGIES}
    pairs = []
    for i, a in enumerate(STRATEGIES):
        for b in STRATEGIES[i + 1:]:
            union = int((cover[a] | cover[b]).sum())
            jac = int((cover[a] & cover[b]).sum()) / union if union else 0.0
            top1 = bool(results[a] and results[b] and overlap_ratio(
                (results[a][0]["start"], results[a][0]["end"]),
                (results[b][0]["start"], results[b][0]["end"])) >= AGREE_RATIO)
            pairs.append({"a": a, "b": b, "top1_agree": top1, "overlap": round(jac, 3)})

    return {
        "question": question,
        "results": results,
        "pairs": pairs,
        "top1_all_agree": all(p["top1_agree"] for p in pairs),
        "max_tokens": max_tokens,
    }
