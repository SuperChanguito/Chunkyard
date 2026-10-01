"""Embedding, retrieval, and the cross-strategy comparison."""

from __future__ import annotations

import bisect
import hashlib
import os
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .chunking import STRATEGIES, Chunk, chunk_all, split_sections

MODEL_NAME = os.environ.get("CHUNKYARD_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_DOCS = 5          # documents kept in memory
AGREE_RATIO = 0.25    # spans "agree" if they share >= 25% of the shorter one
ANSWER_COVERAGE = 0.5  # a chunk "contains the answer" if it covers >= half of it


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
    section_starts: list[int]        # where each detected section begins
    section_paths: list[list[str]]   # and its heading breadcrumb
    fingerprint: str = ""            # hash of the text; marked answers are tied to it

    def section_at(self, pos: int) -> list[str]:
        """Heading breadcrumb of the section containing character `pos`."""
        i = bisect.bisect_right(self.section_starts, pos) - 1
        return self.section_paths[i] if i >= 0 else []


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
        # Section locations let the UI say *where* each strategy's hit came
        # from ("2.1 Recommended Dosage"), whichever strategy produced it.
        sections = [((sec.header_span or sec.body_spans[0])[0], sec.path)
                    for sec in split_sections(text) if sec.header_span or sec.body_spans]
        doc = IndexedDoc(uuid.uuid4().hex[:12], name, text,
                         {"size": size, "overlap": overlap, "max_chars": max_chars},
                         chunks, vectors, tokens,
                         [s for s, _ in sections], [p for _, p in sections],
                         fingerprint(text))
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

def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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


def contains_answer(chunk: tuple[int, int], answer: tuple[int, int]) -> bool:
    """True if the chunk covers at least half of the marked answer span.

    Measured against the answer, not the chunk: a lone heading next to the
    answer covers none of it, while a big section that holds it covers all.
    """
    inter = min(chunk[1], answer[1]) - max(chunk[0], answer[0])
    return inter > 0 and inter / max(1, answer[1] - answer[0]) >= ANSWER_COVERAGE


def answer_rank(results: list[dict], answer: tuple[int, int]) -> int | None:
    """Rank of the first retrieved chunk containing the answer, or None."""
    return next((r["rank"] for r in results if contains_answer((r["start"], r["end"]), answer)), None)


def query(store: Store, doc_id: str, question: str, k: int = 5,
          answer: tuple[int, int] | None = None) -> dict:
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
            "section": doc.section_at(doc.chunks[s][i].start),
            "section_end": doc.section_at(doc.chunks[s][i].end - 1),
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

    out = {
        "question": question,
        "results": results,
        "pairs": pairs,
        "top1_all_agree": all(p["top1_agree"] for p in pairs),
        "max_tokens": max_tokens,
    }
    if answer is not None:
        for s in STRATEGIES:
            for res in results[s]:
                res["has_answer"] = contains_answer((res["start"], res["end"]), answer)
        out["answer_rank"] = {s: answer_rank(results[s], answer) for s in STRATEGIES}
    return out


def evaluate(store: Store, doc_id: str, items: list[tuple[str, tuple[int, int]]], k: int = 5) -> dict:
    """Run every (question, answer span) pair and score each strategy.

    For each strategy: how many answers were found in the top k, and the
    average rank of the ones that were found.
    """
    rows = []
    for question, answer in items:
        res = query(store, doc_id, question, k, answer)
        rows.append({"question": question, "ranks": res["answer_rank"]})
    summary = {}
    for s in STRATEGIES:
        found = [r["ranks"][s] for r in rows if r["ranks"][s] is not None]
        summary[s] = {
            "found": len(found),
            "total": len(rows),
            "avg_rank": round(sum(found) / len(found), 2) if found else None,
        }
    return {"k": k, "rows": rows, "summary": summary}
