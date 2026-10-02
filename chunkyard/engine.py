"""Embedding, retrieval, and the cross-strategy comparison."""

from __future__ import annotations

import bisect
import hashlib
import os
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .chunking import STRATEGIES, Chunk, chunk_all, detect_header, split_sections
from .flags import SHORT_CHARS, chunk_flags

MODEL_NAME = os.environ.get("CHUNKYARD_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_DOCS = 5          # documents kept in memory
EMBED_CACHE_SIZE = 50_000  # chunk embeddings kept (about 1.5 KB each)
AGREE_RATIO = 0.25    # two chunks "agree" if they share >= 25% of the LONGER one
FRAGMENT_INSIDE = 0.5  # a fragment sits "inside" a passage if half of it is covered
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
        # (model, exact chunk text) -> (vector, token count). Re-chunking only
        # embeds chunks it hasn't seen: changing just the fixed size leaves
        # every paragraph and section chunk cached.
        self._cache: OrderedDict[tuple[str, str], tuple[np.ndarray, int]] = OrderedDict()
        self.embedded = 0  # texts actually sent to the model (for tests and curiosity)

    def _embed(self, texts: list[str]) -> tuple[np.ndarray, list[int]]:
        model = self.embedder.name
        with self._lock:
            missing = list(dict.fromkeys(t for t in texts if (model, t) not in self._cache))
        fresh: dict[str, tuple[np.ndarray, int]] = {}
        if missing:
            vecs = self.embedder.encode(missing)
            counts = self.embedder.token_counts(missing)
            fresh = {t: (v, n) for t, v, n in zip(missing, vecs, counts)}
        with self._lock:
            self.embedded += len(missing)
            rows = []
            for t in texts:
                hit = fresh.get(t) or self._cache.get((model, t))
                self._cache[(model, t)] = hit
                self._cache.move_to_end((model, t))
                rows.append(hit)
            while len(self._cache) > EMBED_CACHE_SIZE:
                self._cache.popitem(last=False)
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.stack([r[0] for r in rows]), [r[1] for r in rows]

    def add(self, name: str, text: str, size: int, overlap: int, max_chars: int) -> IndexedDoc:
        chunks = chunk_all(text, size, overlap, max_chars)
        vectors, tokens = {}, {}
        for s in STRATEGIES:
            vectors[s], tokens[s] = self._embed([c.text for c in chunks[s]])
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
    """Shared characters as a fraction of the LONGER span.

    Measuring against the longer span means a lone heading sitting inside a
    big passage shares almost nothing with it, rather than 100%.
    """
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    return inter / max(1, a[1] - a[0], b[1] - b[0])


def same_passage(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Do two chunks from different strategies point at the same passage?

    Never for a chunk under SHORT_CHARS: a heading or fragment that happens to
    sit inside another strategy's passage didn't retrieve that passage.
    """
    if min(a[1] - a[0], b[1] - b[0]) < SHORT_CHARS:
        return False
    return overlap_ratio(a, b) >= AGREE_RATIO


def _fragment_inside(fragment: tuple[int, int], passage: tuple[int, int]) -> bool:
    inter = min(fragment[1], passage[1]) - max(fragment[0], passage[0])
    return inter > 0 and inter / max(1, fragment[1] - fragment[0]) >= FRAGMENT_INSIDE


def _heading_title(doc: "IndexedDoc", start: int, text: str) -> str | None:
    """If a short chunk is just a heading, its title ("What's not covered")."""
    line = text.strip()
    if "\n" in line:
        return None
    title = re.sub(r"^#{1,6}\s+", "", line).strip()
    if (line.startswith("#") or title in doc.section_at(start)
            or detect_header(line, "", True) is not None):
        return title
    return None


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
            "flags": chunk_flags(doc.text, doc.chunks[s][i].start, doc.chunks[s][i].end),
        } for r, i in enumerate(top)]

    # Which other strategies retrieved the same passage, and at what rank?
    # `matches` lists every agreeing card (the UI links them on hover), so the
    # browser never needs its own copy of the rule.
    for s in STRATEGIES:
        for res in results[s]:
            found, matches = [], []
            for t in STRATEGIES:
                if t == s:
                    continue
                ranks = [o["rank"] for o in results[t]
                         if same_passage((res["start"], res["end"]), (o["start"], o["end"]))]
                if ranks:
                    found.append({"strategy": t, "rank": min(ranks)})
                    matches.extend({"strategy": t, "rank": r} for r in ranks)
            res["also_found_by"] = found
            res["matches"] = matches
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
            top1 = bool(results[a] and results[b] and same_passage(
                (results[a][0]["start"], results[a][0]["end"]),
                (results[b][0]["start"], results[b][0]["end"])))
            pairs.append({"a": a, "b": b, "top1_agree": top1, "overlap": round(jac, 3)})

    # A strategy whose #1 is a tiny fragment: say what it is and whether it
    # sits inside another strategy's #1 passage ("only retrieved the heading").
    fragments = []
    for s in STRATEGIES:
        if not results[s]:
            continue
        top = results[s][0]
        if top["end"] - top["start"] >= SHORT_CHARS:
            continue
        span = (top["start"], top["end"])
        fragments.append({
            "strategy": s,
            "text": doc.text[top["start"]:top["end"]].strip(),
            "heading": _heading_title(doc, top["start"], doc.text[top["start"]:top["end"]]),
            "inside": [t for t in STRATEGIES if t != s and results[t]
                       and results[t][0]["end"] - results[t][0]["start"] >= SHORT_CHARS
                       and _fragment_inside(span, (results[t][0]["start"], results[t][0]["end"]))],
        })

    out = {
        "question": question,
        "results": results,
        "pairs": pairs,
        "top1_all_agree": all(p["top1_agree"] for p in pairs),
        "top1_fragments": fragments,
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

    For each strategy: how many answers were found in the top k, the average
    rank of the ones that were found, and MRR (mean of 1/rank, 0 for a miss),
    one number that reflects both hits and misses.
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
            "mrr": round(sum(1 / r for r in found) / len(rows), 3) if rows else None,
        }
    return {"k": k, "rows": rows, "summary": summary}
