"""FastAPI server: upload a document, chunk + embed it, query it."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chunking import STRATEGIES
from .engine import Embedder, IndexedDoc, Store, evaluate, query
from .flags import too_small
from . import generate as gen
from .loaders import load_document
from .samples import SAMPLES, SAMPLES_DIR

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MAX_UPLOAD = 20 * 1024 * 1024

embedder = Embedder()
store = Store(embedder)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    embedder.load_async()  # load in the background so the page opens right away
    yield


app = FastAPI(title="Chunkyard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status() -> dict:
    return {"model": embedder.name, "ready": embedder.ready, "error": embedder.error,
            "generation": gen.status()}


def _summary(doc: IndexedDoc) -> dict:
    max_tokens = embedder.max_tokens
    stats = {}
    for s in STRATEGIES:
        lens = [c.end - c.start for c in doc.chunks[s]]
        stats[s] = {
            "count": len(lens),
            "avg_chars": round(sum(lens) / len(lens)) if lens else 0,
            "min_chars": min(lens, default=0),
            "max_chars": max(lens, default=0),
            "truncated": sum(t > max_tokens for t in doc.tokens[s]),
            "too_small": too_small(sum(lens) / len(lens), min(lens)) if lens else None,
            "spans": [[c.start, c.end] for c in doc.chunks[s]],
        }
    return {"id": doc.id, "name": doc.name, "chars": len(doc.text), "text": doc.text,
            "fingerprint": doc.fingerprint,
            "settings": doc.settings, "stats": stats, "max_tokens": max_tokens}


def _index(name: str, text: str, size: int, overlap: int, max_chars: int) -> dict:
    if not (50 <= size <= 5000):
        raise HTTPException(400, "Chunk size must be between 50 and 5000 characters.")
    if not (0 <= overlap < size):
        raise HTTPException(400, "Overlap must be at least 0 and smaller than the chunk size.")
    if not (200 <= max_chars <= 10000):
        raise HTTPException(400, "Max chunk size must be between 200 and 10000 characters.")
    try:
        doc = store.add(name, text, size, overlap, max_chars)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return _summary(doc)


@app.post("/api/documents")
def upload(file: UploadFile = File(...), size: int = Form(500), overlap: int = Form(100),
           max_chars: int = Form(1200)) -> dict:
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "That file is over 20 MB.")
    try:
        text = load_document(file.filename or "upload.txt", data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Couldn't read that file ({type(e).__name__}).")
    return _index(file.filename or "upload", text, size, overlap, max_chars)


@app.get("/api/samples")
def samples() -> list[dict]:
    return [{"id": k, "title": v["title"], "blurb": v["blurb"], "questions": v["questions"],
             "available": (SAMPLES_DIR / v["file"]).is_file()} for k, v in SAMPLES.items()]


@app.post("/api/sample")
def sample(sample: str = Form("tent"), size: int = Form(500), overlap: int = Form(100),
           max_chars: int = Form(1200)) -> dict:
    if sample not in SAMPLES:
        raise HTTPException(404, "Unknown sample.")
    path = SAMPLES_DIR / SAMPLES[sample]["file"]
    if not path.is_file():
        raise HTTPException(404, f"Sample file not found. Put {path.name} in the samples folder.")
    text = load_document(path.name, path.read_bytes())
    return _index(path.name, text, size, overlap, max_chars) | {"sample": sample}


class Span(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class QueryIn(BaseModel):
    doc_id: str
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=20)
    answer: Span | None = None  # the marked answer for this question, if any


class EvalItem(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: Span


class EvalIn(BaseModel):
    doc_id: str
    items: list[EvalItem] = Field(min_length=1, max_length=500)
    k: int = Field(5, ge=1, le=20)


def _span(doc_id: str, span: Span) -> tuple[int, int]:
    n = len(store.get(doc_id).text)
    if not span.start < span.end <= n:
        raise HTTPException(400, "A marked answer doesn't fit this document. Was it saved for a different file?")
    return span.start, span.end


@app.post("/api/query")
def ask(body: QueryIn) -> dict:
    try:
        answer = _span(body.doc_id, body.answer) if body.answer else None
        return query(store, body.doc_id, body.question.strip(), body.k, answer)
    except KeyError:
        raise HTTPException(404, "That document is no longer loaded. Upload it again.")
    except RuntimeError as e:
        raise HTTPException(503, str(e))


class GenerateIn(BaseModel):
    doc_id: str
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=20)


@app.post("/api/generate")
def generate_answers(body: GenerateIn) -> dict:
    """Optional: answer from each strategy's top chunks with the Claude API."""
    try:
        res = query(store, body.doc_id, body.question.strip(), body.k)
    except KeyError:
        raise HTTPException(404, "That document is no longer loaded. Upload it again.")
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    try:
        import anthropic
    except ImportError:
        raise HTTPException(503, "The anthropic package isn't installed. Run: uv sync")
    try:
        return gen.generate(gen.make_client(), res["question"], res["results"])
    except gen.GenerationError as e:
        raise HTTPException(e.status, str(e))
    except anthropic.AuthenticationError:
        raise HTTPException(401, "The Claude API rejected the key. Check ANTHROPIC_API_KEY and restart Chunkyard.")
    except anthropic.PermissionDeniedError:
        raise HTTPException(403, f"This API key can't use {gen.MODEL}.")
    except anthropic.RateLimitError:
        raise HTTPException(429, "The Claude API is rate-limiting requests. Wait a minute and try again.")
    except anthropic.APIConnectionError:
        raise HTTPException(503, "Couldn't reach the Claude API. Check your internet connection.")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Claude API error ({e.status_code}): {e.message}")
    except TypeError as e:
        # The SDK raises this before sending when it finds no credentials at all.
        if "auth" in str(e).lower() or "api_key" in str(e).lower():
            raise HTTPException(401, "No Claude API key found. Set ANTHROPIC_API_KEY, then restart Chunkyard.")
        raise


@app.post("/api/evaluate")
def run_set(body: EvalIn) -> dict:
    try:
        items = [(it.question.strip(), _span(body.doc_id, it.answer)) for it in body.items]
        return evaluate(store, body.doc_id, items, body.k)
    except KeyError:
        raise HTTPException(404, "That document is no longer loaded. Upload it again.")
    except RuntimeError as e:
        raise HTTPException(503, str(e))
