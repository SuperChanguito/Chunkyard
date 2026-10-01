"""Optional generation step: answer the question from each strategy's chunks.

Off unless the viewer asks for it. Each run makes four Claude API calls: one
answer per chunking strategy (using only that strategy's top chunks), then
one comparison that lists where the three answers differ factually.

Nothing here runs at import time and the `anthropic` package is imported
lazily, so the rest of Chunkyard works without it or without a key.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

from .chunking import STRATEGIES

MODEL = os.environ.get("CHUNKYARD_LLM_MODEL", "claude-opus-5")
# Server-side fallback: if a safety classifier declines, Anthropic re-runs the
# request on its recommended fallback model instead of returning a refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

ANSWER_SYSTEM = """You answer questions about a document using only the numbered passages you are given. \
They are the passages a retrieval system found for the question, and they may be incomplete, cut off, \
or irrelevant.

- Use only facts stated in the passages. Do not use outside knowledge, even if you know the answer.
- If the passages don't contain the answer, say "The passages don't contain the answer." and, in one \
sentence, what they do contain.
- If they contain only part of the answer, give that part and say what is missing.
- Answer in one to three sentences. Cite the passages you used like [2].
- If a number appears without a clear label (for example a table row cut off from its column headings), \
say so rather than guessing what it measures."""

COMPARE_SYSTEM = """You compare three answers to the same question. Each answer was written from a \
different set of retrieved passages, so they may disagree. Identify factual differences only: different \
numbers, doses, durations, conditions, names, yes/no conclusions, or one answer stating a fact that another \
says is missing. Ignore differences in wording, length, order, or citation numbers.

For each difference, give a short label for the point and what each answer says about it ("not stated" if \
it doesn't address it). In "quotes", copy the exact phrases from each answer that carry the differing facts, \
character for character, so they can be highlighted. Keep quotes short (a few words)."""

COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "agree": {"type": "boolean", "description": "True if the answers agree on every fact."},
        "summary": {"type": "string", "description": "One sentence on how the answers differ, or that they agree."},
        "differences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "fixed": {"type": "string"},
                    "paragraph": {"type": "string"},
                    "section": {"type": "string"},
                },
                "required": ["point", "fixed", "paragraph", "section"],
                "additionalProperties": False,
            },
        },
        "quotes": {
            "type": "object",
            "properties": {s: {"type": "array", "items": {"type": "string"}} for s in STRATEGIES},
            "required": list(STRATEGIES),
            "additionalProperties": False,
        },
    },
    "required": ["agree", "summary", "differences", "quotes"],
    "additionalProperties": False,
}


class GenerationError(Exception):
    """A user-facing problem with the optional generation step."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def status() -> dict:
    try:
        import anthropic  # noqa: F401
        installed = True
    except ImportError:
        installed = False
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    return {"installed": installed, "has_key": has_key, "model": MODEL}


def make_client():
    try:
        import anthropic
    except ImportError:
        raise GenerationError("The anthropic package isn't installed. Run: uv sync", 503)
    return anthropic.Anthropic()


def _text(response) -> str:
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _create(client, **kwargs):
    return client.beta.messages.create(
        model=MODEL, betas=[FALLBACK_BETA], fallbacks="default", **kwargs)


def answer_from(client, question: str, passages: list[str]) -> dict:
    numbered = "\n\n".join(f"[{i}]\n{p.strip()}" for i, p in enumerate(passages, 1))
    response = _create(
        client,
        max_tokens=4000,
        output_config={"effort": "low"},
        system=ANSWER_SYSTEM,
        messages=[{"role": "user", "content": f"<passages>\n{numbered}\n</passages>\n\nQuestion: {question}"}],
    )
    if response.stop_reason == "refusal":
        return {"text": None, "refused": True, "model": response.model}
    return {"text": _text(response), "refused": False, "model": response.model,
            "truncated": response.stop_reason == "max_tokens"}


def compare(client, question: str, answers: dict[str, str]) -> dict:
    body = "\n\n".join(f'<answer strategy="{s}">\n{answers[s]}\n</answer>' for s in STRATEGIES)
    response = _create(
        client,
        max_tokens=4000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": COMPARE_SCHEMA}},
        system=COMPARE_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {question}\n\n{body}"}],
    )
    if response.stop_reason == "refusal":
        raise GenerationError("Claude declined to compare these answers.")
    if response.stop_reason == "max_tokens":
        raise GenerationError("The comparison was cut off before it finished.")
    data = json.loads(_text(response))
    # Only keep quotes that really appear in the answer, so highlights are exact.
    data["quotes"] = {s: [q for q in data["quotes"].get(s, []) if q and q in answers[s]] for s in STRATEGIES}
    return data


def generate(client, question: str, results: dict[str, list[dict]]) -> dict:
    """Answer from each strategy's retrieved chunks, then compare the answers."""
    with ThreadPoolExecutor(max_workers=len(STRATEGIES)) as pool:
        futures = {s: pool.submit(answer_from, client, question, [r["text"] for r in results[s]])
                   for s in STRATEGIES}
        answers = {s: f.result() for s, f in futures.items()}
    comparable = {s: a["text"] for s, a in answers.items() if a["text"]}
    comparison = compare(client, question, comparable) if len(comparable) == len(STRATEGIES) else None
    return {"model": MODEL, "answers": answers, "comparison": comparison}
