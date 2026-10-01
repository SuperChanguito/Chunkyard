"""Warning flags for chunks that are likely to be hard to use on their own.

These never change retrieval. They point out *why* a retrieved chunk may be
a poor answer: it was cut mid-sentence, it's tiny, or its numbers have lost
the labels that say what they measure.
"""

from __future__ import annotations

import re

SHORT_CHARS = 100        # chunk flagged as short below this
TOO_SMALL_AVG = 200      # strategy flagged when its average chunk is smaller
TOO_SMALL_MIN = 30       # ...or its smallest chunk is smaller than this

STARTS_MID = "starts_mid_sentence"
ENDS_MID = "ends_mid_sentence"
SHORT = "short"
UNLABELED_NUMBERS = "unlabeled_numbers"

_SENTENCE_END = tuple('.!?:;)"”’')
_BULLET = re.compile(r"^(?:[•▪◦*-]|o\s|\d+[.)]\s|[a-z][.)]\s|#)")
_NUMBER = re.compile(r"[-−+]?\d+(?:[.,]\d+)*%?")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]+")
# Units and filler that don't tell you what a number measures.
_NOT_LABELS = {
    "mg", "ml", "kg", "g", "cm", "mm", "km", "lb", "lbs", "oz", "in", "ft", "mcg", "µg",
    "hr", "hrs", "min", "sec", "ms", "n", "vs", "to", "or", "and", "of", "per", "the",
    "a", "an", "at", "by", "x", "ci", "p", "na", "nr",
}


def starts_mid_sentence(text: str, start: int) -> bool:
    """True if the chunk at `start` begins partway through a sentence."""
    if start <= 0:
        return False
    if text[start - 1].isalnum() and start < len(text) and text[start].isalnum():
        return True  # cut through a word
    before = text[max(0, start - 300):start]
    stripped = before.rstrip()
    if not stripped:
        return False
    gap = before[len(stripped):]
    if "\n\n" in gap or stripped.endswith(_SENTENCE_END):
        return False
    first = text[start:start + 3].lstrip()
    if _BULLET.match(text[start:start + 4]) or first[:1] in "•▪◦":
        return False
    # A new line that starts with a capital is usually a new item or heading.
    if "\n" in gap and first[:1].isupper():
        return False
    return True


def ends_mid_sentence(text: str, end: int) -> bool:
    """True if the chunk ending at `end` stops partway through a sentence."""
    if end >= len(text):
        return False
    if end > 0 and text[end - 1].isalnum() and text[end].isalnum():
        return True  # cut through a word
    body = text[:end].rstrip()
    if not body or body.endswith(_SENTENCE_END):
        return False
    after = text[end:end + 300]
    rest = after.lstrip()
    if not rest:
        return False
    gap = after[:len(after) - len(rest)] + text[len(body):end]
    if "\n\n" in gap:
        return False  # paragraph ended without punctuation (a heading, a list item)
    if "\n" in gap and (rest[:1].isupper() or _BULLET.match(rest[:4])):
        return False
    return True


def _labels(line: str) -> list[str]:
    return [w for w in _WORD.findall(line) if w.lower() not in _NOT_LABELS]


def _numbers(line: str) -> list[str]:
    # Ignore digits inside words like "LDL-C2" or "H2O"; count standalone values.
    return [m.group(0) for m in _NUMBER.finditer(line)
            if not (m.start() > 0 and line[m.start() - 1].isalpha())]


def unlabeled_numbers(chunk: str) -> bool:
    """True if the chunk has a line of numbers with nothing saying what they are.

    A line is "numeric" when it has two or more values and fewer descriptive
    words than values (table rows like "(n = 562) -59 -50 -46 -34"), or is a
    value on its own. It counts as labeled if one of the three lines above it
    in the same chunk looks like a column header: descriptive words, no
    numbers, not a sentence.
    """
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        nums, words = _numbers(line), _labels(line)
        numeric = (len(nums) >= 2 and len(words) < len(nums)) or (len(nums) == 1 and not words)
        if not numeric:
            continue
        header_above = any(
            len(_labels(prev)) >= 2 and not _numbers(prev) and not prev.endswith(".")
            for prev in lines[max(0, i - 3):i])
        if not header_above:
            return True
    return False


def chunk_flags(text: str, start: int, end: int) -> list[str]:
    flags = []
    if starts_mid_sentence(text, start):
        flags.append(STARTS_MID)
    if ends_mid_sentence(text, end):
        flags.append(ENDS_MID)
    if end - start < SHORT_CHARS:
        flags.append(SHORT)
    if unlabeled_numbers(text[start:end]):
        flags.append(UNLABELED_NUMBERS)
    return flags


def too_small(avg_chars: float, min_chars: int) -> str | None:
    """Why a strategy's chunks are too small to be useful, or None.

    "avg": chunks are small across the board. "min": the average is fine,
    but some chunks are tiny fragments.
    """
    if avg_chars < TOO_SMALL_AVG:
        return "avg"
    if min_chars < TOO_SMALL_MIN:
        return "min"
    return None
