"""The three chunking strategies Chunkyard compares.

Every chunk records the character span (start, end) it came from in the
source text. Spans are what let us compare strategies: two chunks from
different strategies "agree" when their spans overlap, even though their
text and boundaries differ.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FIXED = "fixed"
PARAGRAPH = "paragraph"
SECTION = "section"
STRATEGIES = (FIXED, PARAGRAPH, SECTION)


@dataclass
class Chunk:
    strategy: str
    index: int
    start: int
    end: int
    text: str  # what gets embedded and shown; may include a header prefix
    headers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fixed size with overlap
# ---------------------------------------------------------------------------

def chunk_fixed(text: str, size: int = 500, overlap: int = 100) -> list[Chunk]:
    """Slice every `size` characters, stepping back `overlap` each time.

    Deliberately naive: it cuts mid-word and mid-sentence, which is exactly
    the behavior we want to show off.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    overlap = max(0, min(overlap, size - 1))
    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end]
        if piece.strip():
            chunks.append(Chunk(FIXED, len(chunks), start, end, piece))
        if end == len(text):
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# Paragraph boundaries
# ---------------------------------------------------------------------------

_BLANK_LINES = re.compile(r"\n[ \t]*\n+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Spans of blank-line separated blocks, trimmed of outer whitespace."""
    spans = []
    pos = 0
    for m in _BLANK_LINES.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    return [s for s in (_trim(text, a, b) for a, b in spans) if s]


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def _split_long(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split one oversized span at sentence boundaries (hard cut as a last resort)."""
    if end - start <= max_chars:
        return [(start, end)]
    # Greedily pack whole sentences; hard-cut only a sentence longer than max.
    boundaries = [start + m.end() for m in _SENTENCE_END.finditer(text[start:end])] + [end]
    pieces: list[tuple[int, int]] = []
    cur, last_fit = start, None
    for b in boundaries:
        if b - cur <= max_chars:
            last_fit = b
            continue
        if last_fit is not None and last_fit > cur:
            pieces.append((cur, last_fit))
            cur = last_fit
        while b - cur > max_chars:
            pieces.append((cur, cur + max_chars))
            cur += max_chars
        last_fit = b
    if cur < end:
        pieces.append((cur, end))
    return [t for t in (_trim(text, a, b) for a, b in pieces) if t]


def chunk_paragraphs(text: str, max_chars: int = 1200) -> list[Chunk]:
    """One chunk per paragraph. Headers end up as their own tiny chunks,
    detached from the content they describe, which is the point."""
    chunks: list[Chunk] = []
    for a, b in paragraph_spans(text):
        for s, e in _split_long(text, a, b, max_chars):
            chunks.append(Chunk(PARAGRAPH, len(chunks), s, e, text[s:e]))
    return chunks


# ---------------------------------------------------------------------------
# Section aware
# ---------------------------------------------------------------------------

_MD_HEADER = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMBERED = re.compile(r"^((?:\d+\.)*\d+)\.?\s+([A-Z].{0,80})$")
_SETEXT = re.compile(r"^(=+|-+)\s*$")
_DASHED = re.compile(r"^[-=]{3,}\s*([A-Z][^-=]*?[A-Za-z)])\s*[-=]{3,}$")  # ----TITLE----
_NUMERIC = re.compile(r"^[\d.,:%/()+*†±<>=-]*\d[\d.,:%/()+*†±<>=-]*$")


def _has_data_numbers(words: list[str]) -> bool:
    """True for lines like "Nasopharyngitis 9.6 10.5": table rows, not headings."""
    return any(_NUMERIC.match(w) for w in words)


INFERRED = 0  # level placeholder for headings guessed from layout alone


def detect_header(line: str, next_line: str | None, prev_blank: bool) -> tuple[int, str] | None:
    """Return (level, title) if `line` looks like a heading, else None.

    Recognizes Markdown `#` headings, setext (underlined) headings, numbered
    headings like "2.1 Installation", short ALL CAPS lines, and (for PDFs and
    plain text) a short capitalized line standing alone as its own paragraph.
    These are heuristics; plain-text and PDF documents rarely mark headings.
    """
    s = line.strip()
    if not s or len(s) > 100:
        return None
    m = _MD_HEADER.match(s)
    if m:
        return len(m.group(1)), m.group(2).strip()
    if next_line is not None and _SETEXT.match(next_line.strip()) and len(next_line.strip()) >= 3:
        return (1 if next_line.strip()[0] == "=" else 2), s
    m = _DASHED.match(s)
    if m:
        return 2, m.group(1).strip()
    words = s.split()
    m = _NUMBERED.match(s)
    if (m and not s.endswith((".", ",", ";", ":")) and len(words) <= 12
            and not _has_data_numbers(words[1:])
            # "2.1 Dosage" or "3 DOSAGE FORMS" can follow text directly; a plain
            # "3 Something" needs a blank line before it to rule out list items.
            and (prev_blank or "." in m.group(1) or m.group(2).upper() == m.group(2))):
        return m.group(1).count(".") + 1, s
    letters = [c for c in s if c.isalpha()]
    if (prev_blank and len(letters) >= 3 and s.upper() == s and len(words) <= 8
            and not s.endswith((".", ",", ";")) and not _has_data_numbers(words)):
        # One caps word may be a real heading ("TROUBLESHOOTING") or just a
        # table/figure label ("PLACEBO"), so it's treated as a guess and can't
        # reset the document's structure.
        return (1 if len(words) >= 2 else INFERRED), s
    next_blank = next_line is not None and not next_line.strip()
    if (prev_blank and next_blank and s[0].isupper() and len(s) <= 70 and len(words) <= 9
            and not s.endswith((".", ",", ";", ":", "!", "?")) and not re.match(r"^[-*•\d]", s)
            and not _has_data_numbers(words)):
        return INFERRED, s
    return None


@dataclass
class Section:
    path: list[str]          # header breadcrumb, e.g. ["Setup", "Wi-Fi"]
    header_span: tuple[int, int] | None
    body_spans: list[tuple[int, int]]  # paragraph spans inside the section


def split_sections(text: str) -> list[Section]:
    lines = text.splitlines(keepends=True)
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)

    # Find header lines (and setext underlines to skip).
    headers: dict[int, tuple[int, str]] = {}
    skip: set[int] = set()
    for i, ln in enumerate(lines):
        if i in skip:
            continue
        prev_blank = i == 0 or not lines[i - 1].strip()
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        h = detect_header(ln, nxt, prev_blank)
        if h:
            headers[i] = h
            if nxt is not None and _SETEXT.match(nxt.strip()) and not _MD_HEADER.match(ln.strip()):
                skip.add(i + 1)

    sections: list[Section] = []
    stack: list[tuple[int, str, bool]] = []  # (level, title, is_numbered)
    explicit_level = 0  # level of the most recent heading found from real markup
    cur = Section([], None, [])
    body_start = 0

    def close(body_end: int) -> None:
        for a, b in paragraph_spans(text[body_start:body_end]):
            cur.body_spans.append((body_start + a, body_start + b))
        if cur.header_span or cur.body_spans:
            sections.append(cur)

    prev_header_line = -2
    for i in sorted(headers):
        level, title = headers[i]
        if level == INFERRED:
            # No markup says how headings nest. A heading right after another
            # heading is treated as its child; otherwise it sits just under the
            # last real (marked-up or numbered) heading. Nesting is capped so a
            # run of look-alike lines can't build an endless breadcrumb.
            only_blank_between = all(not ln.strip() for ln in lines[prev_header_line + 1:i])
            base = explicit_level + 1
            if stack and prev_header_line >= 0 and only_blank_between:
                level = min(stack[-1][0] + 1, base + 1)
            else:
                level = base
        else:
            explicit_level = level
        prev_header_line = i
        close(offsets[i])
        numbered = not lines[i].lstrip().startswith("#") and bool(_NUMBERED.match(title))
        if numbered:
            # "6.2 Immunogenicity" belongs under "6 ADVERSE REACTIONS", never
            # under some unnumbered heading that happened to come in between.
            while stack and not (stack[-1][2] and stack[-1][0] < level):
                stack.pop()
        else:
            while stack and stack[-1][0] >= level:
                stack.pop()
        stack.append((level, title, numbered))
        h_end = offsets[i] + len(lines[i].rstrip("\r\n"))
        if i + 1 in skip:
            body_start = offsets[i + 1] + len(lines[i + 1])
        else:
            body_start = offsets[i] + len(lines[i])
        cur = Section([t for _, t, _ in stack], (offsets[i], h_end), [])
    close(len(text))
    return sections


def chunk_sections(text: str, max_chars: int = 1200) -> list[Chunk]:
    """Keep each section together with its header.

    Paragraphs inside a section are packed together up to `max_chars`. Every
    chunk gets the full header breadcrumb prepended, so a chunk from deep in
    "Warranty > Exclusions" still carries those words when it is embedded.
    """
    chunks: list[Chunk] = []
    for sec in split_sections(text):
        prefix = " > ".join(sec.path)
        prefix_text = prefix + "\n\n" if prefix else ""
        budget = max(200, max_chars - len(prefix_text))

        pieces: list[tuple[int, int]] = []
        for a, b in sec.body_spans:
            pieces.extend(_split_long(text, a, b, budget))

        if not pieces:  # header with no body (e.g. a chapter title before a subsection)
            continue

        group_start, group_end = pieces[0]
        for a, b in pieces[1:]:
            if b - group_start <= budget:
                group_end = b
            else:
                chunks.append(_section_chunk(text, sec, prefix_text, group_start, group_end, len(chunks)))
                group_start, group_end = a, b
        chunks.append(_section_chunk(text, sec, prefix_text, group_start, group_end, len(chunks)))
    return chunks


def _section_chunk(text: str, sec: Section, prefix_text: str, start: int, end: int, idx: int) -> Chunk:
    # The span covers the header too when this is the section's first chunk,
    # so the document map shows the header as part of what was retrieved.
    span_start = sec.header_span[0] if sec.header_span and start == sec.body_spans[0][0] else start
    return Chunk(SECTION, idx, span_start, end, prefix_text + text[start:end], list(sec.path))


# ---------------------------------------------------------------------------

def chunk_all(text: str, size: int = 500, overlap: int = 100, max_chars: int = 1200) -> dict[str, list[Chunk]]:
    return {
        FIXED: chunk_fixed(text, size, overlap),
        PARAGRAPH: chunk_paragraphs(text, max_chars),
        SECTION: chunk_sections(text, max_chars),
    }
