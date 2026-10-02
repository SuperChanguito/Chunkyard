"""Turn an uploaded file into plain text."""

from __future__ import annotations

import hashlib
import io
import re
import statistics
import threading
import unicodedata
from collections import OrderedDict

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".text"}
NOT_TEXT = "This doesn't look like a text file. Upload a .txt, .md, or .pdf."

# Parsed PDF text by file hash, so re-chunking a PDF doesn't re-parse it.
_PDF_CACHE: OrderedDict[str, str] = OrderedDict()
_PDF_CACHE_SIZE = 8
_pdf_lock = threading.Lock()


def load_document(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        text = _cached_pdf_text(data)
    elif any(name.endswith(ext) for ext in TEXT_EXTENSIONS) or "." not in name:
        text = _decode(data)
    else:
        raise ValueError("Upload a .txt, .md, or .pdf file.")
    text = normalize(text)
    if not text.strip():
        raise ValueError("No text found in that file. If it's a scanned PDF, it has no text layer to read.")
    return text


def _decode(data: bytes) -> str:
    """Bytes to text: UTF-8, UTF-16 only when it has a byte-order mark, then
    Windows-1252, then UTF-8 with replacement characters.

    Trying UTF-16 without a BOM would "succeed" on any even-length file and
    turn ordinary Windows text into CJK garbage, so it isn't guessed.
    """
    text = None
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            text = data.decode("utf-16")
        except UnicodeDecodeError:
            text = None
    if text is None:
        for enc in ("utf-8-sig", "cp1252"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    if text is None:
        text = data.decode("utf-8", errors="replace")
    if not looks_like_text(text):
        raise ValueError(NOT_TEXT)
    return text


def looks_like_text(text: str) -> bool:
    """False when more than 5% of characters are control or replacement
    characters, as with a binary file renamed to .txt."""
    if not text:
        return True
    bad = sum(1 for c in text
              if c == "�" or (unicodedata.category(c) == "Cc" and c not in "\n\r\t\f\v"))
    return bad / len(text) <= 0.05


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = re.sub(r"[  ]+\n", "\n", text)       # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)            # runs of blank lines
    return text.strip() + "\n"


def _cached_pdf_text(data: bytes) -> str:
    key = hashlib.sha256(data).hexdigest()
    with _pdf_lock:
        if key in _PDF_CACHE:
            _PDF_CACHE.move_to_end(key)
            return _PDF_CACHE[key]
    text = _pdf_text(data)
    with _pdf_lock:
        _PDF_CACHE[key] = text
        while len(_PDF_CACHE) > _PDF_CACHE_SIZE:
            _PDF_CACHE.popitem(last=False)
    return text


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return pages_to_text([(page.extract_text() or "") for page in reader.pages])


def pages_to_text(pages: list[str]) -> str:
    """Clean up per-page text extracted from a PDF into one document."""
    pages = strip_running_lines(pages)
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    text = rejoin_hyphens(text)
    return infer_paragraphs(text)


_LINE_HYPHEN = re.compile(r"(\w+)-\n(\w+)")


def rejoin_hyphens(text: str) -> str:
    """Undo hyphens that a PDF put at a line break.

    "subcuta-\nneous" becomes "subcutaneous". The hyphen is kept ("Non-HDL",
    "patient-safety") when either part isn't all lowercase letters, or when
    the document itself shows it's a compound: the hyphenated form appears
    elsewhere, or both parts are words on their own and the joined word isn't.
    """
    # Words used elsewhere in the document (the split fragments themselves don't count).
    words = set(re.findall(r"[a-z]+", _LINE_HYPHEN.sub(" ", text).lower()))
    flat = text.replace("-\n", "-").lower()

    def fix(m: re.Match) -> str:
        left, right = m.group(1), m.group(2)
        lowercase = left.isalpha() and right.isalpha() and left.islower() and right.islower()
        joined = (left + right).lower()
        compound = (
            not lowercase
            or flat.count(f"{left}-{right}".lower()) > 1
            or (joined not in words and left.lower() in words and right.lower() in words))
        return f"{left}-{right}" if compound else left + right

    return _LINE_HYPHEN.sub(fix, text)


def strip_running_lines(pages: list[str], edge: int = 3) -> list[str]:
    """Remove running headers/footers: lines near the top or bottom of a page
    that repeat on many pages ("Reference ID: 12345", "Page 4", a title).
    Left in, they turn into fake headings and noise in every chunk."""
    if len(pages) < 3:
        return pages

    def key(line: str) -> str:
        line = line.strip().lower()
        # Short lines match ignoring numbers ("Page 3 of 9" == "Page 4 of 9");
        # longer lines must repeat exactly, so real text is never removed.
        return re.sub(r"\d+", "#", line) if len(line) <= 40 else line

    split = [p.split("\n") for p in pages]
    counts: dict[str, int] = {}
    for lines in split:
        nonblank = [ln for ln in lines if ln.strip()]
        for k in {key(ln) for ln in nonblank[:edge] + nonblank[-edge:]}:
            counts[k] = counts.get(k, 0) + 1
    # A quarter, not half: one PDF often bundles several separately numbered
    # documents (an FDA label plus its patient leaflet, for example).
    threshold = max(3, len(pages) // 4)
    running = {k for k, n in counts.items() if n >= threshold and k}

    out = []
    for lines in split:
        nonblank_idx = [i for i, ln in enumerate(lines) if ln.strip()]
        edges = set(nonblank_idx[:edge] + nonblank_idx[-edge:])
        out.append("\n".join(ln for i, ln in enumerate(lines) if not (i in edges and key(ln) in running)))
    return out


def infer_paragraphs(text: str) -> str:
    """PDF text usually comes out as one line per printed line with no blank
    lines between paragraphs. If that's the case, guess paragraph breaks: a
    line that ends a sentence and is noticeably shorter than a full line is
    probably the last line of its paragraph."""
    lines = text.split("\n")
    nonblank = [ln for ln in lines if ln.strip()]
    if len(nonblank) < 10:
        return text
    blank_ratio = (len(lines) - len(nonblank)) / len(lines)
    if blank_ratio > 0.05:
        return text  # already has paragraph breaks

    full = statistics.quantiles([len(ln) for ln in nonblank], n=10)[7]  # ~80th percentile
    out = []
    for i, ln in enumerate(lines):
        out.append(ln)
        s = ln.rstrip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        short = len(s) < 0.75 * full
        ends_sentence = s.endswith((".", "!", "?", ":", '."', ".”"))
        looks_like_heading = 0 < len(s) < 60 and not s.endswith((".", ",", ";")) and nxt[:1].isupper()
        next_is_heading = (0 < len(nxt) < 60 and nxt[:1].isupper()
                           and not nxt.endswith((".", ",", ";", ":", "!", "?")))
        if s and nxt and ((short and (ends_sentence or looks_like_heading))
                          or (ends_sentence and next_is_heading)):
            out.append("")
    return "\n".join(out)
