"""Turn an uploaded file into plain text."""

from __future__ import annotations

import io
import re
import statistics

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".text"}


def load_document(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        text = _pdf_text(data)
    elif any(name.endswith(ext) for ext in TEXT_EXTENSIONS) or "." not in name:
        text = _decode(data)
    else:
        raise ValueError("Upload a .txt, .md, or .pdf file.")
    text = normalize(text)
    if not text.strip():
        raise ValueError("No text found in that file. If it's a scanned PDF, it has no text layer to read.")
    return text


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = re.sub(r"[  ]+\n", "\n", text)       # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)            # runs of blank lines
    return text.strip() + "\n"


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return pages_to_text([(page.extract_text() or "") for page in reader.pages])


def pages_to_text(pages: list[str]) -> str:
    """Clean up per-page text extracted from a PDF into one document."""
    pages = strip_running_lines(pages)
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    return infer_paragraphs(text)


def strip_running_lines(pages: list[str], edge: int = 3) -> list[str]:
    """Remove running headers/footers: lines near the top or bottom of a page
    that repeat on most pages ("Reference ID: 12345", "Page 4", a title).
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
    threshold = max(3, len(pages) // 2)
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
