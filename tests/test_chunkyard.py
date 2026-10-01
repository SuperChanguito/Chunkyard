import re
import zlib
from pathlib import Path

import numpy as np
import pytest

from chunkyard.chunking import chunk_fixed, chunk_paragraphs, chunk_sections, split_sections
from chunkyard.engine import Store, overlap_ratio, query
from chunkyard.loaders import infer_paragraphs, load_document

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "trailhead-tent-manual.md"


@pytest.fixture(scope="module")
def sample_text():
    return load_document(SAMPLE.name, SAMPLE.read_bytes())


# --- chunking ---------------------------------------------------------------

def test_fixed_covers_text_with_overlap():
    text = "abcdefghij" * 30  # 300 chars
    chunks = chunk_fixed(text, size=100, overlap=20)
    assert [c.start for c in chunks] == [0, 80, 160, 240]
    assert all(c.text == text[c.start:c.end] for c in chunks)
    assert chunks[-1].end == len(text)


def test_fixed_rejects_bad_size():
    with pytest.raises(ValueError):
        chunk_fixed("abc", size=0)


def test_paragraphs_detach_headers(sample_text):
    chunks = chunk_paragraphs(sample_text)
    assert any(c.text == "## Warranty" for c in chunks)
    assert all(c.text == sample_text[c.start:c.end] for c in chunks)


def test_long_paragraph_splits_at_sentences():
    para = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_paragraphs(para, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_sections_keep_header_breadcrumb(sample_text):
    chunks = chunk_sections(sample_text)
    covered = next(c for c in chunks if c.headers[-1:] == ["What's covered"])
    assert covered.headers[-2] == "Warranty"
    assert covered.text.startswith("Trailhead 2 Tent: Owner's Manual > Warranty > What's covered\n\n")
    assert "two years" in covered.text
    # The span covers the header line in the source, so the map shows it.
    assert sample_text[covered.start:].startswith("### What's covered")


def test_sections_respect_max_size(sample_text):
    for c in chunk_sections(sample_text, max_chars=300):
        body = c.text.split("\n\n", 1)[-1] if c.headers else c.text
        assert len(body) <= 300


def test_header_styles_detected():
    text = ("Intro\n=====\n\nHello there.\n\n1.2 Installation Steps\n\nDo this.\n\n"
            "TROUBLESHOOTING\n\nRestart it.\n\nPlain Heading\n\nBody text.\n")
    titles = [s.path[-1] for s in split_sections(text) if s.path]
    assert titles == ["Intro", "1.2 Installation Steps", "TROUBLESHOOTING", "Plain Heading"]


def test_inferred_headings_nest_only_when_adjacent():
    text = "Setup\n\nChoosing a site\n\nFlat ground.\n\nPitching\n\nStake it.\n"
    paths = [s.path for s in split_sections(text) if s.body_spans]
    assert paths == [["Setup", "Choosing a site"], ["Pitching"]]


# --- loaders ----------------------------------------------------------------

def test_infer_paragraphs_on_pdf_style_text():
    full = "This line is long enough to count as a full printed line of text here"
    lines = [full, full, "and it ends here.", "Next Heading", full, full, "the end."] * 3
    out = infer_paragraphs("\n".join(lines))
    assert "ends here.\n\nNext Heading" in out


def test_rejects_unknown_extension():
    with pytest.raises(ValueError):
        load_document("photo.jpg", b"\xff\xd8")


# --- engine -----------------------------------------------------------------

class FakeEmbedder:
    """Bag-of-words hashing, so tests don't need the real model."""
    max_tokens = 256
    name = "fake"
    ready = True

    def encode(self, texts):
        out = np.zeros((len(texts), 512))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, zlib.crc32(w.encode()) % 512] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)

    def token_counts(self, texts):
        return [len(t.split()) for t in texts]


def test_overlap_ratio():
    assert overlap_ratio((0, 10), (5, 20)) == 0.5
    assert overlap_ratio((0, 10), (10, 20)) == 0
    assert overlap_ratio((0, 100), (40, 50)) == 1.0


def test_query_marks_agreement(sample_text):
    store = Store(FakeEmbedder())
    doc = store.add("s", sample_text, 500, 100, 1200)
    res = query(store, doc.id, "coverage two years receipt", k=5)
    assert set(res["results"]) == {"fixed", "paragraph", "section"}
    assert all(len(v) == 5 for v in res["results"].values())
    assert len(res["pairs"]) == 3
    for s, rs in res["results"].items():
        assert [r["rank"] for r in rs] == [1, 2, 3, 4, 5]
        assert rs == sorted(rs, key=lambda r: -r["score"])
        for r in rs:
            assert r["unique"] == (not r["also_found_by"])
    # The obvious answer passage should be found by everyone.
    assert res["top1_all_agree"]


def test_unknown_doc_raises():
    with pytest.raises(KeyError):
        query(Store(FakeEmbedder()), "nope", "q")


# --- regressions from a real FDA drug label PDF ------------------------------

def test_table_rows_and_lone_caps_are_not_headings():
    text = ("6 ADVERSE REACTIONS\n\n6.1 Clinical Trials\n\nIntro text.\n\nPLACEBO\n\n"
            "Nasopharyngitis 9.6 10.5\n\nBack pain 5.6 6.2\n\nMore text.\n\n6.2 Immunogenicity\n\nBody.\n")
    paths = [s.path for s in split_sections(text)]
    assert ["6 ADVERSE REACTIONS", "6.2 Immunogenicity"] in paths
    assert not any(any("9.6" in t for t in p) for p in paths)
    # A lone caps label may become a guessed subheading, but only under the right parent.
    assert all(p[:2] == ["6 ADVERSE REACTIONS", "6.1 Clinical Trials"] for p in paths if "PLACEBO" in p)


def test_numbered_heading_ignores_unnumbered_parent():
    text = ("6 ADVERSE REACTIONS\n\n6.1 Trials\n\nText.\n\nSome Inferred Label\n\nText.\n\n"
            "6.2 Immunogenicity\n\nBody.\n")
    last = split_sections(text)[-1]
    assert last.path == ["6 ADVERSE REACTIONS", "6.2 Immunogenicity"]


def test_dashed_headings():
    text = "------INDICATIONS AND USAGE------\nUse it daily.\n"
    assert split_sections(text)[0].path == ["INDICATIONS AND USAGE"]


def test_strip_running_lines():
    from chunkyard.loaders import strip_running_lines
    pages = [f"Body text {i} that differs from every other page in this document.\n"
             f"Reference ID: 5647060\nPage {i} of 4" for i in range(1, 5)]
    out = strip_running_lines(pages)
    assert all("Reference ID" not in p and "Page" not in p for p in out)
    assert all("Body text" in p for p in out)
