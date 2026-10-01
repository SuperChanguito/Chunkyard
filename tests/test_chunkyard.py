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
    # A multi-word ALL CAPS table title is a top-level heading, so without the
    # numbered-parent rule "6.2" would end up filed under it.
    text = ("6 ADVERSE REACTIONS\n\n6.1 Trials\n\nText.\n\nSTUDY RESULTS TABLE\n\nText.\n\n"
            "6.2 Immunogenicity\n\nBody.\n")
    last = split_sections(text)[-1]
    assert last.path[-1] == "6.2 Immunogenicity"
    assert "STUDY RESULTS TABLE" not in last.path


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


# --- end to end on a fictional drug label, as extracted from a PDF ----------
# Pages are separated by form feeds. The text mimics the FDA label that first
# exposed these problems: running footers, dashed headings, numbered sections
# right after body text, a results table, and a lone ALL CAPS column label.

LABEL = Path(__file__).resolve().parent / "fixtures" / "fictional-label-pages.txt"


@pytest.fixture(scope="module")
def label_text():
    from chunkyard.loaders import normalize, pages_to_text
    pages = LABEL.read_text(encoding="utf-8").split("\f\n")
    assert len(pages) == 5
    return normalize(pages_to_text(pages))


def test_label_footers_removed(label_text):
    assert "Reference ID" not in label_text
    assert not re.search(r"Page \d of 5", label_text)


def test_label_section_tree(label_text):
    paths = [s.path for s in split_sections(label_text)]
    for expected in (
        ["HIGHLIGHTS OF PRESCRIBING INFORMATION", "INDICATIONS AND USAGE"],
        ["2 DOSAGE AND ADMINISTRATION", "2.1 Recommended Dosage"],
        ["2 DOSAGE AND ADMINISTRATION", "2.2 Missed Doses"],
        ["2 DOSAGE AND ADMINISTRATION", "2.3 Storage Instructions"],  # after a page break
        ["6 ADVERSE REACTIONS", "6.2 Immunogenicity"],                # after the table
        ["8 USE IN SPECIFIC POPULATIONS", "8.2 Lactation"],
        ["17 PATIENT COUNSELING INFORMATION"],
    ):
        assert any(p[:len(expected)] == expected for p in paths), expected
    titles = [t for p in paths for t in p]
    assert not any(re.search(r"\d\.\d", t) and not re.match(r"\d+\.\d+ ", t) for t in titles), titles
    assert "PLACEBO" not in [p[0] for p in paths]
    assert max(len(p) for p in paths) <= 4


def test_label_answer_chunk_carries_its_heading(label_text):
    chunk = next(c for c in chunk_sections(label_text) if "Within 7 days" in c.text)
    assert chunk.headers == ["2 DOSAGE AND ADMINISTRATION", "2.2 Missed Doses"]
    assert chunk.text.startswith("2 DOSAGE AND ADMINISTRATION > 2.2 Missed Doses\n\n")


def test_strip_page_numbers_in_bundled_documents():
    # 12 pages: a 4-page label numbered "N of 4" plus 8 pages of something else.
    from chunkyard.loaders import strip_running_lines
    pages = [f"Label text unique to page {i}, long enough to be real content here.\n{i} of 4"
             for i in range(1, 5)]
    pages += [f"Leaflet text unique to page {i}, long enough to be real content too." for i in range(8)]
    out = strip_running_lines(pages)
    assert not any(" of 4" in p for p in out)
    assert all("unique to page" in p for p in out)


def test_results_report_their_sections(sample_text):
    store = Store(FakeEmbedder())
    doc = store.add("s", sample_text, 500, 100, 1200)
    res = query(store, doc.id, "coverage two years receipt", k=5)
    top = res["results"]["section"][0]
    assert top["section"][-2:] == ["Warranty", "What's covered"]
    assert top["section_end"] == top["section"]
    # Fixed chunks can straddle a section boundary; both ends are reported.
    assert any(r["section"] != r["section_end"] for r in res["results"]["fixed"])


def test_sample_registry_files_and_questions():
    from chunkyard.samples import SAMPLES, SAMPLES_DIR
    assert (SAMPLES_DIR / SAMPLES["tent"]["file"]).is_file()
    for spec in SAMPLES.values():
        assert len(spec["questions"]) >= 3
    fda = " ".join(SAMPLES["fda"]["questions"]).lower()
    assert "dose" in fda and "used to treat" in fda


# --- scoring against a marked answer -----------------------------------------

def test_contains_answer_rule():
    from chunkyard.engine import contains_answer
    answer = (100, 140)                            # e.g. "two years from the date of purchase"
    assert contains_answer((0, 500), answer)       # big chunk holding it
    assert contains_answer((110, 300), answer)     # covers 75% of it
    assert not contains_answer((125, 300), answer)  # covers only 37.5%
    assert not contains_answer((80, 99), answer)   # a heading just before it


def test_header_only_chunk_does_not_count_as_answer(sample_text):
    from chunkyard.engine import answer_rank
    start = sample_text.index("Coverage lasts for two years")
    answer = (start, start + len("Coverage lasts for two years from the date of purchase"))
    store = Store(FakeEmbedder())
    doc = store.add("s", sample_text, 500, 100, 1200)
    header = next(c for c in doc.chunks["paragraph"] if c.text == "## Warranty")
    fake = [{"rank": 1, "start": header.start, "end": header.end}]
    assert answer_rank(fake, answer) is None


def test_query_reports_answer_rank(sample_text):
    store = Store(FakeEmbedder())
    doc = store.add("s", sample_text, 500, 100, 1200)
    start = sample_text.index("Coverage lasts for two years")
    res = query(store, doc.id, "coverage two years receipt", 5, (start, start + 40))
    assert set(res["answer_rank"]) == {"fixed", "paragraph", "section"}
    for s, rank in res["answer_rank"].items():
        hits = [r["rank"] for r in res["results"][s] if r["has_answer"]]
        assert rank == (hits[0] if hits else None)
    assert "answer_rank" not in query(store, doc.id, "coverage two years receipt")


def test_evaluate_summary(sample_text):
    from chunkyard.engine import evaluate
    store = Store(FakeEmbedder())
    doc = store.add("s", sample_text, 500, 100, 1200)
    def span(phrase):
        i = sample_text.index(phrase)
        return (i, i + len(phrase))
    items = [("coverage two years receipt", span("Coverage lasts for two years")),
             ("zippers sticking pliers", span("squeezed gently with pliers")),
             ("nothing relevant xyzzy", span("Peak interior height is 102 cm"))]
    out = evaluate(store, doc.id, items, k=5)
    assert len(out["rows"]) == 3
    for s, summ in out["summary"].items():
        ranks = [r["ranks"][s] for r in out["rows"] if r["ranks"][s] is not None]
        assert summ["found"] == len(ranks) and summ["total"] == 3
        assert summ["avg_rank"] == (round(sum(ranks) / len(ranks), 2) if ranks else None)
    assert doc.fingerprint and len(doc.fingerprint) == 16
