"""Regression tests for the independent QA review (one section per finding).

Retrieval tests use the fake embedder from test_chunkyard, so no model is
needed. Browser-only fixes (answer marking, stale responses, result counts)
are covered by tests/ui/qa_ui_check.js.
"""

import re
from pathlib import Path

import pytest

from chunkyard.engine import SHORT_CHARS, Store, evaluate, query, same_passage
from chunkyard.loaders import load_document
from chunkyard.samples import SAMPLES, SAMPLES_DIR
from test_chunkyard import FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tent():
    p = SAMPLES_DIR / SAMPLES["tent"]["file"]
    text = load_document(p.name, p.read_bytes())
    store = Store(FakeEmbedder())
    return store, store.add(p.name, text, 500, 100, 1200)


# --- 1. A lone heading never counts as "the same passage" ----------------------

def test_short_chunks_never_agree():
    heading, passage = (100, 122), (100, 420)          # 22-char heading inside a passage
    assert not same_passage(heading, passage)
    assert same_passage((0, 400), (100, 420))          # 300 shared of 420
    assert not same_passage((0, 1200), (1000, 1150))   # 150 shared of 1200: under 25%


def test_heading_only_top_hit_is_not_agreement(tent):
    store, doc = tent
    res = query(store, doc.id, "What's not covered")
    para = res["results"]["paragraph"][0]
    assert doc.text[para["start"]:para["end"]] == "### What's not covered"
    assert not res["top1_all_agree"]
    assert all(not p["top1_agree"] for p in res["pairs"] if "paragraph" in (p["a"], p["b"]))
    frag = next(f for f in res["top1_fragments"] if f["strategy"] == "paragraph")
    assert frag["heading"] == "What's not covered"
    assert "section" in frag["inside"]


def test_badges_and_links_never_use_short_chunks(tent):
    store, doc = tent
    for q in SAMPLES["tent"]["questions"] + ["What's not covered", "warranty"]:
        res = query(store, doc.id, q)
        by_key = {(s, r["rank"]): r for s, rs in res["results"].items() for r in rs}
        for s, rs in res["results"].items():
            for r in rs:
                short = r["end"] - r["start"] < SHORT_CHARS
                for m in r["matches"]:
                    other = by_key[(m["strategy"], m["rank"])]
                    assert not short and other["end"] - other["start"] >= SHORT_CHARS
                    assert any(o["strategy"] == s and o["rank"] == r["rank"] for o in other["matches"])
                assert {f["strategy"] for f in r["also_found_by"]} == {m["strategy"] for m in r["matches"]}


# --- 2. Scoring: a short selected answer can be credited to every strategy -----

def test_answer_up_to_half_fixed_size_always_fits_a_fixed_chunk(tent):
    """Answers are limited to half the fixed chunk size (enforced in the UI),
    which guarantees some fixed chunk covers at least half of any answer, so
    Fixed can be credited wherever the answer sits."""
    from chunkyard.engine import contains_answer
    _, doc = tent
    fixed = [(c.start, c.end) for c in doc.chunks["fixed"]]
    length = doc.settings["size"] // 2
    for start in range(0, len(doc.text) - length, 7):
        assert any(contains_answer(c, (start, start + length)) for c in fixed), start


def test_selected_words_are_credited_to_every_strategy_that_has_them(tent):
    store, doc = tent
    phrase = "Coverage lasts for two years from the date of purchase"
    i = doc.text.index(phrase)
    res = query(store, doc.id, "How long does coverage last? two years purchase", 5, (i, i + len(phrase)))
    for s, rs in res["results"].items():
        holders = [r["rank"] for r in rs if r["start"] <= i and i + len(phrase) <= r["end"]]
        if holders:
            assert res["answer_rank"][s] == holders[0], s


# --- 3. No table, figure, or diagram labels in PDF breadcrumbs ----------------

FAKE_HEADINGS = {"REPATHA", "REPATHA†", "Pooled", "Stalins", "Incidence", "Hazard", "No. at Risk",
                 "AMGEN", "Medicine", "Stomach", "WATCH", "GRH0434v1", "1XXXXXX",
                 "Total Treatment Group LDL-C Non-HDL-C Apo B Cholesterol"}
REAL_SUBHEADINGS = {"Risk Summary", "Absorption", "Adverse Reactions in a 52-Week Controlled Trial"}


@pytest.fixture(scope="module")
def fda_breadcrumbs():
    pytest.importorskip("pypdf")
    from chunkyard.chunking import chunk_sections
    p = SAMPLES_DIR / SAMPLES["fda"]["file"]
    if not p.is_file():
        pytest.skip("FDA sample PDF not present")
    text = load_document(p.name, p.read_bytes())
    return {h for c in chunk_sections(text) for h in c.headers}


def test_fda_breadcrumbs_have_no_fake_headings(fda_breadcrumbs):
    assert not (fda_breadcrumbs & FAKE_HEADINGS)
    assert not [h for h in fda_breadcrumbs if len(re.sub(r"[^A-Za-z]", "", h)) < 2]


def test_fda_breadcrumbs_keep_real_subheadings(fda_breadcrumbs):
    assert REAL_SUBHEADINGS <= fda_breadcrumbs


def test_guessed_heading_needs_prose_not_a_table():
    from chunkyard.chunking import split_sections
    text = ("6 ADVERSE REACTIONS\n\nIntro sentence here.\n\nREPATHA\n\n(N = 599)\n\n%\n\n"
            "Hazard\n\nRatio\n\nThe table above shows the hazard ratios.\n\n"
            "Risk Summary\n\nThere is no information about this.\n")
    paths = {p for s in split_sections(text) for p in s.path}
    assert "REPATHA" not in paths and "Hazard" not in paths
    assert "Risk Summary" in paths


# --- 4. Windows-1252 text is never decoded as UTF-16 garbage -------------------

@pytest.mark.parametrize("text", ["Café “quotes” and résumé", "Café “quotes” and résumé!"])
def test_cp1252_decodes_for_even_and_odd_lengths(text):
    data = text.encode("cp1252")
    assert load_document("a.txt", data).strip() == text


def test_utf16_with_bom_still_decodes():
    assert load_document("a.txt", "Héllo wörld".encode("utf-16")).strip() == "Héllo wörld"


def test_binary_file_is_refused():
    with pytest.raises(ValueError, match="doesn't look like a text file"):
        load_document("a.txt", bytes(range(256)) * 8)


# --- 6. Speed: embeddings and parsed PDFs are cached --------------------------

def test_rechunking_only_embeds_new_chunks():
    p = SAMPLES_DIR / SAMPLES["tent"]["file"]
    text = load_document(p.name, p.read_bytes())
    store = Store(FakeEmbedder())
    store.add("t", text, 500, 100, 1200)
    first = store.embedded
    doc2 = store.add("t", text, 300, 50, 1200)  # only the fixed settings changed
    new_fixed = len({c.text for c in doc2.chunks["fixed"]})
    assert store.embedded - first <= new_fixed
    before = store.embedded
    store.add("t", text, 300, 50, 1200)          # identical settings: nothing new
    assert store.embedded == before


def test_pdf_text_is_parsed_once(monkeypatch):
    from chunkyard import loaders
    calls = []
    monkeypatch.setattr(loaders, "_pdf_text", lambda data: calls.append(1) or "Hello there. Some text.")
    data = b"%PDF-fake unique bytes for this test"
    loaders.load_document("x.pdf", data)
    loaders.load_document("x.pdf", data)
    assert len(calls) == 1


def test_upload_limit_is_5mb():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from chunkyard.app import MAX_UPLOAD, app
    assert MAX_UPLOAD == 5 * 1024 * 1024
    r = TestClient(app).post("/api/documents", files={"file": ("big.txt", b"a" * (MAX_UPLOAD + 1))})
    assert r.status_code == 413 and "5 MB" in r.json()["detail"]


# --- 7. Smaller fixes -----------------------------------------------------------

def test_closing_punctuation_needs_a_sentence_end_before_it():
    from chunkyard.flags import ends_mid_sentence
    t = "Inject 140 mg (1 mL) under the skin. Store it in the carton.) Then wait."
    assert ends_mid_sentence(t, t.index(" under"))        # "...(1 mL)" is mid-sentence
    assert not ends_mid_sentence(t, t.index(" Then"))     # "...carton.)" is not
    s = "If a dose is missed: take it now."
    assert ends_mid_sentence(s, s.index(" take"))         # a bare colon isn't a sentence end


def test_unlabeled_numbers_exceptions():
    from chunkyard.flags import unlabeled_numbers
    assert not unlabeled_numbers("The tent weighs 2.1 kg (4.6 lb) and packs to 45 x 15 cm.")
    assert not unlabeled_numbers("Revised: 7/2025")
    assert unlabeled_numbers("(n = 562) -59 -50 -46 -34")  # still flags real table rows


def test_line_break_hyphens():
    from chunkyard.loaders import rejoin_hyphens
    t = ("Non-\nHDL levels and subcuta-\nneous use. Report patient-\nsafety issues. "
         "The patient reports safety data.")
    out = rejoin_hyphens(t)
    assert "Non-HDL" in out and "subcutaneous" in out and "patient-safety" in out


def _lum(hex_color):
    c = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast(a, b):
    hi, lo = sorted([_lum(a), _lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_colour_contrast_meets_4_5():
    html = (ROOT / "chunkyard" / "static" / "index.html").read_text(encoding="utf-8")
    light = html[html.index(":root {"):html.index("@media (prefers-color-scheme: dark)")]
    dark = html[html.index(':root[data-theme="dark"] {'):]
    dark = dark[:dark.index("}")]

    def token(block, name):
        return re.search(rf"--{name}:\s*(#[0-9a-fA-F]{{6}})", block).group(1)

    assert _contrast(token(light, "faint"), token(light, "surface")) >= 4.5
    assert _contrast(token(dark, "faint"), token(dark, "surface")) >= 4.5
    assert _contrast(token(light, "fixed"), token(light, "fixed-soft")) >= 4.5
    assert _contrast(token(light, "warn"), token(light, "warn-soft")) >= 4.5


def test_mrr_counts_misses_as_zero(tent):
    store, doc = tent
    i = doc.text.index("Coverage lasts for two years")
    j = doc.text.index("Peak interior height is 102 cm")
    out = evaluate(store, doc.id, [("coverage two years receipt", (i, i + 28)),
                                   ("nothing relevant xyzzy", (j, j + 30))])
    for s, sm in out["summary"].items():
        ranks = [r["ranks"][s] for r in out["rows"]]
        assert sm["mrr"] == round(sum(1 / r for r in ranks if r) / len(ranks), 3)
