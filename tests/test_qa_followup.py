"""Regression tests for the QA re-check (follow-up to test_qa_fixes.py).

Uses the fake embedder, so no model is needed. The browser-side fixes (saved
answers that are whole passages, verdict locations) are covered by
tests/ui/qa_ui_check.js.
"""

from pathlib import Path

import pytest

from chunkyard.chunking import SECTION, chunk_sections, split_sections
from chunkyard.engine import Store, query
from chunkyard.loaders import load_document
from chunkyard.samples import SAMPLES, SAMPLES_DIR
from test_chunkyard import FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fda_text():
    pytest.importorskip("pypdf")
    p = SAMPLES_DIR / SAMPLES["fda"]["file"]
    if not p.is_file():
        pytest.skip("FDA sample PDF not present")
    return load_document(p.name, p.read_bytes())


@pytest.fixture(scope="module")
def fda_headings(fda_text):
    """Every heading line split_sections accepted: (line text, breadcrumb)."""
    return [(fda_text[s.header_span[0]:s.header_span[1]].strip(), s.path)
            for s in split_sections(fda_text) if s.header_span]


@pytest.fixture(scope="module")
def fda_chunks(fda_text):
    return chunk_sections(fda_text)


# --- 1. Real leaflet and IFU headings survive; address lines never do --------

def test_leaflet_and_ifu_titles_are_headings(fda_headings):
    titles = [t for t, _ in fda_headings]
    assert "Patient Information" in titles
    assert titles.count("Instructions for Use") >= 2  # the two prefilled-syringe IFUs
    assert "Data" in titles and "Animal Data" in titles


def test_question_headings_are_headings(fda_headings):
    titles = {t for t, _ in fda_headings}
    assert {"What is REPATHA?", "How should I store REPATHA?",
            "What are the possible side effects of REPATHA?"} <= titles


def test_no_address_or_admin_lines_in_breadcrumbs(fda_chunks):
    crumbs = {" > ".join(c.headers) for c in fda_chunks}
    for bad in ("Amgen Center", "Thousand Oaks", "License Number", "Revised", "See 17 for"):
        assert not [b for b in crumbs if bad in b], bad


def test_leaflet_is_not_filed_under_pregnancy(fda_text, fda_chunks):
    i = fda_text.index("REPATHA is an injectable prescription medicine used")
    chunk = next(c for c in fda_chunks if c.start <= i < c.end)
    assert "Pregnancy" not in " > ".join(chunk.headers)
    assert "What is REPATHA?" in chunk.headers


def test_pregnancy_breadcrumb_means_pregnancy_text(fda_text):
    store = Store(FakeEmbedder())
    d = store.add("label.pdf", fda_text, 500, 100, 1200)
    res = query(store, d.id, "Is Repatha safe during pregnancy?", k=5)
    labelled = [r for r in res["results"][SECTION] if any("Pregnancy" in h for h in r["headers"])]
    assert labelled  # the real pregnancy sections are still found
    for r in labelled:
        body = fda_text[r["start"]:r["end"]].lower()
        assert "pregnan" in body, r["headers"]


def test_admin_and_question_rules_on_plain_text():
    text = ("Some closing sentence here.\n\nvxx\n\nPatient Information\n\nACME (ak-me)\n(widget)\n\n"
            "What is ACME?\n\nACME is a widget used to fix things around the house.\n\n"
            "One Example Drive\n\nSpringfield, Illinois 62701-1234\n\nU.S. License Number 1080\n\n"
            "See 17 for PATIENT COUNSELING INFORMATION and\n\nMore text follows here today.\n")
    paths = {p for s in split_sections(text) for p in s.path}
    assert {"Patient Information", "What is ACME?"} <= paths
    assert not paths & {"One Example Drive", "Springfield, Illinois 62701-1234",
                        "U.S. License Number 1080", "See 17 for PATIENT COUNSELING INFORMATION and"}


def test_diagram_labels_do_not_hold_each_other_up():
    # A run of short labels must not count as "heading after heading".
    text = ("Read the steps below.\n\nGetting to know the device\n\nGray start button\n\n"
            "Expiration date\n\nWindow Medicine\n\nYellow safety guard\n(needle inside)\n\n"
            "Upper arm\n\nStomach\n\nFront of thigh\n\nYou can use the:\n\n"
            "Important information you need to know before you inject it today.\n")
    paths = {p for s in split_sections(text) for p in s.path}
    assert not paths & {"Window Medicine", "Upper arm", "Front of thigh", "Stomach"}


# --- 4. Line endings are pinned, whatever core.autocrlf says ----------------

def test_gitattributes_pins_line_endings():
    attrs = (ROOT / ".gitattributes").read_text()
    assert "* text=auto eol=lf" in attrs
    assert "*.pdf binary" in attrs
