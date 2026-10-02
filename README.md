# Chunkyard

A local web app that shows how **chunking strategy** changes what a RAG
(retrieval-augmented generation) system retrieves. Upload a document, ask a
question, and see the top 5 chunks from three strategies side by side.

Everything runs on your computer. Embeddings come from a local
[sentence-transformers](https://www.sbert.net/) model, so no API key is needed.
The one exception is an **optional** step, off by default, that asks the
Claude API to answer from each strategy's passages (see
[Optional: AI answers](#optional-ai-answers)).

## Run it

Double-click **`Start Chunkyard.bat`**, or from a terminal in this folder:

```
uv run python -m chunkyard            # opens http://127.0.0.1:8000
uv run python -m chunkyard --port 9000 --no-browser
```

The first start downloads the embedding model (~90 MB) to
`%USERPROFILE%\.cache\huggingface`. After that it works offline.

### Built-in samples

- **Tent manual** (`samples/trailhead-tent-manual.md`): short and tidy, so the
  strategies usually agree. A good baseline.
- **FDA drug label** (`samples/repatha-fda-label.pdf`): the 65-page US
  prescribing information for Repatha, with numbered sections, bullet lists,
  results tables, and a patient leaflet bundled in. This is where the
  strategies disagree. It's a public US government document, included here
  unchanged from the
  [FDA's website](https://www.accessdata.fda.gov/drugsatfda_docs/label/2025/125522s045lbl.pdf).

Each sample has suggested questions. To jump straight to one, open
`http://127.0.0.1:8000/?sample=fda&q=How+should+Repatha+be+stored%3F`
(`sample=tent` for the tent manual).

## The three strategies

| Strategy | How it splits | What tends to go wrong |
|---|---|---|
| **Fixed** | Every *N* characters, stepping back *overlap* characters each time | Cuts mid-word and mid-sentence; one chunk can straddle two topics |
| **Paragraph** | On blank lines; overlong paragraphs split at sentence ends | Headers become their own tiny chunks, detached from their content |
| **Section** | Groups paragraphs under their heading, packs them up to the max size, and prepends the heading breadcrumb (`Warranty > What's covered`) to every chunk | Depends on detecting headings; sections can be bigger than the model reads |

The sample manual is written so that many paragraphs never repeat their
section's topic (the warranty text never says "warranty"), which is where
section-aware chunking shines.

## Reading the results

- **Score**: cosine similarity between the question and the chunk.
- **Agreement**: two chunks from different strategies "agree" (retrieved the
  same passage) when the characters they share are at least 25% of the
  **longer** chunk, and both chunks are at least 100 characters. A heading or
  fragment that merely sits inside another strategy's passage never counts as
  agreement. Each card lists the other strategies that found the same passage
  and at what rank, and hovering a card outlines exactly those cards.
  **Only *X* found this** marks a passage the other strategies missed.
- **Verdict**: when the strategies' #1 passages differ, a warning box names
  who disagrees ("Paragraph disagrees with Fixed and Section…") and says where
  each one's top hit came from. If a #1 is only a heading or a short fragment,
  it says so ("Paragraph only retrieved the heading 'What's not covered',
  which sits inside Section's #1 passage"). Two strategies that agree are
  described together ("Fixed and Section both pick a passage in …") only when
  their #1 passages sit in the same place; otherwise each location is named
  ("Fixed picks a passage spanning …; Section picks a passage in …").
  Agreement gets a quiet one-line note instead.
  Below it: whether each pair picked the same #1 passage, and how much of their
  top-5 text overlaps.
- **Document map**: where each strategy's top-5 hits sit in the document.
  Hover a card to highlight matching cards in the other columns; click a
  block on the map to jump to its card.
- **Show in document**: the full text with retrieved passages highlighted by
  which strategies retrieved them.
- **Warning flags** on retrieved chunks (hover any badge for why it matters):
  - *Starts mid-sentence* / *Ends mid-sentence* (yellow): the chunk is cut
    partway through a sentence. Bullets, headings, and paragraph breaks don't
    count. A closing bracket, quote, colon or semicolon only ends a sentence
    when sentence punctuation comes right before it ("…carton.)" ends one,
    "Inject 140 mg (1 mL)" doesn't).
  - *Short* (red): under 100 characters, usually a lone heading or fragment.
  - *Numbers without labels* (violet): a line of values, like a table row
    ("(n = 562) -59 -50 -46 -34"), with no column heading in the few lines
    above it in the same chunk. Dates and ordinary sentences with units
    ("weighs 2.1 kg (4.6 lb)") aren't flagged.
- **Lengths** on cards and in the stats are *source chars*: how much of the
  document a chunk covers. Section chunks also carry their heading path when
  embedded; hover a section card's length to see the embedded size.
  - *Model saw only 256 tokens*: the chunk is longer than the embedding model
    reads, so the end of it didn't affect its score.
- **Fragments warning** on a strategy's card: its chunks average under 200
  characters, or its smallest is under 30, so it's producing fragments too
  small to be useful on this document.

## Optional: AI answers

Retrieval is only half of RAG. This optional step does the other half: for
each strategy, it sends the question and that strategy's top 5 passages to
Claude and shows the three answers side by side above the chunk columns. A
fourth call compares them and highlights where they **differ factually**
(different numbers, durations, conditions, or one answer saying the
information is missing), with a table of the differences.

- **Off by default.** Nothing is sent unless you click **Generate answers**
  (or tick *Generate automatically for each new question*). Without a key, the
  panel explains how to set one up, and the rest of Chunkyard works as usual.
- **What leaves your computer:** the question and the retrieved passages, sent
  to Anthropic's API. The rest of the document isn't sent.
- **Cost:** 4 paid API calls per question (3 answers + 1 comparison).
- **Setup:** create a key at [console.anthropic.com](https://console.anthropic.com),
  then in a Command Prompt run `setx ANTHROPIC_API_KEY "your-key"`, close it, and
  restart Chunkyard. The model is `claude-opus-5`; set `CHUNKYARD_LLM_MODEL` to
  use another.
- Answers are told to use only the passages and to say when the answer isn't
  there, so a strategy that retrieved the wrong passages produces a visibly
  worse answer. Citations like [2] link to that strategy's passage. AI
  answers can still be wrong; check them against the passages.

## Scoring: which strategy was right?

1. Ask a question, **select the exact words that answer it** in any passage
   (e.g. "two years from the date of purchase"), and click **This is the
   answer** on that passage. Clicking without a selection does nothing but ask
   for one, and a selection can be at most half the fixed chunk size (250
   characters by default). A whole chunk can't be the answer, because it
   would only ever be credited to the strategy that produced it.
2. Each column now says **Answer found at rank N** or **Not in top 5**.
3. Repeat for a few questions. Section 4 lists them. **Save to file** writes
   them to a JSON file; **Load from file** brings them back.
4. **Run saved set** runs every saved question against all three strategies
   and shows, for each strategy, how many answers were found in the top 5,
   their average rank, and **MRR** (mean reciprocal rank: the average of
   1/rank, with a miss counting as 0, so one number reflects both hits and
   misses), plus a per-question breakdown.

How "found" is decided: the marked answer is stored as a span of the
document text, not as a chunk, so the same answers can score all three
strategies at any chunk size. A retrieved chunk counts as containing the
answer if it covers **at least half** of that span. A heading that sits next
to the answer covers none of it and doesn't count. Keeping answers to at most
half the fixed chunk size guarantees that some fixed chunk always covers at
least half of any answer, so every strategy can be credited fairly.

Answer sets are tied to the exact document text (a fingerprint is saved in
the file), so a set made for one document can't be applied to another.
Change the chunk sizes in section 1 and run the set again to compare settings.
The same length limit applies to answers loaded from a file or remembered by
the browser from an older version: whole-passage answers are skipped with a
message asking you to re-mark them. If you lower the fixed size below twice
an answer's length, **Run saved set** still runs but warns that Fixed can't
be credited fairly for those answers.

Current limitation: each question has one accepted answer location. If the
same answer appears in several places (an FDA label repeats dosing in its
highlights, full text, and patient leaflet), a strategy that retrieves a
different copy is scored as a miss.

## Files and limits

- `.txt`, `.md`, and `.pdf` up to **5 MB**, which keeps chunking and
  embedding quick in a live demo. The built-in FDA sample is larger (6 MB),
  so load it with its sample button rather than by dragging the file in.
  Scanned PDFs without a text layer won't work.
- **Text encodings:** UTF-8, UTF-16 (with a byte-order mark), and Windows-1252
  (Notepad's old default). Files that decode to mostly control characters are
  refused as "not a text file".
- **Speed:** embeddings are cached by chunk text, so re-chunking only embeds
  chunks it hasn't seen. Changing only the fixed size leaves every paragraph
  and section chunk cached. Parsed PDF text is cached by file hash.
- **Headings** are detected from Markdown `#`, underlined (`===`/`---`),
  numbered (`2.1 Setup`), ALL CAPS, and short capitalized lines that stand
  alone. PDFs don't record which heading is inside which, so a PDF's
  breadcrumb sometimes drops the parent heading.
- **PDF running headers and footers** (a line like "Reference ID: 12345" or
  "Page 3 of 40" near the top or bottom of most pages) are removed so they
  don't become fake headings.
- **Guessed headings in PDFs** (lines with no `#` or numbering) must look like
  a real subheading: not a single letter, a code ("GRH0434v1"), or a lone ALL
  CAPS word ("AMGEN"); not followed by table content ("(N = 599)", "%", rows
  of numbers); not an address or admin line (a ZIP code, phone number,
  street address, "License Number", "Revised:", "See 17 for", or a line that
  ends in "and"/"of"/"the"); preceded by a finished sentence, another heading,
  or one line of page debris after one ("vxx", a version or copyright line);
  and followed by prose (inside a numbered section, prose or a further
  subheading such as "Data" then "Animal Data"). A Title Case title right
  after a page break ("Patient Information", "Instructions for Use") is kept
  even when a product name or diagram follows it. A standalone question of
  70 characters or fewer followed by prose ("What is REPATHA?") is a heading,
  as patient leaflets use them. On the FDA sample this removes table, figure,
  diagram, and address labels such as "REPATHA", "Hazard", "No. at Risk",
  "Medicine", "Stomach", and "Thousand Oaks, California 91320-1799" while
  keeping "Risk Summary", "Absorption", "Adverse Reactions in a 52-Week
  Controlled Trial", "Patient Information", and the leaflet's questions. It is
  still a heuristic: an unusual document can produce a wrong or
  missing breadcrumb. Numbered headings ("6.2 Immunogenicity") always nest
  under their numbered parent.
- **PDF paragraphs** are guessed from line lengths, because PDF text usually has
  no blank lines between paragraphs. A word hyphenated across a line break is
  re-joined ("subcuta- / neous") only when both parts are lowercase and the
  document doesn't show it's a compound; "Non-HDL" and "patient-safety" keep
  their hyphens.
- The last 5 documents stay in memory; nothing is written to disk.

To use a different model, set `CHUNKYARD_MODEL` (for example
`sentence-transformers/all-mpnet-base-v2`) before starting.

## Code

- `chunkyard/chunking.py`: the three strategies and heading detection
- `chunkyard/loaders.py`: text/PDF loading and PDF paragraph inference
- `chunkyard/engine.py`: embedding, top-k retrieval, cross-strategy agreement, scoring
- `chunkyard/flags.py`: warning flags for chunks and strategies
- `chunkyard/generate.py`: the optional Claude API answers and comparison
- `chunkyard/app.py`: FastAPI endpoints
- `chunkyard/static/index.html`: the single-page front end

Tests: `uv run pytest` (they use a fake embedder, so they're fast and don't
need the model). GitHub Actions runs them on every push.
`tests/test_qa_fixes.py` has a regression test for each finding of the
independent QA review. The page itself has browser checks in `tests/ui/`:
start `uv run python tests/ui/preview_server.py 8766` (fake embedder, no
model), then in `tests/ui` run `npm install` once and `node qa_ui_check.js`.
`tests/fixtures/fictional-label-pages.txt` is an invented drug label laid out
like text extracted from a PDF (running footers, dashed and numbered headings,
a results table) and guards the heading detection against regressions.
