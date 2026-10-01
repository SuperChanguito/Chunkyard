# Chunkyard

A local web app that shows how **chunking strategy** changes what a RAG
(retrieval-augmented generation) system retrieves. Upload a document, ask a
question, and see the top 5 chunks from three strategies side by side.

Everything runs on your computer. Embeddings come from a local
[sentence-transformers](https://www.sbert.net/) model, so no API key is needed.

## Run it

Double-click **`Start Chunkyard.bat`**, or from a terminal in this folder:

```
uv run python -m chunkyard            # opens http://127.0.0.1:8000
uv run python -m chunkyard --port 9000 --no-browser
```

The first start downloads the embedding model (~90 MB) to
`%USERPROFILE%\.cache\huggingface`. After that it works offline.

Don't have the sample in mind? Click **Use the sample tent manual** and try the
suggested questions, or open
`http://127.0.0.1:8000/?sample=1&q=How+long+is+the+warranty%3F`.

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
- **Agreement**: two chunks from different strategies "agree" when they cover
  the same part of the document (they share at least 25% of the shorter
  chunk's characters). Each card lists the other strategies that found the
  same passage and at what rank. **Only *X* found this** marks a passage the
  other strategies missed.
- **Summary line**: whether each pair of strategies picked the same #1
  passage, plus how much of their top-5 text overlaps.
- **Document map**: where each strategy's top-5 hits sit in the document.
  Hover a card to highlight matching cards in the other columns; click a
  block on the map to jump to its card.
- **Show in document**: the full text with retrieved passages highlighted by
  which strategies retrieved them.
- **Badges**: *Very short* flags tiny chunks (usually a lone header). *Model
  saw only 256 tokens* means the chunk is longer than the embedding model
  reads, so the end of it didn't affect its score.

## Files and limits

- `.txt`, `.md`, and `.pdf` up to 20 MB. Scanned PDFs without a text layer
  won't work.
- **Headings** are detected from Markdown `#`, underlined (`===`/`---`),
  numbered (`2.1 Setup`), ALL CAPS, and short capitalized lines that stand
  alone. PDFs don't record which heading is inside which, so a PDF's
  breadcrumb sometimes drops the parent heading.
- **PDF running headers and footers** (a line like "Reference ID: 12345" or
  "Page 3 of 40" near the top or bottom of most pages) are removed so they
  don't become fake headings.
- **Table rows and chart labels** in PDFs can still be mistaken for small
  headings. Lines with data numbers ("Back pain 5.6 6.2") are skipped, and
  numbered headings ("6.2 Immunogenicity") always nest under their numbered
  parent, so mistakes add noise but never a wrong breadcrumb.
- **PDF paragraphs** are guessed from line lengths, because PDF text usually has
  no blank lines between paragraphs. A word hyphenated across a line break is
  re-joined, which occasionally removes a real hyphen.
- The last 5 documents stay in memory; nothing is written to disk.

To use a different model, set `CHUNKYARD_MODEL` (for example
`sentence-transformers/all-mpnet-base-v2`) before starting.

## Code

- `chunkyard/chunking.py`: the three strategies and heading detection
- `chunkyard/loaders.py`: text/PDF loading and PDF paragraph inference
- `chunkyard/engine.py`: embedding, top-k retrieval, cross-strategy agreement
- `chunkyard/app.py`: FastAPI endpoints
- `chunkyard/static/index.html`: the single-page front end

Tests: `uv run pytest` (they use a fake embedder, so they're fast and don't
need the model).
