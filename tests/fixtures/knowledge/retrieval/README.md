# Langley Retrieval Golden Seed

This directory is the Task 3 retrieval ground-truth seed for Langley internal
retrieval baseline, A/B, and regression work. It is not a retriever, an evaluator,
an upstream benchmark reproduction, or a leaderboard dataset.

`dataset.json` is the authority for stable `document_key`, fixture SHA-256, and each
single contiguous UTF-8 evidence span. Offsets are zero-based and half-open. A future
retrieval hit must use the same Golden source identity and a chunk `source_region` that
fully contains the approved evidence span; this fixture does not implement that
evaluation.

## Contents and format

- 40 UTF-8, LF-only Markdown documents: five SciFact-positive documents, 16 SciFact
  unlabeled distractors, 17 Human-provided Xiaolin technical Markdown documents, one
  FastAPI document, and one Langley Japanese sanity document.
- 13 Human-approved one-target RetrievalCases: SciFact 5, Xiaolin 6, FastAPI 1, and
  Japanese sanity 1.
- Every document is listed in `dataset.json`, including distractors that intentionally
  have no case. No `KnowledgeChunk.id`, `DocumentVersion.id`, score, rank, embedding,
  or vector-store identity belongs in this dataset.

Run the deterministic, filesystem-only validation with:

```powershell
uv run pytest tests/fast/test_retrieval_golden_dataset.py -q --basetemp=".test-tmp-task3"
```

The validator checks JSON shape/identity and safe paths, strict UTF-8/LF-only fixture
bytes, SHA-256, source provenance shape, and exact evidence slices. It performs no
network, database, ingestion, retrieval, embedding, or metric work.

## Provenance and attribution

### SciFact-derived Markdown

The SciFact documents are minimally adapted from title plus ordered abstract sentences
into Markdown, without semantic rewriting. The source is the [SciFact repository]
(https://github.com/allenai/scifact) at revision
`68b98a56d93e0f9da0d2aab4e6c3294699a0f72e`, using the official
[data release](https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz).
`dataset.json` records each original document ID and the five source claim IDs plus
their official SUPPORT evidence basis. SciFact claims/evidence annotations are CC BY
4.0; its S2ORC-derived corpus abstracts are ODC-By 1.0. Preserve both source links and
these attribution/license notices when redistributing this fixture.

### FastAPI documentation

`documents/fastapi_async.md` is the unmodified `docs/en/docs/async.md` from
[FastAPI](https://github.com/fastapi/fastapi), revision
`c3f316b7e814667e8ee81e03a7330d00ee61e45c`. It is used under the repository's MIT
License.

```text
Copyright (c) 2018 Sebastián Ramírez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Xiaolin personal technical Markdown

The 17 `xiaolin_*.md` files are Human-provided personal technical Markdown approved
for use as local tracked Task 3 fixtures in this personal Langley project. Their
provenance is recorded as **Human-provided personal technical Markdown**. No
third-party public license is claimed or implied for this material.

### Langley Japanese sanity fixture

`documents/langley_markdown_unicode_ja.md` is a Langley project-owned fixture copied
from `tests/fixtures/knowledge/markdown/golden/unicode_ja.md`.

## Change discipline

Changing document bytes requires updating the affected SHA-256 and mechanically
recomputing/verifying every affected evidence span from the actual tracked bytes. Do
not retain stale coordinates. Changing approved query/evidence semantics requires a
new Human Golden Review; a different valid evidence result is a Golden-labeling issue,
not a retriever failure.
