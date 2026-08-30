"""Fast deterministic checks for Task 5.1 index-build identity mechanics."""

import math

import pytest

from langley.knowledge.index_build import (
    IndexBuildFailure,
    IndexChunk,
    _normalize_embedding_rows,
    _snapshot_sha256,
)


def test_chunk_snapshot_is_independent_of_result_order() -> None:
    first = IndexChunk(11, 4, "first", (), 2, "a" * 64)
    second = IndexChunk(12, 5, "second", (), 3, "b" * 64)

    assert _snapshot_sha256((first, second)) == _snapshot_sha256((second, first))


def test_chunk_snapshot_uses_stable_document_revision_facts() -> None:
    original = IndexChunk(11, 4, "first", (), 2, "a" * 64)
    same_document_facts = IndexChunk(99, 4, "changed", ("Heading",), 2, "a" * 64)
    replaced = IndexChunk(11, 4, "first", (), 3, "b" * 64)

    assert _snapshot_sha256((original,)) == _snapshot_sha256((same_document_facts,))
    assert _snapshot_sha256((original,)) != _snapshot_sha256((replaced,))


@pytest.mark.parametrize(
    "values",
    [
        [[math.nan, 1.0]],
        [[math.inf, 1.0]],
        [[0.0, 0.0]],
        [[1.0, 2.0, 3.0]],
    ],
)
def test_malformed_embeddings_fail_closed(values: list[list[float]]) -> None:
    with pytest.raises(IndexBuildFailure, match="INVALID_EMBEDDING"):
        _normalize_embedding_rows(values, row_count=1, dimension=2)


def test_valid_embedding_is_explicitly_l2_normalized() -> None:
    assert _normalize_embedding_rows([[3.0, 4.0]], row_count=1, dimension=2) == [
        pytest.approx([0.6, 0.8])
    ]
