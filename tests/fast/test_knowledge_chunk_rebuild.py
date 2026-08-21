"""Focused Task 2.4 detached rebuild materialization contracts."""

from dataclasses import replace

from langley.business_time import utc_now
from langley.infrastructure.models import DocumentVersion
from langley.knowledge.chunking import CandidateChunk
from langley.knowledge.commands import (
    _document_version_matches_source_ref,
    _materialize_chunk_rows,
)
from langley.knowledge.contracts import DocumentSourceRef, TextSpanRegion


def test_materialization_maps_candidate_fields_and_existing_region_codec() -> None:
    rows = _materialize_chunk_rows(
        7,
        (
            CandidateChunk(
                ordinal=1,
                content="exact",
                heading_path=("A", "B"),
                source_regions=(TextSpanRegion(3, 8),),
            ),
        ),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.document_version_id == 7
    assert row.ordinal == 1
    assert row.content == "exact"
    assert row.heading_path == ["A", "B"]
    assert row.source_regions == [{"kind": "text_span", "start_byte": 3, "end_byte": 8}]


def test_zero_candidates_materialize_to_no_rows() -> None:
    assert _materialize_chunk_rows(7, ()) == []


def test_source_identity_match_requires_every_immutable_source_fact() -> None:
    version = DocumentVersion(
        id=7,
        document_id=3,
        storage_key="users/1/sources/00000000000000000000000000000000/source",
        source_media_type="text/markdown",
        source_sha256="a" * 64,
        source_size_bytes=9,
        source_filename="document.md",
        created_at=utc_now(),
    )
    source_ref = DocumentSourceRef(
        document_version_id=7,
        storage_key=version.storage_key,
        source_media_type=version.source_media_type,
        source_sha256=version.source_sha256,
        source_size_bytes=version.source_size_bytes,
    )

    assert _document_version_matches_source_ref(version, source_ref)
    assert not _document_version_matches_source_ref(
        version, replace(source_ref, storage_key="users/1/sources/other/source")
    )
    assert not _document_version_matches_source_ref(
        version, replace(source_ref, source_media_type="text/plain")
    )
    assert not _document_version_matches_source_ref(
        version, replace(source_ref, source_sha256="b" * 64)
    )
    assert not _document_version_matches_source_ref(
        version, replace(source_ref, source_size_bytes=10)
    )
