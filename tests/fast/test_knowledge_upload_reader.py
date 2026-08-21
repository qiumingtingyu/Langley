"""Focused bounded-read regressions for Knowledge multipart materialization."""

import asyncio

import pytest
from fastapi import HTTPException

from langley.api.knowledge import MAX_MARKDOWN_UPLOAD_BYTES, _read_markdown_upload


class _BoundedUpload:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = iter(chunks)
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        assert size > 0
        return next(self._chunks, b"")


def test_upload_reader_uses_bounded_reads_and_stops_at_limit() -> None:
    upload = _BoundedUpload([b"x" * (MAX_MARKDOWN_UPLOAD_BYTES + 1)])
    with pytest.raises(HTTPException) as error:
        asyncio.run(_read_markdown_upload(upload))
    assert error.value.status_code == 413
    assert upload.read_sizes == [64 * 1024]
