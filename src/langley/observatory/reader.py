"""Incremental binary JSONL reader for legacy v0 and Trace Contract v1."""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class RecordDisposition(StrEnum):
    EVENT = "EVENT"
    MALFORMED = "MALFORMED"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"


@dataclass(frozen=True)
class JsonlRecord:
    """One consumed complete line and its exact raw JSON locator."""

    offset: int
    byte_length: int
    ingest_order: int
    disposition: RecordDisposition
    event: dict[str, Any] | None
    schema_version: int | None


@dataclass(frozen=True)
class JsonlScan:
    """One bounded scan beginning at a previously committed byte boundary."""

    records: tuple[JsonlRecord, ...]
    indexed_bytes: int
    last_ingest_order: int
    incomplete_tail: bool
    bytes_examined: int


def scan_jsonl(
    path: Path, *, start_offset: int = 0, last_ingest_order: int = 0
) -> JsonlScan:
    """Read complete lines only; defer a final line until its newline arrives."""

    records: list[JsonlRecord] = []
    indexed_bytes = start_offset
    ingest_order = last_ingest_order
    incomplete_tail = False
    bytes_examined = 0

    with path.open("rb") as stream:
        stream.seek(start_offset)
        while True:
            offset = stream.tell()
            raw_line = stream.readline()
            if not raw_line:
                break
            bytes_examined += len(raw_line)
            if not raw_line.endswith(b"\n"):
                incomplete_tail = True
                break

            indexed_bytes = stream.tell()
            ingest_order += 1
            raw_json = raw_line[:-1]
            if raw_json.endswith(b"\r"):
                raw_json = raw_json[:-1]
            records.append(
                _decode_record(
                    raw_json,
                    offset=offset,
                    ingest_order=ingest_order,
                )
            )

    return JsonlScan(
        records=tuple(records),
        indexed_bytes=indexed_bytes,
        last_ingest_order=ingest_order,
        incomplete_tail=incomplete_tail,
        bytes_examined=bytes_examined,
    )


def _decode_record(raw_json: bytes, *, offset: int, ingest_order: int) -> JsonlRecord:
    try:
        value = json.loads(raw_json)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JsonlRecord(
            offset,
            len(raw_json),
            ingest_order,
            RecordDisposition.MALFORMED,
            None,
            None,
        )
    if not isinstance(value, dict):
        return JsonlRecord(
            offset,
            len(raw_json),
            ingest_order,
            RecordDisposition.MALFORMED,
            None,
            None,
        )

    raw_schema_version = value.get("schema_version")
    if raw_schema_version is None:
        schema_version = None
    elif (
        isinstance(raw_schema_version, int)
        and not isinstance(raw_schema_version, bool)
        and raw_schema_version == 1
    ):
        schema_version = 1
    else:
        return JsonlRecord(
            offset,
            len(raw_json),
            ingest_order,
            RecordDisposition.UNSUPPORTED_SCHEMA,
            None,
            raw_schema_version
            if isinstance(raw_schema_version, int)
            and not isinstance(raw_schema_version, bool)
            else None,
        )

    if schema_version == 1 and not _valid_v1_envelope(value):
        return JsonlRecord(
            offset,
            len(raw_json),
            ingest_order,
            RecordDisposition.MALFORMED,
            None,
            schema_version,
        )

    return JsonlRecord(
        offset,
        len(raw_json),
        ingest_order,
        RecordDisposition.EVENT,
        value,
        schema_version,
    )


def _valid_v1_envelope(event: dict[str, Any]) -> bool:
    timestamp = event.get("timestamp")
    run_id = event.get("run_id")
    seq = event.get("seq")
    kind = event.get("kind")
    capture_mode = event.get("capture_mode")
    return (
        isinstance(timestamp, str)
        and bool(timestamp.strip())
        and isinstance(run_id, int)
        and not isinstance(run_id, bool)
        and isinstance(seq, int)
        and not isinstance(seq, bool)
        and seq > 0
        and isinstance(kind, str)
        and bool(kind.strip())
        and isinstance(capture_mode, str)
        and capture_mode in {"FULL_CONTENT", "METADATA_ONLY"}
    )
