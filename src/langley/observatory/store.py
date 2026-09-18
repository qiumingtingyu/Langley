"""Disposable SQLite read model over local Run diagnostic JSONL files."""

import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langley.observatory.reader import RecordDisposition, scan_jsonl

_SCHEMA_VERSION = 1
_FINGERPRINT_BYTES = 4096
_TERMINAL_KINDS = {"run.success", "run.failure"}


@dataclass(frozen=True)
class RefreshResult:
    sources_seen: int = 0
    sources_changed: int = 0
    events_indexed: int = 0
    malformed_lines: int = 0
    unsupported_schema_lines: int = 0
    bytes_examined: int = 0
    source_resets: int = 0
    sources_removed: int = 0

    def __add__(self, other: "RefreshResult") -> "RefreshResult":
        return RefreshResult(
            sources_seen=self.sources_seen + other.sources_seen,
            sources_changed=self.sources_changed + other.sources_changed,
            events_indexed=self.events_indexed + other.events_indexed,
            malformed_lines=self.malformed_lines + other.malformed_lines,
            unsupported_schema_lines=(
                self.unsupported_schema_lines + other.unsupported_schema_lines
            ),
            bytes_examined=self.bytes_examined + other.bytes_examined,
            source_resets=self.source_resets + other.source_resets,
            sources_removed=self.sources_removed + other.sources_removed,
        )


@dataclass(frozen=True)
class TraceSourceSummary:
    source_path: str
    indexed_bytes: int
    file_size: int
    mtime_ns: int
    corruption_count: int
    unsupported_schema_count: int
    incomplete_tail: bool
    generation: int


@dataclass(frozen=True)
class TraceRunSummary:
    run_id: int
    first_observed_timestamp: str | None
    last_observed_timestamp: str | None
    provider: str | None
    configured_model: str | None
    capture_mode: str | None
    observed_outcome: str | None
    trace_complete: bool
    round_count: int
    tool_count: int
    provider_input_tokens: int | None
    provider_output_tokens: int | None
    provider_total_tokens: int | None
    observed_duration_ms: float | None
    raw_trace_size_bytes: int
    source_count: int


@dataclass(frozen=True)
class TraceEventRecord:
    event_id: int
    run_id: int
    source_path: str
    ingest_order: int
    schema_version: int | None
    observed_seq: int | None
    kind: str
    round: int | None
    timestamp: str | None
    tool_call_id: str | None
    parent_tool_call_id: str | None
    tool_ordinal: int | None
    duration_ms: float | None
    raw_byte_offset: int
    raw_byte_length: int


@dataclass(frozen=True)
class TraceRoundRecord:
    run_id: int
    round: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    duration_ms: float | None
    ttft_ms: float | None
    finish_reason: str | None
    provider_model: str | None
    estimated_context_tokens: int | None
    context_estimate_kind: str | None


@dataclass(frozen=True)
class ContextComponentRecord:
    run_id: int
    round: int
    key: str
    kind: str
    source: str | None
    char_count: int
    byte_count: int
    estimated_tokens: int
    content_sha256: str | None


@dataclass(frozen=True)
class RawTraceEvent:
    locator: TraceEventRecord
    raw_json: str
    event: dict[str, Any]


class ObservatoryStore:
    """Synchronous local projection store; callers only mutate via refresh/rebuild."""

    def __init__(
        self,
        database_path: Path = Path(".runtime/observatory.sqlite"),
        traces_root: Path = Path(".runtime/traces"),
    ) -> None:
        self.database_path = database_path
        self.traces_root = traces_root
        self.initialize()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            current_version = connection.execute("PRAGMA user_version").fetchone()[0]
            if current_version not in (0, _SCHEMA_VERSION):
                raise RuntimeError(
                    f"unsupported Observatory SQLite schema {current_version}"
                )
            connection.executescript(_SCHEMA)
            round_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(trace_rounds)")
            }
            if "context_estimate_kind" not in round_columns:
                connection.execute(
                    "ALTER TABLE trace_rounds ADD COLUMN context_estimate_kind TEXT"
                )
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def refresh(self) -> RefreshResult:
        result = RefreshResult()
        root = self.traces_root.resolve()
        current_paths: set[str] = set()
        if self.traces_root.exists():
            for path in sorted(self.traces_root.glob("run-*.jsonl")):
                if path.is_file():
                    current_paths.add(str(path.resolve()))
                    result += self.refresh_source(path)

        with self._connect() as connection:
            indexed_paths = {
                str(row["source_path"])
                for row in connection.execute(
                    "SELECT source_path FROM trace_sources"
                ).fetchall()
                if _is_owned_trace_source(Path(str(row["source_path"])), root)
            }
            removed_paths = indexed_paths - current_paths
            if removed_paths:
                connection.executemany(
                    "DELETE FROM trace_sources WHERE source_path = ?",
                    ((path,) for path in sorted(removed_paths)),
                )
                self._rebuild_run_summaries(connection)
                result += RefreshResult(sources_removed=len(removed_paths))
        return result

    def refresh_source(self, path: Path) -> RefreshResult:
        path = path.resolve()
        stat = path.stat()
        identity = _source_identity(stat)
        result = RefreshResult(sources_seen=1)

        with self._connect() as connection:
            source = connection.execute(
                "SELECT * FROM trace_sources WHERE source_path = ?", (str(path),)
            ).fetchone()
            reset = source is not None and self._source_requires_reset(
                path, stat, source
            )
            generation = (int(source["generation"]) + 1) if reset else 0
            if source is not None and not reset:
                generation = int(source["generation"])
            if reset:
                connection.execute(
                    "DELETE FROM trace_sources WHERE source_path = ?", (str(path),)
                )
                source = None
                result += RefreshResult(source_resets=1)

            indexed_bytes = 0 if source is None else int(source["indexed_bytes"])
            last_ingest_order = (
                0 if source is None else int(source["last_ingest_order"])
            )
            old_corruption_count = (
                0 if source is None else int(source["corruption_count"])
            )
            old_unsupported_count = (
                0 if source is None else int(source["unsupported_schema_count"])
            )

            if source is None:
                self._upsert_source(
                    connection,
                    path,
                    identity=identity,
                    indexed_bytes=0,
                    last_ingest_order=0,
                    file_size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    corruption_count=0,
                    unsupported_schema_count=0,
                    incomplete_tail=stat.st_size > 0,
                    generation=generation,
                )

            if indexed_bytes == stat.st_size:
                self._upsert_source(
                    connection,
                    path,
                    identity=identity,
                    indexed_bytes=indexed_bytes,
                    last_ingest_order=last_ingest_order,
                    file_size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    corruption_count=old_corruption_count,
                    unsupported_schema_count=old_unsupported_count,
                    incomplete_tail=False,
                    generation=generation,
                )
                self._rebuild_run_summaries(connection)
                return result

            scan = scan_jsonl(
                path,
                start_offset=indexed_bytes,
                last_ingest_order=last_ingest_order,
            )
            malformed = 0
            unsupported = 0
            indexed_events = 0
            for record in scan.records:
                if record.disposition is RecordDisposition.MALFORMED:
                    malformed += 1
                    continue
                if record.disposition is RecordDisposition.UNSUPPORTED_SCHEMA:
                    unsupported += 1
                    continue
                if not self._index_event(connection, str(path), record):
                    malformed += 1
                    continue
                indexed_events += 1

            final_stat = path.stat()
            self._upsert_source(
                connection,
                path,
                identity=_source_identity(final_stat),
                indexed_bytes=scan.indexed_bytes,
                last_ingest_order=scan.last_ingest_order,
                file_size=final_stat.st_size,
                mtime_ns=final_stat.st_mtime_ns,
                corruption_count=old_corruption_count + malformed,
                unsupported_schema_count=old_unsupported_count + unsupported,
                incomplete_tail=scan.incomplete_tail,
                generation=generation,
            )
            self._rebuild_run_summaries(connection)
            return result + RefreshResult(
                sources_changed=1,
                events_indexed=indexed_events,
                malformed_lines=malformed,
                unsupported_schema_lines=unsupported,
                bytes_examined=scan.bytes_examined,
            )

    def rebuild(self) -> RefreshResult:
        if self.database_path.exists():
            self.database_path.unlink()
        self.initialize()
        return self.refresh()

    def list_sources(self) -> tuple[TraceSourceSummary, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM trace_sources ORDER BY source_path"
            ).fetchall()
        return tuple(_source_summary(row) for row in rows)

    def list_runs(self) -> tuple[TraceRunSummary, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM trace_runs ORDER BY run_id DESC"
            ).fetchall()
        return tuple(_run_summary(row) for row in rows)

    def get_run(self, run_id: int) -> TraceRunSummary | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM trace_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return None if row is None else _run_summary(row)

    def list_events(self, run_id: int) -> tuple[TraceEventRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM trace_events
                WHERE run_id = ?
                ORDER BY source_path, ingest_order
                """,
                (run_id,),
            ).fetchall()
        return tuple(_event_record(row) for row in rows)

    def list_rounds(self, run_id: int) -> tuple[TraceRoundRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM trace_rounds
                WHERE run_id = ?
                ORDER BY round, source_path
                """,
                (run_id,),
            ).fetchall()
        return tuple(_round_record(row) for row in rows)

    def get_round(self, run_id: int, round_: int) -> TraceRoundRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM trace_rounds
                WHERE run_id = ? AND round = ?
                ORDER BY source_path
                LIMIT 1
                """,
                (run_id, round_),
            ).fetchone()
        return None if row is None else _round_record(row)

    def list_context_components(
        self, run_id: int, round_: int
    ) -> tuple[ContextComponentRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM context_components
                WHERE run_id = ? AND round = ?
                ORDER BY source_path, component_order
                """,
                (run_id, round_),
            ).fetchall()
        return tuple(_component_record(row) for row in rows)

    def get_raw_event(self, event_id: int) -> RawTraceEvent:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM trace_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        if row is None:
            raise KeyError(event_id)
        locator = _event_record(row)
        with Path(locator.source_path).open("rb") as stream:
            stream.seek(locator.raw_byte_offset)
            raw = stream.read(locator.raw_byte_length)
        if len(raw) != locator.raw_byte_length:
            raise RuntimeError("raw trace event is no longer available")
        try:
            raw_json = raw.decode("utf-8")
            event = json.loads(raw_json)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "raw trace event no longer matches its locator"
            ) from error
        if not isinstance(event, dict):
            raise RuntimeError("raw trace event is not a JSON object")
        return RawTraceEvent(locator=locator, raw_json=raw_json, event=event)

    def _source_requires_reset(
        self, path: Path, stat: Any, source: sqlite3.Row
    ) -> bool:
        if str(source["source_identity"]) != _source_identity(stat):
            return True
        indexed_bytes = int(source["indexed_bytes"])
        if stat.st_size < indexed_bytes:
            return True
        prefix_hash, boundary_hash = _fingerprints(path, indexed_bytes)
        return (
            prefix_hash != source["prefix_sha256"]
            or boundary_hash != source["boundary_sha256"]
        )

    def _upsert_source(
        self,
        connection: sqlite3.Connection,
        path: Path,
        *,
        identity: str,
        indexed_bytes: int,
        last_ingest_order: int,
        file_size: int,
        mtime_ns: int,
        corruption_count: int,
        unsupported_schema_count: int,
        incomplete_tail: bool,
        generation: int,
    ) -> None:
        prefix_hash, boundary_hash = _fingerprints(path, indexed_bytes)
        connection.execute(
            """
            INSERT INTO trace_sources (
                source_path, source_identity, indexed_bytes, last_ingest_order,
                file_size, mtime_ns, prefix_sha256, boundary_sha256,
                corruption_count, unsupported_schema_count, incomplete_tail,
                generation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                source_identity = excluded.source_identity,
                indexed_bytes = excluded.indexed_bytes,
                last_ingest_order = excluded.last_ingest_order,
                file_size = excluded.file_size,
                mtime_ns = excluded.mtime_ns,
                prefix_sha256 = excluded.prefix_sha256,
                boundary_sha256 = excluded.boundary_sha256,
                corruption_count = excluded.corruption_count,
                unsupported_schema_count = excluded.unsupported_schema_count,
                incomplete_tail = excluded.incomplete_tail,
                generation = excluded.generation
            """,
            (
                str(path),
                identity,
                indexed_bytes,
                last_ingest_order,
                file_size,
                mtime_ns,
                prefix_hash,
                boundary_hash,
                corruption_count,
                unsupported_schema_count,
                int(incomplete_tail),
                generation,
            ),
        )

    def _index_event(
        self, connection: sqlite3.Connection, source_path: str, record: Any
    ) -> bool:
        event = record.event
        if event is None:
            return False
        run_id = _strict_int(event.get("run_id"))
        kind = event.get("kind")
        if run_id is None or not isinstance(kind, str) or not kind:
            return False
        round_ = _strict_int(event.get("round"))
        observed_seq = (
            _strict_int(event.get("seq")) if record.schema_version == 1 else None
        )
        tool_call_id = event.get("call_id") if kind == "tool" else None
        parent_tool_call_id = event.get("tool_call_id") if kind != "tool" else None
        cursor = connection.execute(
            """
            INSERT INTO trace_events (
                source_path, ingest_order, schema_version, run_id, observed_seq,
                kind, round, timestamp, tool_call_id, parent_tool_call_id,
                tool_ordinal, duration_ms, raw_byte_offset, raw_byte_length,
                capture_mode, provider, configured_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_path,
                record.ingest_order,
                record.schema_version,
                run_id,
                observed_seq,
                kind,
                round_,
                event.get("timestamp")
                if isinstance(event.get("timestamp"), str)
                else None,
                tool_call_id if isinstance(tool_call_id, str) else None,
                parent_tool_call_id if isinstance(parent_tool_call_id, str) else None,
                _strict_int(event.get("tool_ordinal")),
                _number(event.get("duration_ms")),
                record.offset,
                record.byte_length,
                event.get("capture_mode")
                if record.schema_version == 1
                and isinstance(event.get("capture_mode"), str)
                else None,
                event.get("provider")
                if isinstance(event.get("provider"), str)
                else None,
                event.get("configured_model")
                if isinstance(event.get("configured_model"), str)
                else None,
            ),
        )
        if kind == "llm" and round_ is not None:
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return the inserted event identity")
            self._project_round(
                connection,
                source_path=source_path,
                event_id=cursor.lastrowid,
                run_id=run_id,
                round_=round_,
                event=event,
                schema_version=record.schema_version,
            )
        return True

    def _project_round(
        self,
        connection: sqlite3.Connection,
        *,
        source_path: str,
        event_id: int,
        run_id: int,
        round_: int,
        event: dict[str, Any],
        schema_version: int | None,
    ) -> None:
        context_frame = _projectable_context_frame(
            event.get("context_frame"), schema_version=schema_version
        )
        estimated_context_tokens = (
            _strict_int(context_frame["estimated_tokens"])
            if context_frame is not None
            else None
        )
        context_estimate_kind = (
            str(context_frame["estimate_kind"]) if context_frame is not None else None
        )
        connection.execute(
            """
            INSERT INTO trace_rounds (
                source_path, run_id, round, event_id, input_tokens,
                output_tokens, total_tokens, duration_ms, ttft_ms,
                finish_reason, provider_model, estimated_context_tokens,
                context_estimate_kind
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_path, run_id, round) DO UPDATE SET
                event_id = excluded.event_id,
                input_tokens = excluded.input_tokens,
                output_tokens = excluded.output_tokens,
                total_tokens = excluded.total_tokens,
                duration_ms = excluded.duration_ms,
                ttft_ms = excluded.ttft_ms,
                finish_reason = excluded.finish_reason,
                provider_model = excluded.provider_model,
                estimated_context_tokens = excluded.estimated_context_tokens,
                context_estimate_kind = excluded.context_estimate_kind
            """,
            (
                source_path,
                run_id,
                round_,
                event_id,
                _strict_int(event.get("input_tokens")),
                _strict_int(event.get("output_tokens")),
                _strict_int(event.get("total_tokens")),
                _number(event.get("duration_ms")),
                _number(event.get("ttft_ms")),
                event.get("finish_reason")
                if isinstance(event.get("finish_reason"), str)
                else None,
                event.get("provider_model")
                if isinstance(event.get("provider_model"), str)
                else None,
                estimated_context_tokens,
                context_estimate_kind,
            ),
        )
        connection.execute(
            """
            DELETE FROM context_components
            WHERE source_path = ? AND run_id = ? AND round = ?
            """,
            (source_path, run_id, round_),
        )
        if context_frame is None:
            return
        components = context_frame["components"]
        for order, component in enumerate(components):
            if not isinstance(component, dict):
                continue
            key = component.get("key")
            kind = component.get("kind")
            char_count = _strict_int(component.get("char_count"))
            byte_count = _strict_int(component.get("byte_count"))
            estimated_tokens = _strict_int(component.get("estimated_tokens"))
            if (
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(kind, str)
                or not kind.strip()
                or char_count is None
                or char_count < 0
                or byte_count is None
                or byte_count < 0
                or estimated_tokens is None
                or estimated_tokens < 0
            ):
                continue
            connection.execute(
                """
                INSERT INTO context_components (
                    source_path, run_id, round, component_order, component_key,
                    kind, source, char_count, byte_count, estimated_tokens,
                    content_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_path,
                    run_id,
                    round_,
                    order,
                    key,
                    kind,
                    component.get("source")
                    if isinstance(component.get("source"), str)
                    else None,
                    char_count,
                    byte_count,
                    estimated_tokens,
                    component.get("content_sha256")
                    if isinstance(component.get("content_sha256"), str)
                    else None,
                ),
            )

    def _rebuild_run_summaries(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM trace_runs")
        events = connection.execute(
            "SELECT * FROM trace_events ORDER BY event_id"
        ).fetchall()
        grouped: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for event in events:
            grouped[int(event["run_id"])].append(event)
        source_rows = {
            str(row["source_path"]): row
            for row in connection.execute("SELECT * FROM trace_sources").fetchall()
        }

        for run_id, run_events in grouped.items():
            timestamps = [
                str(event["timestamp"])
                for event in run_events
                if event["timestamp"] is not None
            ]
            provider = _last_value(run_events, "provider")
            configured_model = _last_value(run_events, "configured_model")
            capture_mode = _last_value(run_events, "capture_mode")
            last_kind = str(run_events[-1]["kind"])
            observed_outcome = None
            if last_kind == "run.success":
                observed_outcome = "SUCCEEDED"
            elif last_kind == "run.failure":
                observed_outcome = "FAILED"
            source_paths = {str(event["source_path"]) for event in run_events}
            sources_clean = all(
                source_path in source_rows
                and int(source_rows[source_path]["corruption_count"]) == 0
                and int(source_rows[source_path]["unsupported_schema_count"]) == 0
                and int(source_rows[source_path]["incomplete_tail"]) == 0
                for source_path in source_paths
            )
            has_start = any(event["kind"] == "run.start" for event in run_events)
            v1_sequence_clean = _v1_sequences_match_ingest_order(run_events)
            round_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS round_count,
                    CASE
                        WHEN COUNT(input_tokens) = COUNT(*) THEN SUM(input_tokens)
                        ELSE NULL
                    END AS input_tokens,
                    CASE
                        WHEN COUNT(output_tokens) = COUNT(*) THEN SUM(output_tokens)
                        ELSE NULL
                    END AS output_tokens,
                    CASE
                        WHEN COUNT(total_tokens) = COUNT(*) THEN SUM(total_tokens)
                        ELSE NULL
                    END AS total_tokens
                FROM trace_rounds WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            raw_size = sum(
                int(source_rows[source_path]["file_size"])
                for source_path in source_paths
                if source_path in source_rows
            )
            connection.execute(
                """
                INSERT INTO trace_runs (
                    run_id, first_observed_timestamp, last_observed_timestamp,
                    provider, configured_model, capture_mode, observed_outcome,
                    trace_complete, round_count, tool_count,
                    provider_input_tokens, provider_output_tokens,
                    provider_total_tokens, observed_duration_ms,
                    raw_trace_size_bytes, source_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    timestamps[0] if timestamps else None,
                    timestamps[-1] if timestamps else None,
                    provider,
                    configured_model,
                    capture_mode,
                    observed_outcome,
                    int(
                        has_start
                        and last_kind in _TERMINAL_KINDS
                        and sources_clean
                        and v1_sequence_clean
                    ),
                    int(round_row["round_count"]),
                    sum(event["kind"] == "tool" for event in run_events),
                    round_row["input_tokens"],
                    round_row["output_tokens"],
                    round_row["total_tokens"],
                    _duration_ms(timestamps),
                    raw_size,
                    len(source_paths),
                ),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()


def _source_identity(stat: Any) -> str:
    return f"{stat.st_dev}:{stat.st_ino}"


def _projectable_context_frame(
    value: object, *, schema_version: int | None
) -> dict[str, Any] | None:
    if schema_version != 1 or not isinstance(value, dict):
        return None
    estimate_kind = value.get("estimate_kind")
    estimated_tokens = _strict_int(value.get("estimated_tokens"))
    components = value.get("components")
    if (
        not isinstance(estimate_kind, str)
        or not estimate_kind.strip()
        or estimated_tokens is None
        or estimated_tokens < 0
        or not isinstance(components, list)
    ):
        return None
    return value


def _is_owned_trace_source(path: Path, root: Path) -> bool:
    return path.parent == root and path.match("run-*.jsonl")


def _v1_sequences_match_ingest_order(events: list[sqlite3.Row]) -> bool:
    return all(
        event["schema_version"] != 1
        or int(event["observed_seq"]) == int(event["ingest_order"])
        for event in events
    )


def _fingerprints(path: Path, indexed_bytes: int) -> tuple[str, str]:
    prefix_length = min(indexed_bytes, _FINGERPRINT_BYTES)
    boundary_length = min(indexed_bytes, _FINGERPRINT_BYTES)
    with path.open("rb") as stream:
        prefix = stream.read(prefix_length)
        stream.seek(indexed_bytes - boundary_length)
        boundary = stream.read(boundary_length)
    return hashlib.sha256(prefix).hexdigest(), hashlib.sha256(boundary).hexdigest()


def _strict_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _duration_ms(timestamps: list[str]) -> float | None:
    if len(timestamps) < 2:
        return None
    try:
        start = datetime.fromisoformat(timestamps[0].replace("Z", "+00:00"))
        end = datetime.fromisoformat(timestamps[-1].replace("Z", "+00:00"))
    except ValueError:
        return None
    try:
        return round((end - start).total_seconds() * 1000, 3)
    except TypeError:
        return None


def _last_value(rows: list[sqlite3.Row], key: str) -> str | None:
    for row in reversed(rows):
        if row[key] is not None:
            return str(row[key])
    return None


def _source_summary(row: sqlite3.Row) -> TraceSourceSummary:
    return TraceSourceSummary(
        source_path=str(row["source_path"]),
        indexed_bytes=int(row["indexed_bytes"]),
        file_size=int(row["file_size"]),
        mtime_ns=int(row["mtime_ns"]),
        corruption_count=int(row["corruption_count"]),
        unsupported_schema_count=int(row["unsupported_schema_count"]),
        incomplete_tail=bool(row["incomplete_tail"]),
        generation=int(row["generation"]),
    )


def _run_summary(row: sqlite3.Row) -> TraceRunSummary:
    return TraceRunSummary(
        run_id=int(row["run_id"]),
        first_observed_timestamp=row["first_observed_timestamp"],
        last_observed_timestamp=row["last_observed_timestamp"],
        provider=row["provider"],
        configured_model=row["configured_model"],
        capture_mode=row["capture_mode"],
        observed_outcome=row["observed_outcome"],
        trace_complete=bool(row["trace_complete"]),
        round_count=int(row["round_count"]),
        tool_count=int(row["tool_count"]),
        provider_input_tokens=row["provider_input_tokens"],
        provider_output_tokens=row["provider_output_tokens"],
        provider_total_tokens=row["provider_total_tokens"],
        observed_duration_ms=row["observed_duration_ms"],
        raw_trace_size_bytes=int(row["raw_trace_size_bytes"]),
        source_count=int(row["source_count"]),
    )


def _event_record(row: sqlite3.Row) -> TraceEventRecord:
    return TraceEventRecord(
        event_id=int(row["event_id"]),
        run_id=int(row["run_id"]),
        source_path=str(row["source_path"]),
        ingest_order=int(row["ingest_order"]),
        schema_version=row["schema_version"],
        observed_seq=row["observed_seq"],
        kind=str(row["kind"]),
        round=row["round"],
        timestamp=row["timestamp"],
        tool_call_id=row["tool_call_id"],
        parent_tool_call_id=row["parent_tool_call_id"],
        tool_ordinal=row["tool_ordinal"],
        duration_ms=row["duration_ms"],
        raw_byte_offset=int(row["raw_byte_offset"]),
        raw_byte_length=int(row["raw_byte_length"]),
    )


def _round_record(row: sqlite3.Row) -> TraceRoundRecord:
    return TraceRoundRecord(
        run_id=int(row["run_id"]),
        round=int(row["round"]),
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        total_tokens=row["total_tokens"],
        duration_ms=row["duration_ms"],
        ttft_ms=row["ttft_ms"],
        finish_reason=row["finish_reason"],
        provider_model=row["provider_model"],
        estimated_context_tokens=row["estimated_context_tokens"],
        context_estimate_kind=row["context_estimate_kind"],
    )


def _component_record(row: sqlite3.Row) -> ContextComponentRecord:
    return ContextComponentRecord(
        run_id=int(row["run_id"]),
        round=int(row["round"]),
        key=str(row["component_key"]),
        kind=str(row["kind"]),
        source=row["source"],
        char_count=int(row["char_count"]),
        byte_count=int(row["byte_count"]),
        estimated_tokens=int(row["estimated_tokens"]),
        content_sha256=row["content_sha256"],
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trace_sources (
    source_path TEXT PRIMARY KEY,
    source_identity TEXT NOT NULL,
    indexed_bytes INTEGER NOT NULL,
    last_ingest_order INTEGER NOT NULL,
    file_size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    prefix_sha256 TEXT NOT NULL,
    boundary_sha256 TEXT NOT NULL,
    corruption_count INTEGER NOT NULL,
    unsupported_schema_count INTEGER NOT NULL,
    incomplete_tail INTEGER NOT NULL,
    generation INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS trace_runs (
    run_id INTEGER PRIMARY KEY,
    first_observed_timestamp TEXT,
    last_observed_timestamp TEXT,
    provider TEXT,
    configured_model TEXT,
    capture_mode TEXT,
    observed_outcome TEXT,
    trace_complete INTEGER NOT NULL,
    round_count INTEGER NOT NULL,
    tool_count INTEGER NOT NULL,
    provider_input_tokens INTEGER,
    provider_output_tokens INTEGER,
    provider_total_tokens INTEGER,
    observed_duration_ms REAL,
    raw_trace_size_bytes INTEGER NOT NULL,
    source_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS trace_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL REFERENCES trace_sources(source_path) ON DELETE CASCADE,
    ingest_order INTEGER NOT NULL,
    schema_version INTEGER,
    run_id INTEGER NOT NULL,
    observed_seq INTEGER,
    kind TEXT NOT NULL,
    round INTEGER,
    timestamp TEXT,
    tool_call_id TEXT,
    parent_tool_call_id TEXT,
    tool_ordinal INTEGER,
    duration_ms REAL,
    raw_byte_offset INTEGER NOT NULL,
    raw_byte_length INTEGER NOT NULL,
    capture_mode TEXT,
    provider TEXT,
    configured_model TEXT,
    UNIQUE(source_path, ingest_order),
    UNIQUE(source_path, raw_byte_offset)
);

CREATE INDEX IF NOT EXISTS ix_trace_events_run
ON trace_events(run_id, source_path, ingest_order);

CREATE TABLE IF NOT EXISTS trace_rounds (
    source_path TEXT NOT NULL REFERENCES trace_sources(source_path) ON DELETE CASCADE,
    run_id INTEGER NOT NULL,
    round INTEGER NOT NULL,
    event_id INTEGER NOT NULL REFERENCES trace_events(event_id) ON DELETE CASCADE,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    duration_ms REAL,
    ttft_ms REAL,
    finish_reason TEXT,
    provider_model TEXT,
    estimated_context_tokens INTEGER,
    context_estimate_kind TEXT,
    PRIMARY KEY(source_path, run_id, round)
);

CREATE INDEX IF NOT EXISTS ix_trace_rounds_run
ON trace_rounds(run_id, round);

CREATE TABLE IF NOT EXISTS context_components (
    source_path TEXT NOT NULL REFERENCES trace_sources(source_path) ON DELETE CASCADE,
    run_id INTEGER NOT NULL,
    round INTEGER NOT NULL,
    component_order INTEGER NOT NULL,
    component_key TEXT NOT NULL,
    kind TEXT NOT NULL,
    source TEXT,
    char_count INTEGER NOT NULL,
    byte_count INTEGER NOT NULL,
    estimated_tokens INTEGER NOT NULL,
    content_sha256 TEXT,
    PRIMARY KEY(source_path, run_id, round, component_order),
    FOREIGN KEY(source_path, run_id, round)
        REFERENCES trace_rounds(source_path, run_id, round) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_context_components_run_round
ON context_components(run_id, round, component_order);
"""
