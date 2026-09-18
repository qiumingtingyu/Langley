"""Task 2A local Observatory reader, indexer, and derived-store contracts."""

import json
import os
import sqlite3
from pathlib import Path

import pytest

from langley.observatory import ObservatoryStore


def _event(run_id: int, seq: int, kind: str, **fields: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "timestamp": f"2026-09-17T00:00:{seq:02d}+00:00",
        "run_id": run_id,
        "seq": seq,
        "kind": kind,
        "capture_mode": "FULL_CONTENT",
        **fields,
    }


def _line(event: object) -> bytes:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _write(path: Path, events: list[object]) -> list[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_line(event) for event in events]
    path.write_bytes(b"".join(line + b"\n" for line in lines))
    return lines


def _store(tmp_path: Path) -> ObservatoryStore:
    return ObservatoryStore(
        tmp_path / "runtime" / "observatory.sqlite",
        tmp_path / "runtime" / "traces",
    )


def test_v1_index_query_context_and_unknown_kind(tmp_path):
    store = _store(tmp_path)
    trace_path = store.traces_root / "run-41.jsonl"
    lines = _write(
        trace_path,
        [
            _event(
                41,
                1,
                "run.start",
                provider="fake",
                configured_model="configured",
            ),
            _event(
                41,
                2,
                "llm",
                round=1,
                input_tokens=13,
                output_tokens=5,
                total_tokens=18,
                duration_ms=12.5,
                ttft_ms=None,
                finish_reason="STOP",
                provider_model="provider-model",
                context_frame={
                    "estimate_kind": "CONSERVATIVE_MULTILINGUAL_ESTIMATE_V1",
                    "estimated_tokens": 7,
                    "components": [
                        {
                            "key": "system",
                            "kind": "system",
                            "char_count": 3,
                            "byte_count": 7,
                            "estimated_tokens": 3,
                            "content_sha256": "a" * 64,
                        }
                    ],
                },
                request={"system_input": "绝不复制到 SQLite"},
            ),
            _event(41, 3, "future.visible", round=1),
            _event(
                41,
                4,
                "tool",
                round=1,
                call_id="tool-1",
                ordinal=1,
                duration_ms=6.25,
                raw_arguments="secret-tool-body",
            ),
            _event(41, 5, "run.success"),
        ],
    )

    result = store.refresh()
    assert result.events_indexed == 5
    summary = store.get_run(41)
    assert summary is not None
    assert summary.provider == "fake"
    assert summary.configured_model == "configured"
    assert summary.capture_mode == "FULL_CONTENT"
    assert summary.observed_outcome == "SUCCEEDED"
    assert summary.trace_complete is True
    assert summary.round_count == 1
    assert summary.tool_count == 1
    assert summary.provider_input_tokens == 13
    assert summary.provider_output_tokens == 5
    assert summary.provider_total_tokens == 18
    assert summary.observed_duration_ms == 4000

    events = store.list_events(41)
    assert [event.kind for event in events] == [
        "run.start",
        "llm",
        "future.visible",
        "tool",
        "run.success",
    ]
    assert events[3].tool_call_id == "tool-1"
    assert events[3].duration_ms == 6.25
    round_ = store.get_round(41, 1)
    assert round_ is not None
    assert round_.ttft_ms is None
    assert round_.estimated_context_tokens == 7
    assert round_.context_estimate_kind == "CONSERVATIVE_MULTILINGUAL_ESTIMATE_V1"
    component = store.list_context_components(41, 1)[0]
    assert component.char_count == 3
    assert component.byte_count == 7
    assert component.content_sha256 == "a" * 64

    raw = store.get_raw_event(events[1].event_id)
    assert raw.raw_json.encode("utf-8") == lines[1]
    assert raw.event["request"]["system_input"] == "绝不复制到 SQLite"
    sqlite_bytes = store.database_path.read_bytes()
    assert "绝不复制到 SQLite".encode() not in sqlite_bytes
    assert b"secret-tool-body" not in sqlite_bytes


def test_run_usage_aggregates_require_complete_per_dimension_round_coverage(tmp_path):
    store = _store(tmp_path)
    _write(
        store.traces_root / "run-50.jsonl",
        [
            _event(50, 1, "run.start"),
            _event(
                50,
                2,
                "llm",
                round=1,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
            _event(
                50,
                3,
                "llm",
                round=2,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            ),
            _event(50, 4, "run.success"),
        ],
    )
    _write(
        store.traces_root / "run-51.jsonl",
        [
            _event(51, 1, "run.start"),
            _event(
                51,
                2,
                "llm",
                round=1,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
            _event(
                51,
                3,
                "llm",
                round=2,
                input_tokens=None,
                output_tokens=7,
                total_tokens=None,
            ),
            _event(51, 4, "run.success"),
        ],
    )
    _write(
        store.traces_root / "run-52.jsonl",
        [
            _event(52, 1, "run.start"),
            _event(
                52,
                2,
                "llm",
                round=1,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            ),
            _event(52, 3, "run.success"),
        ],
    )

    store.refresh()

    all_known = store.get_run(50)
    assert all_known is not None
    assert all_known.provider_input_tokens == 10
    assert all_known.provider_output_tokens == 5
    assert all_known.provider_total_tokens == 15

    partially_unknown = store.get_run(51)
    assert partially_unknown is not None
    assert partially_unknown.provider_input_tokens is None
    assert partially_unknown.provider_output_tokens == 12
    assert partially_unknown.provider_total_tokens is None

    observed_zero = store.get_run(52)
    assert observed_zero is not None
    assert observed_zero.provider_input_tokens == 0
    assert observed_zero.provider_output_tokens == 0
    assert observed_zero.provider_total_tokens == 0


def test_legacy_v0_keeps_v1_only_fields_unknown(tmp_path):
    store = _store(tmp_path)
    _write(
        store.traces_root / "run-7.jsonl",
        [
            {
                "timestamp": "2026-09-17T00:00:01+00:00",
                "run_id": 7,
                "kind": "run.start",
                "provider": "legacy",
                "configured_model": "old",
                "seq": 900,
            },
            {
                "timestamp": "2026-09-17T00:00:02+00:00",
                "run_id": 7,
                "kind": "llm",
                "round": 1,
                "input_tokens": 8,
                "output_tokens": 2,
                "total_tokens": 10,
                "duration_ms": 4.0,
                "ttft_ms": None,
                "context_frame": {
                    "estimate_kind": "SHOULD_NOT_BE_PROJECTED_FOR_V0",
                    "estimated_tokens": 999,
                    "components": [
                        {
                            "key": "system",
                            "kind": "system",
                            "char_count": 1,
                            "byte_count": 1,
                            "estimated_tokens": 1,
                        }
                    ],
                },
            },
            {
                "timestamp": "2026-09-17T00:00:03+00:00",
                "run_id": 7,
                "kind": "run.success",
            },
        ],
    )

    store.refresh()
    events = store.list_events(7)
    assert all(event.schema_version is None for event in events)
    assert all(event.observed_seq is None for event in events)
    assert store.get_run(7).capture_mode is None  # type: ignore[union-attr]
    assert store.get_round(7, 1).estimated_context_tokens is None  # type: ignore[union-attr]
    assert store.get_round(7, 1).context_estimate_kind is None  # type: ignore[union-attr]
    assert store.list_context_components(7, 1) == ()


@pytest.mark.parametrize(
    "context_frame",
    [
        {"estimated_tokens": 123, "components": []},
        {
            "estimate_kind": "CONSERVATIVE_MULTILINGUAL_ESTIMATE_V1",
            "estimated_tokens": -1,
            "components": [],
        },
        {
            "estimate_kind": "CONSERVATIVE_MULTILINGUAL_ESTIMATE_V1",
            "estimated_tokens": "invalid",
            "components": [],
        },
        {
            "estimate_kind": "CONSERVATIVE_MULTILINGUAL_ESTIMATE_V1",
            "estimated_tokens": 3,
            "components": {},
        },
    ],
)
def test_invalid_v1_context_frame_degrades_without_dropping_round(
    tmp_path, context_frame
):
    store = _store(tmp_path)
    _write(
        store.traces_root / "run-invalid-frame.jsonl",
        [_event(25, 1, "llm", round=1, context_frame=context_frame)],
    )

    result = store.refresh()

    assert result.events_indexed == 1
    assert result.malformed_lines == 0
    assert [event.kind for event in store.list_events(25)] == ["llm"]
    round_ = store.get_round(25, 1)
    assert round_ is not None
    assert round_.estimated_context_tokens is None
    assert round_.context_estimate_kind is None
    assert store.list_context_components(25, 1) == ()


def test_invalid_context_components_are_omitted_without_dropping_valid_frame(
    tmp_path,
):
    store = _store(tmp_path)

    def component(key: str, kind: str = "system", **fields: object):
        return {
            "key": key,
            "kind": kind,
            "char_count": 1,
            "byte_count": 1,
            "estimated_tokens": 1,
            **fields,
        }

    _write(
        store.traces_root / "run-invalid-components.jsonl",
        [
            _event(
                26,
                1,
                "llm",
                round=1,
                context_frame={
                    "estimate_kind": "CONSERVATIVE_MULTILINGUAL_ESTIMATE_V1",
                    "estimated_tokens": 9,
                    "components": [
                        component("valid"),
                        component("negative-char", char_count=-1),
                        component("negative-byte", byte_count=-1),
                        component("negative-tokens", estimated_tokens=-1),
                        component(""),
                        component("blank-kind", kind=" "),
                    ],
                },
            )
        ],
    )

    store.refresh()

    round_ = store.get_round(26, 1)
    assert round_ is not None
    assert round_.estimated_context_tokens == 9
    assert round_.context_estimate_kind == "CONSERVATIVE_MULTILINGUAL_ESTIMATE_V1"
    assert [item.key for item in store.list_context_components(26, 1)] == ["valid"]


def test_incremental_append_and_unchanged_file_do_not_duplicate(tmp_path):
    store = _store(tmp_path)
    path = store.traces_root / "run-8.jsonl"
    _write(path, [_event(8, 1, "run.start")])
    first = store.refresh()
    first_size = path.stat().st_size
    assert first.events_indexed == 1
    assert first.bytes_examined == first_size

    with path.open("ab") as stream:
        appended = _line(_event(8, 2, "run.success")) + b"\n"
        stream.write(appended)
    second = store.refresh()
    assert second.events_indexed == 1
    assert second.bytes_examined == len(appended)
    assert [event.observed_seq for event in store.list_events(8)] == [1, 2]

    unchanged = store.refresh()
    assert unchanged.events_indexed == 0
    assert unchanged.bytes_examined == 0
    assert len(store.list_events(8)) == 2


def test_missing_terminal_and_torn_final_line_are_not_failures(tmp_path):
    store = _store(tmp_path)
    path = store.traces_root / "run-9.jsonl"
    start = _line(_event(9, 1, "run.start"))
    terminal = _line(_event(9, 2, "run.success"))
    path.parent.mkdir(parents=True)
    path.write_bytes(start + b"\n" + terminal)

    first = store.refresh()
    assert first.events_indexed == 1
    source = store.list_sources()[0]
    assert source.incomplete_tail is True
    summary = store.get_run(9)
    assert summary is not None
    assert summary.observed_outcome is None
    assert summary.trace_complete is False

    with path.open("ab") as stream:
        stream.write(b"\n")
    second = store.refresh()
    assert second.events_indexed == 1
    assert store.list_sources()[0].incomplete_tail is False
    assert store.get_run(9).observed_outcome == "SUCCEEDED"  # type: ignore[union-attr]
    assert store.get_run(9).trace_complete is True  # type: ignore[union-attr]


def test_malformed_middle_and_future_schema_are_explicit_and_nonfatal(tmp_path):
    store = _store(tmp_path)
    path = store.traces_root / "run-10.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        _line(_event(10, 1, "run.start"))
        + b"\n{malformed}\n"
        + _line(
            {
                "schema_version": 2,
                "run_id": 10,
                "kind": "future.schema",
            }
        )
        + b"\n"
        + _line(_event(10, 4, "future.kind"))
        + b"\n"
        + _line(_event(10, 5, "run.success"))
        + b"\n"
    )

    result = store.refresh()
    assert result.malformed_lines == 1
    assert result.unsupported_schema_lines == 1
    assert [event.kind for event in store.list_events(10)] == [
        "run.start",
        "future.kind",
        "run.success",
    ]
    source = store.list_sources()[0]
    assert source.corruption_count == 1
    assert source.unsupported_schema_count == 1
    assert store.get_run(10).trace_complete is False  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("field", "value", "remove"),
    [
        ("timestamp", None, True),
        ("timestamp", "", False),
        ("run_id", None, True),
        ("seq", None, True),
        ("seq", 0, False),
        ("seq", -1, False),
        ("kind", None, True),
        ("kind", "", False),
        ("capture_mode", None, True),
        ("capture_mode", "REDACTED", False),
        ("capture_mode", [], False),
    ],
)
def test_invalid_v1_envelope_is_corruption_and_not_projected(
    tmp_path, field, value, remove
):
    store = _store(tmp_path)
    invalid = _event(18, 1, "run.start")
    if remove:
        invalid.pop(field)
    else:
        invalid[field] = value
    _write(store.traces_root / "run-18.jsonl", [invalid])

    result = store.refresh()

    assert result.malformed_lines == 1
    assert result.events_indexed == 0
    assert store.list_events(18) == ()
    assert store.get_run(18) is None
    assert store.list_sources()[0].corruption_count == 1


@pytest.mark.parametrize("sequences", [(1, 1), (2, 1), (1, 3)])
def test_v1_duplicate_decreasing_or_gapped_sequence_is_incomplete(tmp_path, sequences):
    store = _store(tmp_path)
    _write(
        store.traces_root / "run-19.jsonl",
        [
            _event(19, sequences[0], "run.start"),
            _event(19, sequences[1], "run.success"),
        ],
    )

    store.refresh()

    summary = store.get_run(19)
    assert summary is not None
    assert summary.observed_outcome == "SUCCEEDED"
    assert summary.trace_complete is False


def test_pure_v1_sequence_shifted_from_ingest_order_is_incomplete(tmp_path):
    store = _store(tmp_path)
    _write(
        store.traces_root / "run-27.jsonl",
        [_event(27, 2, "run.start"), _event(27, 3, "run.success")],
    )

    store.refresh()

    summary = store.get_run(27)
    assert summary is not None
    assert summary.observed_outcome == "SUCCEEDED"
    assert summary.trace_complete is False


def test_mixed_legacy_then_v1_sequence_matching_ingest_order_is_complete(tmp_path):
    store = _store(tmp_path)
    _write(
        store.traces_root / "run-28.jsonl",
        [
            {
                "timestamp": "2026-09-17T00:00:01+00:00",
                "run_id": 28,
                "kind": "run.start",
            },
            {
                "timestamp": "2026-09-17T00:00:02+00:00",
                "run_id": 28,
                "kind": "legacy.visible",
            },
            _event(28, 3, "llm", round=1),
            _event(28, 4, "run.success"),
        ],
    )

    store.refresh()

    events = store.list_events(28)
    assert [event.observed_seq for event in events] == [None, None, 3, 4]
    assert [event.ingest_order for event in events] == [1, 2, 3, 4]
    assert store.get_run(28).trace_complete is True  # type: ignore[union-attr]


def test_terminal_only_trace_is_not_complete(tmp_path):
    store = _store(tmp_path)
    _write(store.traces_root / "run-20.jsonl", [_event(20, 1, "run.success")])

    store.refresh()

    summary = store.get_run(20)
    assert summary is not None
    assert summary.observed_outcome == "SUCCEEDED"
    assert summary.trace_complete is False


def test_valid_v1_start_terminal_sequence_is_complete(tmp_path):
    store = _store(tmp_path)
    _write(
        store.traces_root / "run-21.jsonl",
        [_event(21, 1, "run.start"), _event(21, 2, "run.success")],
    )

    store.refresh()

    assert store.get_run(21).trace_complete is True  # type: ignore[union-attr]


def test_replaced_file_resets_only_that_source_projection(tmp_path):
    store = _store(tmp_path)
    path = store.traces_root / "run-shared.jsonl"
    _write(path, [_event(11, 1, "run.start"), _event(11, 2, "run.success")])
    store.refresh()
    old_generation = store.list_sources()[0].generation

    replacement = path.with_suffix(".replacement")
    _write(
        replacement,
        [_event(12, 1, "run.start"), _event(12, 2, "run.failure")],
    )
    os.replace(replacement, path)
    result = store.refresh()

    assert result.source_resets == 1
    assert store.get_run(11) is None
    assert store.get_run(12).observed_outcome == "FAILED"  # type: ignore[union-attr]
    assert store.list_sources()[0].generation == old_generation + 1


def test_truncated_file_resets_projection_and_ingest_order(tmp_path):
    store = _store(tmp_path)
    path = store.traces_root / "run-truncated.jsonl"
    _write(
        path,
        [
            _event(16, 1, "run.start"),
            _event(16, 2, "llm", round=1),
            _event(16, 3, "run.success"),
        ],
    )
    store.refresh()

    _write(path, [_event(17, 1, "run.start")])
    result = store.refresh()

    assert result.source_resets == 1
    assert store.get_run(16) is None
    assert [event.ingest_order for event in store.list_events(17)] == [1]
    summary = store.get_run(17)
    assert summary is not None
    assert summary.observed_outcome is None
    assert summary.trace_complete is False


def test_refresh_reconciles_deleted_owned_source_and_preserves_other_run(tmp_path):
    store = _store(tmp_path)
    removed_path = store.traces_root / "run-22.jsonl"
    retained_path = store.traces_root / "run-23.jsonl"
    _write(
        removed_path,
        [
            _event(22, 1, "run.start"),
            _event(
                22,
                2,
                "llm",
                round=1,
                context_frame={
                    "estimate_kind": "CONSERVATIVE_MULTILINGUAL_ESTIMATE_V1",
                    "estimated_tokens": 1,
                    "components": [
                        {
                            "key": "system",
                            "kind": "system",
                            "char_count": 1,
                            "byte_count": 1,
                            "estimated_tokens": 1,
                        }
                    ],
                },
            ),
            _event(22, 3, "run.success"),
        ],
    )
    _write(
        retained_path,
        [_event(23, 1, "run.start"), _event(23, 2, "run.success")],
    )
    store.refresh()
    removed_path.unlink()

    result = store.refresh()

    assert result.sources_removed == 1
    assert store.get_run(22) is None
    assert store.list_events(22) == ()
    assert store.list_rounds(22) == ()
    assert store.list_context_components(22, 1) == ()
    assert store.get_run(23).trace_complete is True  # type: ignore[union-attr]
    assert [Path(source.source_path).name for source in store.list_sources()] == [
        "run-23.jsonl"
    ]
    assert store.refresh().sources_removed == 0


def test_refresh_does_not_remove_explicit_source_outside_traces_root(tmp_path):
    store = _store(tmp_path)
    external = tmp_path / "external" / "run-24.jsonl"
    _write(
        external,
        [_event(24, 1, "run.start"), _event(24, 2, "run.success")],
    )
    store.refresh_source(external)

    result = store.refresh()

    assert result.sources_removed == 0
    assert store.get_run(24).trace_complete is True  # type: ignore[union-attr]


def test_utf8_locator_round_trip_uses_bytes_not_characters(tmp_path):
    store = _store(tmp_path)
    path = store.traces_root / "run-13.jsonl"
    first = _line(_event(13, 1, "run.start", note="中文"))
    second = _line(_event(13, 2, "run.success"))
    path.parent.mkdir(parents=True)
    path.write_bytes(first + b"\n" + second + b"\n")
    store.refresh()

    events = store.list_events(13)
    assert events[1].raw_byte_offset == len(first) + 1
    assert events[0].raw_byte_length == len(first)
    assert len(first) > len(first.decode("utf-8"))
    assert store.get_raw_event(events[0].event_id).raw_json.encode() == first


def test_metadata_only_context_projects_without_content_hash(tmp_path):
    store = _store(tmp_path)
    event = _event(
        14,
        1,
        "llm",
        round=1,
        capture_mode="METADATA_ONLY",
        ttft_ms=None,
        context_frame={
            "estimate_kind": "CONSERVATIVE_MULTILINGUAL_ESTIMATE_V1",
            "estimated_tokens": 2,
            "components": [
                {
                    "key": "system",
                    "kind": "system",
                    "char_count": 4,
                    "byte_count": 4,
                    "estimated_tokens": 1,
                }
            ],
        },
    )
    _write(store.traces_root / "run-14.jsonl", [event])
    store.refresh()

    component = store.list_context_components(14, 1)[0]
    assert component.content_sha256 is None
    assert store.get_round(14, 1).ttft_ms is None  # type: ignore[union-attr]
    assert store.get_round(14, 1).estimated_context_tokens == 2  # type: ignore[union-attr]
    assert (  # type: ignore[union-attr]
        store.get_round(14, 1).context_estimate_kind
        == "CONSERVATIVE_MULTILINGUAL_ESTIMATE_V1"
    )
    assert store.list_events(14)[0].observed_seq == 1


def test_rebuild_and_sqlite_recreation_leave_raw_jsonl_unchanged(tmp_path):
    store = _store(tmp_path)
    path = store.traces_root / "run-15.jsonl"
    _write(path, [_event(15, 1, "run.start"), _event(15, 2, "run.success")])
    raw_before = path.read_bytes()
    store.refresh()
    runs_before = store.list_runs()
    events_before = [
        (event.ingest_order, event.observed_seq, event.kind)
        for event in store.list_events(15)
    ]

    rebuilt = store.rebuild()
    assert rebuilt.events_indexed == 2
    assert store.list_runs() == runs_before
    assert [
        (event.ingest_order, event.observed_seq, event.kind)
        for event in store.list_events(15)
    ] == events_before
    assert path.read_bytes() == raw_before

    store.database_path.unlink()
    recreated = ObservatoryStore(store.database_path, store.traces_root)
    assert recreated.list_runs() == ()
    recreated.refresh()
    assert recreated.list_runs() == runs_before
    assert path.read_bytes() == raw_before


def test_schema_is_local_explicit_and_contains_required_tables(tmp_path):
    store = _store(tmp_path)
    with sqlite3.connect(store.database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        event_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(trace_events)")
        }
        round_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(trace_rounds)")
        }
    assert {
        "trace_sources",
        "trace_runs",
        "trace_events",
        "trace_rounds",
        "context_components",
    } <= tables
    assert "request" not in event_columns
    assert "result_content" not in event_columns
    assert "context_estimate_kind" in round_columns
