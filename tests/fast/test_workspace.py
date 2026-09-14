"""Focused filesystem semantics and executor observation barrier tests."""

import asyncio
import json
import os
import subprocess

import pytest

from langley.answering.contracts import ToolCall, ToolResultKind
from langley.answering.tools import ToolContext, ToolExecutor
from langley.answering.workspace_tools import WORKSPACE_TOOL_NAMES, WorkspaceTool
from langley.settings import Settings
from langley.workspace_storage import WorkspaceFileError, WorkspaceStorage

KEY = "a" * 32


@pytest.fixture
def storage(tmp_path):
    storage = WorkspaceStorage(Settings(workspace_storage_root=tmp_path))
    storage.workspace_root(KEY).mkdir()
    return storage


def test_confinement_and_symlinks(storage, tmp_path):
    for path in (
        "../escape",
        "/etc/passwd",
        "C:/file",
        "a/../b",
        "a\\b",
        "file:stream",
    ):
        with pytest.raises(WorkspaceFileError, match="PATH_OUTSIDE"):
            storage.resolve(KEY, path)
    outside = tmp_path / "outside"
    outside.write_text("private", encoding="utf-8")
    link = storage.workspace_root(KEY) / "link"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip(
            "Host does not grant symlink creation; exercised by Docker acceptance"
        )
    with pytest.raises(WorkspaceFileError, match="PATH_OUTSIDE"):
        storage.read_file(KEY, "link")


def test_write_read_and_manifest(storage):
    before = storage.manifest(KEY)
    storage.write_file(KEY, "src/a.txt", "a\r\nb\r\nc")
    assert storage.read_file(KEY, "src/a.txt", 2, 1) == {
        "path": "src/a.txt",
        "start_line": 2,
        "end_line": 2,
        "total_lines": 3,
        "content": "b\r\n",
    }
    with pytest.raises(WorkspaceFileError, match="FILE_ALREADY_EXISTS"):
        storage.write_file(KEY, "src/a.txt", "oops")
    assert storage.read_file(KEY, "src/a.txt")["content"] == "a\r\nb\r\nc"
    assert storage.changes(before, storage.manifest(KEY))["added"] == ["src/a.txt"]
    middle = storage.manifest(KEY)
    storage.write_file(KEY, "src/a.txt", "new", True)
    assert storage.changes(middle, storage.manifest(KEY))["modified"] == ["src/a.txt"]
    storage.resolve(KEY, "src/a.txt").unlink()
    assert storage.changes(middle, storage.manifest(KEY))["deleted"] == ["src/a.txt"]


@pytest.mark.skipif(os.name != "nt", reason="Windows junction boundary")
def test_windows_junction_escape(storage, tmp_path):
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    (outside / "secret").write_text("private", encoding="utf-8")
    link = storage.workspace_root(KEY) / "junction"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
        capture_output=True,
        check=False,
    )
    assert created.returncode == 0
    with pytest.raises(WorkspaceFileError, match="PATH_OUTSIDE"):
        storage.read_file(KEY, "junction/secret")


def test_text_bounds_binary(storage):
    storage.resolve(KEY, "binary").write_bytes(b"\xff\x00")
    with pytest.raises(WorkspaceFileError, match="TEXT_FILE_UNSUPPORTED"):
        storage.read_file(KEY, "binary")
    storage.write_file(KEY, "binary", "text replacement", overwrite=True)
    assert storage.read_file(KEY, "binary")["content"] == "text replacement"
    storage.settings.workspace_max_text_read_bytes = 2
    with pytest.raises(WorkspaceFileError, match="FILE_TOO_LARGE"):
        storage.read_file(KEY, "binary")
    baseline = storage.manifest(KEY)
    storage.settings.workspace_manifest_max_bytes = 2
    summary = storage.changes(baseline, storage.manifest(KEY))
    assert not summary["complete"] and summary["deleted"] == []
    storage.settings.workspace_max_write_bytes = 3
    with pytest.raises(WorkspaceFileError, match="FILE_TOO_LARGE"):
        storage.write_file(KEY, "new", "long")
    assert not storage.resolve(KEY, "new").exists()


def test_snapshot_edits_are_atomic_and_ambiguity_has_hints(storage):
    storage.write_file(KEY, "a", "alpha beta\nalpha gamma\n")
    with pytest.raises(WorkspaceFileError) as caught:
        storage.edit_file(KEY, "a", [{"old_text": "alpha", "new_text": "x"}])
    assert caught.value.hints == {"match_count": 2, "candidate_lines": [1, 2]}
    original = storage.read_file(KEY, "a")["content"]
    for edits in (
        [
            {"old_text": "beta", "new_text": "x"},
            {"old_text": "missing", "new_text": "y"},
        ],
        [
            {"old_text": "alpha beta", "new_text": "x"},
            {"old_text": "beta", "new_text": "y"},
        ],
    ):
        with pytest.raises(WorkspaceFileError):
            storage.edit_file(KEY, "a", edits)
        assert storage.read_file(KEY, "a")["content"] == original
    storage.edit_file(
        KEY,
        "a",
        [
            {"old_text": "beta", "new_text": "gamma"},
            {"old_text": "gamma", "new_text": "delta"},
        ],
    )
    assert storage.read_file(KEY, "a")["content"] == "alpha gamma\nalpha delta\n"


def test_batch_barrier_and_scope(storage):
    executor = ToolExecutor(
        tools=[WorkspaceTool(name, storage) for name in sorted(WORKSPACE_TOOL_NAMES)]
    )
    context = ToolContext(1, 1, None, workspace_id=1, workspace_storage_key=KEY)
    write = ToolCall("w", "write_file", '{"path":"a","content":"ok"}')
    read = ToolCall("r", "list_files", "{}")

    async def check():
        blocked = await executor.execute_batch((read, write), context=context)
        assert all(
            "SIDE_EFFECT_BATCH_UNSUPPORTED" in result.content for result in blocked
        )
        assert not storage.resolve(KEY, "a").exists()
        denied = await executor.execute_batch((write,))
        assert "WORKSPACE_UNAVAILABLE" in denied[0].content
        assert (await executor.execute_batch((write,), context=context))[
            0
        ].kind == ToolResultKind.SUCCESS
        results = await executor.execute_batch(
            (read, ToolCall("r2", "read_file", '{"path":"a"}')), context=context
        )
        assert all(result.kind == ToolResultKind.SUCCESS for result in results)
        assert json.loads(results[1].content)["content"] == "ok"

    asyncio.run(check())


@pytest.mark.parametrize(
    "content,offset,limit,expected",
    [
        ("one\r\ntwo\r\nthree\n", 1, 1, "one\r\n"),
        ("one\r\ntwo\r\nthree\n", 2, 10, "two\r\n"),
        ("界面\n中文\n第三行\n", 1, 10, "界面\n中文\n"),
    ],
)
def test_read_output_budget_preserves_complete_utf8_lines(
    storage, content, offset, limit, expected
):
    storage.write_file(KEY, "text", content)
    end_line = offset + len(expected.splitlines()) - 1
    expected_result = {
        "path": "text",
        "start_line": offset,
        "end_line": end_line,
        "total_lines": len(content.splitlines()),
        "content": expected,
    }
    bound = len(json.dumps(expected_result, ensure_ascii=False).encode("utf-8"))
    storage.settings.workspace_max_read_output_bytes = bound
    result = storage.read_file(KEY, "text", offset, limit)
    assert result == expected_result
    assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= bound


def test_oversized_requested_line_is_recoverable_tool_error(storage):
    storage.write_file(KEY, "large", "ok\n" + "中" * 1000 + "\n")
    storage.settings.workspace_max_read_output_bytes = 200
    executor = ToolExecutor(tools=[WorkspaceTool("read_file", storage)])
    result = asyncio.run(
        executor.execute_batch(
            (ToolCall("read", "read_file", '{"path":"large","offset":2,"limit":1}'),),
            context=ToolContext(1, 1, None, workspace_id=1, workspace_storage_key=KEY),
        )
    )[0]
    assert result.kind is ToolResultKind.TOOL_ERROR
    assert json.loads(result.content)["error"]["code"] == "READ_OUTPUT_TOO_LARGE"
