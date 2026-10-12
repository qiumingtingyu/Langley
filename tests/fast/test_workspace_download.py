"""Owned API + real isolated filesystem download boundaries; no DB reset or LLM."""

import asyncio
import os
import shutil
import stat
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import anyio
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from langley.api.dependencies import (
    get_current_user_id,
    get_session_factory,
    get_workspace_storage,
)
from langley.api.workspaces import _DownloadResponse, router
from langley.settings import Settings
from langley.workspace_storage import WorkspaceFileError, WorkspaceStorage

KEY = "a" * 32


@pytest.fixture
def owned(tmp_path):
    storage = WorkspaceStorage(Settings(workspace_storage_root=tmp_path))
    storage.workspace_root(KEY).mkdir()
    app = FastAPI()
    app.include_router(router)
    open_sessions = []

    class Session:
        async def __aenter__(self):
            open_sessions.append(self)
            return self

        async def __aexit__(self, *args):
            open_sessions.remove(self)

        async def get(self, model, identity):
            return {
                1: SimpleNamespace(user_id=1, storage_key=KEY),
                2: SimpleNamespace(user_id=2, storage_key=KEY),
            }.get(identity)

    app.dependency_overrides[get_current_user_id] = lambda: 1
    app.dependency_overrides[get_session_factory] = lambda: Session
    app.dependency_overrides[get_workspace_storage] = lambda: storage
    return app, storage, open_sessions


def test_binary_chinese_headers_and_cleanup(owned, monkeypatch):
    app, storage, sessions = owned
    raw = b"PK\x03\x04\x00\xff\x80binary"
    storage.resolve(KEY, "报告.docx").write_bytes(raw)
    snapshots = []
    original = storage.download_snapshot

    def capture(key, path):
        assert not sessions and storage.execution_lock(key).locked()
        result = original(key, path)
        snapshots.append(result[0])
        return result

    monkeypatch.setattr(storage, "download_snapshot", capture)
    with TestClient(app) as client:
        response = client.get(
            "/api/workspaces/1/files/download", params={"path": "报告.docx"}
        )
    assert response.status_code == 200 and response.content == raw
    assert response.headers["content-type"] == "application/octet-stream"
    assert "attachment" in response.headers["content-disposition"]
    assert (
        unquote(response.headers["content-disposition"].split("UTF-8''")[1])
        == "报告.docx"
    )
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-length"] == str(len(raw))
    assert all(s.closed for s in snapshots)


@pytest.mark.parametrize("identity", [2, 999])
def test_owner_or_missing_workspace_denied(owned, identity):
    app, storage, _ = owned
    storage.resolve(KEY, "file").write_bytes(b"private")
    with TestClient(app) as client:
        response = client.get(
            f"/api/workspaces/{identity}/files/download", params={"path": "file"}
        )
    assert response.status_code == 404


@pytest.mark.parametrize(
    "path,status",
    [
        ("../outside", 422),
        ("C:/Windows/a", 422),
        (r"C:\Windows\a", 422),
        ("/workspace/a", 422),
        (r"\\server\share\a", 422),
        ("x:ads", 422),
        ("NUL", 422),
        ("dir", 422),
        ("missing", 404),
    ],
)
def test_path_errors(owned, path, status):
    app, storage, _ = owned
    storage.resolve(KEY, "dir").mkdir()
    with TestClient(app) as client:
        response = client.get("/api/workspaces/1/files/download", params={"path": path})
    assert response.status_code == status


def test_hardlink_and_oversize(owned, monkeypatch, tmp_path):
    app, storage, _ = owned
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    os.link(outside, storage.resolve(KEY, "hard"))
    with TestClient(app) as client:
        assert (
            client.get(
                "/api/workspaces/1/files/download", params={"path": "hard"}
            ).status_code
            == 422
        )
        monkeypatch.setattr("langley.workspace_storage.MAX_DOWNLOAD_BYTES", 4)
        storage.resolve(KEY, "large").write_bytes(b"12345")
        assert (
            client.get(
                "/api/workspaces/1/files/download", params={"path": "large"}
            ).status_code
            == 413
        )


def test_junction_or_symlink(owned, tmp_path):
    app, storage, _ = owned
    outside = tmp_path / "external"
    outside.mkdir()
    (outside / "data").write_bytes(b"secret")
    link = storage.workspace_root(KEY) / "link"
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True
        )
        assert result.returncode == 0
    else:
        link.symlink_to(outside, target_is_directory=True)
    with TestClient(app) as client:
        assert (
            client.get(
                "/api/workspaces/1/files/download", params={"path": "link/data"}
            ).status_code
            == 422
        )


def test_snapshot_survives_later_writer_and_waits_for_lock(owned, monkeypatch):
    app, storage, _ = owned
    target = storage.resolve(KEY, "data")
    target.write_bytes(b"A" * 150000)

    async def check():
        lock = storage.execution_lock(KEY)
        ready = asyncio.Event()
        original = storage.download_snapshot
        loop = asyncio.get_running_loop()

        def capture(key, path):
            snapshot = original(key, path)
            loop.call_soon_threadsafe(ready.set)
            return snapshot

        monkeypatch.setattr(storage, "download_snapshot", capture)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            await lock.acquire()
            pending = asyncio.create_task(
                client.get("/api/workspaces/1/files/download", params={"path": "data"})
            )
            await asyncio.sleep(0)
            assert not ready.is_set()
            lock.release()
            await ready.wait()
            async with lock:
                target.write_bytes(b"B" * 150000)
            response = await pending
            assert response.content == b"A" * 150000
            assert target.read_bytes() == b"B" * 150000

    asyncio.run(check())


def test_slow_copy_does_not_block_light_request(owned, monkeypatch):
    app, storage, _ = owned
    storage.resolve(KEY, "data").write_bytes(b"binary")
    entered, release = threading.Event(), threading.Event()
    snapshots = []
    original = storage.download_snapshot
    main_thread = threading.get_ident()

    @app.get("/light")
    async def light():
        return {"ready": True}

    def slow(key, path):
        assert threading.get_ident() != main_thread
        result = original(key, path)
        snapshots.append(result[0])
        entered.set()
        assert release.wait(3), "test did not release the worker"
        return result

    monkeypatch.setattr(storage, "download_snapshot", slow)

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            task = asyncio.create_task(
                client.get("/api/workspaces/1/files/download", params={"path": "data"})
            )
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                response = await asyncio.wait_for(client.get("/light"), timeout=0.5)
                assert response.json() == {"ready": True}
                assert not task.done() and storage.execution_lock(KEY).locked()
            finally:
                release.set()
                await task

    asyncio.run(check())
    assert snapshots and all(s.closed for s in snapshots)


@pytest.mark.parametrize("copy_fails", [False, True])
def test_cancelled_copy_keeps_lock_until_worker_and_cleans(
    owned, monkeypatch, copy_fails
):
    app, storage, _ = owned
    storage.resolve(KEY, "data").write_bytes(b"binary")
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    snapshots = []
    original = storage.download_snapshot

    def slow(key, path):
        result = original(key, path)
        snapshots.append(result[0])
        entered.set()
        try:
            assert release.wait(3), "test did not release the worker"
            if copy_fails:
                # A real storage error closes its incomplete snapshot itself.
                result[0].close()
                raise WorkspaceFileError("FILE_CHANGED")
            return result
        finally:
            finished.set()

    monkeypatch.setattr(storage, "download_snapshot", slow)

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            task = asyncio.create_task(
                client.get("/api/workspaces/1/files/download", params={"path": "data"})
            )
            lock = storage.execution_lock(KEY)
            waiter = None
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()  # A second cancellation must not free the lock.
                await asyncio.sleep(0)
                assert not task.done() and lock.locked() and not finished.is_set()
                waiter = asyncio.create_task(lock.acquire())
                await asyncio.sleep(0)
                assert not waiter.done()
            finally:
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                if waiter is not None:
                    await asyncio.wait_for(waiter, 1)
                    lock.release()
            assert finished.is_set() and not lock.locked()

    asyncio.run(check())
    assert snapshots and all(s.closed for s in snapshots)


def test_anyio_scope_cancellation_drains_copy(owned, monkeypatch):
    app, storage, _ = owned
    storage.resolve(KEY, "data").write_bytes(b"binary")
    entered, release = threading.Event(), threading.Event()
    snapshots, scopes = [], []
    original = storage.download_snapshot

    def slow(key, path):
        result = original(key, path)
        snapshots.append(result[0])
        entered.set()
        assert release.wait(3)
        return result

    monkeypatch.setattr(storage, "download_snapshot", slow)

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:

            async def request():
                with anyio.CancelScope() as scope:
                    scopes.append(scope)
                    await client.get(
                        "/api/workspaces/1/files/download", params={"path": "data"}
                    )

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(request)
                try:
                    assert await asyncio.to_thread(entered.wait, 2)
                    scopes[0].cancel()
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    assert storage.execution_lock(KEY).locked()
                    assert snapshots and not snapshots[0].closed
                finally:
                    release.set()
            assert not storage.execution_lock(KEY).locked()

    asyncio.run(check())
    assert snapshots and all(s.closed for s in snapshots)


def test_external_mutation_rejected_and_snapshot_cleaned(owned, monkeypatch):
    _, storage, _ = owned
    target = storage.resolve(KEY, "data")
    target.write_bytes(b"original")
    import tempfile

    original = tempfile.TemporaryFile
    captured = []

    class ChangingSnapshot:
        def __init__(self):
            self.file = original(mode="w+b")
            captured.append(self.file)

        def write(self, chunk):
            target.write_bytes(b"changed size")
            return self.file.write(chunk)

        def close(self):
            self.file.close()

    monkeypatch.setattr(
        "langley.workspace_storage.tempfile.TemporaryFile",
        lambda **kw: ChangingSnapshot(),
    )
    with pytest.raises(WorkspaceFileError, match="FILE_CHANGED"):
        storage.download_snapshot(KEY, "data")
    assert all(s.closed for s in captured)


def test_response_closes_on_send_failure(owned):
    _, storage, _ = owned
    storage.resolve(KEY, "data").write_bytes(b"hello")
    stream, size, name = storage.download_snapshot(KEY, "data")

    async def check():
        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            raise RuntimeError("client gone")

        with pytest.raises(RuntimeError, match="client gone"):
            await _DownloadResponse(stream, size, name)(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
            )

    asyncio.run(check())
    assert stream.closed


def test_download_permission_error(owned, monkeypatch):
    app, storage, _ = owned
    storage.resolve(KEY, "file").write_bytes(b"bytes")
    original = os.open

    def denied(path, flags, *args, **kwargs):
        if str(path).endswith("file"):
            raise PermissionError("denied")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr("langley.workspace_storage.os.open", denied)
    with TestClient(app) as client:
        assert (
            client.get(
                "/api/workspaces/1/files/download", params={"path": "file"}
            ).status_code
            == 403
        )


def test_source_swapped_before_open_is_not_served(owned, monkeypatch, tmp_path):
    _, storage, _ = owned
    target = storage.resolve(KEY, "file")
    target.write_bytes(b"inside")
    outside = tmp_path / "outside-data"
    outside.write_bytes(b"private outside")
    original = os.open

    def swapped(path, flags, *args, **kwargs):
        return original(
            outside if Path(path) == target else path, flags, *args, **kwargs
        )

    monkeypatch.setattr("langley.workspace_storage.os.open", swapped)
    with pytest.raises(WorkspaceFileError, match="FILE_CHANGED"):
        storage.download_snapshot(KEY, "file")


def test_response_closes_on_cancellation(owned):
    _, storage, _ = owned
    storage.resolve(KEY, "file").write_bytes(b"hello")
    stream, size, name = storage.download_snapshot(KEY, "file")

    async def check():
        started = asyncio.Event()

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(
            _DownloadResponse(stream, size, name)(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(check())
    assert stream.closed


def test_path_and_handle_ctime_can_differ_without_content_change(owned, monkeypatch):
    _, storage, _ = owned
    storage.resolve(KEY, "file").write_bytes(b"hello")
    original = os.fstat

    def different_ctime(fd):
        info = original(fd)
        fields = {
            k: getattr(info, k)
            for k in (
                "st_dev",
                "st_ino",
                "st_size",
                "st_mtime_ns",
                "st_nlink",
                "st_mode",
            )
        }
        return SimpleNamespace(**fields, st_ctime_ns=info.st_ctime_ns + 1000)

    monkeypatch.setattr("langley.workspace_storage.os.fstat", different_ctime)
    stream, _, _ = storage.download_snapshot(KEY, "file")
    with stream:
        assert stream.read() == b"hello"


def test_symlink_file_is_denied(owned, tmp_path):
    app, storage, _ = owned
    outside = tmp_path / "external-file"
    outside.write_bytes(b"secret")
    link = storage.workspace_root(KEY) / "symlink-file"
    try:
        link.symlink_to(outside)
    except OSError as error:
        if os.name != "nt" or getattr(error, "winerror", None) != 1314:
            raise
        pytest.skip("Windows symlink privilege missing; junction tested separately")
    with TestClient(app) as client:
        assert (
            client.get(
                "/api/workspaces/1/files/download", params={"path": "symlink-file"}
            ).status_code
            == 422
        )


def test_special_file_type_is_denied_before_open(owned, monkeypatch):
    _, storage, _ = owned
    target = storage.resolve(KEY, "special")
    target.write_bytes(b"not a pipe")
    original = Path.stat

    def special(path, *args, **kwargs):
        if path == target and kwargs.get("follow_symlinks") is False:
            return SimpleNamespace(st_mode=stat.S_IFIFO, st_nlink=1)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", special)
    with pytest.raises(WorkspaceFileError, match="DOWNLOAD_FILE_UNSUPPORTED"):
        storage.download_snapshot(KEY, "special")


def test_generated_docx_is_downloaded_byte_for_byte(owned):
    artifact = os.environ.get("LANGLEY_DOWNLOAD_TEST_ARTIFACT")
    if not artifact:
        pytest.skip("Explicit isolated Agent artifact acceptance opt-in only")
    source = Path(artifact)
    assert source.is_file()
    app, storage, _ = owned
    shutil.copyfile(source, storage.resolve(KEY, "生成报告.docx"))
    with TestClient(app) as client:
        response = client.get(
            "/api/workspaces/1/files/download", params={"path": "生成报告.docx"}
        )
    assert response.status_code == 200
    assert response.content == source.read_bytes()
