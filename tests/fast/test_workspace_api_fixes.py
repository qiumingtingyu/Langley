"""Workspace read lock-domain and publication outcome regression tests."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from langley.answering.fake_provider import FakeProvider
from langley.api.dependencies import get_current_user_id, get_session_factory
from langley.api.workspaces import _publish
from langley.infrastructure.models import Workspace
from langley.main import create_app
from langley.settings import Settings
from langley.workspace_storage import WorkspaceStorage


@pytest.mark.parametrize("preview", [False, True])
def test_file_api_shares_execution_lock_and_releases_db(tmp_path, monkeypatch, preview):
    async def check():
        app = create_app(
            Settings(
                database_url="mysql+asyncmy://unused/langley_test",
                workspace_storage_root=tmp_path,
            ),
            provider=FakeProvider([]),
        )
        storage = app.state.workspace_storage
        execution = app.state.execution_manager._workflow_factory()
        assert execution._workspace_storage is storage
        keys = {1: "a" * 32, 2: "b" * 32}
        for key in keys.values():
            storage.workspace_root(key).mkdir()
            storage.write_file(key, "file.txt", "hello")
        db_closed = asyncio.Event()
        opened = 0

        class ReadSession:
            async def __aenter__(self):
                nonlocal opened
                opened += 1
                return self

            async def __aexit__(self, *args):
                nonlocal opened
                opened -= 1
                db_closed.set()

            async def get(self, model, identity):
                assert model is Workspace
                return SimpleNamespace(user_id=1, storage_key=keys[identity])

        app.dependency_overrides[get_current_user_id] = lambda: 1
        app.dependency_overrides[get_session_factory] = lambda: ReadSession
        entered = []
        operation_name = "read_file" if preview else "list_files"
        original = getattr(storage, operation_name)

        def observed(key, *args):
            assert opened == 0
            entered.append(key)
            return original(key, *args)

        monkeypatch.setattr(storage, operation_name, observed)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            params = {
                "preview": str(preview).lower(),
                "path": "file.txt" if preview else "",
            }
            lock = execution._workspace_storage.execution_lock(keys[1])
            assert lock is not storage.execution_lock(keys[2])
            async with lock:
                waiting = asyncio.create_task(
                    client.get("/api/workspaces/1/files", params=params)
                )
                await asyncio.wait_for(db_closed.wait(), timeout=2)
                assert entered == [] and not waiting.done()
                other = await asyncio.wait_for(
                    client.get("/api/workspaces/2/files", params=params), timeout=2
                )
                assert other.status_code == 200
                assert entered == [keys[2]] and not waiting.done()
            result = await asyncio.wait_for(waiting, timeout=2)
            assert result.status_code == 200
            assert entered == [keys[2], keys[1]]
        await app.state.database_engine.dispose()

    asyncio.run(check())


@pytest.mark.parametrize(
    "outcome", ["before_commit", "unknown", "committed", "cancelled"]
)
def test_publish_cleanup_preserves_uncertain_commit(tmp_path, outcome):
    storage = WorkspaceStorage(Settings(workspace_storage_root=tmp_path))
    key = "a" * 32
    storage.workspace_root(key).mkdir()
    storage.write_file(key, "artifact.txt", "keep")
    added = []
    published = []

    class PublicationSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def add(self, row):
            row.id = len(added) + 1
            added.append(row)

        async def flush(self):
            if outcome == "before_commit":
                raise RuntimeError("flush failed before commit")

        def begin(self):
            return Transaction()

    class Transaction:
        async def __aenter__(self):
            pass

        async def __aexit__(self, error_type, *args):
            if error_type is not None:
                return False
            if outcome == "committed":
                published.extend(added)
            if outcome == "cancelled":
                raise asyncio.CancelledError()
            raise RuntimeError("commit acknowledgement unavailable")

    async def check():
        error = asyncio.CancelledError if outcome == "cancelled" else RuntimeError
        with pytest.raises(error):
            await _publish(PublicationSession, storage, key, 1, "demo", {})
        final_root = storage.workspace_root(added[0].storage_key)
        if outcome == "before_commit":
            assert not final_root.exists() and published == []
        else:
            assert (final_root / "artifact.txt").read_text() == "keep"
        if outcome == "committed":
            assert published[0].storage_key == final_root.name

    asyncio.run(check())
