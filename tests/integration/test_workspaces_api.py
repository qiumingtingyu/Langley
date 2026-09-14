"""One real-MySQL/API slice for binding, import publication and detached scope."""

import asyncio
import json
import os
import time
from argparse import Namespace
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSessionTransaction

from langley.answering.contracts import (
    LLMFinishReason,
    LLMResponseCompleted,
    ToolCall,
    ToolResult,
)
from langley.answering.conversation_context_builder import ConversationContextBuilder
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.api.workspaces import _publish
from langley.bootstrap import bootstrap_local_user
from langley.infrastructure.database import (
    create_database_engine,
    create_session_factory,
    dispose_database_engine,
)
from langley.infrastructure.models import Workspace
from langley.main import create_app
from langley.settings import Settings
from langley.workspace_storage import WorkspaceStorage


def _round(name=None, args=None):
    return ScriptedProviderRound(
        events=(
            LLMResponseCompleted(
                assistant_content="done" if name is None else "",
                tool_calls=()
                if name is None
                else (ToolCall(name, name, json.dumps(args)),),
                finish_reason=LLMFinishReason.STOP
                if name is None
                else LLMFinishReason.TOOL_CALLS,
                usage=None,
            ),
        )
    )


def _await_run(client, run_id):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        result = client.get(f"/api/runs/{run_id}").json()
        if result["run"]["status"] not in {"PENDING", "RUNNING"}:
            return result
        time.sleep(0.05)
    raise AssertionError("Run failed to terminate")


def test_workspace_api_binding_import_and_owner(
    test_database_url, reset_database, tmp_path, monkeypatch
):
    reset_database()
    config = Config("alembic.ini")
    config.cmd_opts = Namespace(x=["use_test_database=true"])
    command.upgrade(config, "head")
    settings = Settings(
        database_url=test_database_url,
        local_user_id=1,
        workspace_storage_root=tmp_path / "workspaces",
        workspace_max_import_file_bytes=100,
    )
    asyncio.run(bootstrap_local_user(settings))
    provider = FakeProvider(
        [
            _round(
                "write_file",
                {"path": "calc.py", "content": "def add(a, b): return a-b\n"},
            ),
            _round(
                "write_file",
                {
                    "path": "test_calc.py",
                    "content": (
                        "from calc import add\ndef test_add(): assert add(2, 3) == 5\n"
                    ),
                },
            ),
            _round("run_command", {"command": "python -m pytest -q"}),
            _round(
                "edit_file",
                {"path": "calc.py", "edits": [{"old_text": "a-b", "new_text": "a+b"}]},
            ),
            _round("run_command", {"command": "python -m pytest -q"}),
            _round(),
            _round("write_file", {"path": "aftermath.txt", "content": "keep"}),
            ScriptedProviderRound(events=(), failure=RuntimeError("scripted failure")),
            _round("read_file", {"path": "aftermath.txt"}),
            _round(),
        ]
    )
    with TestClient(
        create_app(
            settings,
            provider=provider,
            memory_provider=FakeProvider([]),
            conversation_compactor_provider=FakeProvider([]),
        )
    ) as client:
        standalone = client.post("/api/conversations", json={}).json()
        assert standalone["workspace_id"] is None
        created = client.post("/api/workspaces", json={"name": "demo"})
        assert created.status_code == 201
        workspace = created.json()
        conversations = client.get("/api/conversations").json()
        first = next(
            c for c in conversations if c["id"] == workspace["conversation_id"]
        )
        assert first["workspace_id"] == workspace["id"]
        if os.getenv("RUN_SANDBOX_IT") == "true":
            answer = client.post(
                f"/api/conversations/{first['id']}/messages",
                json={"content": "repair", "client_request_id": "repair-1"},
            )
            assert answer.status_code == 202, answer.text
            done = _await_run(client, answer.json()["run"]["id"])
            assert done["run"]["status"] == "SUCCEEDED", done
            assert {"calc.py", "test_calc.py"} <= set(
                done["workspace_changes"]["added"]
            )
            commands = [
                json.loads(item.content)
                for item in provider.requests[5].transcript
                if isinstance(item, ToolResult) and item.name == "run_command"
            ]
            assert [item["exit_code"] for item in commands] == [1, 0]
            assert "1 passed" in commands[-1]["output"]
            failed = client.post(
                f"/api/conversations/{first['id']}/messages",
                json={"content": "fail after write", "client_request_id": "fail-1"},
            )
            failure = _await_run(client, failed.json()["run"]["id"])
            assert (
                failure["run"]["status"] == "FAILED"
                and failure["assistant_message"] is None
            )
            retried = client.post(
                f"/api/conversations/{first['id']}/retry",
                json={"client_request_id": "retry-1"},
            )
            retry = _await_run(client, retried.json()["run"]["id"])
            assert (
                retry["run"]["id"] != failure["run"]["id"]
                and retry["run"]["status"] == "SUCCEEDED"
            )
            assert not any(
                isinstance(item, ToolResult) for item in provider.requests[8].transcript
            )
            assert (
                json.loads(provider.requests[9].transcript[-1].content)["content"]
                == "keep"
            )
        another = client.post(
            "/api/conversations", json={"workspace_id": workspace["id"]}
        )
        assert another.status_code == 201
        assert (
            client.patch(
                f"/api/conversations/{standalone['id']}",
                json={"title": "bad", "workspace_id": workspace["id"]},
            ).status_code
            == 422
        )
        imported = client.post(
            "/api/workspaces/import",
            data={"name": "copy"},
            files=[
                ("files", ("nested/a.txt", b"hello", "text/plain")),
                ("files", ("image.bin", b"\xff\x00", "application/octet-stream")),
                ("files", (".env", b"secret", "text/plain")),
                ("files", ("node_modules/cache", b"cache", "text/plain")),
            ],
        )
        assert imported.status_code == 201, imported.text
        imported_id = imported.json()["id"]
        assert imported.json()["skipped"] == {
            "secret_filename": 1,
            "cache_or_repository": 1,
        }
        assert (
            client.get(
                f"/api/workspaces/{imported_id}/files",
                params={"path": "nested/a.txt", "preview": True},
            ).json()["content"]
            == "hello"
        )
        assert (
            client.get(
                f"/api/workspaces/{imported_id}/files",
                params={"path": "image.bin", "preview": True},
            ).status_code
            == 422
        )
        before = list(settings.workspace_storage_root.iterdir())
        invalid = client.post(
            "/api/workspaces/import",
            files=[("files", ("ok", b"ok")), ("files", ("../escape", b"bad"))],
        )
        assert invalid.status_code == 422
        oversized = client.post(
            "/api/workspaces/import", files=[("files", ("large", b"a" * 101))]
        )
        assert oversized.status_code == 422
        assert list(settings.workspace_storage_root.iterdir()) == before
        assert len(client.get("/api/workspaces").json()) == 2
    settings.local_user_id = 2
    asyncio.run(bootstrap_local_user(settings))
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/workspaces").json() == []
        assert client.get(f"/api/workspaces/{workspace['id']}/files").status_code == 404
        assert (
            client.post(
                "/api/conversations", json={"workspace_id": workspace["id"]}
            ).status_code
            == 404
        )

    async def detached_scope():
        engine = create_database_engine(test_database_url)
        try:
            factory = create_session_factory(engine)
            builder = ConversationContextBuilder(working_context_budget_estimate=16000)
            facts = await builder._read_facts(factory, workspace["conversation_id"])
            assert facts.user_id == 1 and facts.workspace_id == workspace["id"]
            assert facts.workspace_name == "demo" and facts.workspace_storage_key
            async with factory() as session:
                row = await session.scalar(
                    select(Workspace).where(Workspace.id == imported_id)
                )
                assert (
                    settings.workspace_storage_root / row.storage_key / "image.bin"
                ).read_bytes() == b"\xff\x00"
            storage = WorkspaceStorage(settings)
            staging_key = uuid4().hex
            storage.workspace_root(staging_key).mkdir()
            storage.write_file(staging_key, "receipt.txt", "committed")
            original_exit = AsyncSessionTransaction.__aexit__

            async def commit_then_disconnect(transaction, error_type, error, traceback):
                await original_exit(transaction, error_type, error, traceback)
                if error_type is None:
                    raise RuntimeError("lost commit acknowledgement")

            with monkeypatch.context() as injection:
                injection.setattr(
                    AsyncSessionTransaction, "__aexit__", commit_then_disconnect
                )
                with pytest.raises(RuntimeError, match="lost commit acknowledgement"):
                    await _publish(
                        factory, storage, staging_key, 1, "uncertain-publication", {}
                    )
            async with factory() as session:
                committed = await session.scalar(
                    select(Workspace).where(Workspace.name == "uncertain-publication")
                )
                assert committed is not None
                assert (
                    storage.read_file(committed.storage_key, "receipt.txt")["content"]
                    == "committed"
                )
        finally:
            await dispose_database_engine(engine)

    asyncio.run(detached_scope())
