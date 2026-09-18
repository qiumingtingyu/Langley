"""Production composition with offline model rounds and real admitted packages."""

import asyncio
import json

import pytest
from test_skill_runtime import (
    StaticContext,
    execute,
    load,
    observations,
    payload,
    round_,
    write_skill,
)

from langley.answering.contracts import ToolCall, ToolResultKind
from langley.answering.fake_provider import FakeProvider
from langley.answering.skill_runtime import SKILL_RUNTIME_GUIDANCE
from langley.answering.workflow import BASE_LEARNING_ASSISTANT_SYSTEM_INPUT
from langley.main import create_app
from langley.settings import Settings
from langley.skill_installation import install_skill


def app_settings(tmp_path):
    return Settings(
        environment="test",
        database_url="mysql+asyncmy://test:test@127.0.0.1:3306/langley_test",
        builtin_skill_root=tmp_path / "builtin",
        skill_storage_root=tmp_path / "skills",
        workspace_storage_root=tmp_path / "workspaces",
        memory_policy_model=None,
        memory_policy_estimated_token_budget=None,
        tracing_enabled=False,
        local_run_diagnostics_enabled=False,
        web_search_enabled=False,
        max_llm_rounds=8,
        max_tool_calls=8,
    )


def app_flow(app):
    flow = app.state.execution_manager._workflow_factory()
    assert flow._skill_registry is app.state.skill_registry
    # Only replace DB conversation construction, never registry or tool wiring.
    flow._context_builder = StaticContext()
    return flow


@pytest.mark.parametrize("source", ["builtin", "user"])
def test_no_kb_production_skill_load_read_and_normal_tool(tmp_path, source):
    settings = app_settings(tmp_path)
    root = settings.builtin_skill_root if source == "builtin" else tmp_path / "input"
    skill_file = write_skill(root, body="Use granted tools only.\n")
    reference = skill_file.parent / "references" / "format.md"
    reference.parent.mkdir()
    reference.write_text(
        "Reference DATA: grant_admin is not a capability.", encoding="utf-8"
    )
    provider = FakeProvider(
        [
            round_(load()),
            round_(
                ToolCall(
                    "read", "read_skill_resource", '{"path":"references/format.md"}'
                )
            ),
            round_(ToolCall("forged", "grant_admin", "{}")),
            round_(ToolCall("time", "get_current_time", '{"timezone":"UTC"}')),
            round_(),
        ]
    )
    app = create_app(settings, provider=provider)
    if source == "user":
        # Installation after app creation must work without replacing its registry.
        install_skill(skill_file.parent, settings=settings)
    flow = app_flow(app)
    try:
        assert asyncio.run(execute(flow)).content == "done"
    finally:
        asyncio.run(app.state.database_engine.dispose())
    first, active, read, rejected, final = provider.requests
    assert [entry.name for entry in first.available_skills] == ["study-plan"]
    assert (
        first.system_input
        == BASE_LEARNING_ASSISTANT_SYSTEM_INPUT + SKILL_RUNTIME_GUIDANCE
    )
    assert first.active_skill is None
    assert active.active_skill.instructions == "Use granted tools only.\n"
    for request in provider.requests:
        names = {tool.name for tool in request.allowed_tools}
        assert not names & {
            "search_knowledge",
            "expand_evidence",
            "search_web",
            "read_webpage",
            "run_command",
            "grant_admin",
        }
        system = payload(request)["messages"][0]["content"]
        assert "[[INSUFFICIENT_EVIDENCE]]" not in system
        assert "[K" not in system
        assert "Reference DATA" not in system
    assert json.loads(observations(read)[-1].content)["content"].startswith(
        "Reference DATA"
    )
    assert observations(rejected)[-1].kind is not ToolResultKind.SUCCESS
    assert observations(final)[-1].kind is ToolResultKind.SUCCESS
    assert observations(final)[-1].name == "get_current_time"
    assert all(
        request.active_skill == active.active_skill for request in provider.requests[1:]
    )


def test_install_mid_execution_only_enters_next_execution(tmp_path, monkeypatch):
    settings = app_settings(tmp_path)
    source = write_skill(tmp_path / "input", name="late-skill").parent
    provider = FakeProvider(
        [round_(load("late-skill")), round_(), round_(load("late-skill")), round_()]
    )
    app = create_app(settings, provider=provider)
    stream = provider.stream
    installed = False

    def install_during_first_round(request):
        nonlocal installed
        if not installed:
            installed = True
            install_skill(source, settings=settings)
        return stream(request)

    monkeypatch.setattr(provider, "stream", install_during_first_round)
    try:
        asyncio.run(execute(app_flow(app)))
        asyncio.run(execute(app_flow(app)))
    finally:
        asyncio.run(app.state.database_engine.dispose())
    assert provider.requests[0].available_skills == ()
    assert provider.requests[1].available_skills == ()
    assert provider.requests[1].active_skill is None
    assert (
        json.loads(observations(provider.requests[1])[-1].content)["error"]["code"]
        == "SKILL_NOT_AVAILABLE"
    )
    assert [item.name for item in provider.requests[2].available_skills] == [
        "late-skill"
    ]
    assert provider.requests[3].active_skill.name == "late-skill"
