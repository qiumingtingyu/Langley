"""Triggered/reused/incremental Conversation Context assembly contracts."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from langley.answering.contracts import LLMFinishReason, LLMResponseCompleted, LLMUsage
from langley.answering.conversation_context import (
    ConversationCompactState,
    LLMConversationCompactor,
)
from langley.answering.conversation_context_builder import (
    ConversationContextBuilder,
    _AuthoritativeContextFacts,
)
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.answering.tracing import context_compaction_trace_context
from langley.infrastructure.models import (
    ConversationContextSnapshot,
    Memory,
    Message,
    Run,
)

NOW = datetime(2026, 8, 28, tzinfo=UTC)


def _user(
    message_id: int,
    sequence_no: int,
    content: str,
    *,
    regenerated_from_message_id: int | None = None,
) -> Message:
    return Message(
        id=message_id,
        conversation_id=1,
        sequence_no=sequence_no,
        role="USER",
        content=content,
        run_id=None,
        regenerated_from_message_id=regenerated_from_message_id,
        created_at=NOW,
    )


def _assistant(message_id: int, sequence_no: int, run_id: int, content: str) -> Message:
    return Message(
        id=message_id,
        conversation_id=1,
        sequence_no=sequence_no,
        role="ASSISTANT",
        content=content,
        run_id=run_id,
        regenerated_from_message_id=None,
        created_at=NOW,
    )


def _run(run_id: int, input_message_id: int, status: str = "SUCCEEDED") -> Run:
    started_at = NOW if status in {"RUNNING", "SUCCEEDED"} else None
    finished_at = NOW if status in {"SUCCEEDED", "FAILED", "CANCELLED"} else None
    return Run(
        id=run_id,
        conversation_id=1,
        input_message_id=input_message_id,
        client_request_id=f"request-{run_id}",
        attempt_no=1,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        error_code="ANSWER_EXECUTION_FAILED" if status == "FAILED" else None,
        created_at=NOW,
        updated_at=NOW,
    )


def _memory(
    memory_id: int,
    content: str,
    *,
    source_message_id: int | None = None,
) -> Memory:
    return Memory(
        id=memory_id,
        user_id=1,
        content=content,
        source_message_id=source_message_id,
        valid_until=None,
        created_at=NOW,
        updated_at=NOW,
    )


def _state_payload(
    decisions: list[tuple[str, list[int]]],
) -> dict[str, object]:
    return {
        "current_goals": [],
        "active_decisions": [
            {"content": content, "source_message_ids": source_ids}
            for content, source_ids in decisions
        ],
        "active_constraints": [],
        "open_loops": [],
        "important_facts": [],
        "artifacts": [],
    }


def _completion(
    payload: dict[str, object],
    *,
    usage: LLMUsage | None = None,
    provider_model: str | None = None,
) -> LLMResponseCompleted:
    return LLMResponseCompleted(
        assistant_content=json.dumps(payload),
        tool_calls=(),
        finish_reason=LLMFinishReason.STOP,
        usage=usage,
        provider_model=provider_model,
    )


class _RecordingBuilder(ConversationContextBuilder):
    def __init__(self, facts: _AuthoritativeContextFacts, **kwargs) -> None:
        super().__init__(**kwargs)
        self.facts = facts
        self.persisted: list[tuple[int, object]] = []
        self.compaction_events: list[dict[str, object]] = []

    async def _read_facts(self, session_factory, conversation_id):
        del session_factory, conversation_id
        return self.facts

    async def _persist_snapshot(
        self,
        session_factory,
        *,
        conversation_id,
        through_message_id,
        compaction,
        observed_snapshot_through_message_id,
        observed_snapshot_valid,
    ) -> None:
        del (
            session_factory,
            conversation_id,
            observed_snapshot_through_message_id,
            observed_snapshot_valid,
        )
        self.persisted.append((through_message_id, compaction.state))

    def _log_compaction(self, **event) -> None:
        self.compaction_events.append(event)
        super()._log_compaction(**event)


class _RecordingContextCompactionTrace:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def context_compact(self, **event) -> None:
        self.events.append(event)


def _builder(
    facts: _AuthoritativeContextFacts,
    provider: FakeProvider,
    *,
    trigger: int = 25,
    recent_target: int = 12,
    memory_budget: int = 8_192,
) -> _RecordingBuilder:
    compactor = LLMConversationCompactor(
        provider=provider,
        model="qwen-compactor",
        compact_state_target_estimate=10,
    )
    return _RecordingBuilder(
        facts,
        working_context_budget_estimate=60,
        conversation_compaction_trigger_estimate=trigger,
        recent_raw_target_estimate=recent_target,
        compact_state_target_estimate=10,
        memory_estimated_token_budget=memory_budget,
        compactor=compactor,
    )


def _facts(
    messages: tuple[Message, ...],
    runs: tuple[Run, ...],
    snapshot: ConversationContextSnapshot | None = None,
    memories: tuple[Memory, ...] = (),
) -> _AuthoritativeContextFacts:
    return _AuthoritativeContextFacts(messages, runs, memories, snapshot)


def _build(builder: ConversationContextBuilder, current_user_message_id: int):
    return asyncio.run(
        builder.build(
            object(),
            conversation_id=1,
            current_user_message_id=current_user_message_id,
        )
    )


def test_short_history_does_not_invoke_compactor() -> None:
    provider = FakeProvider([])
    facts = _facts(
        (_user(1, 1, "hello"), _assistant(2, 2, 10, "hi"), _user(3, 3, "now")),
        (_run(10, 1), _run(11, 3, "RUNNING")),
    )
    builder = _builder(facts, provider, trigger=50)
    trace = _RecordingContextCompactionTrace()

    with context_compaction_trace_context(trace):
        context = _build(builder, 3)

    assert provider.requests == []
    assert trace.events == []
    assert context.conversation_compact_context is None
    assert [turn.user_content for turn in context.completed_turns] == ["hello"]


def test_orion_canary_compact_while_recent_is_raw() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        _state_payload(
                            [
                                ("Codename: ORION.", [1]),
                                ("Deployment: CANARY.", [3]),
                            ]
                        )
                    ),
                )
            )
        ]
    )
    messages = (
        _user(
            1,
            1,
            "The current project codename is ORION; reject APOLLO. "
            + "Background context. " * 40,
        ),
        _assistant(2, 2, 10, "Recorded ORION."),
        _user(
            3,
            3,
            "Use CANARY; BLUE-GREEN is rejected. " + "Deployment context. " * 40,
        ),
        _assistant(4, 4, 11, "Recorded CANARY."),
        _user(5, 5, "I meant B, not A."),
        _assistant(6, 6, 12, "Understood: B."),
        _user(7, 7, "What is current?"),
    )
    builder = _builder(
        _facts(
            messages,
            (_run(10, 1), _run(11, 3), _run(12, 5), _run(13, 7, "RUNNING")),
        ),
        provider,
        trigger=25,
        recent_target=12,
    )
    trace = _RecordingContextCompactionTrace()

    with context_compaction_trace_context(trace):
        context = _build(builder, 7)

    assert "ORION" in (context.conversation_compact_context or "")
    assert "CANARY" in (context.conversation_compact_context or "")
    assert [turn.user_content for turn in context.completed_turns] == [
        "I meant B, not A."
    ]
    assert builder.persisted[0][0] == 4
    assert len(trace.events) == 1
    assert trace.events[0] == {
        "estimated_before": builder.compaction_events[-1]["estimated_before"],
        "estimated_after": builder.compaction_events[-1]["estimated_after"],
        "newly_compacted_turn_count": 2,
        "recent_raw_turn_count": 1,
        "compactor_model": "qwen-compactor",
        "provider_model": None,
        "provider_input_tokens": None,
        "provider_output_tokens": None,
        "duration_ms": builder.compaction_events[-1]["duration_ms"],
        "success": True,
        "outcome": "COMPACTED",
    }
    request_payload = json.loads(provider.requests[0].transcript[0].content)
    assert [
        message["message_id"] for message in request_payload["newly_aged_out_messages"]
    ] == [1, 2, 3, 4]


def test_existing_snapshot_is_incrementally_updated_with_only_newly_aged_turns() -> (
    None
):
    old_state = _state_payload([("Project codename is ORION.", [1])])
    snapshot = ConversationContextSnapshot(
        conversation_id=1,
        through_message_id=2,
        structured_state=old_state,
        compactor_model="older-model",
        prompt_version="older-prompt",
        created_at=NOW,
        updated_at=NOW,
    )
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        _state_payload(
                            [
                                ("Project codename is ORION.", [1]),
                                ("Production deployment strategy is CANARY.", [3]),
                            ]
                        )
                    ),
                )
            )
        ]
    )
    messages = (
        _user(1, 1, "ORION"),
        _assistant(2, 2, 10, "recorded"),
        _user(3, 3, "CANARY is final; BLUE-GREEN rejected."),
        _assistant(4, 4, 11, "recorded"),
        _user(5, 5, "recent wording stays raw"),
        _assistant(6, 6, 12, "yes"),
        _user(7, 7, "current"),
    )
    builder = _builder(
        _facts(
            messages,
            (_run(10, 1), _run(11, 3), _run(12, 5), _run(13, 7, "RUNNING")),
            snapshot,
        ),
        provider,
        trigger=20,
        recent_target=10,
    )

    context = _build(builder, 7)

    request_payload = json.loads(provider.requests[0].transcript[0].content)
    assert request_payload["previous_compact_state"] == old_state
    assert [
        message["message_id"] for message in request_payload["newly_aged_out_messages"]
    ] == [3, 4]
    assert builder.persisted[0][0] == 4
    assert context.completed_turns[0].user_content == "recent wording stays raw"


def test_invalid_output_keeps_snapshot_and_lossless_raw_fallback() -> None:
    old_state = _state_payload([("Project codename is ORION.", [1])])
    snapshot = ConversationContextSnapshot(
        conversation_id=1,
        through_message_id=2,
        structured_state=old_state,
        compactor_model="older-model",
        prompt_version="older-prompt",
        created_at=NOW,
        updated_at=NOW,
    )
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    LLMResponseCompleted(
                        assistant_content="not-json",
                        tool_calls=(),
                        finish_reason=LLMFinishReason.STOP,
                        usage=None,
                    ),
                )
            )
        ]
    )
    messages = (
        _user(1, 1, "ORION"),
        _assistant(2, 2, 10, "recorded"),
        _user(3, 3, "new decision"),
        _assistant(4, 4, 11, "recorded"),
        _user(5, 5, "I meant B, not A."),
        _assistant(6, 6, 12, "Understood exactly."),
        _user(7, 7, "current"),
    )
    builder = _builder(
        _facts(
            messages,
            (_run(10, 1), _run(11, 3), _run(12, 5), _run(13, 7, "RUNNING")),
            snapshot,
        ),
        provider,
        trigger=15,
        recent_target=8,
    )

    context = _build(builder, 7)

    assert builder.persisted == []
    assert "ORION" in (context.conversation_compact_context or "")
    assert [turn.user_content for turn in context.completed_turns] == [
        "new decision",
        "I meant B, not A.",
    ]


def test_non_beneficial_output_is_rejected_before_snapshot_persistence() -> None:
    old_state = _state_payload([("Project codename is ORION.", [1])])
    snapshot = ConversationContextSnapshot(
        conversation_id=1,
        through_message_id=2,
        structured_state=old_state,
        compactor_model="older-model",
        prompt_version="older-prompt",
        created_at=NOW,
        updated_at=NOW,
    )
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        _state_payload([("X" * 500, [3])]),
                        usage=LLMUsage(input_tokens=321, output_tokens=45),
                        provider_model="provider-compactor",
                    ),
                )
            )
        ]
    )
    messages = (
        _user(1, 1, "ORION"),
        _assistant(2, 2, 10, "recorded"),
        _user(3, 3, "new decision"),
        _assistant(4, 4, 11, "recorded"),
        _user(5, 5, "recent wording stays raw"),
        _assistant(6, 6, 12, "yes"),
        _user(7, 7, "current"),
    )
    builder = _builder(
        _facts(
            messages,
            (_run(10, 1), _run(11, 3), _run(12, 5), _run(13, 7, "RUNNING")),
            snapshot,
        ),
        provider,
        trigger=15,
        recent_target=8,
    )
    trace = _RecordingContextCompactionTrace()

    with context_compaction_trace_context(trace):
        context = _build(builder, 7)

    assert builder.persisted == []
    assert "ORION" in (context.conversation_compact_context or "")
    assert [turn.user_content for turn in context.completed_turns] == [
        "new decision",
        "recent wording stays raw",
    ]
    assert builder.compaction_events[-1]["success"] is False
    assert builder.compaction_events[-1]["outcome"] == "NON_BENEFICIAL_OUTPUT"
    assert builder.compaction_events[-1]["provider_model"] == "provider-compactor"
    assert builder.compaction_events[-1]["provider_input_tokens"] == 321
    assert builder.compaction_events[-1]["provider_output_tokens"] == 45
    assert trace.events[0]["provider_model"] == "provider-compactor"
    assert trace.events[0]["provider_input_tokens"] == 321
    assert trace.events[0]["provider_output_tokens"] == 45


def test_beneficial_candidate_over_working_budget_is_rejected() -> None:
    old_state = _state_payload([("Project codename is ORION.", [1])])
    snapshot = ConversationContextSnapshot(
        conversation_id=1,
        through_message_id=2,
        structured_state=old_state,
        compactor_model="older-model",
        prompt_version="older-prompt",
        created_at=NOW,
        updated_at=NOW,
    )
    candidate_payload = _state_payload([("X" * 240, [3])])
    provider = FakeProvider(
        [ScriptedProviderRound(events=(_completion(candidate_payload),))]
    )
    messages = (
        _user(1, 1, "ORION"),
        _assistant(2, 2, 10, "recorded"),
        _user(3, 3, "Y" * 800),
        _assistant(4, 4, 11, "recorded"),
        _user(5, 5, "recent raw"),
        _assistant(6, 6, 12, "yes"),
        _user(7, 7, "current"),
    )
    builder = _builder(
        _facts(
            messages,
            (_run(10, 1), _run(11, 3), _run(12, 5), _run(13, 7, "RUNNING")),
            snapshot,
        ),
        provider,
        trigger=15,
        recent_target=8,
    )

    context = _build(builder, 7)

    estimated_before = builder.compaction_events[-1]["estimated_before"]
    candidate_after = builder._estimate_visible_conversation(
        ConversationCompactState.model_validate(candidate_payload),
        context.completed_turns,
    )
    assert isinstance(estimated_before, int)
    assert 60 < candidate_after < estimated_before
    assert builder.persisted == []
    assert "ORION" in (context.conversation_compact_context or "")
    assert [turn.user_content for turn in context.completed_turns] == ["recent raw"]
    assert builder.compaction_events[-1]["success"] is False
    assert builder.compaction_events[-1]["outcome"] == "NON_BENEFICIAL_OUTPUT"


def test_builder_includes_only_completed_turns_and_current_user() -> None:
    messages = (
        _user(1, 1, "completed user"),
        _assistant(2, 2, 10, "completed answer"),
        _user(3, 3, "failed"),
        _user(4, 4, "cancelled"),
        _user(5, 5, "active"),
        _user(6, 6, "orphan"),
        _user(7, 7, "current"),
    )
    facts = _facts(
        messages,
        (
            _run(10, 1),
            _run(11, 3, "FAILED"),
            _run(12, 4, "CANCELLED"),
            _run(13, 5, "RUNNING"),
            _run(14, 7, "PENDING"),
        ),
    )

    context = _build(_builder(facts, FakeProvider([]), trigger=50), 7)

    assert context.current_user_content == "current"
    assert [
        (turn.user_content, turn.assistant_content) for turn in context.completed_turns
    ] == [("completed user", "completed answer")]


def test_builder_preserves_retry_and_regenerate_linear_context() -> None:
    retry_facts = _facts(
        (
            _user(1, 1, "first"),
            _assistant(2, 2, 10, "first answer"),
            _user(3, 3, "retry"),
        ),
        (_run(10, 1), _run(11, 3, "FAILED"), _run(12, 3, "PENDING")),
    )
    regenerate_facts = _facts(
        (
            _user(1, 1, "first"),
            _assistant(2, 2, 10, "first answer"),
            _user(3, 3, "first", regenerated_from_message_id=1),
        ),
        (_run(10, 1), _run(11, 3, "PENDING")),
    )

    retry_context = _build(_builder(retry_facts, FakeProvider([]), trigger=50), 3)
    regenerate_context = _build(
        _builder(regenerate_facts, FakeProvider([]), trigger=50), 3
    )

    assert retry_context.current_user_content == "retry"
    assert regenerate_context.current_user_content == "first"
    assert [turn.user_content for turn in retry_context.completed_turns] == ["first"]
    assert [turn.assistant_content for turn in regenerate_context.completed_turns] == [
        "first answer"
    ]


def test_builder_loads_all_personal_context_or_marks_it_unavailable() -> None:
    memory = _memory(10, "preference")
    facts = _facts(
        (_user(1, 1, "current"),),
        (_run(1, 1, "PENDING"),),
        memories=(memory,),
    )

    exact_fit = _build(
        _builder(
            facts,
            FakeProvider([]),
            trigger=50,
            memory_budget=len(memory.content) + 32,
        ),
        1,
    )
    over_budget = _build(
        _builder(
            facts,
            FakeProvider([]),
            trigger=50,
            memory_budget=len(memory.content) + 31,
        ),
        1,
    )

    assert [
        (item.memory_id, item.content) for item in exact_fit.personal_context or ()
    ] == [(10, "preference")]
    assert over_budget.personal_context is None


def test_builder_suppresses_only_exposed_canonical_source_identity() -> None:
    facts = _facts(
        (
            _user(1, 1, "history user"),
            _assistant(2, 2, 10, "history answer"),
            _user(3, 3, "current"),
        ),
        (_run(10, 1), _run(11, 3, "PENDING")),
        memories=(
            _memory(10, "from exposed original", source_message_id=1),
            _memory(11, "same meaning is not identity", source_message_id=99),
        ),
    )

    context = _build(_builder(facts, FakeProvider([]), trigger=50), 3)

    assert [item.content for item in context.personal_context or ()] == [
        "same meaning is not identity"
    ]


def test_builder_maps_regenerated_current_user_to_canonical_source_identity() -> None:
    facts = _facts(
        (
            _user(1, 1, "original"),
            _user(2, 2, "original", regenerated_from_message_id=1),
        ),
        (_run(10, 2, "PENDING"),),
        memories=(_memory(10, "from original", source_message_id=1),),
    )

    context = _build(_builder(facts, FakeProvider([]), trigger=50), 2)

    assert context.personal_context == ()


def test_builder_releases_its_database_scope_before_returning_context() -> None:
    class ScalarResult:
        def __init__(self, values: tuple[object, ...]) -> None:
            self._values = values

        def all(self) -> tuple[object, ...]:
            return self._values

    class RecordingSessionFactory:
        def __init__(self) -> None:
            self.exited = False
            self._calls = 0

        def __call__(self) -> "RecordingSessionFactory":
            return self

        async def __aenter__(self) -> "RecordingSessionFactory":
            return self

        async def __aexit__(self, *args: object) -> None:
            self.exited = True

        @asynccontextmanager
        async def begin(self):
            yield

        async def scalars(self, statement: object) -> ScalarResult:
            del statement
            self._calls += 1
            if self._calls == 1:
                return ScalarResult((_user(1, 1, "current"),))
            if self._calls == 2:
                return ScalarResult((_run(10, 1, "PENDING"),))
            return ScalarResult(())

        async def get(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            return None

    factory = RecordingSessionFactory()
    builder = ConversationContextBuilder(
        working_context_budget_estimate=60,
        conversation_compaction_trigger_estimate=50,
        recent_raw_target_estimate=20,
        compact_state_target_estimate=10,
    )

    async def build_context():
        return await builder.build(
            factory,
            conversation_id=1,
            current_user_message_id=1,
        )

    context = asyncio.run(build_context())

    assert factory.exited is True
    assert context.current_user_content == "current"
