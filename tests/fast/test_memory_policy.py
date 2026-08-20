"""Deterministic contract tests for the Slice 5 Memory Policy boundary."""

import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from langley.answering.contracts import (
    LLMFinishReason,
    LLMResponseCompleted,
    UserRuntimeMessage,
)
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.memory.policy import (
    MemoryPolicy,
    MemoryPolicyContextInfeasibleError,
    MemoryPolicyConversationMessage,
    MemoryPolicyInput,
    MemoryPolicyInvalidOutputError,
    MemoryPolicyItem,
    MemoryPolicyUnavailableError,
)

_EVIDENCE_CREATED_AT = datetime(2026, 8, 20, 2, 0)
_LOCAL_REFERENCE = datetime(2026, 8, 20, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def _memory(
    memory_id: int = 7,
    content: str = "I prefer concise Python examples.",
) -> MemoryPolicyItem:
    return MemoryPolicyItem(memory_id=memory_id, content=content)


def _input(
    *,
    evidence_content: str = "Please remember that I prefer concise Python examples.",
    previous_messages: tuple[MemoryPolicyConversationMessage, ...] = (),
    current_memories: tuple[MemoryPolicyItem, ...] = (),
    auto_memory_enabled: bool = True,
    local_temporal_reference: datetime = _LOCAL_REFERENCE,
) -> MemoryPolicyInput:
    return MemoryPolicyInput(
        evidence_message_id=11,
        evidence_content=evidence_content,
        evidence_created_at=_EVIDENCE_CREATED_AT,
        previous_messages=previous_messages,
        current_memories=current_memories,
        auto_memory_enabled=auto_memory_enabled,
        local_temporal_reference=local_temporal_reference,
    )


def _completion(content: str) -> LLMResponseCompleted:
    return LLMResponseCompleted(
        assistant_content=content,
        tool_calls=(),
        finish_reason=LLMFinishReason.STOP,
        usage=None,
    )


def _policy(
    provider: FakeProvider,
    *,
    memory_policy_estimated_token_budget: int | None = 10_000,
) -> MemoryPolicy:
    return MemoryPolicy(
        provider=provider,
        memory_policy_estimated_token_budget=memory_policy_estimated_token_budget,
    )


def _decide(policy: MemoryPolicy, policy_input: MemoryPolicyInput):
    return asyncio.run(policy.decide(policy_input))


@pytest.mark.parametrize(
    ("payload", "current_memories", "expected_operations", "expected_hint"),
    [
        (
            {
                "mutations": [
                    {
                        "operation": "NEW",
                        "content": "I prefer concise Python examples.",
                    }
                ],
                "user_requested_memory_action": True,
            },
            (),
            ("NEW",),
            True,
        ),
        (
            {
                "mutations": [
                    {
                        "operation": "CHANGE",
                        "target_memory_id": 7,
                        "content": "I prefer detailed Python examples.",
                    }
                ],
                "user_requested_memory_action": True,
            },
            (_memory(),),
            ("CHANGE",),
            True,
        ),
        (
            {
                "mutations": [{"operation": "FORGET", "target_memory_id": 7}],
                "user_requested_memory_action": True,
            },
            (_memory(),),
            ("FORGET",),
            True,
        ),
        (
            {"mutations": [], "user_requested_memory_action": False},
            (),
            (),
            False,
        ),
    ],
)
def test_policy_parses_complete_mutation_result(
    payload: dict[str, object],
    current_memories: tuple[MemoryPolicyItem, ...],
    expected_operations: tuple[str, ...],
    expected_hint: bool,
) -> None:
    provider = FakeProvider(
        [ScriptedProviderRound(events=(_completion(json.dumps(payload)),))]
    )

    result = _decide(_policy(provider), _input(current_memories=current_memories))

    assert (
        tuple(mutation.operation for mutation in result.mutations)
        == expected_operations
    )
    assert result.user_requested_memory_action is expected_hint
    assert result.is_no_change is (not expected_operations)
    assert len(provider.requests) == 1


def test_policy_instructs_auto_memory_semantics_and_serializes_the_mode() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        '{"mutations":[],"user_requested_memory_action":false}'
                    ),
                )
            )
        ]
    )

    _decide(_policy(provider), _input(auto_memory_enabled=False))

    request = provider.requests[0]
    assert request.allowed_tools == ()
    assert (
        "When it is true (ON), ordinary implicit eligible Personal Context may produce"
        in (request.system_input)
    )
    assert (
        "When it is false (OFF), ordinary implicit information must produce NO_CHANGE"
        in (request.system_input)
    )
    assert (
        "Explicit conversational remember, correct, or forget requests may still"
        in (request.system_input)
    )
    assert "An explicit request never bypasses Personal Context eligibility" in (
        request.system_input
    )
    assert "Resolve temporal language relative to local_temporal_reference" in (
        request.system_input
    )
    assert request.transcript == (
        UserRuntimeMessage(
            content=json.dumps(
                _input(auto_memory_enabled=False).model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        ),
    )


def test_policy_input_rejects_a_local_reference_for_a_different_instant() -> None:
    with pytest.raises(ValidationError, match="match evidence_created_at"):
        _input(
            local_temporal_reference=datetime(
                2026, 8, 20, 11, 0, tzinfo=ZoneInfo("Asia/Shanghai")
            )
        )


def test_policy_allows_an_explicit_mutation_while_auto_memory_is_off() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        json.dumps(
                            {
                                "mutations": [
                                    {
                                        "operation": "NEW",
                                        "content": "I prefer concise Python examples.",
                                    }
                                ],
                                "user_requested_memory_action": True,
                            }
                        )
                    ),
                )
            )
        ]
    )

    result = _decide(
        _policy(provider),
        _input(
            evidence_content="记住，我喜欢简洁的 Python 示例。",
            auto_memory_enabled=False,
        ),
    )

    assert tuple(mutation.operation for mutation in result.mutations) == ("NEW",)
    assert result.user_requested_memory_action is True


def test_policy_allows_off_implicit_no_change() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        '{"mutations":[],"user_requested_memory_action":false}'
                    ),
                )
            )
        ]
    )

    result = _decide(
        _policy(provider),
        _input(
            evidence_content="I prefer working in the morning.",
            auto_memory_enabled=False,
        ),
    )

    assert result.is_no_change is True
    assert result.user_requested_memory_action is False


def test_policy_requires_calibrated_budget_before_invocation() -> None:
    provider = FakeProvider([])

    with pytest.raises(MemoryPolicyUnavailableError, match="budget is not calibrated"):
        _decide(
            _policy(
                provider,
                memory_policy_estimated_token_budget=None,
            ),
            _input(),
        )

    assert provider.requests == []


def test_policy_rejects_load_all_context_that_exceeds_the_configured_budget() -> None:
    memory = _memory(content="abcdefgh")
    provider = FakeProvider([])

    with pytest.raises(
        MemoryPolicyContextInfeasibleError, match="all current memories"
    ):
        _decide(
            _policy(provider, memory_policy_estimated_token_budget=len(memory.content)),
            _input(current_memories=(memory,)),
        )

    assert provider.requests == []


def test_policy_retries_malformed_result_then_returns_the_next_whole_result() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(events=(_completion("not JSON"),)),
            ScriptedProviderRound(
                events=(
                    _completion(
                        '{"mutations":[],"user_requested_memory_action":false}'
                    ),
                )
            ),
        ]
    )

    result = _decide(_policy(provider), _input())

    assert result.is_no_change is True
    assert len(provider.requests) == 2


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {
            "mutations": [
                {
                    "operation": "CHANGE",
                    "target_memory_id": 999,
                    "content": "Never supplied.",
                }
            ],
            "user_requested_memory_action": True,
        },
        {
            "mutations": [
                {
                    "operation": "CHANGE",
                    "target_memory_id": 7,
                    "content": "First replacement.",
                },
                {"operation": "FORGET", "target_memory_id": 7},
            ],
            "user_requested_memory_action": True,
        },
        {
            "mutations": [
                {
                    "operation": "NEW",
                    "target_memory_id": 7,
                    "content": "An invalid NEW target.",
                }
            ],
            "user_requested_memory_action": True,
        },
        {
            "mutations": [
                {
                    "operation": "NEW",
                    "content": "I am mainly preparing Python for 14 days.",
                    "valid_until": "2026-09-03T10:00:00",
                }
            ],
            "user_requested_memory_action": False,
        },
    ],
)
def test_policy_rejects_an_illegal_whole_result_before_retrying(
    invalid_payload: dict[str, object],
) -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(events=(_completion(json.dumps(invalid_payload)),)),
            ScriptedProviderRound(
                events=(
                    _completion(
                        '{"mutations":[],"user_requested_memory_action":false}'
                    ),
                )
            ),
        ]
    )

    result = _decide(_policy(provider), _input(current_memories=(_memory(),)))

    assert result.is_no_change is True
    assert len(provider.requests) == 2


def test_policy_rejects_exact_duplicate_new_decisions_before_retrying() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        json.dumps(
                            {
                                "mutations": [
                                    {
                                        "operation": "NEW",
                                        "content": "I prefer concise Python examples.",
                                    },
                                    {
                                        "operation": "NEW",
                                        "content": "I prefer concise Python examples.",
                                    },
                                ],
                                "user_requested_memory_action": True,
                            }
                        )
                    ),
                )
            ),
            ScriptedProviderRound(
                events=(
                    _completion(
                        '{"mutations":[],"user_requested_memory_action":false}'
                    ),
                )
            ),
        ]
    )

    result = _decide(_policy(provider), _input())

    assert result.is_no_change is True
    assert len(provider.requests) == 2


def test_policy_rejects_new_content_equal_to_supplied_memory_before_retrying() -> None:
    existing_memory = _memory()
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        json.dumps(
                            {
                                "mutations": [
                                    {
                                        "operation": "NEW",
                                        "content": existing_memory.content,
                                    }
                                ],
                                "user_requested_memory_action": True,
                            }
                        )
                    ),
                )
            ),
            ScriptedProviderRound(
                events=(
                    _completion(
                        '{"mutations":[],"user_requested_memory_action":false}'
                    ),
                )
            ),
        ]
    )

    result = _decide(_policy(provider), _input(current_memories=(existing_memory,)))

    assert result.is_no_change is True
    assert len(provider.requests) == 2


def test_policy_exposes_bounded_invalid_output_after_two_completed_attempts() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(events=(_completion("invalid first"),)),
            ScriptedProviderRound(events=(_completion("invalid second"),)),
        ]
    )

    with pytest.raises(MemoryPolicyInvalidOutputError, match="invalid JSON"):
        _decide(_policy(provider), _input())

    assert len(provider.requests) == 2


def test_policy_does_not_retry_provider_failure() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(),
                failure=WorkflowFailure(RunErrorCode.LLM_PROVIDER_FAILED),
            )
        ]
    )

    with pytest.raises(WorkflowFailure) as raised:
        _decide(_policy(provider), _input())

    assert raised.value.error_code is RunErrorCode.LLM_PROVIDER_FAILED
    assert len(provider.requests) == 1


def test_policy_normalizes_an_absolute_temporal_result_to_utc_naive() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        json.dumps(
                            {
                                "mutations": [
                                    {
                                        "operation": "NEW",
                                        "content": (
                                            "I am mainly preparing Python for 14 days."
                                        ),
                                        "valid_until": "2026-09-03T10:00:00+08:00",
                                    }
                                ],
                                "user_requested_memory_action": False,
                            }
                        )
                    ),
                )
            )
        ]
    )

    result = _decide(
        _policy(provider),
        _input(evidence_content="从现在起 14 天内，我主要准备 Python。"),
    )

    assert result.mutations[0].valid_until == datetime(2026, 9, 3, 2, 0)


def test_no_change_preserves_explicit_request_hint() -> None:
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion('{"mutations":[],"user_requested_memory_action":true}'),
                )
            )
        ]
    )

    result = _decide(_policy(provider), _input(evidence_content="记住，2+2=4。"))

    assert result.is_no_change is True
    assert result.user_requested_memory_action is True


def test_assistant_context_can_explain_user_confirmation_but_not_become_evidence() -> (
    None
):
    previous_messages = (
        MemoryPolicyConversationMessage(
            role="ASSISTANT", content="You said you want concise examples."
        ),
    )
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        json.dumps(
                            {
                                "mutations": [
                                    {
                                        "operation": "NEW",
                                        "content": "I prefer concise Python examples.",
                                    }
                                ],
                                "user_requested_memory_action": False,
                            }
                        )
                    ),
                )
            )
        ]
    )

    result = _decide(
        _policy(provider),
        _input(
            evidence_content="是的，今后请给我简洁的 Python 示例。",
            previous_messages=previous_messages,
        ),
    )

    assert tuple(mutation.operation for mutation in result.mutations) == ("NEW",)


def test_assistant_inference_without_user_confirmation_can_be_no_change() -> None:
    previous_messages = (
        MemoryPolicyConversationMessage(
            role="ASSISTANT", content="I infer that you prefer detailed examples."
        ),
    )
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        '{"mutations":[],"user_requested_memory_action":false}'
                    ),
                )
            )
        ]
    )

    result = _decide(
        _policy(provider),
        _input(
            evidence_content="What should I study next?",
            previous_messages=previous_messages,
        ),
    )

    assert result.is_no_change is True
    assert result.user_requested_memory_action is False
