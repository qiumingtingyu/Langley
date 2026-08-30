"""Focused deterministic contracts for single-KnowledgeBase QA completion."""

import asyncio
import re

import pytest

import langley.answering.knowledge_qa as knowledge_qa
from langley.answering.contracts import LLMFinishReason, LLMResponseCompleted, ToolCall
from langley.answering.errors import (
    InvalidResponseSubtype,
    RunErrorCode,
    WorkflowFailure,
)
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.knowledge.retrieval import RetrievalHit, RetrievalResult


def _hit(identifier: int) -> RetrievalHit:
    return RetrievalHit(
        knowledge_chunk_id=identifier,
        rank=identifier,
        retrieval_rank=identifier,
        score=1.0 / identifier,
        rerank_score=None,
        chunk_ordinal=identifier,
        content=f"evidence {identifier}",
        heading_path=("Heading",),
        source_regions=({"kind": "text", "start_byte": 0, "end_byte": 1},),
        document_id=10,
        document_version_id=20,
        source_display_name="notes.md",
        source_sha256="a" * 64,
    )


def _completion(
    content: str, *, tool_calls: tuple[ToolCall, ...] = ()
) -> LLMResponseCompleted:
    return LLMResponseCompleted(
        assistant_content=content,
        tool_calls=tool_calls,
        finish_reason=LLMFinishReason.STOP,
        usage=None,
    )


def test_validated_completion_preserves_handles_and_deduplicates() -> None:
    completion = knowledge_qa._validated_completion(
        "Use [K2], then [K1], and [K2] again.",
        (_hit(1), _hit(2)),
        abstention_control_token=None,
    )

    assert completion.abstained is False
    actual = [
        (
            item.evidence_handle,
            item.document_version_id,
            item.evidence_text,
            item.source_display_name_snapshot,
            item.heading_path_snapshot,
            item.source_regions_snapshot,
        )
        for item in completion.citations
    ]
    assert actual == [
        (
            2,
            20,
            "evidence 2",
            "notes.md",
            ("Heading",),
            ({"kind": "text", "start_byte": 0, "end_byte": 1},),
        ),
        (
            1,
            20,
            "evidence 1",
            "notes.md",
            ("Heading",),
            ({"kind": "text", "start_byte": 0, "end_byte": 1},),
        ),
    ]


@pytest.mark.parametrize(
    ("content", "subtype"),
    [
        ("Unknown [K3]", InvalidResponseSubtype.UNKNOWN_CITATION_HANDLE),
        ("An uncited answer", InvalidResponseSubtype.MISSING_REQUIRED_CITATION),
    ],
)
def test_normal_completion_rejects_unknown_or_missing_handles(
    content: str, subtype: InvalidResponseSubtype
) -> None:
    with pytest.raises(WorkflowFailure) as raised:
        knowledge_qa._validated_completion(
            content,
            (_hit(1), _hit(2)),
            abstention_control_token=None,
        )

    assert raised.value.error_code is RunErrorCode.LLM_RESPONSE_INVALID
    assert raised.value.invalid_response_subtype is subtype


def test_auto_exact_legacy_sentinel_abstains_without_strip_normalization() -> None:
    completion = knowledge_qa._validated_completion(
        knowledge_qa.AUTO_INSUFFICIENT_EVIDENCE_SENTINEL,
        (_hit(1),),
        abstention_control_token=None,
    )

    assert completion.content == knowledge_qa.INSUFFICIENT_EVIDENCE_ANSWER
    assert completion.abstained is True
    assert completion.citations == ()

    with pytest.raises(WorkflowFailure) as raised:
        knowledge_qa._validated_completion(
            f" {knowledge_qa.AUTO_INSUFFICIENT_EVIDENCE_SENTINEL} ",
            (_hit(1),),
            abstention_control_token=None,
        )
    assert (
        raised.value.invalid_response_subtype
        is InvalidResponseSubtype.MISSING_REQUIRED_CITATION
    )


def test_auto_embedded_legacy_sentinel_is_ordinary_content() -> None:
    content = "该字符串 [[INSUFFICIENT_EVIDENCE]] 可以被讨论 [K1]。"

    completion = knowledge_qa._validated_completion(
        content,
        (_hit(1),),
        abstention_control_token=None,
    )

    assert completion.content == content
    assert completion.abstained is False
    assert [citation.evidence_handle for citation in completion.citations] == [1]


_CURRENT_TOKEN = "[[LANGLEY_ABSTAIN_0123456789abcdef0123456789abcdef]]"


@pytest.mark.parametrize("content", [_CURRENT_TOKEN, f"  {_CURRENT_TOKEN}\n"])
def test_exact_current_run_token_abstains_without_citations(content: str) -> None:
    completion = knowledge_qa._validated_completion(
        content,
        (_hit(1),),
        abstention_control_token=_CURRENT_TOKEN,
    )

    assert completion.content == knowledge_qa.INSUFFICIENT_EVIDENCE_ANSWER
    assert completion.abstained is True
    assert completion.citations == ()


@pytest.mark.parametrize(
    "content",
    [
        f"解释文字 [K1] {_CURRENT_TOKEN}",
        f"{_CURRENT_TOKEN} 补充文字",
    ],
)
def test_current_run_token_mixed_with_text_abstains_without_citations(
    content: str,
) -> None:
    completion = knowledge_qa._validated_completion(
        content,
        (_hit(1),),
        abstention_control_token=_CURRENT_TOKEN,
    )

    assert completion.content == knowledge_qa.INSUFFICIENT_EVIDENCE_ANSWER
    assert completion.abstained is True
    assert completion.citations == ()


@pytest.mark.parametrize(
    "content",
    [
        "`[[INSUFFICIENT_EVIDENCE]]` 是一个表示证据不足的字符串 [K1]。",
        "另一个标记 [[LANGLEY_ABSTAIN_deadbeef]] 不是本轮控制信号 [K1]。",
    ],
)
def test_legacy_or_other_run_token_like_text_is_ordinary_content(content: str) -> None:
    completion = knowledge_qa._validated_completion(
        content,
        (_hit(1),),
        abstention_control_token=_CURRENT_TOKEN,
    )

    assert completion.content == content
    assert completion.abstained is False
    assert [citation.evidence_handle for citation in completion.citations] == [1]


def test_run_local_control_tokens_are_compact_unique_and_prompted_as_control() -> None:
    first = knowledge_qa.new_abstention_control_token()
    second = knowledge_qa.new_abstention_control_token()

    assert first != second
    assert re.fullmatch(r"\[\[LANGLEY_ABSTAIN_[0-9a-f]{32}\]\]", first)
    prompt = knowledge_qa.required_grounding_system_input(first)
    assert first in prompt
    assert "any material part cannot be answered" in prompt
    assert "nothing else" in prompt
    assert "Do not provide a partial answer before or after the token" in prompt
    assert "Do not cite the token" in prompt


def test_flow_rejects_provider_tool_completion() -> None:
    class FakeRetrievalService:
        async def search(self, **kwargs: object) -> RetrievalResult:
            assert kwargs == {
                "user_id": 1,
                "knowledge_base_id": 1,
                "query": "question",
                "top_k": 5,
            }
            return RetrievalResult(knowledge_base_id=1, hits=(_hit(1),))

    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    _completion(
                        "[K1]",
                        tool_calls=(ToolCall("call", "unexpected", "{}"),),
                    ),
                )
            )
        ]
    )
    flow = knowledge_qa.KnowledgeQAFlow(FakeRetrievalService(), provider)  # type: ignore[arg-type]

    async def execute() -> None:
        with pytest.raises(WorkflowFailure) as raised:
            await flow.execute(
                user_id=1,
                knowledge_base_id=1,
                question="question",
                on_assistant_delta=lambda _: asyncio.sleep(0),
            )
        assert raised.value.error_code is RunErrorCode.LLM_RESPONSE_INVALID

    asyncio.run(execute())


def test_flow_keeps_run_local_token_out_of_delta_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRetrievalService:
        async def search(self, **kwargs: object) -> RetrievalResult:
            return RetrievalResult(knowledge_base_id=1, hits=(_hit(1),))

    monkeypatch.setattr(
        knowledge_qa, "new_abstention_control_token", lambda: _CURRENT_TOKEN
    )
    provider = FakeProvider(
        [
            ScriptedProviderRound(
                events=(
                    knowledge_qa.AssistantContentDelta(_CURRENT_TOKEN),
                    _completion(_CURRENT_TOKEN),
                )
            )
        ]
    )
    flow = knowledge_qa.KnowledgeQAFlow(FakeRetrievalService(), provider)  # type: ignore[arg-type]
    deltas: list[str] = []

    async def execute() -> None:
        completion = await flow.execute(
            user_id=1,
            knowledge_base_id=1,
            question="question",
            on_assistant_delta=lambda content: asyncio.sleep(
                0, result=deltas.append(content)
            ),
        )
        assert completion.content == knowledge_qa.INSUFFICIENT_EVIDENCE_ANSWER

    asyncio.run(execute())
    assert deltas == [knowledge_qa.INSUFFICIENT_EVIDENCE_ANSWER]
    assert _CURRENT_TOKEN in provider.requests[0].system_input
