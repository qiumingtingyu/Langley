"""Focused deterministic contracts for single-KnowledgeBase QA completion."""

import asyncio

import pytest

import langley.answering.knowledge_qa as knowledge_qa
from langley.answering.contracts import LLMFinishReason, LLMResponseCompleted, ToolCall
from langley.answering.errors import RunErrorCode, WorkflowFailure
from langley.answering.fake_provider import FakeProvider, ScriptedProviderRound
from langley.knowledge.retrieval import RetrievalHit, RetrievalResult


def _hit(identifier: int) -> RetrievalHit:
    return RetrievalHit(
        knowledge_chunk_id=identifier,
        rank=identifier,
        score=1.0 / identifier,
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
        "Use [K2], then [K1], and [K2] again.", (_hit(1), _hit(2))
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


@pytest.mark.parametrize("content", ["Unknown [K3]", "An uncited answer"])
def test_normal_completion_rejects_unknown_or_missing_handles(content: str) -> None:
    with pytest.raises(WorkflowFailure) as raised:
        knowledge_qa._validated_completion(content, (_hit(1), _hit(2)))

    assert raised.value.error_code is RunErrorCode.LLM_RESPONSE_INVALID


def test_exact_insufficient_evidence_sentinel_abstains_without_citations() -> None:
    completion = knowledge_qa._validated_completion(
        "[[INSUFFICIENT_EVIDENCE]]", (_hit(1),)
    )

    assert completion.content == knowledge_qa.INSUFFICIENT_EVIDENCE_ANSWER
    assert completion.abstained is True
    assert completion.citations == ()


def test_sentinel_embedded_in_explanation_is_not_an_abstention() -> None:
    completion = knowledge_qa._validated_completion(
        "解释文字 [K1] [[INSUFFICIENT_EVIDENCE]] 补充文字", (_hit(1),)
    )

    assert completion.content == "解释文字 [K1] [[INSUFFICIENT_EVIDENCE]] 补充文字"
    assert completion.abstained is False
    assert [citation.evidence_handle for citation in completion.citations] == [1]


def test_flow_rejects_provider_tool_completion() -> None:
    class FakeRetrievalService:
        async def search(self, **kwargs: object) -> RetrievalResult:
            assert kwargs == {
                "user_id": 1,
                "knowledge_base_id": 1,
                "query": "question",
                "top_k": 5,
            }
            return RetrievalResult(
                knowledge_base_id=1, generation_id="generation", hits=(_hit(1),)
            )

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
