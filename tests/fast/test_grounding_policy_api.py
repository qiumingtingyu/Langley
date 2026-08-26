"""Focused HTTP DTO contract for explicit Run grounding policy."""

from datetime import datetime
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

from langley.answering.grounding import GroundingPolicy
from langley.api.conversations import NewQuestionRequest
from langley.api.responses import run_response
from langley.infrastructure.models import Run


def test_new_question_policy_is_exact_and_auto_by_default() -> None:
    default = NewQuestionRequest(content="q", client_request_id="request-1")
    required = NewQuestionRequest(
        content="q",
        client_request_id="request-2",
        knowledge_base_id=7,
        grounding_policy="REQUIRED",
    )

    assert default.grounding_policy is GroundingPolicy.AUTO
    assert required.grounding_policy is GroundingPolicy.REQUIRED
    with pytest.raises(ValidationError):
        NewQuestionRequest(
            content="q",
            client_request_id="request-3",
            knowledge_base_id=7,
            grounding_policy="required",
        )


def test_run_response_exposes_scope_and_policy_for_recovery() -> None:
    run = cast(
        Run,
        SimpleNamespace(
            id=1,
            input_message_id=2,
            attempt_no=1,
            knowledge_base_id=7,
            grounding_policy="REQUIRED",
            status="PENDING",
            started_at=None,
            finished_at=None,
            error_code=None,
            created_at=datetime(2026, 8, 26),
        ),
    )

    response = run_response(run)

    assert response.knowledge_base_id == 7
    assert response.grounding_policy == "REQUIRED"
