"""Structured Personal Context Memory policy contract.

The policy owns semantic judgement.  This module only supplies complete,
detached context to the configured provider and validates its structured
response before a later task decides whether and how to persist it.
"""

import json
from collections.abc import Iterable
from datetime import datetime
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from langley.answering.contracts import (
    LLMFinishReason,
    LLMProvider,
    LLMRequest,
    LLMResponseCompleted,
    UserRuntimeMessage,
)
from langley.business_time import normalize_aware_datetime_to_utc_naive

MEMORY_POLICY_PROMPT_VERSION = "slice5-t3-v1"
_MAX_STRUCTURED_OUTPUT_ATTEMPTS = 2
# Structural context estimate only; calibration still supplies the actual budget.
_MEMORY_ITEM_FRAMING_ALLOWANCE = 32


def estimate_load_all_memory_contribution(contents: Iterable[str]) -> int:
    """Estimate complete Memory contribution without claiming tokenizer parity."""

    return sum(len(content) + _MEMORY_ITEM_FRAMING_ALLOWANCE for content in contents)


_MEMORY_POLICY_SYSTEM_INPUT = """You are Langley's Personal Context Memory Policy.

Return exactly one JSON object and no Markdown, prose, or tool calls. You own
all semantic decisions about Personal Context eligibility, durable versus
temporary information, explicit versus implicit intent, proposition splitting,
target selection, self-contained content normalization, and temporal intent.

The input's evidence_content is canonical USER evidence. previous_messages are
only earlier persisted conversation context. Assistant context can explain a
USER confirmation, reference, correction, scope, or temporal intent, but an
Assistant statement alone cannot establish Personal Context evidence.

auto_memory_enabled is a semantic instruction, not merely metadata:
- When it is true (ON), ordinary implicit eligible Personal Context may produce
  mutations. Explicit conversational remember, correct, or forget requests may
  also produce mutations.
- When it is false (OFF), ordinary implicit information must produce NO_CHANGE.
  Explicit conversational remember, correct, or forget requests may still
  produce mutations.
- An explicit request never bypasses Personal Context eligibility. If a request
  is not eligible, return NO_CHANGE while preserving
  user_requested_memory_action=true when the USER explicitly requested a
  remember, correct, or forget action.

The supplied current_memories are the complete current/effective Personal
Memory context. A NEW mutation has no target_memory_id and requires non-empty,
self-contained content. A CHANGE mutation targets a supplied memory and
requires full replacement content. A FORGET mutation targets a supplied memory
and has neither content nor valid_until. Supply valid_until only when reliable
USER evidence supports a temporal boundary; it must be an absolute,
offset-aware ISO 8601 timestamp.

For temporal interpretation, evidence_created_at is UTC-naive storage and
local_temporal_reference is the same instant in the configured local timezone.
Resolve temporal language relative to local_temporal_reference, never the host
or provider invocation time.

Use this JSON shape. Each mutation operation is one of NEW, CHANGE, or FORGET:
{
  "mutations": [
    {
      "operation": "NEW",
      "content": "self-contained current Personal Context",
      "valid_until": "2026-08-20T10:00:00+08:00"
    }
  ],
  "user_requested_memory_action": false
}

Represent NO_CHANGE with an empty mutations array. Do not include unused
fields in a mutation."""


class MemoryPolicyUnavailableError(RuntimeError):
    """The write-side Memory Policy configuration is not ready for invocation."""


class MemoryPolicyContextInfeasibleError(RuntimeError):
    """All current Memory facts cannot fit the configured write-side budget."""


class MemoryPolicyInvalidOutputError(RuntimeError):
    """A normally completed provider response failed whole-result validation."""


class _FrozenPolicyModel(BaseModel):
    """Shared strict boundary behavior for detached policy values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MemoryPolicyItem(_FrozenPolicyModel):
    """One supplied current/effective Memory fact available to the policy."""

    memory_id: StrictInt = Field(gt=0)
    content: StrictStr = Field(min_length=1)
    valid_until: datetime | None = None

    @field_validator("valid_until")
    @classmethod
    def valid_until_must_use_utc_naive_storage(
        cls, value: datetime | None
    ) -> datetime | None:
        """Keep detached supplied facts aligned with the MySQL business convention."""

        if value is not None and (
            value.tzinfo is not None or value.utcoffset() is not None
        ):
            raise ValueError("valid_until must be UTC-naive")
        return value


class MemoryPolicyConversationMessage(_FrozenPolicyModel):
    """One persisted message preceding the canonical USER evidence."""

    role: Literal["USER", "ASSISTANT"]
    content: StrictStr = Field(min_length=1)


class MemoryPolicyInput(_FrozenPolicyModel):
    """Complete detached context for one canonical USER evidence decision."""

    evidence_message_id: StrictInt = Field(gt=0)
    evidence_role: Literal["USER"] = "USER"
    evidence_content: StrictStr = Field(min_length=1)
    evidence_created_at: datetime
    previous_messages: tuple[MemoryPolicyConversationMessage, ...] = Field(max_length=4)
    current_memories: tuple[MemoryPolicyItem, ...]
    auto_memory_enabled: StrictBool
    local_temporal_reference: datetime

    @field_validator("evidence_created_at")
    @classmethod
    def evidence_created_at_must_use_utc_naive_storage(
        cls, value: datetime
    ) -> datetime:
        """Require the existing UTC-naive MySQL business-time convention."""

        if value.tzinfo is not None or value.utcoffset() is not None:
            raise ValueError("evidence_created_at must be UTC-naive")
        return value

    @field_validator("local_temporal_reference")
    @classmethod
    def local_temporal_reference_must_be_offset_aware(cls, value: datetime) -> datetime:
        """Reject host-timezone-dependent temporal references."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("local_temporal_reference must be offset-aware")
        return value

    @model_validator(mode="after")
    def current_memory_ids_must_be_unique(self) -> Self:
        """Make every targetable supplied Memory identity unambiguous."""

        memory_ids = tuple(memory.memory_id for memory in self.current_memories)
        if len(memory_ids) != len(set(memory_ids)):
            raise ValueError(
                "current_memories must not contain duplicate memory_id values"
            )
        return self

    @model_validator(mode="after")
    def local_reference_must_represent_the_evidence_instant(self) -> Self:
        """Keep temporal interpretation tied to the durable evidence creation time."""

        if (
            normalize_aware_datetime_to_utc_naive(self.local_temporal_reference)
            != self.evidence_created_at
        ):
            raise ValueError("local_temporal_reference must match evidence_created_at")
        return self


class MemoryMutationDecision(_FrozenPolicyModel):
    """One validated NEW, CHANGE, or FORGET candidate from the policy."""

    operation: Literal["NEW", "CHANGE", "FORGET"]
    target_memory_id: StrictInt | None = Field(default=None, gt=0)
    content: StrictStr | None = None
    valid_until: datetime | None = None

    @field_validator("valid_until")
    @classmethod
    def valid_until_must_be_absolute_and_normalized(
        cls, value: datetime | None
    ) -> datetime | None:
        """Require an absolute instant before later persistence code sees it."""

        if value is None:
            return None
        try:
            return normalize_aware_datetime_to_utc_naive(value)
        except ValueError as error:
            raise ValueError("valid_until must be offset-aware") from error

    @model_validator(mode="after")
    def fields_must_match_operation(self) -> Self:
        """Reject incomplete or contradictory mutations as a whole-result failure."""

        has_content = self.content is not None and bool(self.content.strip())
        if self.operation == "NEW":
            if self.target_memory_id is not None:
                raise ValueError("NEW must not target an existing memory")
            if not has_content:
                raise ValueError("NEW requires non-empty content")
        elif self.operation == "CHANGE":
            if self.target_memory_id is None:
                raise ValueError("CHANGE requires target_memory_id")
            if not has_content:
                raise ValueError("CHANGE requires non-empty content")
        else:
            if self.target_memory_id is None:
                raise ValueError("FORGET requires target_memory_id")
            if self.content is not None:
                raise ValueError("FORGET must not include content")
            if self.valid_until is not None:
                raise ValueError("FORGET must not include valid_until")
        return self


class MemoryPolicyResult(_FrozenPolicyModel):
    """A NO_CHANGE result or a complete batch of validated mutations."""

    mutations: tuple[MemoryMutationDecision, ...]
    user_requested_memory_action: StrictBool

    @model_validator(mode="after")
    def target_memory_ids_must_not_repeat(self) -> Self:
        """Reject same-target batches instead of attempting partial interpretation."""

        target_ids = tuple(
            mutation.target_memory_id
            for mutation in self.mutations
            if mutation.target_memory_id is not None
        )
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("a result must not target the same memory more than once")
        return self

    @model_validator(mode="after")
    def new_decisions_must_not_repeat(self) -> Self:
        """Reject exact duplicate NEW decisions without semantic similarity rules."""

        new_decisions = tuple(
            (mutation.content, mutation.valid_until)
            for mutation in self.mutations
            if mutation.operation == "NEW" and mutation.content is not None
        )
        if len(new_decisions) != len(set(new_decisions)):
            raise ValueError("a result must not contain duplicate NEW decisions")
        return self

    @property
    def is_no_change(self) -> bool:
        """Expose the frozen zero-mutation NO_CHANGE semantic without inference."""

        return not self.mutations


class MemoryPolicy:
    """Ask an already-configured provider for a structured Memory decision."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        memory_policy_estimated_token_budget: int | None,
    ) -> None:
        """Keep provider/model composition outside this provider-neutral boundary."""

        if (
            memory_policy_estimated_token_budget is not None
            and memory_policy_estimated_token_budget < 1
        ):
            raise ValueError("memory_policy_estimated_token_budget must be positive")
        self._provider = provider
        self._memory_policy_estimated_token_budget = (
            memory_policy_estimated_token_budget
        )

    @property
    def estimated_token_budget(self) -> int:
        """Expose the configured write-capacity contract to persistence guards."""

        self._ensure_budget_configured()
        budget = self._memory_policy_estimated_token_budget
        assert budget is not None
        return budget

    async def decide(self, policy_input: MemoryPolicyInput) -> MemoryPolicyResult:
        """Return one validated decision or expose the bounded failure category."""

        self._ensure_budget_configured()
        self._ensure_current_memories_fit(policy_input)
        request = self._request_for(policy_input)
        last_error: MemoryPolicyInvalidOutputError | None = None

        for _ in range(_MAX_STRUCTURED_OUTPUT_ATTEMPTS):
            try:
                return await self._decide_once(request, policy_input)
            except MemoryPolicyInvalidOutputError as error:
                last_error = error

        if last_error is None:
            raise RuntimeError(
                "memory policy structured-output retry was not attempted"
            )
        raise last_error

    def _ensure_budget_configured(self) -> None:
        """Fail before invocation until the independent write budget is calibrated."""

        if self._memory_policy_estimated_token_budget is None:
            raise MemoryPolicyUnavailableError(
                "memory policy estimated token budget is not calibrated"
            )

    def _ensure_current_memories_fit(self, policy_input: MemoryPolicyInput) -> None:
        """Use the existing deterministic character estimator without truncation."""

        estimated_tokens = estimate_load_all_memory_contribution(
            memory.content for memory in policy_input.current_memories
        )
        if estimated_tokens > self.estimated_token_budget:
            raise MemoryPolicyContextInfeasibleError(
                "all current memories exceed the memory policy token budget"
            )

    @staticmethod
    def _request_for(policy_input: MemoryPolicyInput) -> LLMRequest:
        """Build one provider-neutral JSON-only policy request."""

        payload = json.dumps(
            policy_input.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return LLMRequest(
            system_input=_MEMORY_POLICY_SYSTEM_INPUT,
            transcript=(UserRuntimeMessage(content=payload),),
            allowed_tools=(),
        )

    async def _decide_once(
        self, request: LLMRequest, policy_input: MemoryPolicyInput
    ) -> MemoryPolicyResult:
        """Use only the canonical completion from one normally completed stream."""

        completion: LLMResponseCompleted | None = None
        async for event in self._provider.stream(request):
            if isinstance(event, LLMResponseCompleted):
                if completion is not None:
                    raise MemoryPolicyInvalidOutputError(
                        "memory policy returned more than one completion"
                    )
                completion = event

        if completion is None:
            raise MemoryPolicyInvalidOutputError(
                "memory policy returned no canonical completion"
            )
        if (
            completion.tool_calls
            or completion.finish_reason is not LLMFinishReason.STOP
        ):
            raise MemoryPolicyInvalidOutputError(
                "memory policy completion must stop without tool calls"
            )
        return self._parse_and_validate_result(
            completion.assistant_content,
            supplied_memory_ids={
                memory.memory_id for memory in policy_input.current_memories
            },
            supplied_memory_contents={
                memory.content for memory in policy_input.current_memories
            },
        )

    @staticmethod
    def _parse_and_validate_result(
        assistant_content: str,
        *,
        supplied_memory_ids: set[int],
        supplied_memory_contents: set[str],
    ) -> MemoryPolicyResult:
        """Reject malformed or illegal output without exposing content in errors."""

        try:
            payload = json.loads(assistant_content)
        except (json.JSONDecodeError, TypeError) as error:
            raise MemoryPolicyInvalidOutputError(
                "memory policy returned invalid JSON"
            ) from error
        if not isinstance(payload, dict):
            raise MemoryPolicyInvalidOutputError(
                "memory policy result must be a JSON object"
            )
        try:
            result = MemoryPolicyResult.model_validate(payload)
        except ValidationError as error:
            raise MemoryPolicyInvalidOutputError(
                "memory policy returned an invalid structured result"
            ) from error

        for mutation in result.mutations:
            target_memory_id = mutation.target_memory_id
            if (
                target_memory_id is not None
                and target_memory_id not in supplied_memory_ids
            ):
                raise MemoryPolicyInvalidOutputError(
                    "memory policy targeted a memory not supplied in context"
                )
            if (
                mutation.operation == "NEW"
                and mutation.content is not None
                and mutation.content in supplied_memory_contents
            ):
                raise MemoryPolicyInvalidOutputError(
                    "memory policy NEW content duplicates supplied current memory"
                )
        return result
