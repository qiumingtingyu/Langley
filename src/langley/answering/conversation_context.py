"""Structured, rebuildable projection of older authoritative conversation facts."""

import json
from dataclasses import dataclass
from typing import Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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
    LLMUsage,
    UserRuntimeMessage,
)

CONVERSATION_COMPACTOR_PROMPT_VERSION = "conversation-compactor-v1-sticky-r1"
_MESSAGE_STRUCTURAL_OVERHEAD_ESTIMATE = 4

_FIELD_LABELS = (
    ("current_goals", "Current goals"),
    ("active_decisions", "Active decisions"),
    ("active_constraints", "Active constraints"),
    ("open_loops", "Open loops"),
    ("important_facts", "Important facts"),
    ("artifacts", "Artifacts"),
)

_COMPACTOR_SYSTEM_INPUT = """You maintain Langley's derived conversation compact state.

Return exactly one JSON object and no Markdown, prose, or tool calls. Do not
answer the user. Update the previous compact state using only the newly aged-out
canonical USER/ASSISTANT messages supplied in the input.

Previous compact-state items are sticky by default. Keep each previous item
current unless a later USER-authored message explicitly corrects, replaces,
revokes, invalidates, or supersedes it. Ordinary later discussion must not
remove a previous item. A later ASSISTANT restatement or paraphrase alone must
not supersede, weaken, replace, or remove an existing item. If a previous item
remains semantically unchanged, preserve its existing source_message_ids; do
not replace or extend that provenance with later Assistant-generated
restatements. An explicit later USER correction still wins normally.

Preserve active goals, explicit decisions, active constraints, unresolved or
open work, important names, numbers, versions, artifacts, and entity
relationships. Preserve rejection and negation semantics. Favor recall over
aggressive compression. Later explicit correction or replacement overrides
stale state: keep the current fact and its supporting source_message_ids, not a
second active stale value. Discussion does not automatically become a decision.
An Assistant suggestion alone must not become a USER fact or decision. Do not
invent facts.

Treat words and relations such as current, final, only, must, must not,
rejected, no longer, correction, replace, change, and update as important.
source_message_ids are provenance and must contain only IDs supplied in the
previous state or newly aged-out messages. Aim for the supplied soft
compact_state_target_estimate; it is an estimate, not exact provider tokens.

Use exactly this shape. Every field is an array; every item has only content and
source_message_ids:
{
  "current_goals": [{"content": "...", "source_message_ids": [1]}],
  "active_decisions": [],
  "active_constraints": [],
  "open_loops": [],
  "important_facts": [],
  "artifacts": []
}"""


def estimate_multilingual_tokens(text: str) -> int:
    """Estimate tokens conservatively without pretending to vendor tokenization."""

    ascii_count = sum(character.isascii() for character in text)
    non_ascii_count = len(text) - ascii_count
    return (ascii_count + 3) // 4 + non_ascii_count


def estimate_message_tokens(content: str) -> int:
    """Add a small explicit estimate for one chat message's structural framing."""

    return estimate_multilingual_tokens(content) + _MESSAGE_STRUCTURAL_OVERHEAD_ESTIMATE


class _CompactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConversationCompactItem(_CompactModel):
    """One active compact-state fact with authoritative Message provenance."""

    content: StrictStr = Field(min_length=1)
    source_message_ids: tuple[StrictInt, ...] = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_must_be_nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content must be nonblank")
        return value

    @model_validator(mode="after")
    def source_message_ids_must_be_positive_and_unique(self) -> Self:
        if any(message_id < 1 for message_id in self.source_message_ids):
            raise ValueError("source_message_ids must be positive")
        if len(self.source_message_ids) != len(set(self.source_message_ids)):
            raise ValueError("source_message_ids must be unique per item")
        return self


class ConversationCompactState(_CompactModel):
    """Current structured projection of older conversation history."""

    current_goals: tuple[ConversationCompactItem, ...]
    active_decisions: tuple[ConversationCompactItem, ...]
    active_constraints: tuple[ConversationCompactItem, ...]
    open_loops: tuple[ConversationCompactItem, ...]
    important_facts: tuple[ConversationCompactItem, ...]
    artifacts: tuple[ConversationCompactItem, ...]

    @property
    def source_message_ids(self) -> frozenset[int]:
        return frozenset(
            message_id
            for field_name, _ in _FIELD_LABELS
            for item in getattr(self, field_name)
            for message_id in item.source_message_ids
        )


@dataclass(frozen=True)
class ConversationContextMessage:
    """One detached canonical Message supplied to the internal compactor."""

    message_id: int
    role: Literal["USER", "ASSISTANT"]
    content: str


@dataclass(frozen=True)
class ConversationCompactionResult:
    """One validated compact state and optional provider-reported usage."""

    state: ConversationCompactState
    usage: LLMUsage | None
    provider_model: str | None


class ConversationCompactionInvalidOutputError(RuntimeError):
    """The compactor completed normally but returned no valid whole state."""


class ConversationContextCompactor(Protocol):
    @property
    def model(self) -> str: ...

    async def compact(
        self,
        *,
        previous_state: ConversationCompactState | None,
        newly_aged_out_messages: tuple[ConversationContextMessage, ...],
    ) -> ConversationCompactionResult: ...


class LLMConversationCompactor:
    """Run one provider-neutral, JSON-only internal compaction operation."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        compact_state_target_estimate: int,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be nonblank")
        if compact_state_target_estimate < 1:
            raise ValueError("compact_state_target_estimate must be positive")
        self._provider = provider
        self._model = model
        self._compact_state_target_estimate = compact_state_target_estimate

    @property
    def model(self) -> str:
        return self._model

    async def compact(
        self,
        *,
        previous_state: ConversationCompactState | None,
        newly_aged_out_messages: tuple[ConversationContextMessage, ...],
    ) -> ConversationCompactionResult:
        if not newly_aged_out_messages:
            raise ValueError("newly_aged_out_messages must not be empty")
        request = self._request_for(previous_state, newly_aged_out_messages)
        completion: LLMResponseCompleted | None = None
        async for event in self._provider.stream(request):
            if isinstance(event, LLMResponseCompleted):
                if completion is not None:
                    raise ConversationCompactionInvalidOutputError(
                        "compactor returned more than one completion"
                    )
                completion = event
        if completion is None:
            raise ConversationCompactionInvalidOutputError(
                "compactor returned no canonical completion"
            )
        if (
            completion.tool_calls
            or completion.finish_reason is not LLMFinishReason.STOP
        ):
            raise ConversationCompactionInvalidOutputError(
                "compactor must stop without tool calls"
            )
        allowed_source_ids = {message.message_id for message in newly_aged_out_messages}
        if previous_state is not None:
            allowed_source_ids.update(previous_state.source_message_ids)
        state = self._parse_state(
            completion.assistant_content,
            allowed_source_ids=allowed_source_ids,
        )
        return ConversationCompactionResult(
            state=state,
            usage=completion.usage,
            provider_model=completion.provider_model,
        )

    def _request_for(
        self,
        previous_state: ConversationCompactState | None,
        newly_aged_out_messages: tuple[ConversationContextMessage, ...],
    ) -> LLMRequest:
        payload = {
            "previous_compact_state": (
                None
                if previous_state is None
                else previous_state.model_dump(mode="json")
            ),
            "newly_aged_out_messages": [
                {
                    "message_id": message.message_id,
                    "role": message.role,
                    "content": message.content,
                }
                for message in newly_aged_out_messages
            ],
            "compact_state_target_estimate": self._compact_state_target_estimate,
        }
        return LLMRequest(
            system_input=_COMPACTOR_SYSTEM_INPUT,
            transcript=(
                UserRuntimeMessage(
                    content=json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ),
            ),
            allowed_tools=(),
        )

    @staticmethod
    def _parse_state(
        assistant_content: str,
        *,
        allowed_source_ids: set[int],
    ) -> ConversationCompactState:
        try:
            payload = json.loads(assistant_content)
        except (json.JSONDecodeError, TypeError) as error:
            raise ConversationCompactionInvalidOutputError(
                "compactor returned invalid JSON"
            ) from error
        try:
            state = ConversationCompactState.model_validate(payload)
        except ValidationError as error:
            raise ConversationCompactionInvalidOutputError(
                "compactor returned an invalid structured state"
            ) from error
        if not state.source_message_ids <= allowed_source_ids:
            raise ConversationCompactionInvalidOutputError(
                "compactor returned unsupported source_message_ids"
            )
        return state


def render_conversation_compact_context(state: ConversationCompactState) -> str:
    """Render human-readable compact content without internal provenance IDs."""

    sections = ["Conversation compact context (derived from older history):"]
    for field_name, label in _FIELD_LABELS:
        items: tuple[ConversationCompactItem, ...] = getattr(state, field_name)
        if not items:
            continue
        sections.append(f"{label}:")
        sections.extend(f"- {item.content}" for item in items)
    if len(sections) == 1:
        sections.append("No active older conversation state.")
    return "\n".join(sections)
