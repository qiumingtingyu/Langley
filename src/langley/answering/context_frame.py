"""Provider-neutral context measurements for local Run diagnostics only."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from langley.answering.contracts import (
    AssistantRuntimeMessage,
    LLMRequest,
    ToolResult,
    UserRuntimeMessage,
)
from langley.answering.conversation_context import estimate_multilingual_tokens

CONTEXT_ESTIMATE_KIND = "CONSERVATIVE_MULTILINGUAL_ESTIMATE_V1"


@dataclass(frozen=True)
class ContextComponent:
    """One body-free measurement of a semantic request component."""

    key: str
    kind: str
    source: str | None
    char_count: int
    byte_count: int
    estimated_tokens: int
    content_sha256: str | None

    def as_json(self) -> dict[str, str | int]:
        value: dict[str, str | int] = {
            "key": self.key,
            "kind": self.kind,
            "char_count": self.char_count,
            "byte_count": self.byte_count,
            "estimated_tokens": self.estimated_tokens,
        }
        if self.source is not None:
            value["source"] = self.source
        if self.content_sha256 is not None:
            value["content_sha256"] = self.content_sha256
        return value


@dataclass(frozen=True)
class ContextFrame:
    """Measurements for one completed provider-neutral ``LLMRequest``."""

    components: tuple[ContextComponent, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "estimate_kind": CONTEXT_ESTIMATE_KIND,
            "char_count": sum(item.char_count for item in self.components),
            "byte_count": sum(item.byte_count for item in self.components),
            "estimated_tokens": sum(item.estimated_tokens for item in self.components),
            "components": [item.as_json() for item in self.components],
        }


@dataclass(frozen=True)
class _ComponentValue:
    key: str
    kind: str
    value: str
    source: str | None = None


ContextExtractor = Callable[[LLMRequest], list[_ComponentValue]]


def build_context_frame(
    request: LLMRequest, *, include_content_hashes: bool
) -> ContextFrame:
    """Classify and measure every non-empty context-bearing request field."""

    values: list[_ComponentValue] = []
    runtime_field_names = {field.name for field in fields(request)}
    for field_name, extractor in _CONTEXT_EXTRACTORS.items():
        if field_name in runtime_field_names:
            values.extend(extractor(request))

    classified = set(_CONTEXT_EXTRACTORS) | _REQUEST_METADATA_FIELDS
    for field in fields(request):
        if field.name in classified:
            continue
        value = getattr(request, field.name)
        if _has_content(value):
            values.append(
                _ComponentValue(
                    key=f"unclassified.request_field.{field.name}",
                    kind="unclassified.request_field",
                    source=field.name,
                    value=_canonical_json(value),
                )
            )

    return ContextFrame(
        tuple(
            _measure(value, include_content_hash=include_content_hashes)
            for value in values
        )
    )


def _measure(
    component: _ComponentValue, *, include_content_hash: bool
) -> ContextComponent:
    encoded = component.value.encode("utf-8")
    return ContextComponent(
        key=component.key,
        kind=component.kind,
        source=component.source,
        char_count=len(component.value),
        byte_count=len(encoded),
        estimated_tokens=estimate_multilingual_tokens(component.value),
        content_sha256=(
            hashlib.sha256(encoded).hexdigest() if include_content_hash else None
        ),
    )


def _system(request: LLMRequest) -> list[_ComponentValue]:
    return _text_component("system", "system", request.system_input)


def _personal_context(request: LLMRequest) -> list[_ComponentValue]:
    if request.personal_context is None:
        return []
    return [
        _ComponentValue(
            key=f"personal_context.{index}",
            kind="personal_context",
            source=f"personal_context[{index}]",
            value=value,
        )
        for index, value in enumerate(request.personal_context)
        if value
    ]


def _conversation_compact(request: LLMRequest) -> list[_ComponentValue]:
    return _text_component(
        "conversation_compact",
        "conversation_compact",
        request.conversation_compact_context,
    )


def _active_skill(request: LLMRequest) -> list[_ComponentValue]:
    if request.active_skill is None:
        return []
    return [
        _ComponentValue(
            key="skill.active",
            kind="skill.active",
            source=request.active_skill.name,
            value=_canonical_json(request.active_skill),
        )
    ]


def _skill_catalog(request: LLMRequest) -> list[_ComponentValue]:
    return [
        _ComponentValue(
            key=f"skill.catalog.{index}",
            kind="skill.catalog",
            source=skill.name,
            value=_canonical_json(skill),
        )
        for index, skill in enumerate(request.available_skills)
    ]


def _skill_resources(request: LLMRequest) -> list[_ComponentValue]:
    return [
        _ComponentValue(
            key=f"skill.resources.{index}",
            kind="skill.resources",
            source=resource.path,
            value=_canonical_json(resource),
        )
        for index, resource in enumerate(request.active_skill_resources)
    ]


def _tool_schemas(request: LLMRequest) -> list[_ComponentValue]:
    return [
        _ComponentValue(
            key=f"tools.schema.{index}",
            kind="tools.schema",
            source=tool.name,
            value=_canonical_json(tool),
        )
        for index, tool in enumerate(request.allowed_tools)
    ]


def _transcript(request: LLMRequest) -> list[_ComponentValue]:
    values: list[_ComponentValue] = []
    for index, item in enumerate(request.transcript):
        if isinstance(item, UserRuntimeMessage):
            values.extend(
                _text_component(
                    f"transcript.{index}.user",
                    "transcript.user",
                    item.content,
                    source=f"transcript[{index}]",
                )
            )
        elif isinstance(item, AssistantRuntimeMessage):
            values.extend(
                _text_component(
                    f"transcript.{index}.assistant",
                    "transcript.assistant",
                    item.content,
                    source=f"transcript[{index}]",
                )
            )
            values.extend(
                _ComponentValue(
                    key=f"transcript.{index}.tool_call.{call_index}",
                    kind="transcript.tool_call",
                    source=call.name,
                    value=_canonical_json(call),
                )
                for call_index, call in enumerate(item.tool_calls)
            )
        elif isinstance(item, ToolResult):
            values.append(
                _ComponentValue(
                    key=f"transcript.{index}.tool_result",
                    kind="transcript.tool_result",
                    source=item.name,
                    value=_canonical_json(item),
                )
            )
        elif _has_content(item):
            values.append(
                _ComponentValue(
                    key=f"unclassified.request_field.transcript.{index}",
                    kind="unclassified.request_field",
                    source=f"transcript[{index}]",
                    value=_canonical_json(item),
                )
            )
    return values


def _evidence_context(request: LLMRequest) -> list[_ComponentValue]:
    return _text_component(
        "evidence_context", "evidence_context", request.evidence_context
    )


def _text_component(
    key: str, kind: str, value: str | None, *, source: str | None = None
) -> list[_ComponentValue]:
    if not value:
        return []
    return [_ComponentValue(key=key, kind=kind, source=source, value=value)]


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_value(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _has_content(value: object) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return bool(value)
    return True


_CONTEXT_EXTRACTORS: dict[str, ContextExtractor] = {
    "system_input": _system,
    "transcript": _transcript,
    "allowed_tools": _tool_schemas,
    "personal_context": _personal_context,
    "conversation_compact_context": _conversation_compact,
    "evidence_context": _evidence_context,
    "available_skills": _skill_catalog,
    "active_skill": _active_skill,
    "active_skill_resources": _skill_resources,
}

_REQUEST_METADATA_FIELDS = frozenset({"current_user_message_index"})
