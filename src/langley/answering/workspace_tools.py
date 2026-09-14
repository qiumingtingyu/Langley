"""The five Workspace capabilities. Scope is supplied only by the runtime."""

import json
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from langley.answering.contracts import JSONValue, ToolSpec
from langley.answering.tools import ToolContext, ToolExecutionError, ToolExecutionOutput
from langley.workspace_storage import WorkspaceFileError, WorkspaceStorage


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ListFilesArguments(_Arguments):
    path: str = ""


class ReadFileArguments(_Arguments):
    path: str = Field(min_length=1)
    offset: int = Field(default=1, ge=1)
    limit: int = Field(default=200, ge=1)


class WriteFileArguments(_Arguments):
    path: str = Field(min_length=1)
    content: str
    overwrite: bool = False


class ExactEdit(_Arguments):
    old_text: str = Field(min_length=1)
    new_text: str


class EditFileArguments(_Arguments):
    path: str = Field(min_length=1)
    edits: list[ExactEdit] = Field(min_length=1, max_length=100)


class RunCommandArguments(_Arguments):
    command: str = Field(min_length=1, max_length=16384)
    timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        description="Integer seconds; omit/null for default; must not be a string.",
    )


_DEFINITIONS: dict[str, tuple[type[BaseModel], str]] = {
    "list_files": (
        ListFilesArguments,
        "List direct children of a Workspace-relative directory.",
    ),
    "read_file": (
        ReadFileArguments,
        "Read UTF-8 Workspace text; offset is a 1-based line, "
        "content has no line-number prefixes.",
    ),
    "write_file": (
        WriteFileArguments,
        "Create UTF-8 text. Set overwrite=true only for intentional whole-file "
        "replacement. Call alone and observe the result.",
    ),
    "edit_file": (
        EditFileArguments,
        "Apply unique exact non-overlapping anchors against the same pre-edit "
        "file. All edits validate before any mutation. Call alone and observe "
        "the result.",
    ),
    "run_command": (
        RunCommandArguments,
        "Run one foreground bash command in /workspace, offline Linux with "
        "Python and pytest. Nonzero exit/timeout may leave changes. Call alone "
        "and observe the result before planning another action.",
    ),
}
WORKSPACE_TOOL_NAMES = frozenset(_DEFINITIONS)


class WorkspaceTool:
    """Small concrete binding of a named schema to the shared storage/runtime."""

    def __init__(self, name: str, storage: WorkspaceStorage) -> None:
        self.arguments, description = _DEFINITIONS[name]
        self.storage = storage
        schema = self.arguments.model_json_schema()
        if name == "run_command":
            # Model-facing optional integer; runtime still accepts int | None.
            schema["properties"]["timeout_seconds"] = {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Timeout in whole seconds. Omit this field to use the default."
                ),
            }
        self.spec = ToolSpec(
            name=name,
            description=description,
            arguments_schema=cast(dict[str, JSONValue], schema),
            side_effecting=name in {"write_file", "edit_file", "run_command"},
        )

    def validate_arguments(self, arguments: dict[str, JSONValue]) -> bool:
        try:
            self.arguments.model_validate(arguments)
        except ValidationError:
            return False
        return True

    def explain_invalid_arguments(
        self, arguments: dict[str, JSONValue]
    ) -> list[dict[str, JSONValue]]:
        """Expose only schema-owned paths and fixed messages, never input/ctx."""
        try:
            self.arguments.model_validate(arguments)
        except ValidationError as error:
            messages = {
                "int_type": "Input should be a valid integer",
                "string_type": "Input should be a valid string",
                "bool_type": "Input should be a valid boolean",
                "list_type": "Input should be a valid list",
                "model_type": "Input should be an object",
                "missing": "Field required",
                "extra_forbidden": "Extra fields are not permitted",
                "greater_than_equal": "Value is below the allowed minimum",
                "string_too_short": "String is shorter than the allowed minimum",
                "string_too_long": "String exceeds the allowed maximum",
                "too_short": "List is shorter than the allowed minimum",
                "too_long": "List exceeds the allowed maximum",
            }
            issues: list[dict[str, JSONValue]] = []
            for detail in error.errors(
                include_input=False, include_context=False, include_url=False
            )[:5]:
                path = ""
                for part in detail["loc"]:
                    if isinstance(part, int):
                        path += f"[{part}]"
                    elif part in self.arguments.model_fields or (
                        path.startswith("edits[") and part in ExactEdit.model_fields
                    ):
                        path += ("." if path else "") + part
                    else:
                        # Extra keys are model input too; do not echo them.
                        path = "$"
                        break
                issues.append(
                    {
                        "field": (path or "$")[:128],
                        "message": messages.get(detail["type"], "Invalid field value")[
                            :160
                        ],
                    }
                )
            return issues
        return []

    async def execute(
        self, arguments: dict[str, JSONValue], context: ToolContext | None
    ) -> ToolExecutionOutput:
        if (
            context is None
            or context.workspace_id is None
            or context.workspace_storage_key is None
        ):
            raise ToolExecutionError("WORKSPACE_UNAVAILABLE", retryable=False)
        args = self.arguments.model_validate(arguments).model_dump()
        try:
            if self.spec.name == "run_command":
                if context.sandbox_runtime is None:
                    raise ToolExecutionError("WORKSPACE_UNAVAILABLE", retryable=False)
                result = await context.sandbox_runtime.execute(**args)
            else:
                operation = getattr(self.storage, self.spec.name)
                result = operation(context.workspace_storage_key, **args)
        except WorkspaceFileError as error:
            raise ToolExecutionError(
                error.code, retryable=False, hints=error.hints
            ) from error
        return ToolExecutionOutput(json.dumps(result, ensure_ascii=False))
