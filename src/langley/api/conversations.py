"""HTTP routes for owned Conversation creation and readback."""

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from langley.answer_execution import AnswerExecutionManager
from langley.answering.grounding import GroundingPolicy
from langley.api.dependencies import (
    get_current_user_id,
    get_execution_manager,
    get_session,
)
from langley.api.responses import (
    MessageResponse,
    RunResponse,
    as_optional_utc,
    as_utc,
    message_response,
    run_response,
)
from langley.conversation_commands import (
    ActiveRunExistsError,
    ClientRequestIdReusedError,
    ConversationNotFoundError,
    RegenerateNotAllowedError,
    RetryNotAllowedError,
    admit_new_question,
    admit_regenerate,
    admit_retry,
)
from langley.conversations import (
    ConversationHasActiveRunError,
    create_conversation,
    delete_conversation,
    get_conversation_messages,
    list_conversations,
    rename_conversation,
)
from langley.infrastructure.models import Conversation, Message, Run
from langley.knowledge.commands import KnowledgeBaseNotFoundError

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class CreateConversationRequest(BaseModel):
    """Client-supplied Conversation fields; ownership is server-derived."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=255)


class NewQuestionRequest(BaseModel):
    """The only client fields accepted for a new answer command."""

    model_config = ConfigDict(extra="forbid")

    content: str
    client_request_id: str = Field(min_length=1, max_length=64)
    knowledge_base_id: int | None = None
    grounding_policy: GroundingPolicy = GroundingPolicy.AUTO


class RenameConversationRequest(BaseModel):
    """The manual title change allowed for one owned active Conversation."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=255)


class ExistingUserCommandRequest(BaseModel):
    """The only client field accepted for Retry and Regenerate commands."""

    model_config = ConfigDict(extra="forbid")

    client_request_id: str = Field(min_length=1, max_length=64)


class ConversationResponse(BaseModel):
    """A user-visible Conversation projection."""

    id: int
    title: str | None
    created_at: str
    updated_at: str
    last_message_at: str | None


class ConversationMessagesResponse(BaseModel):
    """Ordered persisted messages plus the latest run for the latest user input."""

    messages: list[MessageResponse]
    latest_run: RunResponse | None


class AnswerCommandResponse(BaseModel):
    """The persisted acceptance facts for an asynchronous answer command."""

    user_message: MessageResponse
    run: RunResponse


def _conversation_response(conversation: Conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=as_utc(conversation.created_at),
        updated_at=as_utc(conversation.updated_at),
        last_message_at=as_optional_utc(conversation.last_message_at),
    )


def _answer_command_response(user_message: Message, run: Run) -> AnswerCommandResponse:
    return AnswerCommandResponse(
        user_message=message_response(user_message),
        run=run_response(run),
    )


def _command_response_status(run: Run) -> int:
    """Map the persisted command snapshot to the Frozen Slice 3 HTTP status."""

    if run.status in {"PENDING", "RUNNING"}:
        return status.HTTP_202_ACCEPTED
    return status.HTTP_200_OK


def _raise_command_http_error(error: Exception) -> None:
    if isinstance(error, ConversationNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CONVERSATION_NOT_FOUND"},
        ) from error
    if isinstance(error, KnowledgeBaseNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "KNOWLEDGE_BASE_NOT_FOUND"},
        ) from error
    if isinstance(error, ActiveRunExistsError):
        code = "ACTIVE_RUN_EXISTS"
    elif isinstance(error, ClientRequestIdReusedError):
        code = "CLIENT_REQUEST_ID_REUSED"
    elif isinstance(error, RetryNotAllowedError):
        code = "RETRY_NOT_ALLOWED"
    elif isinstance(error, RegenerateNotAllowedError):
        code = "REGENERATE_NOT_ALLOWED"
    elif isinstance(error, ValueError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "VALIDATION_ERROR"},
        ) from error
    else:
        raise error
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": code})


@router.post(
    "", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED
)
async def post_conversation(
    body: CreateConversationRequest,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> ConversationResponse:
    """Create a Conversation for the configured current user."""

    conversation = await create_conversation(session, current_user_id, body.title)
    return _conversation_response(conversation)


@router.get("", response_model=list[ConversationResponse])
async def get_conversations(
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> list[ConversationResponse]:
    """List the configured user's non-deleted Conversations."""

    conversations = await list_conversations(session, current_user_id)
    return [_conversation_response(conversation) for conversation in conversations]


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def patch_conversation(
    conversation_id: int,
    body: RenameConversationRequest,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> ConversationResponse:
    """Manually rename one owned, non-deleted Conversation."""

    try:
        conversation = await rename_conversation(
            session,
            user_id=current_user_id,
            conversation_id=conversation_id,
            title=body.title,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "VALIDATION_ERROR"},
        ) from error
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CONVERSATION_NOT_FOUND"},
        )
    return _conversation_response(conversation)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation_route(
    conversation_id: int,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> Response:
    """Logically delete one owned inactive Conversation without cancelling a Run."""

    try:
        deleted = await delete_conversation(
            session,
            user_id=current_user_id,
            conversation_id=conversation_id,
        )
    except ConversationHasActiveRunError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "ACTIVE_RUN_EXISTS"},
        ) from error
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CONVERSATION_NOT_FOUND"},
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{conversation_id}/messages", response_model=ConversationMessagesResponse)
async def get_messages(
    conversation_id: int,
    session: AsyncSession = Depends(get_session),
    current_user_id: int = Depends(get_current_user_id),
) -> ConversationMessagesResponse:
    """Return ordered owned messages and latest execution state for UI refresh."""

    result = await get_conversation_messages(session, current_user_id, conversation_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CONVERSATION_NOT_FOUND"},
        )

    _, messages, latest_run, citations_by_message = result
    return ConversationMessagesResponse(
        messages=[
            message_response(message, citations_by_message.get(message.id))
            for message in messages
        ],
        latest_run=run_response(latest_run) if latest_run is not None else None,
    )


@router.post("/{conversation_id}/messages", response_model=AnswerCommandResponse)
async def post_new_question(
    conversation_id: int,
    body: NewQuestionRequest,
    response: Response,
    session: AsyncSession = Depends(get_session),
    execution_manager: AnswerExecutionManager = Depends(get_execution_manager),
    current_user_id: int = Depends(get_current_user_id),
) -> AnswerCommandResponse:
    """Accept a new question and independently schedule only new execution."""

    try:
        command = await admit_new_question(
            session,
            user_id=current_user_id,
            conversation_id=conversation_id,
            content=body.content,
            client_request_id=body.client_request_id,
            knowledge_base_id=body.knowledge_base_id,
            grounding_policy=body.grounding_policy,
        )
        await execution_manager.schedule(command)
    except Exception as error:
        _raise_command_http_error(error)

    response.status_code = _command_response_status(command.run)
    return _answer_command_response(command.user_message, command.run)


@router.post("/{conversation_id}/retry", response_model=AnswerCommandResponse)
async def post_retry(
    conversation_id: int,
    body: ExistingUserCommandRequest,
    response: Response,
    session: AsyncSession = Depends(get_session),
    execution_manager: AnswerExecutionManager = Depends(get_execution_manager),
    current_user_id: int = Depends(get_current_user_id),
) -> AnswerCommandResponse:
    """Accept a Retry and independently schedule only new execution."""

    try:
        command = await admit_retry(
            session,
            user_id=current_user_id,
            conversation_id=conversation_id,
            client_request_id=body.client_request_id,
        )
        await execution_manager.schedule(command)
    except Exception as error:
        _raise_command_http_error(error)

    response.status_code = _command_response_status(command.run)
    return _answer_command_response(command.user_message, command.run)


@router.post("/{conversation_id}/regenerate", response_model=AnswerCommandResponse)
async def post_regenerate(
    conversation_id: int,
    body: ExistingUserCommandRequest,
    response: Response,
    session: AsyncSession = Depends(get_session),
    execution_manager: AnswerExecutionManager = Depends(get_execution_manager),
    current_user_id: int = Depends(get_current_user_id),
) -> AnswerCommandResponse:
    """Accept a Regenerate and independently schedule only new execution."""

    try:
        command = await admit_regenerate(
            session,
            user_id=current_user_id,
            conversation_id=conversation_id,
            client_request_id=body.client_request_id,
        )
        await execution_manager.schedule(command)
    except Exception as error:
        _raise_command_http_error(error)

    response.status_code = _command_response_status(command.run)
    return _answer_command_response(command.user_message, command.run)
