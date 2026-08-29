"""SQLAlchemy mappings for Langley's authoritative business facts."""

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import DATETIME, LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column

from langley.infrastructure.database import Base


class User(Base):
    """Minimal ownership anchor for Langley business resources."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    auto_memory_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default="1",
        nullable=False,
    )


class KnowledgeBase(Base):
    """A user-owned grouping of logically related knowledge documents."""

    __tablename__ = "knowledge_bases"
    __table_args__ = (
        Index("ix_knowledge_bases_user", "user_id"),
        CheckConstraint(
            "CHAR_LENGTH(TRIM(name)) > 0", name="ck_knowledge_bases_name_nonblank"
        ),
        CheckConstraint(
            "index_status IN ('CHUNKED', 'INDEXING', 'READY', 'FAILED', 'STALE')",
            name="ck_knowledge_bases_index_status_valid",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", name="fk_knowledge_bases_user"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    index_status: Mapped[str] = mapped_column(
        String(16, collation="utf8mb4_0900_bin"),
        default="CHUNKED",
        server_default="CHUNKED",
        nullable=False,
    )
    active_generation_id: Mapped[str | None] = mapped_column(
        String(36, collation="utf8mb4_0900_bin"), nullable=True
    )
    building_generation_id: Mapped[str | None] = mapped_column(
        String(36, collation="utf8mb4_0900_bin"), nullable=True
    )
    active_embedding_model: Mapped[str | None] = mapped_column(
        String(255, collation="utf8mb4_0900_bin"), nullable=True
    )
    active_embedding_revision: Mapped[str | None] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=True
    )
    active_embedding_dimension: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )
    active_embedding_representation: Mapped[str | None] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=True
    )
    active_chunk_snapshot_sha256: Mapped[str | None] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)


class KnowledgeIndexJob(Base):
    """One durable manual attempt to build a complete Knowledge dense index."""

    __tablename__ = "knowledge_index_jobs"
    __table_args__ = (
        UniqueConstraint("generation_id", name="uq_knowledge_index_jobs_generation"),
        Index(
            "ix_knowledge_index_jobs_base_created", "knowledge_base_id", "created_at"
        ),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'INTERRUPTED')",
            name="ck_knowledge_index_jobs_status_valid",
        ),
        CheckConstraint(
            "stage IS NULL OR stage IN ('SNAPSHOT', 'EMBEDDING', 'UPLOADING_INDEX', "
            "'VERIFYING', 'ACTIVATING')",
            name="ck_knowledge_index_jobs_stage_valid",
        ),
        CheckConstraint(
            "processed_chunk_count >= 0 AND total_chunk_count >= 0 AND "
            "processed_chunk_count <= total_chunk_count",
            name="ck_knowledge_index_jobs_progress_valid",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    knowledge_base_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "knowledge_bases.id",
            name="fk_knowledge_index_jobs_knowledge_base",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    generation_id: Mapped[str] = mapped_column(
        String(36, collation="utf8mb4_0900_bin"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(16, collation="utf8mb4_0900_bin"), nullable=False
    )
    stage: Mapped[str | None] = mapped_column(
        String(32, collation="utf8mb4_0900_bin"), nullable=True
    )
    processed_chunk_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    total_chunk_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    embedding_model: Mapped[str] = mapped_column(
        String(255, collation="utf8mb4_0900_bin"), nullable=False
    )
    embedding_revision: Mapped[str] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=False
    )
    embedding_dimension: Mapped[int] = mapped_column(BigInteger, nullable=False)
    embedding_representation: Mapped[str] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=False
    )
    chunk_snapshot_sha256: Mapped[str] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=False
    )
    error_code: Mapped[str | None] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6), nullable=True)


class Document(Base):
    """A logical user-visible document within one KnowledgeBase."""

    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_knowledge_base", "knowledge_base_id"),
        CheckConstraint(
            "CHAR_LENGTH(TRIM(name)) > 0", name="ck_documents_name_nonblank"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    knowledge_base_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("knowledge_bases.id", name="fk_documents_knowledge_base"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)


class DocumentVersion(Base):
    """One immutable source upload belonging to a logical Document."""

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("storage_key", name="uq_document_versions_storage_key"),
        Index("ix_document_versions_document", "document_id"),
        CheckConstraint(
            "CHAR_LENGTH(TRIM(source_filename)) > 0",
            name="ck_document_versions_source_filename_nonblank",
        ),
        CheckConstraint(
            "CHAR_LENGTH(TRIM(source_media_type)) > 0",
            name="ck_document_versions_source_media_type_nonblank",
        ),
        CheckConstraint(
            "CHAR_LENGTH(source_sha256) = 64",
            name="ck_document_versions_source_sha256_length",
        ),
        CheckConstraint(
            "source_size_bytes > 0",
            name="ck_document_versions_source_size_bytes_positive",
        ),
        CheckConstraint(
            "CHAR_LENGTH(TRIM(storage_key)) > 0",
            name="ck_document_versions_storage_key_nonblank",
        ),
        CheckConstraint(
            "chunk_max_chars IS NULL OR chunk_max_chars > 0",
            name="ck_document_versions_chunk_max_chars_positive",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("documents.id", name="fk_document_versions_document"),
        nullable=False,
    )
    source_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    source_media_type: Mapped[str] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=False
    )
    source_sha256: Mapped[str] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=False
    )
    source_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_key: Mapped[str] = mapped_column(
        String(512, collation="utf8mb4_0900_bin"), nullable=False
    )
    chunk_max_chars: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)


class DocumentProcessingJob(Base):
    """One durable processing attempt for an immutable DocumentVersion."""

    __tablename__ = "document_processing_jobs"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "attempt_no",
            name="uq_doc_processing_version_attempt",
        ),
        Index("ix_doc_processing_status_id", "status", "id"),
        CheckConstraint("attempt_no > 0", name="ck_doc_processing_attempt_positive"),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'INTERRUPTED')",
            name="ck_doc_processing_status_valid",
        ),
        CheckConstraint(
            "stage IS NULL OR stage IN ('VERIFYING_SOURCE', 'PARSING', 'CHUNKING', "
            "'VALIDATING', 'PUBLISHING')",
            name="ck_doc_processing_stage_valid",
        ),
        CheckConstraint(
            "CHAR_LENGTH(TRIM(recipe_id)) > 0",
            name="ck_doc_processing_recipe_nonblank",
        ),
        CheckConstraint(
            "error_code IS NULL OR error_code IN ("
            "'SOURCE_MISSING', 'SOURCE_INTEGRITY_MISMATCH', 'PDF_PROCESS_TIMEOUT', "
            "'PDF_PROCESS_RESOURCE_LIMIT', 'PDF_PROCESS_LAUNCH_FAILED', "
            "'PDF_PROCESS_WORKER_EXITED', 'PDF_PARSE_FAILED', "
            "'PDF_CHUNKING_FAILED', 'PDF_OUTPUT_INVALID', "
            "'SOURCE_CHANGED_DURING_PROCESSING', 'PUBLICATION_FAILED', "
            "'PROCESS_INTERRUPTED')",
            name="ck_doc_processing_error_valid",
        ),
        CheckConstraint(
            "(status = 'PENDING' AND stage IS NULL AND started_at IS NULL "
            "AND finished_at IS NULL AND error_code IS NULL "
            "AND error_message IS NULL) OR "
            "(status = 'RUNNING' AND stage IS NOT NULL AND started_at IS NOT NULL "
            "AND finished_at IS NULL AND error_code IS NULL "
            "AND error_message IS NULL) OR "
            "(status = 'SUCCEEDED' AND stage = 'PUBLISHING' "
            "AND started_at IS NOT NULL AND finished_at IS NOT NULL "
            "AND error_code IS NULL AND error_message IS NULL) OR "
            "(status = 'FAILED' AND stage IS NOT NULL AND started_at IS NOT NULL "
            "AND finished_at IS NOT NULL AND error_code IS NOT NULL "
            "AND error_code <> 'PROCESS_INTERRUPTED') OR "
            "(status = 'INTERRUPTED' AND finished_at IS NOT NULL "
            "AND error_code = 'PROCESS_INTERRUPTED' "
            "AND ((stage IS NULL AND started_at IS NULL) "
            "OR (stage IS NOT NULL AND started_at IS NOT NULL)))",
            name="ck_doc_processing_lifecycle_valid",
        ),
        CheckConstraint(
            "(started_at IS NULL OR started_at >= created_at) AND "
            "(finished_at IS NULL OR finished_at >= created_at) AND "
            "(started_at IS NULL OR finished_at IS NULL "
            "OR finished_at >= started_at)",
            name="ck_doc_processing_timestamp_order",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_version_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "document_versions.id",
            name="fk_doc_processing_document_version",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    attempt_no: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16, collation="utf8mb4_0900_bin"), nullable=False
    )
    stage: Mapped[str | None] = mapped_column(
        String(32, collation="utf8mb4_0900_bin"), nullable=True
    )
    recipe_id: Mapped[str] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=False
    )
    error_code: Mapped[str | None] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6), nullable=True)


class KnowledgeChunk(Base):
    """One current authoritative chunk with typed JSON provenance."""

    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id", "ordinal", name="uq_knowledge_chunks_version_ordinal"
        ),
        CheckConstraint("ordinal > 0", name="ck_knowledge_chunks_ordinal_positive"),
        CheckConstraint(
            "CHAR_LENGTH(content) > 0", name="ck_knowledge_chunks_content_nonempty"
        ),
        CheckConstraint(
            "JSON_TYPE(heading_path) = 'ARRAY'",
            name="ck_knowledge_chunks_heading_path_array",
        ),
        CheckConstraint(
            "JSON_TYPE(source_regions) = 'ARRAY' AND JSON_LENGTH(source_regions) > 0",
            name="ck_knowledge_chunks_source_regions_nonempty_array",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_version_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "document_versions.id",
            name="fk_knowledge_chunks_document_version",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    heading_path: Mapped[list[object]] = mapped_column(JSON, nullable=False)
    source_regions: Mapped[list[object]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)


class Conversation(Base):
    """A user-owned, linearly ordered conversation."""

    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_user", "user_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", name="fk_conversations_user"),
        nullable=False,
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    last_message_at: Mapped[datetime | None] = mapped_column(
        DATETIME(fsp=6), nullable=True
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6), nullable=True)


class ConversationContextSnapshot(Base):
    """One rebuildable current compact-state projection per Conversation."""

    __tablename__ = "conversation_context_snapshots"
    __table_args__ = (
        CheckConstraint(
            "JSON_TYPE(structured_state) = 'OBJECT'",
            name="ck_conversation_context_snapshots_state_object",
        ),
    )

    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "conversations.id",
            name="fk_conversation_context_snapshots_conversation",
            ondelete="RESTRICT",
        ),
        primary_key=True,
    )
    through_message_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "messages.id",
            name="fk_conversation_context_snapshots_through_message",
            ondelete="RESTRICT",
            use_alter=True,
        ),
        nullable=False,
    )
    structured_state: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    compactor_model: Mapped[str] = mapped_column(
        String(255, collation="utf8mb4_0900_bin"), nullable=False
    )
    prompt_version: Mapped[str] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)


class Message(Base):
    """An immutable user-visible message in a conversation."""

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "sequence_no", name="uq_messages_conversation_sequence"
        ),
        UniqueConstraint("run_id", name="uq_messages_run"),
        CheckConstraint("sequence_no > 0", name="ck_messages_sequence_positive"),
        CheckConstraint("role IN ('USER', 'ASSISTANT')", name="ck_messages_role_valid"),
        CheckConstraint(
            "(role = 'USER' AND run_id IS NULL) OR "
            "(role = 'ASSISTANT' AND run_id IS NOT NULL)",
            name="ck_messages_role_run",
        ),
        CheckConstraint(
            "regenerated_from_message_id IS NULL OR role = 'USER'",
            name="ck_messages_regenerated_from_role",
        ),
        CheckConstraint(
            "memory_processed_at IS NULL OR "
            "(role = 'USER' AND regenerated_from_message_id IS NULL)",
            name="ck_messages_memory_processed_canonical",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("conversations.id", name="fk_messages_conversation"),
        nullable=False,
    )
    sequence_no: Mapped[int] = mapped_column(BigInteger, nullable=False)
    role: Mapped[str] = mapped_column(
        String(16, collation="utf8mb4_0900_bin"), nullable=False
    )
    content: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    run_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("runs.id", name="fk_messages_run", use_alter=True),
        nullable=True,
    )
    regenerated_from_message_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("messages.id", name="fk_messages_regenerated_from"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    memory_processed_at: Mapped[datetime | None] = mapped_column(
        DATETIME(fsp=6), nullable=True
    )


class MessageCitation(Base):
    """One durable answer-time evidence/provenance snapshot."""

    __tablename__ = "message_citations"
    __table_args__ = (
        UniqueConstraint(
            "message_id",
            "evidence_handle",
            name="uq_message_citations_message_evidence_handle",
        ),
        CheckConstraint(
            "evidence_handle > 0", name="ck_message_citations_evidence_handle_positive"
        ),
        CheckConstraint(
            "CHAR_LENGTH(TRIM(evidence_text)) > 0",
            name="ck_message_citations_evidence_text_nonblank",
        ),
        CheckConstraint(
            "CHAR_LENGTH(TRIM(source_display_name_snapshot)) > 0",
            name="ck_message_citations_source_display_name_nonblank",
        ),
        CheckConstraint(
            "JSON_TYPE(heading_path_snapshot) = 'ARRAY'",
            name="ck_message_citations_heading_path_snapshot_array",
        ),
        CheckConstraint(
            "JSON_TYPE(source_regions_snapshot) = 'ARRAY' "
            "AND JSON_LENGTH(source_regions_snapshot) > 0",
            name="ck_message_citations_source_regions_snapshot_nonempty_array",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "messages.id", name="fk_message_citations_message", ondelete="RESTRICT"
        ),
        nullable=False,
    )
    document_version_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "document_versions.id",
            name="fk_message_citations_document_version",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    evidence_handle: Mapped[int] = mapped_column(BigInteger, nullable=False)
    evidence_text: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    source_display_name_snapshot: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    heading_path_snapshot: Mapped[list[object]] = mapped_column(JSON, nullable=False)
    source_regions_snapshot: Mapped[list[object]] = mapped_column(JSON, nullable=False)


class Memory(Base):
    """A current, user-owned Personal Context Memory item."""

    __tablename__ = "memories"
    __table_args__ = (Index("ix_memories_user", "user_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", name="fk_memories_user"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    source_message_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("messages.id", name="fk_memories_source_message"),
        nullable=True,
    )
    valid_until: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)


class Run(Base):
    """A persisted answer execution attempt for one user message."""

    __tablename__ = "runs"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "client_request_id",
            name="uq_runs_conversation_client_request",
        ),
        UniqueConstraint(
            "input_message_id", "attempt_no", name="uq_runs_input_attempt"
        ),
        Index("ix_runs_conversation_status", "conversation_id", "status"),
        CheckConstraint("attempt_no > 0", name="ck_runs_attempt_positive"),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED')",
            name="ck_runs_status_valid",
        ),
        CheckConstraint(
            "grounding_policy IN ('AUTO', 'REQUIRED')",
            name="ck_runs_grounding_policy_valid",
        ),
        CheckConstraint(
            "grounding_policy != 'REQUIRED' OR knowledge_base_id IS NOT NULL",
            name="ck_runs_required_grounding_has_knowledge_base",
        ),
        CheckConstraint(
            "(status = 'PENDING' AND started_at IS NULL AND finished_at IS NULL "
            "AND error_code IS NULL) OR "
            "(status = 'RUNNING' AND started_at IS NOT NULL AND finished_at IS NULL "
            "AND error_code IS NULL) OR "
            "(status = 'SUCCEEDED' AND started_at IS NOT NULL "
            "AND finished_at IS NOT NULL AND error_code IS NULL) OR "
            "(status = 'FAILED' AND finished_at IS NOT NULL "
            "AND error_code IS NOT NULL) OR "
            "(status = 'CANCELLED' AND finished_at IS NOT NULL AND error_code IS NULL)",
            name="ck_runs_status_timestamps_error",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("conversations.id", name="fk_runs_conversation"),
        nullable=False,
    )
    input_message_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("messages.id", name="fk_runs_input_message"),
        nullable=False,
    )
    knowledge_base_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey(
            "knowledge_bases.id", name="fk_runs_knowledge_base", ondelete="RESTRICT"
        ),
        nullable=True,
    )
    grounding_policy: Mapped[str] = mapped_column(
        String(16, collation="utf8mb4_0900_bin"),
        nullable=False,
        default="AUTO",
        server_default="AUTO",
    )
    client_request_id: Mapped[str] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=False
    )
    attempt_no: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16, collation="utf8mb4_0900_bin"), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6), nullable=True)
    error_code: Mapped[str | None] = mapped_column(
        String(64, collation="utf8mb4_0900_bin"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
