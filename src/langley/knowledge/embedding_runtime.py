"""Application-local BGE residency shared by every Knowledge embedding path."""

import threading
from math import isfinite
from typing import Any


class KnowledgeEmbeddingError(Exception):
    """One local embedding request failed deterministic validation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code)
        self.code = code
        self.message = message


class KnowledgeEmbeddingRuntime:
    """Keep one SentenceTransformer resident per configured model identity."""

    def __init__(self, *, device: str) -> None:
        self._device = device
        self._lock = threading.Lock()
        self._identity: tuple[str, str, str] | None = None
        self._model: Any | None = None

    def set_device(self, device: str) -> None:
        """Keep the shared runtime aligned with the application setting."""

        with self._lock:
            self._device = device

    def encode_documents(
        self,
        contents: list[str],
        *,
        model: str,
        revision: str,
        dimension: int,
    ) -> list[list[float]]:
        """Encode and L2-normalize document representations under one model lock."""

        with self._lock:
            embedding_model = self._model_for(model, revision)
            values = embedding_model.encode_document(
                contents, convert_to_numpy=True, show_progress_bar=False
            )
        return normalize_embedding_rows(
            values, row_count=len(contents), dimension=dimension
        )

    def encode_query(
        self,
        query: str,
        *,
        model: str,
        revision: str,
        dimension: int,
    ) -> list[float]:
        """Encode and L2-normalize one retrieval query under the same model lock."""

        with self._lock:
            embedding_model = self._model_for(model, revision)
            values = embedding_model.encode_query(
                [query], convert_to_numpy=True, show_progress_bar=False
            )
        return normalize_query_embedding(values, dimension=dimension)

    def _model_for(self, model: str, revision: str) -> Any:
        identity = (model, revision, self._device)
        if self._identity == identity and self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(
            model, revision=revision, device=self._device
        )
        if str(embedding_model.device) != self._device:
            raise KnowledgeEmbeddingError("EMBEDDING_FAILED", "配置的嵌入设备不可用。")
        self._identity = identity
        self._model = embedding_model
        return embedding_model


def normalize_embedding_rows(
    values: object, *, row_count: int, dimension: int
) -> list[list[float]]:
    """Reject malformed document embeddings before any vector-store write."""

    import numpy as np

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape != (row_count, dimension):
        raise KnowledgeEmbeddingError("INVALID_EMBEDDING", "嵌入维度不符合索引配置。")
    if not np.isfinite(matrix).all():
        raise KnowledgeEmbeddingError("INVALID_EMBEDDING", "嵌入包含无效数值。")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise KnowledgeEmbeddingError("INVALID_EMBEDDING", "嵌入向量不能为空。")
    matrix /= norms[:, None]
    return matrix.tolist()


def normalize_query_embedding(values: object, *, dimension: int) -> list[float]:
    """Reject a malformed query vector before dense retrieval."""

    import numpy as np

    matrix = np.asarray(values)
    if matrix.dtype != np.float32 or matrix.shape != (1, dimension):
        raise KnowledgeEmbeddingError("INVALID_EMBEDDING", "嵌入维度不符合索引配置。")
    if not np.isfinite(matrix).all():
        raise KnowledgeEmbeddingError("INVALID_EMBEDDING", "嵌入包含无效数值。")
    norm = float(np.linalg.norm(matrix[0]))
    if not isfinite(norm) or norm <= 0:
        raise KnowledgeEmbeddingError("INVALID_EMBEDDING", "嵌入向量不能为空。")
    normalized = np.asarray(matrix[0] / norm, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise KnowledgeEmbeddingError("INVALID_EMBEDDING", "嵌入包含无效数值。")
    return normalized.tolist()
