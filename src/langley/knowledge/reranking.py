"""Technology-neutral reranking contract and one local BGE adapter."""

from __future__ import annotations

import asyncio
import math
import re
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

_CUDA_DEVICE = re.compile(r"cuda:[0-9]+")


class Reranker(Protocol):
    """Score detached passages for one query without knowing retrieval storage."""

    async def score(
        self,
        *,
        query: str,
        passages: tuple[str, ...],
    ) -> tuple[float, ...]: ...


class RerankerError(Exception):
    """Fail-closed reranker configuration, loading, inference, or output error."""


def validate_reranker_scores(
    values: object, *, expected_count: int
) -> tuple[float, ...]:
    """Require one finite real float for every passage."""

    if not isinstance(values, tuple) or len(values) != expected_count:
        raise RerankerError("reranker returned an invalid score count")
    if any(
        not isinstance(value, float) or not math.isfinite(value) for value in values
    ):
        raise RerankerError("reranker returned a malformed score")
    return values


class LocalBGEReranker:
    """Lazy, process-local adapter for a local BGE sequence-classification model."""

    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        max_length: int = 2048,
    ) -> None:
        resolved_path = model_path.expanduser().resolve()
        if not resolved_path.is_dir():
            raise RerankerError("configured reranker model path is not a directory")
        if device != "cpu" and _CUDA_DEVICE.fullmatch(device) is None:
            raise RerankerError("reranker device must be 'cpu' or an explicit 'cuda:N'")
        if max_length < 1:
            raise RerankerError("reranker max_length must be positive")

        self._model_path = resolved_path
        self._device = device
        self._max_length = max_length
        self._worker_lock = Lock()
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    async def score(
        self,
        *,
        query: str,
        passages: tuple[str, ...],
    ) -> tuple[float, ...]:
        if not passages:
            return ()
        try:
            values = await asyncio.to_thread(self._score_sync, query, passages)
        except RerankerError:
            raise
        except Exception as error:
            raise RerankerError("local reranker inference failed") from error
        return validate_reranker_scores(values, expected_count=len(passages))

    def _score_sync(self, query: str, passages: tuple[str, ...]) -> tuple[float, ...]:
        # Cancellation cannot terminate a worker thread. Holding this lock for both
        # load and inference keeps a cancelled worker serialized with later calls.
        with self._worker_lock:
            self._load_once()
            assert self._tokenizer is not None
            assert self._model is not None
            assert self._torch is not None

            encoded = self._tokenizer(
                [[query, passage] for passage in passages],
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            device_inputs = {
                name: tensor.to(self._device) for name, tensor in encoded.items()
            }
            with self._torch.inference_mode():
                logits = self._model(**device_inputs, return_dict=True).logits
            return tuple(
                float(value) for value in logits.reshape(-1).float().cpu().tolist()
            )

    def _load_once(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as error:
            raise RerankerError(
                "local reranker dependencies are unavailable"
            ) from error

        dtype = torch.float32
        if self._device.startswith("cuda:"):
            device_index = int(self._device.partition(":")[2])
            if (
                not torch.cuda.is_available()
                or device_index >= torch.cuda.device_count()
            ):
                raise RerankerError("configured reranker CUDA device is unavailable")
            dtype = torch.float16

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(self._model_path),
                local_files_only=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                str(self._model_path),
                local_files_only=True,
                dtype=dtype,
            )
            model.to(self._device)
            model.eval()
        except Exception as error:
            raise RerankerError("local reranker model loading failed") from error

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
