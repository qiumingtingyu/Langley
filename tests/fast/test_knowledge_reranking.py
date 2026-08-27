"""Focused R1 reranking invariants without loading real model weights."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

import langley.knowledge.retrieval as retrieval
import langley.knowledge.retrieval_service as retrieval_service
import langley.main as main
from langley.knowledge.index_build import DenseSearchHit
from langley.knowledge.reranking import LocalBGEReranker, RerankerError
from langley.knowledge.retrieval import (
    ActiveRetrievalContext,
    RetrievalGenerationChangedError,
    RetrievalIndexInconsistentError,
    RetrievalResult,
    _AuthoritativeChunk,
    retrieve_reranked,
)
from langley.knowledge.retrieval_service import (
    KnowledgeRetrievalService,
    KnowledgeSearchError,
)
from langley.settings import Settings


def _context() -> ActiveRetrievalContext:
    return ActiveRetrievalContext(
        knowledge_base_id=4,
        user_id=7,
        generation_id="11111111-1111-4111-8111-111111111111",
        model="active-model",
        revision="a" * 40,
        dimension=2,
        representation="content_only",
        chunk_snapshot_sha256="b" * 64,
    )


def _chunk(chunk_id: int, *, fresh: bool = False) -> _AuthoritativeChunk:
    return _AuthoritativeChunk(
        knowledge_chunk_id=chunk_id,
        chunk_ordinal=chunk_id,
        content=f"{'fresh' if fresh else 'content'}-{chunk_id}",
        heading_path=("Heading",),
        source_regions=({"kind": "text", "start_byte": 0, "end_byte": 1},),
        document_id=100 + chunk_id,
        document_version_id=200 + chunk_id,
        source_display_name=f"source-{chunk_id}.md",
        source_sha256="c" * 64,
    )


class _FakeRuntime:
    def __init__(self, search_hits: tuple[DenseSearchHit, ...]) -> None:
        self.search_hits = search_hits
        self.search_top_ks: list[int] = []

    async def encode_query(self, query: str, **kwargs: object) -> list[float]:
        del query, kwargs
        return [0.6, 0.8]

    async def search_dense(
        self, vector: list[float], **kwargs: object
    ) -> tuple[DenseSearchHit, ...]:
        del vector
        self.search_top_ks.append(kwargs["top_k"])  # type: ignore[arg-type]
        return self.search_hits


class _FakeReranker:
    def __init__(self, scores: tuple[float, ...] | Exception) -> None:
        self.scores = scores
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def score(
        self, *, query: str, passages: tuple[str, ...]
    ) -> tuple[float, ...]:
        self.calls.append((query, passages))
        if isinstance(self.scores, Exception):
            raise self.scores
        return self.scores


def _patch_reads(
    monkeypatch: pytest.MonkeyPatch,
    *,
    final_error: Exception | None = None,
) -> list[tuple[int, ...]]:
    calls: list[tuple[int, ...]] = []

    async def initial(*args: object, **kwargs: object) -> ActiveRetrievalContext:
        del args, kwargs
        return _context()

    async def final(
        *args: object, returned_chunk_ids: tuple[int, ...], **kwargs: object
    ) -> tuple[_AuthoritativeChunk, ...]:
        del args, kwargs
        calls.append(returned_chunk_ids)
        if len(calls) == 2 and final_error is not None:
            raise final_error
        return tuple(
            _chunk(chunk_id, fresh=len(calls) == 2) for chunk_id in returned_chunk_ids
        )

    monkeypatch.setattr(retrieval, "_read_active_context", initial)
    monkeypatch.setattr(retrieval, "_final_revalidate", final)
    return calls


def test_rerank_preserves_dense_provenance_and_uses_deterministic_tie_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_reads(monkeypatch)
    runtime = _FakeRuntime(
        tuple(
            DenseSearchHit(chunk_id, dense_score)
            for chunk_id, dense_score in zip(
                range(10, 16), (0.99, 0.89, 0.79, 0.69, 0.59, 0.49), strict=True
            )
        )
    )
    reranker = _FakeReranker((0.1, 5.0, 5.0, 2.0, 4.0, 3.0))

    result = asyncio.run(
        retrieve_reranked(
            None,  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
            reranker,
            user_id=7,
            knowledge_base_id=4,
            query="exact query",
            candidate_k=20,
            top_k=5,
        )
    )

    assert runtime.search_top_ks == [20]
    assert reranker.calls == [
        ("exact query", tuple(f"content-{chunk_id}" for chunk_id in range(10, 16)))
    ]
    assert calls == [tuple(range(10, 16)), (11, 12, 14, 15, 13)]
    assert [
        (
            hit.rank,
            hit.retrieval_rank,
            hit.knowledge_chunk_id,
            hit.score,
            hit.rerank_score,
            hit.content,
        )
        for hit in result.hits
    ] == [
        (1, 2, 11, 0.89, 5.0, "fresh-11"),
        (2, 3, 12, 0.79, 5.0, "fresh-12"),
        (3, 5, 14, 0.59, 4.0, "fresh-14"),
        (4, 6, 15, 0.49, 3.0, "fresh-15"),
        (5, 4, 13, 0.69, 2.0, "fresh-13"),
    ]


@pytest.mark.parametrize(
    "scores",
    [
        (0.1,),
        (float("nan"), 0.2),
        (float("inf"), 0.2),
    ],
)
def test_malformed_reranker_output_fails_closed_before_final_selection(
    monkeypatch: pytest.MonkeyPatch, scores: tuple[float, ...]
) -> None:
    calls = _patch_reads(monkeypatch)
    runtime = _FakeRuntime((DenseSearchHit(10, 0.9), DenseSearchHit(11, 0.8)))

    with pytest.raises(RerankerError):
        asyncio.run(
            retrieve_reranked(
                None,  # type: ignore[arg-type]
                runtime,  # type: ignore[arg-type]
                _FakeReranker(scores),
                user_id=7,
                knowledge_base_id=4,
                query="query",
                candidate_k=20,
                top_k=2,
            )
        )

    assert calls == [(10, 11)]


@pytest.mark.parametrize(
    "final_error",
    [RetrievalGenerationChangedError(), RetrievalIndexInconsistentError()],
)
def test_active_context_change_across_rerank_boundary_returns_no_stale_result(
    monkeypatch: pytest.MonkeyPatch, final_error: Exception
) -> None:
    calls = _patch_reads(monkeypatch, final_error=final_error)
    runtime = _FakeRuntime((DenseSearchHit(10, 0.9), DenseSearchHit(11, 0.8)))
    reranker = _FakeReranker((2.0, 1.0))

    with pytest.raises(type(final_error)):
        asyncio.run(
            retrieve_reranked(
                None,  # type: ignore[arg-type]
                runtime,  # type: ignore[arg-type]
                reranker,
                user_id=7,
                knowledge_base_id=4,
                query="query",
                candidate_k=20,
                top_k=1,
            )
        )

    assert calls == [(10, 11), (10,)]
    assert len(reranker.calls) == 1


@pytest.mark.anyio
async def test_disabled_service_keeps_the_dense_path_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = RetrievalResult(4, "generation", ())
    calls: list[dict[str, object]] = []

    async def dense(*args: object, **kwargs: object) -> RetrievalResult:
        del args
        calls.append(dict(kwargs))
        return expected

    async def reranked(*args: object, **kwargs: object) -> RetrievalResult:
        del args, kwargs
        raise AssertionError("disabled service must not call reranked retrieval")

    monkeypatch.setattr(retrieval_service, "retrieve_dense", dense)
    monkeypatch.setattr(retrieval_service, "retrieve_reranked", reranked)

    result = await KnowledgeRetrievalService(None, None).search(  # type: ignore[arg-type]
        user_id=7,
        knowledge_base_id=4,
        query="query",
        top_k=5,
    )

    assert result is expected
    assert calls == [
        {"user_id": 7, "knowledge_base_id": 4, "query": "query", "top_k": 5}
    ]


@pytest.mark.anyio
async def test_enabled_service_over_retrieves_and_maps_reranker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reranker = _FakeReranker(())
    calls: list[dict[str, object]] = []

    async def reranked(*args: object, **kwargs: object) -> RetrievalResult:
        del args
        calls.append(dict(kwargs))
        raise RerankerError("model failed")

    monkeypatch.setattr(retrieval_service, "retrieve_reranked", reranked)
    service = KnowledgeRetrievalService(  # type: ignore[arg-type]
        None,
        None,
        reranker=reranker,
        reranker_candidate_k=20,
    )

    with pytest.raises(KnowledgeSearchError) as raised:
        await service.search(
            user_id=7,
            knowledge_base_id=4,
            query="query",
            top_k=5,
        )

    assert raised.value.code == "KNOWLEDGE_SEARCH_UNAVAILABLE"
    assert raised.value.retryable is False
    assert calls == [
        {
            "user_id": 7,
            "knowledge_base_id": 4,
            "query": "query",
            "candidate_k": 20,
            "top_k": 5,
        }
    ]


def test_empty_passages_do_not_enter_the_model_worker(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reranker = LocalBGEReranker(model_path=tmp_path, device="cpu")

    def unexpected(*args: object) -> tuple[float, ...]:
        del args
        raise AssertionError("empty candidates must not invoke model work")

    monkeypatch.setattr(reranker, "_score_sync", unexpected)
    assert asyncio.run(reranker.score(query="query", passages=())) == ()


def test_disabled_composition_does_not_construct_a_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("disabled reranking must not construct the adapter")

    monkeypatch.setattr(main, "LocalBGEReranker", unexpected)
    assert main._reranker_for(Settings()) is None


def test_enabled_composition_requires_a_model_path() -> None:
    with pytest.raises(ValueError, match="model_path is required"):
        main._reranker_for(Settings(knowledge_reranking_enabled=True))


def test_workflow_factory_reuses_one_unloaded_reranker_per_application(
    tmp_path,
) -> None:
    factory = main._workflow_factory_for(
        Settings(
            knowledge_reranking_enabled=True,
            knowledge_reranker_model_path=tmp_path,
            knowledge_reranker_device="cpu",
        ),
        main._UnavailableProvider(),
        None,
        None,
        None,
    )

    first = factory()
    second = factory()

    assert first._retrieval_service is second._retrieval_service
    assert first._retrieval_service is not None
    assert first._retrieval_service._reranker is second._retrieval_service._reranker
    assert first._retrieval_service._reranker._model is None


@pytest.mark.parametrize(
    ("device", "expected_dtype"),
    [("cpu", "float32"), ("cuda:0", "float16")],
)
def test_local_bge_loads_and_transfers_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected_dtype: str,
) -> None:
    events: list[tuple[object, ...]] = []
    batch_size = 0

    class FakeTensor:
        def to(self, target: str) -> FakeTensor:
            events.append(("tensor.to", target))
            return self

    class FakeLogits:
        def reshape(self, value: int) -> FakeLogits:
            assert value == -1
            return self

        def float(self) -> FakeLogits:
            return self

        def cpu(self) -> FakeLogits:
            return self

        def tolist(self) -> list[float]:
            return [float(index) for index in range(batch_size)]

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> FakeTokenizer:
            events.append(("tokenizer.load", path, kwargs))
            return cls()

        def __call__(self, pairs: list[list[str]], **kwargs: object):
            nonlocal batch_size
            batch_size = len(pairs)
            events.append(("tokenize", tuple(tuple(pair) for pair in pairs), kwargs))
            return {"input_ids": FakeTensor()}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> FakeModel:
            events.append(("model.load", path, kwargs))
            return cls()

        def to(self, target: str) -> None:
            events.append(("model.to", target))

        def eval(self) -> None:
            events.append(("model.eval",))

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            events.append(("infer", kwargs))
            return SimpleNamespace(logits=FakeLogits())

    class InferenceMode:
        def __enter__(self) -> None:
            events.append(("inference.enter",))

        def __exit__(self, *args: object) -> None:
            del args
            events.append(("inference.exit",))

    fake_torch = SimpleNamespace(
        float32="float32",
        float16="float16",
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        inference_mode=InferenceMode,
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForSequenceClassification=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    reranker = LocalBGEReranker(model_path=tmp_path, device=device)
    first = asyncio.run(reranker.score(query="q1", passages=("p1", "p2")))
    second = asyncio.run(reranker.score(query="q2", passages=("p3",)))

    assert first == (0.0, 1.0)
    assert second == (0.0,)
    assert sum(event[0] == "tokenizer.load" for event in events) == 1
    assert sum(event[0] == "model.load" for event in events) == 1
    assert sum(event[0] == "model.to" for event in events) == 1
    assert sum(event[0] == "model.eval" for event in events) == 1
    model_load = next(event for event in events if event[0] == "model.load")
    assert model_load[2] == {"local_files_only": True, "dtype": expected_dtype}


def test_unavailable_cuda_fails_without_cpu_fallback(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_calls: list[str] = []

    class Loader:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            del args, kwargs
            load_calls.append("called")
            return object()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32="float32",
            float16="float16",
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=Loader,
            AutoModelForSequenceClassification=Loader,
        ),
    )

    reranker = LocalBGEReranker(model_path=tmp_path, device="cuda:0")
    with pytest.raises(RerankerError, match="CUDA device is unavailable"):
        asyncio.run(reranker.score(query="query", passages=("passage",)))

    assert load_calls == []
