"""Explicit offline entrypoint for the Slice 6 Retrieval Golden Eval.

Run preflight without model activity:
``uv run --group eval python -m langley.knowledge.retrieval_eval_entrypoint preflight``.
The ``experiment`` subcommand remains explicitly gated and must only be used
after the Human Gate recorded in the current handoff is approved.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from langley.knowledge.retrieval_eval import (
    EmbeddingBoundary,
    EmbeddingMetadata,
    RetrievalEvalError,
    SentenceTransformersEmbedding,
    corpus_preflight,
    load_golden_corpus,
    normalize_embeddings,
    render_result_json,
    render_result_markdown,
    run_evaluation,
)

_REPOSITORY_ROOT = Path(__file__).parents[3]
_DEFAULT_DATASET_ROOT = (
    _REPOSITORY_ROOT / "tests" / "fixtures" / "knowledge" / "retrieval"
)
_MODEL_ID = "BAAI/bge-m3"
_IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}")
_TINY_QUERIES = (
    "What validates the operational query embedding path?",
    "Which facts belong in a tiny document embedding check?",
)
_TINY_DOCUMENTS = (
    "This document is only for the operational embedding preflight.",
    "The preflight validates finite, normalized float32 vectors.",
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _code_identity() -> tuple[str, bool, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPOSITORY_ROOT, text=True
    ).strip()
    status_lines = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=_REPOSITORY_ROOT, text=True
    ).splitlines()
    tracked_dirty = any(not line.startswith("??") for line in status_lines)
    return commit, tracked_dirty, bool(status_lines)


def _run_preflight(dataset_root: Path, output: Path) -> None:
    corpus = load_golden_corpus(dataset_root)
    preflight = corpus_preflight(corpus)
    payload: dict[str, object] = {
        "task": "Slice 6 Task 4.6 Real Corpus Preflight",
        "dataset": corpus.dataset_identity,
        "dataset_manifest_sha256": corpus.dataset_manifest_sha256,
        "chunking_config": {"max_chunk_chars": corpus.chunking_config.max_chunk_chars},
        "model_activity": "none",
        **asdict(preflight),
    }
    _write_json(output, payload)
    print(
        "retrieval-corpus-preflight: "
        f"documents={preflight.document_count} "
        f"candidates={preflight.candidate_chunk_count} "
        f"matchable={preflight.matchable_case_count}/{preflight.approved_case_count} "
        f"no_matchable={len(preflight.no_matchable_case_ids)}"
    )


def _resolve_model_revision() -> str:
    """Resolve the public model's immutable Hugging Face commit before loading it."""

    from huggingface_hub import HfApi

    return _require_immutable_model_revision(HfApi().model_info(_MODEL_ID).sha)


def _require_immutable_model_revision(value: object) -> str:
    """Fail closed unless a supplied revision is an immutable Hugging Face SHA."""

    if not isinstance(value, str) or _IMMUTABLE_REVISION.fullmatch(value) is None:
        raise RetrievalEvalError(
            "model revision must be an immutable 40-character lowercase hexadecimal SHA"
        )
    return value


def _operational_vector_facts(embedder: EmbeddingBoundary) -> dict[str, object]:
    """Exercise only the two frozen embedding roles with intentionally tiny inputs."""

    query_vectors = embedder.embed_queries(_TINY_QUERIES)
    document_vectors = embedder.embed_documents(_TINY_DOCUMENTS)
    if (
        str(query_vectors.dtype) != "float32"
        or str(document_vectors.dtype) != "float32"
    ):
        raise RetrievalEvalError("operational preflight embeddings must be float32")

    normalized_queries = normalize_embeddings(
        query_vectors, expected_rows=len(_TINY_QUERIES), role="query"
    )
    normalized_documents = normalize_embeddings(
        document_vectors, expected_rows=len(_TINY_DOCUMENTS), role="document"
    )
    observed_dimension = normalized_queries.shape[1]
    if normalized_documents.shape[1] != observed_dimension:
        raise RetrievalEvalError("query and document embedding dimensions differ")
    return {
        "observed_dimension": observed_dimension,
        "dtype": str(normalized_queries.dtype),
        "query_row_count": normalized_queries.shape[0],
        "document_row_count": normalized_documents.shape[0],
        "finite": True,
        "non_zero_norm": True,
        "normalized_norm_check": "pass",
    }


def _run_model_preflight(output: Path, *, device: str | None) -> None:
    """Run the authorized Task 4.7a path; never load the Golden corpus or rank it."""

    import sentence_transformers

    resolved_model_revision = _resolve_model_revision()
    embedder = SentenceTransformersEmbedding(
        model_id=_MODEL_ID, revision=resolved_model_revision, device=device
    )
    vector_facts = _operational_vector_facts(embedder)
    code_commit, tracked_worktree_dirty, worktree_dirty = _code_identity()
    payload: dict[str, object] = {
        "task": "Slice 6 Task 4.7a BGE-M3 Operational Preflight",
        "model_activity": "operational_preflight_only",
        "model_id": _MODEL_ID,
        "resolved_model_revision": resolved_model_revision,
        "sentence_transformers_version": sentence_transformers.__version__,
        "device": embedder.observed_device,
        "code_commit": code_commit,
        "tracked_worktree_dirty": tracked_worktree_dirty,
        "worktree_dirty": worktree_dirty,
        **vector_facts,
    }
    _write_json(output, payload)
    print(
        "retrieval-model-preflight: "
        f"model={_MODEL_ID}@{resolved_model_revision} "
        f"dimension={vector_facts['observed_dimension']} "
        f"device={embedder.observed_device}"
    )


def _run_experiment(
    dataset_root: Path,
    output_json: Path,
    output_markdown: Path,
    *,
    model_revision: str,
    device: str | None,
) -> None:
    """Run Experiment #0 only after the caller has passed the Human Gate."""

    model_revision = _require_immutable_model_revision(model_revision)
    import sentence_transformers

    corpus = load_golden_corpus(dataset_root)
    code_commit, tracked_worktree_dirty, worktree_dirty = _code_identity()
    embedder = SentenceTransformersEmbedding(
        model_id=_MODEL_ID, revision=model_revision, device=device
    )
    provisional_metadata = EmbeddingMetadata(
        model_id=embedder.model_id,
        model_revision=model_revision,
        sentence_transformers_version=sentence_transformers.__version__,
        device=device or "auto",
    )
    result = run_evaluation(
        corpus,
        embedder,
        code_commit=code_commit,
        tracked_worktree_dirty=tracked_worktree_dirty,
        worktree_dirty=worktree_dirty,
        embedding_metadata=provisional_metadata,
    )
    result = replace(
        result,
        embedding_metadata=replace(
            provisional_metadata, device=embedder.observed_device
        ),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(render_result_json(result), encoding="utf-8")
    output_markdown.write_text(render_result_markdown(result), encoding="utf-8")
    print(
        "retrieval-experiment: "
        f"Hit@1={result.hit_at_1.value:.6f} "
        f"Hit@3={result.hit_at_3.value:.6f} "
        f"Hit@5={result.hit_at_5.value:.6f} "
        f"MRR={result.mean_reciprocal_rank:.6f}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline Langley Retrieval Eval"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    preflight = subcommands.add_parser("preflight")
    preflight.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET_ROOT)
    preflight.add_argument("--output", type=Path, required=True)
    model_preflight = subcommands.add_parser("model-preflight")
    model_preflight.add_argument("--output", type=Path, required=True)
    model_preflight.add_argument("--device")
    model_preflight.add_argument(
        "--human-gate-approved",
        action="store_true",
        help=(
            "required after explicit Task 4.7a authorization; "
            "this flag does not grant approval"
        ),
    )
    experiment = subcommands.add_parser("experiment")
    experiment.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET_ROOT)
    experiment.add_argument("--output-json", type=Path, required=True)
    experiment.add_argument("--output-markdown", type=Path, required=True)
    experiment.add_argument("--model-revision", required=True)
    experiment.add_argument("--device")
    experiment.add_argument(
        "--human-gate-approved",
        action="store_true",
        help=(
            "required after the separate Human Gate; this flag does not grant approval"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "preflight":
        _run_preflight(args.dataset_root, args.output)
        return
    if args.command == "model-preflight":
        if not args.human_gate_approved:
            raise SystemExit(
                "model-preflight requires --human-gate-approved "
                "after explicit Human approval"
            )
        _run_model_preflight(args.output, device=args.device)
        return
    if not args.human_gate_approved:
        raise SystemExit(
            "experiment requires --human-gate-approved after explicit Human approval"
        )
    _run_experiment(
        args.dataset_root,
        args.output_json,
        args.output_markdown,
        model_revision=args.model_revision,
        device=args.device,
    )


if __name__ == "__main__":
    main()
