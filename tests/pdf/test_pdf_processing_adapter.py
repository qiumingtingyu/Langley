"""Real Docling/Hybrid path over a tiny controlled text PDF."""

import asyncio
from hashlib import sha256
from pathlib import Path

from langley.knowledge.document_processing import PDF_PROCESSING_RECIPE_ID
from langley.knowledge.pdf_processing import PersistentPdfWorker
from langley.knowledge.pdf_processing_result import load_pdf_processing_result

FIXTURE = Path("tests/fixtures/knowledge/pdf/text_pdf_fixture.pdf")


def test_controlled_pdf_reaches_candidate_compatible_staging(tmp_path: Path) -> None:
    async def run() -> None:
        source_bytes = FIXTURE.read_bytes()
        source_sha256 = sha256(source_bytes).hexdigest()
        result_paths = (tmp_path / "result-1.json", tmp_path / "result-2.json")
        stages: list[str] = []

        async def chunking() -> None:
            stages.append("CHUNKING")

        worker = PersistentPdfWorker(
            tokenizer_id="BAAI/bge-m3",
            tokenizer_revision="5617a9f61b028005a4858fdac845db406aefb181",
            worker_marker_path=tmp_path / "worker.json",
        )
        completions = []
        try:
            for offset, result_path in enumerate(result_paths):
                job_id = 7 + offset
                document_version_id = 11 + offset
                completions.append(
                    await worker.process(
                        {
                            "command": "process",
                            "job_id": job_id,
                            "attempt_no": 1,
                            "document_version_id": document_version_id,
                            "source_path": str(FIXTURE.resolve()),
                            "source_sha256": source_sha256,
                            "source_size_bytes": len(source_bytes),
                            "recipe_id": PDF_PROCESSING_RECIPE_ID,
                            "staging_path": str(result_path),
                        },
                        timeout_seconds=600,
                        on_chunking=chunking,
                    )
                )
                result = load_pdf_processing_result(
                    result_path,
                    expected_recipe_id=PDF_PROCESSING_RECIPE_ID,
                    expected_job_id=job_id,
                    expected_attempt_no=1,
                    expected_document_version_id=document_version_id,
                    expected_source_sha256=source_sha256,
                    expected_source_size_bytes=len(source_bytes),
                )
                combined = "\n".join(
                    candidate.content for candidate in result.candidates
                )
                assert result.page_count == 1
                assert "Hybrid chunking preserves raw evidence text." in combined
                assert all(candidate.source_regions for candidate in result.candidates)
                assert all(
                    1 <= region.page_start <= region.page_end <= 1
                    for candidate in result.candidates
                    for region in candidate.source_regions
                )
                assert any(candidate.heading_path for candidate in result.candidates)
        finally:
            await worker.stop()
        assert stages == ["CHUNKING", "CHUNKING"]
        assert completions[0].pid == completions[1].pid
        assert completions[1].total_ms < completions[0].total_ms

    asyncio.run(run())
