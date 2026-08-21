"""Structural golden regression coverage for Markdown candidate chunks."""

from pathlib import Path

from langley.knowledge.chunking import (
    CandidateChunk,
    ChunkingConfig,
    build_candidate_chunks,
)
from langley.knowledge.contracts import TextSpanRegion
from langley.knowledge.markdown import parse_markdown

_GOLDEN_FIXTURES = (
    Path(__file__).parents[1] / "fixtures" / "knowledge" / "markdown" / "golden"
)


def _build_fixture(name: str, config: ChunkingConfig | None = None):
    source_bytes = (_GOLDEN_FIXTURES / name).read_bytes()
    assert b"\r\n" not in source_bytes
    parsed = parse_markdown(source_bytes)
    return source_bytes, build_candidate_chunks(parsed, config or ChunkingConfig())


def _assert_provenance(
    source_bytes: bytes, candidates: tuple[CandidateChunk, ...]
) -> None:
    assert [candidate.ordinal for candidate in candidates] == list(
        range(1, len(candidates) + 1)
    )
    previous_end = 0
    for candidate in candidates:
        assert len(candidate.source_regions) == 1
        region = candidate.source_regions[0]
        assert isinstance(region, TextSpanRegion)
        assert 0 <= region.start_byte < region.end_byte <= len(source_bytes)
        assert source_bytes[region.start_byte : region.end_byte].decode("utf-8") == (
            candidate.content
        )
        assert previous_end <= region.start_byte
        previous_end = region.end_byte


def _assert_exact_candidates(
    source_bytes: bytes,
    candidates: tuple[CandidateChunk, ...],
    expected: tuple[tuple[int, str, tuple[str, ...], tuple[int, int]], ...],
) -> None:
    _assert_provenance(source_bytes, candidates)
    assert [
        (
            candidate.ordinal,
            candidate.content,
            candidate.heading_path,
            (
                candidate.source_regions[0].start_byte,
                candidate.source_regions[0].end_byte,
            ),
        )
        for candidate in candidates
    ] == list(expected)


def test_normal_structured_exact_golden() -> None:
    source_bytes, candidates = _build_fixture("normal.md")

    _assert_exact_candidates(
        source_bytes,
        candidates,
        (
            (1, "操作系统负责管理计算机资源。\n\n", ("操作系统",), (15, 59)),
            (2, "进程是资源分配的基本单位。\n\n", ("操作系统", "进程"), (69, 110)),
            (
                3,
                "PCB 保存进程运行所需的状态信息。\n\n",
                ("操作系统", "进程", "PCB"),
                (118, 166),
            ),
            (4, "线程是处理器调度的基本单位。\n", ("操作系统", "线程"), (176, 219)),
        ),
    )


def test_rich_markdown_body_exact_golden() -> None:
    source_bytes, candidates = _build_fixture("rich_body.md")

    _assert_exact_candidates(
        source_bytes,
        candidates,
        (
            (1, "工具调用需要保留原始 Markdown 字节。\n\n", ("工具调用",), (15, 66)),
            (
                2,
                "1. **准备** `name`\n"
                "2. 记录参数\n"
                "\n"
                "> 引用说明。\n"
                "\n"
                "```python\n"
                "def run_tool(name, args):\n"
                "    # not a heading\n"
                "    return registry[name](**args)\n"
                "```\n"
                "\n"
                "| 字段 | 说明 |\n"
                "| --- | --- |\n"
                "| name | 工具名称 |\n"
                "\n"
                "正文保持 **bold** 与 `inline code`。\n",
                ("工具调用", "示例"),
                (76, 330),
            ),
        ),
    )


def test_unicode_japanese_exact_golden() -> None:
    source_bytes, candidates = _build_fixture("unicode_ja.md")

    _assert_exact_candidates(
        source_bytes,
        candidates,
        (
            (1, "プロセスは資源管理の単位です。\n\n", ("プロセス",), (15, 62)),
            (
                2,
                "スレッドは実行の単位です。🙂\n\n",
                ("プロセス", "スレッド"),
                (78, 123),
            ),
            (3, "进程与线程的职责不同。\n", ("プロセス", "中文补充"), (139, 173)),
        ),
    )
    assert len(candidates[0].content.encode("utf-8")) != len(candidates[0].content)


def test_oversize_contract_golden() -> None:
    source_bytes, candidates = _build_fixture("oversize.md", ChunkingConfig(20))
    expected_heading_path = ("并发控制",)
    expected_body = (
        "并发控制需要协调事务、锁与共享状态。"
        "合理的边界可以避免竞争条件，也让多个执行步骤在同一份事实基础上稳定协作。\n"
    )

    _assert_provenance(source_bytes, candidates)
    assert len(candidates) > 1
    assert all(0 < len(candidate.content) <= 20 for candidate in candidates)
    assert all(
        candidate.heading_path == expected_heading_path for candidate in candidates
    )
    assert source_bytes[15:178].decode("utf-8") == expected_body
    assert candidates[0].source_regions[0].start_byte == 15
    assert candidates[-1].source_regions[0].end_byte == 178
    assert all(
        previous.source_regions[0].end_byte == current.source_regions[0].start_byte
        for previous, current in zip(candidates, candidates[1:])
    )
    assert "".join(candidate.content for candidate in candidates) == expected_body
