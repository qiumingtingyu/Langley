"""Focused strict contracts for persisted Knowledge source provenance."""

import pytest

from langley.knowledge.contracts import (
    TextSpanRegion,
    decode_source_region,
    encode_source_region,
    validate_heading_path,
    validate_source_regions,
)


def test_text_span_round_trip_and_intrinsic_validation() -> None:
    region = TextSpanRegion(0, 1)
    assert decode_source_region(encode_source_region(region)) == region
    for args in ((-1, 1), (1, 1), (2, 1)):
        with pytest.raises(ValueError):
            TextSpanRegion(*args)
    with pytest.raises(ValueError):
        decode_source_region({"kind": "other", "start_byte": 0, "end_byte": 1})


def test_heading_path_and_source_regions_are_strict_and_ordered() -> None:
    assert validate_heading_path([]) == []
    assert validate_heading_path(["操作系统", "进程", "PCB"]) == [
        "操作系统",
        "进程",
        "PCB",
    ]
    for value in ("not-list", [""], [1]):
        with pytest.raises(ValueError):
            validate_heading_path(value)
    regions = validate_source_regions(
        [
            {"kind": "text_span", "start_byte": 0, "end_byte": 1},
            {"kind": "text_span", "start_byte": 1, "end_byte": 2},
        ]
    )
    assert regions == [TextSpanRegion(0, 1), TextSpanRegion(1, 2)]
    assert validate_source_regions(
        [
            {"kind": "text_span", "start_byte": 0, "end_byte": 10},
            {"kind": "text_span", "start_byte": 20, "end_byte": 30},
        ]
    ) == [TextSpanRegion(0, 10), TextSpanRegion(20, 30)]
    with pytest.raises(ValueError):
        validate_source_regions(
            [
                {"kind": "text_span", "start_byte": 20, "end_byte": 30},
                {"kind": "text_span", "start_byte": 0, "end_byte": 10},
            ]
        )
    for value in (
        [],
        {},
        [{"kind": "text_span", "start_byte": "0", "end_byte": 1}],
        [{"kind": "text_span", "start_byte": 0}],
        [
            {
                "kind": "text_span",
                "start_byte": 0,
                "end_byte": 1,
                "unexpected": "field",
            }
        ],
    ):
        with pytest.raises(ValueError):
            validate_source_regions(value)
