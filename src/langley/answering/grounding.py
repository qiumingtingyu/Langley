"""Run-level grounding policy controlled by explicit product state."""

from enum import StrEnum


class GroundingPolicy(StrEnum):
    """Whether knowledge retrieval is optional or a deterministic prerequisite."""

    AUTO = "AUTO"
    REQUIRED = "REQUIRED"
