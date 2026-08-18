"""Small process-local state for one live answer."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

StreamItem = tuple[str, dict[str, Any]]


@dataclass
class ActiveAnswer:
    """The transient state needed while one answer is running."""

    task: asyncio.Task[None] | None = None
    partial_text: str = ""
    streams: set[asyncio.Queue[StreamItem | None]] = field(default_factory=set)
    closed: bool = False
