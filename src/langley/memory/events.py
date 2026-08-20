"""Small process-local delivery for conversational Memory outcomes."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MemoryOutcome:
    """One detached post-commit outcome safe for best-effort presentation."""

    user_id: int
    conversation_id: int
    source_message_id: int
    kind: Literal["updated", "no_change", "retry_pending", "not_saved"]
    user_requested_memory_action: bool | None = None
    created_count: int = 0
    changed_count: int = 0
    forgotten_count: int = 0


class MemoryEventSubscribers:
    """Own only live subscriber queues; Memory durability remains in MySQL."""

    def __init__(self) -> None:
        self._queues: dict[int, set[asyncio.Queue[MemoryOutcome]]] = defaultdict(set)

    def subscribe(self, user_id: int) -> asyncio.Queue[MemoryOutcome]:
        queue: asyncio.Queue[MemoryOutcome] = asyncio.Queue()
        self._queues[user_id].add(queue)
        return queue

    def unsubscribe(self, user_id: int, queue: asyncio.Queue[MemoryOutcome]) -> None:
        queues = self._queues.get(user_id)
        if queues is None:
            return
        queues.discard(queue)
        if not queues:
            del self._queues[user_id]

    def publish(self, outcome: MemoryOutcome) -> None:
        for queue in tuple(self._queues.get(outcome.user_id, ())):
            queue.put_nowait(outcome)
