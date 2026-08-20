"""Fast checks for process-local Memory outcome delivery."""

import pytest

from langley.api.memories import _encode_outcome
from langley.memory_events import MemoryEventSubscribers, MemoryOutcome


def test_memory_event_subscribers_are_user_scoped_and_unsubscribable() -> None:
    subscribers = MemoryEventSubscribers()
    owner_queue = subscribers.subscribe(1)
    other_queue = subscribers.subscribe(2)
    outcome = MemoryOutcome(
        user_id=1,
        conversation_id=3,
        source_message_id=4,
        kind="updated",
        created_count=1,
    )

    subscribers.publish(outcome)

    assert owner_queue.get_nowait() == outcome
    assert other_queue.empty()
    subscribers.unsubscribe(1, owner_queue)
    subscribers.publish(outcome)
    assert owner_queue.empty()


def test_memory_outcome_is_detached_from_memory_content() -> None:
    outcome = MemoryOutcome(
        user_id=1,
        conversation_id=2,
        source_message_id=3,
        kind="no_change",
        user_requested_memory_action=True,
    )

    assert outcome == MemoryOutcome(
        user_id=1,
        conversation_id=2,
        source_message_id=3,
        kind="no_change",
        user_requested_memory_action=True,
    )


@pytest.mark.parametrize("kind", ["updated", "no_change", "retry_pending", "not_saved"])
def test_memory_outcome_sse_encodes_each_frozen_kind(kind: str) -> None:
    encoded = _encode_outcome(
        MemoryOutcome(
            user_id=1,
            conversation_id=2,
            source_message_id=3,
            kind=kind,  # type: ignore[arg-type]
        )
    )

    assert encoded.startswith(f"event: memory.{kind}\n")
    assert '"conversation_id":2' in encoded
