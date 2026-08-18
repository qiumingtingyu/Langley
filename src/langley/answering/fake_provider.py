"""Deterministic scripted implementation of the normalized provider contract."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from langley.answering.contracts import LLMRequest, LLMStreamEvent


@dataclass(frozen=True)
class ScriptedProviderRound:
    """Exactly one deterministic provider round for a test scenario."""

    events: tuple[LLMStreamEvent, ...]
    failure: BaseException | None = None
    started: asyncio.Event | None = None
    blocked_until: asyncio.Event | None = None
    blocked_after_event_count: int | None = None
    additional_blocked_after_events: tuple[tuple[int, asyncio.Event], ...] = ()
    event_reached: asyncio.Event | None = None
    event_reached_after_count: int | None = None


class FakeProvider:
    """Consume one explicit script entry per normalized provider call."""

    def __init__(self, rounds: list[ScriptedProviderRound]) -> None:
        self._rounds = list(rounds)
        self.requests: list[LLMRequest] = []

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """Capture one request, then yield its next scripted round exactly once."""

        return self._stream(request)

    async def _stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.requests.append(request)
        if not self._rounds:
            raise AssertionError("FakeProvider script exhausted")

        round_ = self._rounds.pop(0)
        if round_.started is not None:
            round_.started.set()
        if round_.blocked_until is not None:
            if round_.blocked_after_event_count is None:
                await round_.blocked_until.wait()
        for event_count, event in enumerate(round_.events, start=1):
            yield event
            if (
                round_.event_reached is not None
                and round_.event_reached_after_count == event_count
            ):
                round_.event_reached.set()
            if (
                round_.blocked_until is not None
                and round_.blocked_after_event_count == event_count
            ):
                await round_.blocked_until.wait()
            for blocked_after_count, gate in round_.additional_blocked_after_events:
                if blocked_after_count == event_count:
                    await gate.wait()
        if round_.failure is not None:
            raise round_.failure
