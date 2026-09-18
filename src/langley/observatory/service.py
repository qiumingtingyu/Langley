"""Async coordination for the synchronous local Observatory read model."""

import asyncio
from collections.abc import Callable
from typing import TypeVar

from langley.observatory.store import ObservatoryStore

_Result = TypeVar("_Result")


class ObservatoryUnavailableError(RuntimeError):
    """Raised when the optional local diagnostic projection is unavailable."""


class ObservatoryRuntime:
    """Serialize refresh/read snapshots and keep blocking work off the event loop."""

    def __init__(
        self,
        store: ObservatoryStore | None = None,
        *,
        store_factory: Callable[[], ObservatoryStore] | None = None,
    ) -> None:
        self.store = store
        self._store_factory = store_factory
        self._refresh_lock = asyncio.Lock()

    async def read(self, operation: Callable[[ObservatoryStore], _Result]) -> _Result:
        async with self._refresh_lock:
            return await asyncio.to_thread(self._refresh_and_read, operation)

    def _refresh_and_read(
        self, operation: Callable[[ObservatoryStore], _Result]
    ) -> _Result:
        if self.store is None:
            if self._store_factory is None:
                raise ObservatoryUnavailableError("Observatory store is unavailable")
            try:
                self.store = self._store_factory()
            except Exception as error:
                raise ObservatoryUnavailableError(
                    "Observatory initialization is unavailable"
                ) from error
        try:
            self.store.refresh()
        except Exception as error:
            raise ObservatoryUnavailableError(
                "Observatory refresh is unavailable"
            ) from error
        return operation(self.store)
