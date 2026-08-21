"""Safe local persistence for immutable Knowledge source bytes."""

import asyncio
import os
import stat
from hashlib import sha256
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from langley.knowledge.contracts import StoredSource


class InvalidStorageKeyError(ValueError):
    """Raised when an opaque source key is not a generated local source key."""


class LocalFileStorage:
    """Store exact source bytes under generated, root-contained locations."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    async def store_source(self, user_id: int, source_bytes: bytes) -> StoredSource:
        """Finalize source bytes before returning detached integrity facts."""
        return await asyncio.to_thread(self._store_source, user_id, source_bytes)

    async def read_source(self, storage_key: str) -> bytes:
        """Read exact bytes for a valid generated source key."""
        source_path = self._path_for_storage_key(storage_key)
        return await asyncio.to_thread(source_path.read_bytes)

    async def cleanup_partial_sources(self) -> None:
        """Remove managed partial files without deleting finalized sources."""
        await asyncio.to_thread(self._cleanup_partial_sources)

    async def list_finalized_source_keys(self) -> frozenset[str]:
        """Return generated finalized-source keys without consulting MySQL."""
        return await asyncio.to_thread(self._list_finalized_source_keys)

    def _store_source(self, user_id: int, source_bytes: bytes) -> StoredSource:
        if user_id <= 0:
            raise ValueError("user_id must be positive")

        storage_identity = uuid4().hex
        storage_key = f"users/{user_id}/sources/{storage_identity}/source"
        final_path = self._path_for_storage_key(storage_key)
        source_directory = final_path.parent
        partial_path = source_directory / "source.part"
        digest = sha256(source_bytes).hexdigest()

        try:
            source_directory.mkdir(parents=True, exist_ok=False)
            with partial_path.open("xb") as source_file:
                source_file.write(source_bytes)
                source_file.flush()
                os.fsync(source_file.fileno())
            os.replace(partial_path, final_path)
        except Exception:
            self._remove_partial_file(partial_path)
            raise

        return StoredSource(
            storage_key=storage_key,
            sha256=digest,
            size_bytes=len(source_bytes),
        )

    def _path_for_storage_key(self, storage_key: str) -> Path:
        if "\\" in storage_key:
            raise InvalidStorageKeyError("storage key must use POSIX separators")

        key_path = PurePosixPath(storage_key)
        parts = key_path.parts
        if (
            key_path.is_absolute()
            or len(parts) != 5
            or parts[0] != "users"
            or parts[2] != "sources"
            or parts[4] != "source"
        ):
            raise InvalidStorageKeyError("storage key has an invalid structure")

        user_id, storage_identity = parts[1], parts[3]
        if not user_id.isdecimal() or str(int(user_id)) != user_id or int(user_id) <= 0:
            raise InvalidStorageKeyError("storage key has an invalid user component")
        try:
            parsed_identity = UUID(hex=storage_identity)
        except ValueError as error:
            raise InvalidStorageKeyError(
                "storage key has an invalid generated identity"
            ) from error
        if parsed_identity.hex != storage_identity:
            raise InvalidStorageKeyError(
                "storage key has an invalid generated identity"
            )

        candidate = (self._root / Path(*parts)).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError as error:
            raise InvalidStorageKeyError(
                "storage key escapes the storage root"
            ) from error
        return candidate

    def _cleanup_partial_sources(self) -> None:
        for source_directory in self._generated_source_directories():
            final_path = source_directory / "source"
            partial_path = source_directory / "source.part"
            if partial_path.is_file():
                partial_path.unlink()
            if not final_path.exists() and not any(source_directory.iterdir()):
                source_directory.rmdir()

    def _list_finalized_source_keys(self) -> frozenset[str]:
        finalized_keys: set[str] = set()
        for source_directory in self._generated_source_directories():
            final_path = source_directory / "source"
            if self._managed_finalized_source(final_path):
                finalized_keys.add(final_path.relative_to(self._root).as_posix())
        return frozenset(finalized_keys)

    def _generated_source_directories(self):
        users_directory = self._root / "users"
        if not users_directory.is_dir():
            return
        for user_directory in users_directory.iterdir():
            if not user_directory.is_dir() or not _is_canonical_positive_integer(
                user_directory.name
            ):
                continue
            sources_directory = user_directory / "sources"
            if not sources_directory.is_dir():
                continue
            for source_directory in sources_directory.iterdir():
                if not source_directory.is_dir() or not _is_uuid_hex(
                    source_directory.name
                ):
                    continue
                managed_directory = self._managed_source_directory(source_directory)
                if managed_directory is not None:
                    yield managed_directory

    def _managed_source_directory(self, candidate: Path) -> Path | None:
        if _is_redirected_path(candidate):
            return None
        try:
            resolved_candidate = candidate.resolve()
            resolved_candidate.relative_to(self._root)
        except OSError:
            return None
        except ValueError:
            return None
        return resolved_candidate if resolved_candidate.is_dir() else None

    def _managed_finalized_source(self, candidate: Path) -> bool:
        if _is_redirected_path(candidate):
            return False
        try:
            resolved_candidate = candidate.resolve()
            resolved_candidate.relative_to(self._root)
        except OSError:
            return False
        except ValueError:
            return False
        return resolved_candidate.is_file()

    @staticmethod
    def _remove_partial_file(partial_path: Path) -> None:
        try:
            partial_path.unlink(missing_ok=True)
        except OSError:
            pass


def _is_canonical_positive_integer(value: str) -> bool:
    return value.isdecimal() and str(int(value)) == value and int(value) > 0


def _is_uuid_hex(value: str) -> bool:
    try:
        return UUID(hex=value).hex == value
    except ValueError:
        return False


def _is_redirected_path(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return True
    return path.is_symlink() or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
