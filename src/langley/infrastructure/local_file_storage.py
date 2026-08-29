"""Safe local persistence for immutable Knowledge source bytes."""

import asyncio
import json
import os
import re
import stat
from hashlib import sha256
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from langley.knowledge.contracts import StoredSource


def process_is_alive(pid: int) -> bool:
    """Return process liveness without waiting for or signalling the process."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    import ctypes
    from ctypes import wintypes

    synchronize = 0x00100000
    wait_object_0 = 0x00000000
    wait_timeout = 0x00000102
    error_access_denied = 5
    error_invalid_parameter = 87

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    wait_for_single_object = kernel32.WaitForSingleObject
    wait_for_single_object.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    wait_for_single_object.restype = wintypes.DWORD
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = open_process(synchronize, False, pid)
    if not handle:
        error_code = ctypes.get_last_error()
        if error_code == error_invalid_parameter:
            return False
        if error_code == error_access_denied:
            return True
        raise ctypes.WinError(error_code)
    try:
        wait_result = wait_for_single_object(handle, 0)
        if wait_result == wait_timeout:
            return True
        if wait_result == wait_object_0:
            return False
        raise ctypes.WinError(ctypes.get_last_error())
    finally:
        close_handle(handle)


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

    def source_path(self, storage_key: str) -> Path:
        """Resolve one validated opaque key for reference-based local processing."""
        return self._path_for_storage_key(storage_key)

    async def prepare_processing_result_path(
        self, job_id: int, attempt_no: int
    ) -> Path:
        """Create one exact job-scoped staging directory and return its result path."""
        result_path = self._processing_result_path(job_id, attempt_no)
        await asyncio.to_thread(result_path.parent.mkdir, parents=True, exist_ok=False)
        return result_path

    async def cleanup_processing_attempt(self, job_id: int, attempt_no: int) -> None:
        """Remove only the known staging files for one completed local execution."""
        result_path = self._processing_result_path(job_id, attempt_no)
        await asyncio.to_thread(self._cleanup_processing_attempt, result_path)

    def processing_worker_marker_path(self) -> Path:
        """Return the contained marker used only to confirm worker process lifetime."""
        return (self._root / "_processing" / "worker.json").resolve()

    async def confirm_no_processing_worker(self) -> None:
        """Fail closed if a prior marked worker PID is still alive."""
        await asyncio.to_thread(
            self._confirm_no_processing_worker, self.processing_worker_marker_path()
        )

    async def cleanup_stale_processing_artifacts(self) -> None:
        """Remove known job staging artifacts while no dispatcher is running."""
        await asyncio.to_thread(self._cleanup_stale_processing_artifacts)

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

    def _processing_result_path(self, job_id: int, attempt_no: int) -> Path:
        if job_id <= 0 or attempt_no <= 0:
            raise ValueError("processing identity must be positive")
        candidate = (
            self._root
            / "_processing"
            / f"job-{job_id}-attempt-{attempt_no}"
            / "result.json"
        ).resolve()
        candidate.relative_to(self._root)
        return candidate

    @staticmethod
    def _cleanup_processing_attempt(result_path: Path) -> None:
        for candidate in (result_path, result_path.with_suffix(".json.tmp")):
            candidate.unlink(missing_ok=True)
        try:
            result_path.parent.rmdir()
        except FileNotFoundError:
            return
        try:
            result_path.parent.parent.rmdir()
        except OSError:
            pass

    @staticmethod
    def _confirm_no_processing_worker(marker_path: Path) -> None:
        temporary_path = marker_path.with_suffix(".json.tmp")
        if not marker_path.exists():
            temporary_path.unlink(missing_ok=True)
            return
        try:
            value = json.loads(marker_path.read_text(encoding="ascii"))
            pid = value["pid"]
            if set(value) != {"pid"} or type(pid) is not int or pid <= 0:
                raise ValueError
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise RuntimeError("invalid PDF worker marker") from error
        if not process_is_alive(pid):
            marker_path.unlink(missing_ok=True)
            temporary_path.unlink(missing_ok=True)
            return
        raise RuntimeError("a prior PDF worker process is still running")

    def _cleanup_stale_processing_artifacts(self) -> None:
        processing_root = self._root / "_processing"
        if not processing_root.is_dir():
            return
        for attempt_directory in processing_root.iterdir():
            if (
                not attempt_directory.is_dir()
                or re.fullmatch(
                    r"job-[1-9][0-9]*-attempt-[1-9][0-9]*", attempt_directory.name
                )
                is None
            ):
                continue
            self._cleanup_processing_attempt(attempt_directory / "result.json")
        try:
            processing_root.rmdir()
        except OSError:
            pass

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
