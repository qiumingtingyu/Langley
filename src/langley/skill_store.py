"""Bounded private copying and atomic publication of user procedure packages.

Only complete validated directories enter user/. Staging is never discoverable.
Published packages have no update or removal API; existing snapshots retain paths.
"""

import os
import re
import shutil
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import structlog

from langley.settings import Settings
from langley.skill_paths import MAX_SKILL_PATH_CHARS, valid_skill_package_path
from langley.skill_resources import discover_skill_resources
from langley.skills import (
    SkillConfigurationError,
    SkillDescriptor,
    SkillIntegrityError,
    SkillSource,
    inspect_skill_package,
)

MAX_USER_SKILL_PACKAGE_FILES = 256
MAX_USER_SKILL_PACKAGE_FILE_BYTES = 4 * 1024 * 1024
MAX_USER_SKILL_PACKAGE_TOTAL_BYTES = 8 * 1024 * 1024
MAX_USER_SKILL_PACKAGE_PATH_CHARS = MAX_SKILL_PATH_CHARS
MAX_USER_SKILL_PACKAGE_ENTRIES = 512
logger = structlog.get_logger(__name__)


def _best_effort_cleanup(
    cleanup: Callable[[str], None],
    key: str,
) -> None:
    """Cleanup and its diagnostics cannot replace the authoritative outcome."""

    try:
        cleanup(key)
    except Exception:
        try:
            # Fixed event/category only: no exception text, traceback or identity.
            logger.warning("skill_install_cleanup_failed", target="staging")
        except Exception:
            # A broken diagnostic sink is also non-authoritative.
            pass


class SkillInstallError(ValueError):
    """Bounded domain code only; never include paths, YAML or SQL diagnostics."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class SkillStaging:
    """Server-side transport handle, never an API/model response."""

    key: str
    root: Path


@dataclass(frozen=True)
class PreparedSkill:
    staging_key: str
    descriptor: SkillDescriptor


def _unaliased(path: Path) -> None:
    if path.is_symlink() or path.is_junction():
        raise SkillInstallError("INVALID_SKILL_PACKAGE")


def _directory(path: Path) -> None:
    _unaliased(path)
    if path.resolve(strict=True) != path or not path.is_dir():
        raise SkillInstallError("INVALID_SKILL_PACKAGE")


def _read_file(path: Path) -> bytes:
    _unaliased(path)
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SkillInstallError("INVALID_SKILL_PACKAGE")
    if info.st_size > MAX_USER_SKILL_PACKAGE_FILE_BYTES:
        raise SkillInstallError("SKILL_PACKAGE_TOO_LARGE")
    # A file opened for copying must still be the regular, unaliased file checked
    # above. O_NOFOLLOW also closes the final-component link race on POSIX.
    fd = os.open(
        path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)
        ):
            raise SkillInstallError("INVALID_SKILL_PACKAGE")
        raw = stream.read(MAX_USER_SKILL_PACKAGE_FILE_BYTES + 1)
    if len(raw) > MAX_USER_SKILL_PACKAGE_FILE_BYTES:
        raise SkillInstallError("SKILL_PACKAGE_TOO_LARGE")
    return raw


def _copy_package(source: Path, destination: Path) -> None:
    """Copy the entire bounded tree without trusting paths or source handles."""

    _directory(source)
    destination.mkdir()
    pending = [source]
    files = total = visited = 0
    while pending:
        directory = pending.pop()
        _directory(directory)
        with os.scandir(directory) as children:
            for child in children:
                visited += 1
                if visited > MAX_USER_SKILL_PACKAGE_ENTRIES:
                    raise SkillInstallError("SKILL_PACKAGE_TOO_LARGE")
                path = Path(child.path)
                relative = path.relative_to(source).as_posix()
                if not valid_skill_package_path(relative):
                    raise SkillInstallError("INVALID_SKILL_PACKAGE")
                _unaliased(path)
                if path.resolve(strict=True) != path or not path.is_relative_to(source):
                    raise SkillInstallError("INVALID_SKILL_PACKAGE")
                target = destination / relative
                if child.is_dir(follow_symlinks=False):
                    target.mkdir()
                    pending.append(path)
                    continue
                if files >= MAX_USER_SKILL_PACKAGE_FILES:
                    raise SkillInstallError("SKILL_PACKAGE_TOO_LARGE")
                raw = _read_file(path)
                total += len(raw)
                if total > MAX_USER_SKILL_PACKAGE_TOTAL_BYTES:
                    raise SkillInstallError("SKILL_PACKAGE_TOO_LARGE")
                with target.open("xb") as stream:
                    stream.write(raw)
                files += 1


class SkillStore:
    def __init__(self, settings: Settings) -> None:
        configured = settings.skill_storage_root.absolute()
        _unaliased(configured)
        # Resolve application configuration once; never discover packages via CWD.
        self.root = configured.resolve()
        self.user_root = self.root / "user"
        # Neither model-writable Workspace storage nor builtin deployment may
        # overlap the admission area (including its private staging directory).
        builtin = settings.builtin_skill_root.resolve()
        workspace = settings.workspace_storage_root.resolve()
        for left, right in (
            (self.root, builtin),
            (self.root, workspace),
            (builtin, workspace),
        ):
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise SkillInstallError("INVALID_SKILL_ROOTS")

    def _base(self, kind: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        _directory(self.root)
        base = self.root / kind
        _unaliased(base)
        base.mkdir(exist_ok=True)
        _directory(base)
        return base

    def _key_root(self, kind: str, key: str) -> Path:
        if re.fullmatch(r"[a-f0-9]{32}", key) is None:
            raise SkillInstallError("INVALID_SKILL_PACKAGE")
        base = self._base(kind)
        root = base / key
        _unaliased(root)
        if root.resolve() != root or root.parent != base:
            raise SkillInstallError("INVALID_SKILL_PACKAGE")
        return root

    def allocate_staging(self) -> SkillStaging:
        key = uuid4().hex
        root = self._key_root("staging", key)
        root.mkdir()  # exclusive; even a generated-key collision cannot overwrite
        return SkillStaging(key=key, root=root)

    @contextmanager
    def publication_lock(self) -> Iterator[None]:
        """Serialize only inventory checks + rename across local installers.

        OS locks release on close/process exit. The small persistent lock file is
        outside user/ and is not an installation record or a stale lock marker.
        Package copying/validation happens before taking this lock.
        """

        self._base("staging")
        path = self.root / ".publication.lock"
        _unaliased(path)
        if path.exists():
            info = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise SkillInstallError("INVALID_SKILL_PACKAGE")
        with path.open("a+b") as stream:
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield  # Closing this handle releases the lock, also on exceptions.

    def discard_staging(self, key: str) -> None:
        self._remove_key("staging", key)

    def _remove_key(self, kind: str, key: str) -> None:
        # Resolve/check the exact absolute target before recursive removal.
        root = self._key_root(kind, key)
        if root.exists():
            _directory(root)
            shutil.rmtree(root)

    def prepare(self, source: Path) -> PreparedSkill:
        """Validate a private copy, so source handles cannot mutate published bytes.

        The local caller must finish source writes before installation. A failed
        preparation removes its partial private copy, never partially publishes.
        """

        private: SkillStaging | None = None
        prepared = False
        try:
            _unaliased(source)
            source = source.resolve(strict=True)
            _directory(source)
            if source.is_relative_to(self.root) or self.root.is_relative_to(source):
                raise SkillInstallError("INVALID_SKILL_PACKAGE")
            if not valid_skill_package_path(source.name):
                raise SkillInstallError("INVALID_SKILL_PACKAGE")
            private = self.allocate_staging()
            copied = private.root / source.name
            _copy_package(source, copied)
            descriptor = inspect_skill_package(
                private.root, copied, source=SkillSource.USER_INSTALLED
            )
            discover_skill_resources(copied)
            prepared = True
            return PreparedSkill(private.key, descriptor)
        except (SkillConfigurationError, SkillIntegrityError, OSError, RuntimeError):
            raise SkillInstallError("INVALID_SKILL_PACKAGE") from None
        finally:
            if private is not None and not prepared:
                _best_effort_cleanup(self.discard_staging, private.key)

    def publish(self, prepared: PreparedSkill) -> SkillDescriptor:
        """Atomically publish a complete package, never replace an installed one.

        Every admitted directory contains SKILL.md and is nonempty. Directory
        rename cannot replace a competing admitted package on Windows or POSIX.
        No discoverable empty directory is reserved before the rename.
        """

        source = self._key_root("staging", prepared.staging_key)
        package = source / prepared.descriptor.name
        if package != prepared.descriptor.package_root:
            raise SkillInstallError("INVALID_SKILL_PACKAGE")
        _directory(package)
        final = self._base("user") / prepared.descriptor.name
        if os.path.lexists(final):
            raise SkillInstallError("SKILL_NAME_CONFLICT")
        try:
            os.rename(package, final)
        except OSError:
            if os.path.lexists(final):
                raise SkillInstallError("SKILL_NAME_CONFLICT") from None
            raise SkillInstallError("SKILL_INSTALL_FAILED") from None
        # No fallible filesystem work after the publication commit point.
        return SkillDescriptor(
            name=prepared.descriptor.name,
            description=prepared.descriptor.description,
            source=SkillSource.USER_INSTALLED,
            package_root=final,
            skill_file=final / "SKILL.md",
            sha256=prepared.descriptor.sha256,
        )
