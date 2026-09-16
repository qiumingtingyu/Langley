"""Bounded supporting-text identity for one activated Skill package."""

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from langley.skill_paths import MAX_SKILL_PATH_CHARS, valid_skill_package_path
from langley.skills import SkillIntegrityError

MAX_SKILL_RESOURCES = 64
MAX_SKILL_RESOURCE_FILE_BYTES = 256 * 1024
MAX_SKILL_RESOURCE_TOTAL_BYTES = 2 * 1024 * 1024
MAX_SKILL_RESOURCE_PATH_CHARS = MAX_SKILL_PATH_CHARS
# Include directories so an arbitrarily large empty tree cannot bypass the cap.
MAX_SKILL_RESOURCE_ENTRIES = 256
_RESOURCE_SUBTREES = ("references",)


@dataclass(frozen=True)
class SkillResourceDescriptor:
    relative_path: str
    file_path: Path
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class SkillResourceSnapshot:
    package_root: Path
    entries: tuple[SkillResourceDescriptor, ...]

    def get(self, path: str) -> SkillResourceDescriptor | None:
        return next(
            (entry for entry in self.entries if entry.relative_path == path), None
        )


def valid_skill_resource_path(path: str) -> bool:
    """Accept exact portable relative paths, without normalizing aliases."""

    parts = path.split("/")
    return (
        valid_skill_package_path(path)
        and len(parts) >= 2
        and parts[0] in _RESOURCE_SUBTREES
    )


def _reject_alias(path: Path) -> None:
    if path.is_symlink() or path.is_junction():
        raise SkillIntegrityError("Skill resource paths must not be links or junctions")


def _check_root(root: Path) -> None:
    _reject_alias(root)
    if root.resolve(strict=True) != root or not root.is_dir():
        raise SkillIntegrityError("Skill resource package identity changed")


def _checked_path(root: Path, relative_path: str) -> Path:
    target = root
    for part in relative_path.split("/"):
        target = target / part
        _reject_alias(target)
    resolved = target.resolve(strict=True)
    if resolved != target or not resolved.is_relative_to(root):
        raise SkillIntegrityError("Skill resource must stay inside its package")
    return resolved


def _resource_bytes(path: Path) -> bytes:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SkillIntegrityError("Skill resources must be regular, unaliased files")
    if info.st_size > MAX_SKILL_RESOURCE_FILE_BYTES:
        raise SkillIntegrityError("Skill resource exceeds 256 KiB")
    with path.open("rb") as stream:
        raw = stream.read(MAX_SKILL_RESOURCE_FILE_BYTES + 1)
    if len(raw) > MAX_SKILL_RESOURCE_FILE_BYTES:
        raise SkillIntegrityError("Skill resource exceeds 256 KiB")
    raw.decode("utf-8")
    return raw


def discover_skill_resources(package_root: Path) -> SkillResourceSnapshot:
    """Freeze selected references after SKILL.md verification.

    Unsupported subtrees and root files are never inspected. Any invalid supported
    resource fails activation; the model must not receive a partial manifest.
    """

    try:
        _check_root(package_root)
        pending: list[Path] = []
        for subtree in _RESOURCE_SUBTREES:
            path = package_root / subtree
            _reject_alias(path)
            try:
                info = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(info.st_mode):
                raise SkillIntegrityError("Skill resource subtree must be a directory")
            pending.append(path)
        files: list[str] = []
        visited = len(pending)
        while pending:
            directory = pending.pop()
            _checked_path(package_root, directory.relative_to(package_root).as_posix())
            with os.scandir(directory) as children:
                for child in children:
                    visited += 1
                    if visited > MAX_SKILL_RESOURCE_ENTRIES:
                        raise SkillIntegrityError(
                            "Skill resource traversal exceeds 256 entries"
                        )
                    path = Path(child.path)
                    relative_path = path.relative_to(package_root).as_posix()
                    if not valid_skill_resource_path(relative_path):
                        raise SkillIntegrityError(
                            "Skill resource path is not canonical"
                        )
                    _reject_alias(path)
                    if child.is_dir(follow_symlinks=False):
                        pending.append(path)
                    else:
                        files.append(relative_path)
                        if len(files) > MAX_SKILL_RESOURCES:
                            raise SkillIntegrityError("Skill resource count exceeds 64")
        entries: list[SkillResourceDescriptor] = []
        total_bytes = 0
        for relative_path in sorted(files):
            path = _checked_path(package_root, relative_path)
            raw = _resource_bytes(path)
            total_bytes += len(raw)
            if total_bytes > MAX_SKILL_RESOURCE_TOTAL_BYTES:
                raise SkillIntegrityError("Skill resource total exceeds 2 MiB")
            entries.append(
                SkillResourceDescriptor(
                    relative_path=relative_path,
                    file_path=path,
                    byte_size=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                )
            )
        return SkillResourceSnapshot(package_root=package_root, entries=tuple(entries))
    except (OSError, RuntimeError, UnicodeError):
        raise SkillIntegrityError(
            "Skill resource configuration is unavailable or invalid"
        ) from None


def read_verified_skill_resource(
    snapshot: SkillResourceSnapshot, descriptor: SkillResourceDescriptor
) -> str:
    """Re-read a frozen descriptor, never discover a model-supplied path."""

    try:
        _check_root(snapshot.package_root)
        path = _checked_path(snapshot.package_root, descriptor.relative_path)
        if path != descriptor.file_path:
            raise SkillIntegrityError("Skill resource path identity changed")
        raw = _resource_bytes(path)
        if (
            len(raw) != descriptor.byte_size
            or hashlib.sha256(raw).hexdigest() != descriptor.sha256
        ):
            raise SkillIntegrityError("Skill resource bytes changed")
        return raw.decode("utf-8")
    except (OSError, RuntimeError, UnicodeError):
        raise SkillIntegrityError(
            "Skill resource bytes are unavailable or invalid"
        ) from None
