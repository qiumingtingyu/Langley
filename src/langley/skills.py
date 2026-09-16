"""Controlled procedure discovery; no activation or execution authority.

The caller supplies application-owned roots, never Workspace or upload paths.
Builtins are deployment-stable; published user packages are discovered for each
new snapshot. Snapshots freeze descriptors, not on-disk package bytes.
"""

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml
from yaml.nodes import MappingNode

MAX_SKILLS_PER_SOURCE = 64
MAX_SKILL_FILE_BYTES = 64 * 1024
SKILL_NAME_PATTERN = r"[a-z0-9]+(?:-[a-z0-9]+)*"


class SkillConfigurationError(ValueError):
    """Invalid application-owned Skill configuration, with content-free reasons."""


class SkillIntegrityError(ValueError):
    """A snapshotted package can no longer supply its approved bytes."""


class SkillSource(StrEnum):
    BUILTIN = "BUILTIN"
    USER_INSTALLED = "USER_INSTALLED"


@dataclass(frozen=True)
class SkillDescriptor:
    """Server-side identity. Filesystem paths must not be projected to a model."""

    name: str
    description: str
    source: SkillSource
    package_root: Path
    skill_file: Path
    sha256: str


@dataclass(frozen=True)
class SkillSnapshot:
    """Immutable, lexically ordered view for a future Run; contains no file body."""

    entries: tuple[SkillDescriptor, ...]

    def get(self, name: str) -> SkillDescriptor | None:
        return next((entry for entry in self.entries if entry.name == name), None)


class SkillRegistry:
    """Merge two controlled sources into one immutable execution inventory.

    A missing root is empty. Every direct directory must be a valid package;
    ordinary root files are ignored. Links/junctions at the root, direct entry,
    or SKILL.md are rejected, even if their destination stays within the root.
    Ancestors of the configured root are resolved as application configuration.
    No watcher, recursive search, resource reads, or import-time IO occurs.
    """

    def __init__(self, builtin_root: Path, user_root: Path | None = None) -> None:
        try:
            if user_root is not None:
                builtin = builtin_root.resolve()
                user = user_root.resolve()
                if builtin.is_relative_to(user) or user.is_relative_to(builtin):
                    raise SkillConfigurationError("Skill roots must not overlap")
            self._entries = _discover_packages(builtin_root, SkillSource.BUILTIN)
            self._user_root = user_root
            self.snapshot()  # Validate the configured publication root at startup.
        except (OSError, RuntimeError):
            # Filesystem exceptions can contain absolute paths; do not propagate
            # their diagnostics or cause into configuration errors.
            raise SkillConfigurationError("Skill filesystem is unavailable") from None

    def snapshot(self) -> SkillSnapshot:
        try:
            users = (
                ()
                if self._user_root is None
                else _discover_packages(self._user_root, SkillSource.USER_INSTALLED)
            )
            entries = self._entries + users
            if len({entry.name for entry in entries}) != len(entries):
                raise SkillConfigurationError("duplicate Skill name across sources")
            return SkillSnapshot(entries=tuple(sorted(entries, key=lambda e: e.name)))
        except (OSError, RuntimeError):
            raise SkillConfigurationError("Skill filesystem is unavailable") from None

    def get(self, name: str) -> SkillDescriptor | None:
        return self.snapshot().get(name)


def _reject_link(path: Path) -> None:
    if path.is_symlink() or path.is_junction():
        raise SkillConfigurationError("Skill paths must not be symlinks or junctions")


def _discover_packages(
    configured_root: Path, source: SkillSource
) -> tuple[SkillDescriptor, ...]:
    _reject_link(configured_root)
    try:
        root = configured_root.resolve(strict=True)
    except FileNotFoundError:
        return ()
    if not root.is_dir():
        raise SkillConfigurationError("Skill root must be a directory")

    # Bound the directory list before sorting/reading any package. Do not collect
    # arbitrarily many directory entries just to reject them after discovery.
    packages: list[Path] = []
    with os.scandir(root) as children:
        for child in children:
            path = Path(child.path)
            _reject_link(path)
            if child.is_dir(follow_symlinks=False):
                packages.append(path)
                if len(packages) > MAX_SKILLS_PER_SOURCE:
                    raise SkillConfigurationError("Skill source count exceeds 64")

    entries: list[SkillDescriptor] = []
    names: set[str] = set()
    for package in sorted(packages, key=lambda path: path.name):
        entry = inspect_skill_package(root, package, source=source)
        if entry.name in names:
            raise SkillConfigurationError("duplicate Skill name")
        names.add(entry.name)
        entries.append(entry)
    return tuple(entries)


def inspect_skill_package(
    root: Path, package: Path, *, source: SkillSource = SkillSource.BUILTIN
) -> SkillDescriptor:
    """Inspect one package with the shared SKILL.md contract, without activation."""

    package_root, skill_file, raw = _read_skill_bytes(root, package)
    name, description = _parse_metadata(raw, package.name)
    return SkillDescriptor(
        name=name,
        description=description,
        source=source,
        package_root=package_root,
        skill_file=skill_file,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _read_skill_bytes(root: Path, package: Path) -> tuple[Path, Path, bytes]:
    _reject_link(package)
    package_root = package.resolve(strict=True)
    if package_root.parent != root:
        raise SkillConfigurationError("Skill package must be a direct child of root")
    skill_file = package / "SKILL.md"
    _reject_link(skill_file)
    if not skill_file.is_file():
        raise SkillConfigurationError("Skill package must contain a regular SKILL.md")
    info = skill_file.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SkillConfigurationError("SKILL.md must be a regular, unaliased file")
    skill_file = skill_file.resolve(strict=True)
    if skill_file.parent != package_root or not skill_file.is_relative_to(root):
        raise SkillConfigurationError("SKILL.md must stay inside its package and root")

    with skill_file.open("rb") as stream:
        raw = stream.read(MAX_SKILL_FILE_BYTES + 1)
    if len(raw) > MAX_SKILL_FILE_BYTES:
        raise SkillConfigurationError("SKILL.md exceeds 65536 raw bytes")
    return package_root, skill_file, raw


def read_verified_skill_body(descriptor: SkillDescriptor) -> str:
    """Re-read only a frozen identity; never discover or activate new metadata.

    The returned text is detached from disk. Later rounds use these verified
    bytes even if the deployment changes after activation.
    """

    try:
        root = descriptor.package_root.parent
        _reject_link(root)
        if root.resolve() != root:
            raise SkillIntegrityError("Skill package identity changed")
        package_root, skill_file, raw = _read_skill_bytes(root, descriptor.package_root)
        if (
            package_root != descriptor.package_root
            or skill_file != descriptor.skill_file
            or hashlib.sha256(raw).hexdigest() != descriptor.sha256
        ):
            raise SkillIntegrityError("Skill package identity changed")
        _, body = _split_frontmatter(raw)
        return body
    except (SkillConfigurationError, OSError, RuntimeError):
        raise SkillIntegrityError(
            "Skill package bytes are unavailable or invalid"
        ) from None


class _SkillLoader(yaml.SafeLoader):
    """Safe YAML, additionally rejecting ambiguous duplicate mapping keys."""

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[object, object]:
        # SafeLoader handles tags, aliases, and key structure. Keep YAML merge
        # support, including explicit overrides, but reject duplicate local keys.
        keys: set[object] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in keys:
                    raise SkillConfigurationError("duplicate YAML metadata key")
                keys.add(key)
            except TypeError:
                raise SkillConfigurationError(
                    "YAML metadata keys must be scalar"
                ) from None
        return super().construct_mapping(node, deep=deep)


def _split_frontmatter(raw: bytes) -> tuple[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise SkillConfigurationError("SKILL.md must be UTF-8") from None
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise SkillConfigurationError("SKILL.md must start with YAML frontmatter")
    end = next(
        (i for i in range(1, len(lines)) if lines[i].rstrip("\r\n") == "---"),
        None,
    )
    if end is None:
        raise SkillConfigurationError("YAML frontmatter closing delimiter is missing")
    return "".join(lines[1:end]), "".join(lines[end + 1 :])


def _parse_metadata(raw: bytes, directory_name: str) -> tuple[str, str]:
    frontmatter, _ = _split_frontmatter(raw)
    try:
        metadata = yaml.load(frontmatter, Loader=_SkillLoader)
    except SkillConfigurationError:
        raise
    except (yaml.YAMLError, ValueError, RecursionError):
        # PyYAML's exception includes source excerpts, which must stay private.
        raise SkillConfigurationError("invalid or unsafe YAML frontmatter") from None
    if not isinstance(metadata, dict):
        raise SkillConfigurationError("YAML frontmatter must be a mapping")
    name = metadata.get("name")
    if (
        not isinstance(name, str)
        or not 1 <= len(name) <= 64
        or re.fullmatch(SKILL_NAME_PATTERN, name) is None
    ):
        raise SkillConfigurationError(
            "name must be 1..64 lowercase ASCII letters/digits "
            "with single inner hyphens"
        )
    if name != directory_name:
        raise SkillConfigurationError("name must equal the Skill directory name")
    description = metadata.get("description")
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > 1024
    ):
        raise SkillConfigurationError(
            "description must be a nonblank string of <=1024 chars"
        )
    return name, description
