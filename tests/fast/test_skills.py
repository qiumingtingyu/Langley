"""Built-in Skill identity/discovery only; no provider or external storage."""

import errno
import hashlib
import os
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from langley.skills import (
    MAX_SKILL_FILE_BYTES,
    MAX_SKILLS_PER_SOURCE,
    SkillConfigurationError,
    SkillRegistry,
    SkillSource,
)


def write_skill(root: Path, name: str = "study-plan", raw: bytes | None = None) -> Path:
    package = root / name
    package.mkdir(parents=True)
    skill_file = package / "SKILL.md"
    if raw is None:
        raw = f"---\nname: {name}\ndescription: Make a plan.\n---\n".encode()
    skill_file.write_bytes(raw)
    return skill_file


def test_minimal_package_exact_hash_and_body_is_not_interpreted(tmp_path):
    raw = (
        b"---\r\nname: study-plan\r\ndescription: Make a plan.\r\n---\r\n"
        b"# Instructions\r\n!!invalid YAML [\r\nprint('not executed')\r\n"
    )
    skill_file = write_skill(tmp_path, raw=raw)
    registry = SkillRegistry(tmp_path)
    entry = registry.get("study-plan")
    assert entry is not None
    assert entry.name == "study-plan"
    assert entry.description == "Make a plan."
    assert entry.source is SkillSource.BUILTIN
    assert entry.package_root == skill_file.parent.resolve()
    assert entry.skill_file == skill_file.resolve()
    assert entry.sha256 == hashlib.sha256(raw).hexdigest()
    assert entry.sha256 != hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    assert registry.get("absent") is None
    assert registry.snapshot().get("study-plan") == entry


def test_optional_yaml_metadata_and_multiline_description(tmp_path):
    raw = b"""---
name: study-plan
description: |-
  First line.
  Second line.
license: MIT
compatibility: Python
metadata:
  defaults: &defaults {key: value}
  overridden: {<<: *defaults, key: override}
unknown: [true, 7, null]
---
"""
    write_skill(tmp_path, raw=raw)
    assert SkillRegistry(tmp_path).get("study-plan").description == (
        "First line.\nSecond line."
    )


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Study-plan",
        "-study",
        "study-",
        "study--plan",
        "study_plan",
        "学习",
        "a" * 65,
    ],
)
def test_invalid_name(tmp_path, name):
    write_skill(
        tmp_path,
        raw=f'---\nname: "{name}"\ndescription: valid\n---\n'.encode(),
    )
    with pytest.raises(SkillConfigurationError, match="name must be"):
        SkillRegistry(tmp_path)


@pytest.mark.parametrize("field", ["", "name: null", "name: 123", "name: [study-plan]"])
def test_missing_or_non_string_name(tmp_path, field):
    write_skill(tmp_path, raw=f"---\n{field}\ndescription: valid\n---\n".encode())
    with pytest.raises(SkillConfigurationError, match="name must be"):
        SkillRegistry(tmp_path)


def test_directory_mismatch_and_effective_name_conflict_fail_closed(tmp_path):
    write_skill(tmp_path, "first")
    write_skill(tmp_path, "second", b"---\nname: first\ndescription: valid\n---\n")
    # Equal directory/name validation makes two valid packages with the same
    # effective name impossible on a real filesystem, even with case folding.
    with pytest.raises(SkillConfigurationError, match="name must equal"):
        SkillRegistry(tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "",
        "description: ''",
        "description: '  '",
        "description: null",
        "description: 123",
        "description: [valid]",
        "description: " + "a" * 1025,
    ],
)
def test_invalid_description(tmp_path, field):
    write_skill(tmp_path, raw=f"---\nname: study-plan\n{field}\n---\n".encode())
    with pytest.raises(SkillConfigurationError, match="description must be"):
        SkillRegistry(tmp_path)


def test_name_and_description_limits_are_inclusive_and_count_characters(tmp_path):
    name = "a" * 64
    description = "学" * 1024
    write_skill(
        tmp_path, name, f"---\nname: {name}\ndescription: {description}\n---".encode()
    )
    assert SkillRegistry(tmp_path).get(name).description == description


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"name: study-plan\ndescription: valid",
        b"\n---\nname: study-plan\n---",
        b"---\nname: study-plan\ndescription: valid",
        b"---\nname: [\n---",
        b"---\n[]\n---",
        b"---\n---",
        b"---\n!!python/object:object {}\n---",
        b"---\nname: study-plan\ndescription: valid\nextra: !!bad secret\n---",
        b"---\nname: study-plan\ndescription: valid\nextra: {[a, b]: value}\n---",
    ],
)
def test_invalid_frontmatter(tmp_path, raw):
    write_skill(tmp_path, raw=raw)
    with pytest.raises(SkillConfigurationError):
        SkillRegistry(tmp_path)


@pytest.mark.parametrize("field", ["name: other", "description: other", "extra: other"])
def test_duplicate_metadata_keys_rejected(tmp_path, field):
    write_skill(
        tmp_path,
        raw=(
            "---\nname: study-plan\ndescription: valid\nextra: valid\n"
            + field
            + "\n---"
        ).encode(),
    )
    with pytest.raises(SkillConfigurationError, match="duplicate YAML"):
        SkillRegistry(tmp_path)


def test_errors_do_not_expose_yaml_or_body(tmp_path):
    secret = "private-content-sentinel"
    write_skill(tmp_path, raw=f"---\nname: [ {secret}\n---\n{secret}".encode())
    with pytest.raises(SkillConfigurationError) as caught:
        SkillRegistry(tmp_path)
    assert secret not in str(caught.value)
    assert caught.value.__suppress_context__
    assert caught.value.__cause__ is None


def test_filesystem_errors_do_not_expose_paths(tmp_path, monkeypatch):
    write_skill(tmp_path)

    def denied(*args, **kwargs):
        raise PermissionError("private-path-sentinel")

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(
        SkillConfigurationError, match="filesystem is unavailable"
    ) as caught:
        SkillRegistry(tmp_path)
    assert "private-path-sentinel" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert caught.value.__cause__ is None


def test_non_utf8_rejected_even_in_body(tmp_path):
    write_skill(tmp_path, raw=b"---\nname: study-plan\ndescription: valid\n---\n\xff")
    with pytest.raises(SkillConfigurationError, match="UTF-8"):
        SkillRegistry(tmp_path)


def test_raw_size_bound_rejects_instead_of_truncating(tmp_path):
    skill_file = write_skill(tmp_path)
    raw = skill_file.read_bytes()
    exact_limit = raw + b"x" * (MAX_SKILL_FILE_BYTES - len(raw))
    skill_file.write_bytes(exact_limit)
    assert (
        SkillRegistry(tmp_path).get("study-plan").sha256
        == hashlib.sha256(exact_limit).hexdigest()
    )
    skill_file.write_bytes(exact_limit + b"x")
    with pytest.raises(SkillConfigurationError, match="exceeds 65536"):
        SkillRegistry(tmp_path)


def test_missing_root_is_empty_and_is_not_created(tmp_path):
    root = tmp_path / "missing"
    assert SkillRegistry(root).snapshot().entries == ()
    assert not root.exists()


def test_root_must_be_a_directory(tmp_path):
    root = tmp_path / "root"
    root.write_bytes(b"file")
    with pytest.raises(SkillConfigurationError, match="root must be a directory"):
        SkillRegistry(root)


def test_only_direct_packages_are_discovered_in_lexical_order(tmp_path):
    root = tmp_path / "builtin"
    for name in ("zebra", "alpha", "middle"):
        write_skill(root, name)
    # Root files, package resources and unrelated Workspace trees are not parsed.
    (root / "SKILL.md").write_bytes(b"invalid")
    write_skill(root / "alpha" / "resources", "nested", b"invalid")
    write_skill(tmp_path / "workspace", "workspace-skill", b"invalid")
    snapshot = SkillRegistry(root).snapshot()
    assert tuple(entry.name for entry in snapshot.entries) == (
        "alpha",
        "middle",
        "zebra",
    )
    assert snapshot.get("workspace-skill") is None
    assert snapshot.get("nested") is None


def test_direct_directory_without_skill_file_fails(tmp_path):
    (tmp_path / "broken").mkdir()
    with pytest.raises(SkillConfigurationError, match="regular SKILL.md"):
        SkillRegistry(tmp_path)


def test_skill_file_must_be_regular(tmp_path):
    (tmp_path / "broken" / "SKILL.md").mkdir(parents=True)
    with pytest.raises(SkillConfigurationError, match="regular SKILL.md"):
        SkillRegistry(tmp_path)


def test_snapshot_and_descriptor_are_frozen_across_reconstruction(tmp_path):
    skill_file = write_skill(tmp_path)
    registry = SkillRegistry(tmp_path)
    before = registry.snapshot()
    entry = before.get("study-plan")
    skill_file.write_bytes(
        skill_file.read_bytes().replace(b"Make a plan.", b"New plan.")
    )
    write_skill(tmp_path, "new-skill")
    after = SkillRegistry(tmp_path).snapshot()
    assert before.entries == (entry,)
    assert before == registry.snapshot()
    assert entry.description == "Make a plan."
    assert entry.sha256 != after.get("study-plan").sha256
    assert after.get("study-plan").description == "New plan."
    assert before.get("new-skill") is None
    with pytest.raises(FrozenInstanceError):
        entry.name = "changed"
    with pytest.raises(FrozenInstanceError):
        before.entries = ()


def test_skill_count_limit_is_inclusive_and_checked_before_package_read(tmp_path):
    for i in range(MAX_SKILLS_PER_SOURCE):
        write_skill(tmp_path, f"skill-{i:02}")
    assert len(SkillRegistry(tmp_path).snapshot().entries) == MAX_SKILLS_PER_SOURCE
    (tmp_path / "overflow").mkdir()
    with pytest.raises(SkillConfigurationError, match="count exceeds 64"):
        SkillRegistry(tmp_path)


def create_symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        if (
            error.errno in {errno.EPERM, errno.EACCES}
            or getattr(error, "winerror", None) == 1314
        ):
            pytest.skip("Host does not grant symlink creation")
        raise


@pytest.mark.parametrize("kind", ["root", "package", "file", "dangling"])
def test_symlinks_cannot_escape_trusted_root(tmp_path, kind):
    root = tmp_path / "builtin"
    outside_file = write_skill(tmp_path / "outside")
    if kind == "root":
        create_symlink(root, outside_file.parent.parent, directory=True)
    else:
        root.mkdir()
        if kind == "package":
            create_symlink(root / "study-plan", outside_file.parent, directory=True)
        else:
            (root / "study-plan").mkdir()
            target = outside_file if kind == "file" else tmp_path / "missing"
            create_symlink(root / "study-plan" / "SKILL.md", target)
    with pytest.raises(SkillConfigurationError, match="symlinks or junctions"):
        SkillRegistry(root)


def test_internal_symlink_is_also_rejected(tmp_path):
    skill_file = write_skill(tmp_path, "original")
    create_symlink(tmp_path / "alias", skill_file.parent, directory=True)
    with pytest.raises(SkillConfigurationError, match="symlinks or junctions"):
        SkillRegistry(tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction boundary")
@pytest.mark.parametrize("kind", ["root", "package"])
def test_windows_junction_cannot_escape_root(tmp_path, kind):
    outside_file = write_skill(tmp_path / "outside")
    root = tmp_path / "builtin"
    link = root
    target = outside_file.parent.parent
    if kind == "package":
        root.mkdir()
        link = root / "study-plan"
        target = outside_file.parent
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=True,
        capture_output=True,
    )
    assert link.is_junction()
    with pytest.raises(SkillConfigurationError, match="symlinks or junctions"):
        SkillRegistry(root)


@pytest.mark.skipif(os.name == "nt", reason="POSIX FIFO boundary")
def test_fifo_is_rejected_without_reading(tmp_path):
    (tmp_path / "study-plan").mkdir()
    os.mkfifo(tmp_path / "study-plan" / "SKILL.md")
    with pytest.raises(SkillConfigurationError, match="regular SKILL.md"):
        SkillRegistry(tmp_path)
