"""Filesystem-only admission: real copying/publication and bounded failures."""

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

import langley.skill_store as module
from langley.settings import Settings
from langley.skill_installation import install_skill
from langley.skill_store import SkillInstallError, SkillStore
from langley.skills import (
    SkillConfigurationError,
    SkillRegistry,
    SkillSource,
    read_verified_skill_body,
)


def package(root, name="example", files=None):
    target = root / name
    target.mkdir(parents=True)
    (target / "SKILL.md").write_bytes(
        (
            f"---\r\nname: {name}\r\ndescription: Example procedure\r\n"
            "---\r\nDo work.\r\n"
        ).encode()
    )
    for relative, raw in (files or {}).items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return target


@pytest.fixture
def settings(tmp_path):
    return Settings(
        skill_storage_root=tmp_path / "skills",
        builtin_skill_root=tmp_path / "builtin",
        workspace_storage_root=tmp_path / "workspaces",
    )


def test_private_copy_publication_and_new_snapshot(settings, tmp_path):
    files = {
        "references/week.md": b"reference\r\n",
        "scripts/never.py": b"raise RuntimeError('must not execute')",
        "assets/binary.dat": b"\x00\xff",
    }
    source = package(tmp_path / "input", files=files)
    store = SkillStore(settings)
    registry = SkillRegistry(settings.builtin_skill_root, store.user_root)
    before = registry.snapshot()
    prepared = store.prepare(source)
    (source / "SKILL.md").write_bytes(b"changed source")
    (source / "assets/binary.dat").write_bytes(b"changed asset")
    assert registry.snapshot().entries == ()
    installed = store.publish(prepared)
    store.discard_staging(prepared.staging_key)
    assert before.entries == ()
    descriptor = registry.snapshot().get("example")
    assert descriptor == installed and descriptor.source is SkillSource.USER_INSTALLED
    assert (
        descriptor.sha256
        == hashlib.sha256(descriptor.skill_file.read_bytes()).hexdigest()
    )
    assert read_verified_skill_body(descriptor) == "Do work.\r\n"
    for path, raw in files.items():
        assert (descriptor.package_root / path).read_bytes() == raw


@pytest.mark.parametrize(
    "invalid",
    ["metadata", "name", "skill_utf8", "resource_utf8", "resource_size", "missing"],
)
def test_invalid_package_never_becomes_discoverable(settings, tmp_path, invalid):
    source = package(tmp_path / "input")
    if invalid == "metadata":
        (source / "SKILL.md").write_bytes(
            b"---\nname: example\nname: example\ndescription: okay\n---\n"
        )
    elif invalid == "name":
        (source / "SKILL.md").write_bytes(b"---\nname: other\ndescription: okay\n---\n")
    elif invalid == "skill_utf8":
        (source / "SKILL.md").write_bytes(b"\xff")
    elif invalid.startswith("resource"):
        (source / "references").mkdir()
        (source / "references/a").write_bytes(
            b"\xff" if invalid == "resource_utf8" else b"a" * (256 * 1024 + 1)
        )
    else:
        (source / "SKILL.md").unlink()
    with pytest.raises(SkillInstallError, match="^INVALID_SKILL_PACKAGE$"):
        install_skill(source, settings=settings)
    assert (
        SkillRegistry(settings.builtin_skill_root, settings.skill_storage_root / "user")
        .snapshot()
        .entries
        == ()
    )
    assert list((settings.skill_storage_root / "staging").iterdir()) == []


@pytest.mark.parametrize("bound", ["file", "total", "count", "entries"])
def test_package_bounds_are_inclusive(settings, tmp_path, monkeypatch, bound):
    source = package(tmp_path / "input")
    if bound == "file":
        monkeypatch.setattr(module, "MAX_USER_SKILL_PACKAGE_FILE_BYTES", 100)
        (source / "asset").write_bytes(b"x" * 100)
    elif bound == "total":
        monkeypatch.setattr(
            module,
            "MAX_USER_SKILL_PACKAGE_TOTAL_BYTES",
            (source / "SKILL.md").stat().st_size + 10,
        )
        (source / "asset").write_bytes(b"x" * 10)
    elif bound == "count":
        monkeypatch.setattr(module, "MAX_USER_SKILL_PACKAGE_FILES", 2)
        (source / "asset").touch()
    else:
        monkeypatch.setattr(module, "MAX_USER_SKILL_PACKAGE_ENTRIES", 2)
        (source / "empty").mkdir()
    store = SkillStore(settings)
    prepared = store.prepare(source)
    store.discard_staging(prepared.staging_key)
    if bound in ("file", "total"):
        with (source / "asset").open("ab") as stream:
            stream.write(b"x")
    else:
        (source / "overflow").mkdir() if bound == "entries" else (
            source / "overflow"
        ).touch()
    with pytest.raises(SkillInstallError, match="^SKILL_PACKAGE_TOO_LARGE$"):
        install_skill(source, settings=settings)
    assert not store.user_root.exists()


@pytest.mark.parametrize(
    "kind", ["hardlink", "file_symlink", "dir_symlink", "junction", "fifo"]
)
def test_aliases_and_special_files_are_rejected(settings, tmp_path, kind):
    source = package(tmp_path / "input")
    target = tmp_path / "outside"
    target.mkdir()
    (target / "data").write_bytes(b"private")
    alias = source / "alias"
    if kind == "hardlink":
        os.link(target / "data", alias)
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("POSIX FIFO unavailable")
        os.mkfifo(alias)
    elif kind == "junction":
        if os.name != "nt":
            pytest.skip("Windows junction only")
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            check=True,
            capture_output=True,
        )
    else:
        directory = kind == "dir_symlink"
        try:
            alias.symlink_to(
                target if directory else target / "data", target_is_directory=directory
            )
        except OSError as error:
            if getattr(error, "winerror", None) == 1314 or error.errno in (1, 13):
                pytest.skip("Symlink creation requires platform privilege")
            raise
    with pytest.raises(SkillInstallError, match="^INVALID_SKILL_PACKAGE$"):
        install_skill(source, settings=settings)
    assert not (settings.skill_storage_root / "user").exists()


@pytest.mark.parametrize("collision", ["builtin", "user"])
def test_name_collision_rejects_without_overwriting(settings, tmp_path, collision):
    source = package(tmp_path / "input")
    if collision == "builtin":
        existing = package(settings.builtin_skill_root)
    else:
        existing = install_skill(source, settings=settings).package_root
    original = (existing / "SKILL.md").read_bytes()
    with pytest.raises(SkillInstallError, match="^SKILL_NAME_CONFLICT$"):
        install_skill(source, settings=settings)
    assert (existing / "SKILL.md").read_bytes() == original
    assert list((settings.skill_storage_root / "staging").iterdir()) == []


def test_later_builtin_collision_fails_discovery(settings, tmp_path):
    source = package(tmp_path / "input")
    install_skill(source, settings=settings)
    package(settings.builtin_skill_root)
    with pytest.raises(SkillConfigurationError, match="duplicate Skill name"):
        SkillRegistry(settings.builtin_skill_root, settings.skill_storage_root / "user")


@pytest.mark.parametrize("collision", ["name", "capacity"])
def test_concurrent_publication_has_one_winner(
    settings, tmp_path, monkeypatch, collision
):
    import langley.skill_installation as installation

    first = package(tmp_path / "input")
    second = first if collision == "name" else package(tmp_path / "input", "other")
    monkeypatch.setattr(installation, "MAX_SKILLS_PER_SOURCE", 1)
    barrier = Barrier(2)
    copy = module._copy_package

    def racing_copy(source, destination):
        copy(source, destination)
        barrier.wait(timeout=10)

    monkeypatch.setattr(module, "_copy_package", racing_copy)

    def install(source):
        try:
            return install_skill(source, settings=settings)
        except SkillInstallError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(install, (first, second)))
    code = "SKILL_NAME_CONFLICT" if collision == "name" else "SKILL_COUNT_LIMIT"
    assert results.count(code) == 1
    entries = (
        SkillRegistry(settings.builtin_skill_root, settings.skill_storage_root / "user")
        .snapshot()
        .entries
    )
    assert len(entries) == 1 and read_verified_skill_body(entries[0]) == "Do work.\r\n"


@pytest.mark.parametrize("failure", ["copy", "publish"])
def test_failure_and_broken_cleanup_never_publish_partial_package(
    settings, tmp_path, monkeypatch, failure
):
    source = package(tmp_path / "input")

    def fail(*args):
        raise OSError("private path details")

    def broken_logger(*args, **kwargs):
        raise RuntimeError("private log details")

    monkeypatch.setattr(SkillStore, "discard_staging", fail)
    monkeypatch.setattr(module, "logger", SimpleNamespace(warning=broken_logger))
    if failure == "copy":
        original = module._copy_package

        def partial(source, destination):
            original(source, destination)
            fail()

        monkeypatch.setattr(module, "_copy_package", partial)
    else:
        monkeypatch.setattr(module.os, "rename", fail)
    with pytest.raises(SkillInstallError) as caught:
        install_skill(source, settings=settings)
    assert caught.value.code == (
        "INVALID_SKILL_PACKAGE" if failure == "copy" else "SKILL_INSTALL_FAILED"
    )
    assert (
        SkillRegistry(settings.builtin_skill_root, settings.skill_storage_root / "user")
        .snapshot()
        .entries
        == ()
    )


def test_cleanup_failure_does_not_change_success(settings, tmp_path, monkeypatch):
    source = package(tmp_path / "input")

    def fail(*args):
        raise PermissionError("private path")

    monkeypatch.setattr(SkillStore, "discard_staging", fail)
    with capture_logs() as logs:
        descriptor = install_skill(source, settings=settings)
    assert read_verified_skill_body(descriptor) == "Do work.\r\n"
    assert logs == [
        {
            "event": "skill_install_cleanup_failed",
            "target": "staging",
            "log_level": "warning",
        }
    ]


@pytest.mark.parametrize("overlap", ["storage", "builtin"])
def test_workspace_cannot_contain_skill_storage(settings, overlap):
    settings.workspace_storage_root = (
        settings.skill_storage_root
        if overlap == "storage"
        else settings.builtin_skill_root
    )
    with pytest.raises(SkillInstallError, match="INVALID_SKILL_ROOTS"):
        SkillStore(settings)


def test_source_cannot_overlap_controlled_storage(settings):
    settings.skill_storage_root.mkdir()
    with pytest.raises(SkillInstallError, match="INVALID_SKILL_PACKAGE"):
        install_skill(settings.skill_storage_root, settings=settings)


def test_copy_rejects_noncanonical_paths(settings, tmp_path):
    source = package(tmp_path / "input")
    # A representable filename that violates the shared portable-path contract.
    (source / "a\u202e").write_bytes(b"data")
    with pytest.raises(SkillInstallError, match="INVALID_SKILL_PACKAGE"):
        install_skill(source, settings=settings)


def test_module_entrypoint_installs_and_reports_safe_conflict(settings, tmp_path):
    source = package(tmp_path / "input")
    env = {
        **os.environ,
        "LANGLEY_SKILL_STORAGE_ROOT": str(settings.skill_storage_root),
        "LANGLEY_BUILTIN_SKILL_ROOT": str(settings.builtin_skill_root),
        "LANGLEY_WORKSPACE_STORAGE_ROOT": str(settings.workspace_storage_root),
    }
    command = [sys.executable, "-m", "langley.skill_installation", str(source)]
    success = subprocess.run(command, env=env, capture_output=True, text=True)
    assert success.returncode == 0 and json.loads(success.stdout) == {
        "ok": True,
        "name": "example",
    }
    duplicate = subprocess.run(command, env=env, capture_output=True, text=True)
    assert duplicate.returncode == 1 and json.loads(duplicate.stdout) == {
        "ok": False,
        "code": "SKILL_NAME_CONFLICT",
    }
    assert str(source) not in duplicate.stdout and duplicate.stderr == ""
