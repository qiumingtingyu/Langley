"""Read-only Builtin Skill Library API contracts."""

from pathlib import Path

from fastapi.testclient import TestClient

from langley.main import create_app
from langley.settings import Settings
from langley.skills import SkillConfigurationError, SkillSource


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str | None = None,
    body: str = "# Procedure\n\nFollow the verified steps.\n",
) -> Path:
    package = root / name
    package.mkdir(parents=True)
    skill_file = package / "SKILL.md"
    with skill_file.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "---\n"
            f"name: {name}\n"
            f"description: {description or f'{name} description'}\n"
            "---\n"
            f"{body}"
        )
    return skill_file


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        database_url=None,
        builtin_skill_root=tmp_path / "builtin",
        skill_storage_root=tmp_path / "skills",
        workspace_storage_root=tmp_path / "workspaces",
        tracing_enabled=False,
        local_run_diagnostics_enabled=False,
    )


def test_list_is_lexical_builtin_only_and_exposes_safe_fields(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_skill(settings.builtin_skill_root, "zeta")
    _write_skill(settings.builtin_skill_root, "alpha")
    _write_skill(settings.skill_storage_root / "user", "private-local")

    app = create_app(settings)
    snapshot = app.state.skill_registry.snapshot()

    response = TestClient(app).get("/api/skills")

    assert [(item.name, item.source) for item in snapshot.entries] == [
        ("alpha", SkillSource.BUILTIN),
        ("private-local", SkillSource.USER_INSTALLED),
        ("zeta", SkillSource.BUILTIN),
    ]
    assert response.status_code == 200
    assert response.json() == [
        {"name": "alpha", "description": "alpha description", "source": "BUILTIN"},
        {"name": "zeta", "description": "zeta description", "source": "BUILTIN"},
    ]
    assert TestClient(app).get("/api/skills/private-local").json() == {
        "detail": {"code": "SKILL_NOT_FOUND"}
    }


def test_detail_returns_verified_body_and_resource_manifest_without_identity(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    skill_file = _write_skill(
        settings.builtin_skill_root,
        "interview-review",
        description="Review an interview with a repeatable rubric.",
        body="# Review\n\nUse the attached rubric.\n",
    )
    reference = skill_file.parent / "references" / "rubric.md"
    reference.parent.mkdir()
    with reference.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("Evidence before conclusions.\n")

    response = TestClient(create_app(settings)).get("/api/skills/interview-review")

    assert response.status_code == 200
    assert response.json() == {
        "name": "interview-review",
        "description": "Review an interview with a repeatable rubric.",
        "source": "BUILTIN",
        "instructions": "# Review\n\nUse the attached rubric.\n",
        "resources": [
            {
                "path": "references/rubric.md",
                "byte_size": len("Evidence before conclusions.\n".encode()),
            }
        ],
    }
    serialized = response.text
    for private_name in (
        "package_root",
        "skill_file",
        "file_path",
        "sha256",
        str(tmp_path),
    ):
        assert private_name not in serialized


def test_detail_fails_closed_when_frozen_skill_body_changes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    skill_file = _write_skill(settings.builtin_skill_root, "stable-skill")
    app = create_app(settings)
    skill_file.write_text(skill_file.read_text() + "changed\n", encoding="utf-8")

    response = TestClient(app).get("/api/skills/stable-skill")

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "SKILL_CONTENT_UNAVAILABLE"}}
    assert "changed" not in response.text
    assert str(tmp_path) not in response.text


def test_detail_returns_no_partial_manifest_for_invalid_resources(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    skill_file = _write_skill(settings.builtin_skill_root, "resource-skill")
    references = skill_file.parent / "references"
    references.mkdir()
    (references / "good.md").write_text("valid", encoding="utf-8")
    (references / "invalid.bin").write_bytes(b"\xff")
    app = create_app(settings)

    response = TestClient(app).get("/api/skills/resource-skill")

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "SKILL_CONTENT_UNAVAILABLE"}}
    assert "good.md" not in response.text


def test_catalog_failure_is_safe_and_never_returns_partial_inventory(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    _write_skill(settings.builtin_skill_root, "visible")
    app = create_app(settings)

    class UnavailableRegistry:
        def snapshot(self):
            raise SkillConfigurationError("private filesystem detail")

    app.state.skill_registry = UnavailableRegistry()
    response = TestClient(app).get("/api/skills")

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "SKILL_CATALOG_UNAVAILABLE"}}
    assert "private filesystem detail" not in response.text
