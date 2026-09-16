"""Local admission: private copy -> validate -> conflict check -> publish.

Run with ``python -m langley.skill_installation <local-skill-directory>``.
This is a local operator entry point, never an Agent Tool or HTTP endpoint.
"""

import argparse
import json
from pathlib import Path

from langley.settings import Settings
from langley.skill_store import SkillInstallError, SkillStore, _best_effort_cleanup
from langley.skills import (
    MAX_SKILLS_PER_SOURCE,
    SkillConfigurationError,
    SkillDescriptor,
    SkillRegistry,
    SkillSource,
)


def install_skill(source: Path, *, settings: Settings) -> SkillDescriptor:
    """Admit a procedure without granting tools, permissions or Workspace scope."""

    store = SkillStore(settings)
    registry = SkillRegistry(settings.builtin_skill_root, store.user_root)
    prepared = store.prepare(source)
    try:
        with store.publication_lock():
            snapshot = registry.snapshot()
            if snapshot.get(prepared.descriptor.name) is not None:
                raise SkillInstallError("SKILL_NAME_CONFLICT")
            if (
                sum(e.source is SkillSource.USER_INSTALLED for e in snapshot.entries)
                >= MAX_SKILLS_PER_SOURCE
            ):
                raise SkillInstallError("SKILL_COUNT_LIMIT")
            return store.publish(prepared)
    finally:
        _best_effort_cleanup(store.discard_staging, prepared.staging_key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install one local Skill directory")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        installed = install_skill(args.directory, settings=Settings())
    except SkillInstallError as error:
        print(json.dumps({"ok": False, "code": error.code}))
        return 1
    except (SkillConfigurationError, OSError, RuntimeError, ValueError):
        print(json.dumps({"ok": False, "code": "SKILL_INSTALL_FAILED"}))
        return 1
    print(json.dumps({"ok": True, "name": installed.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
