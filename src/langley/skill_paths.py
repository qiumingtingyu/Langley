"""Exact relative Skill package paths, shared by install and resource admission."""

import unicodedata
from pathlib import PureWindowsPath

MAX_SKILL_PATH_CHARS = 512


def valid_skill_package_path(path: str) -> bool:
    """Reject aliases without normalizing user names into different identities."""

    return (
        1 <= len(path) <= MAX_SKILL_PATH_CHARS
        and "\\" not in path
        and ":" not in path
        and not any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in path)
        and all(
            part not in {"", ".", ".."}
            and part.rstrip(" .") == part
            and not PureWindowsPath(part).is_reserved()
            for part in path.split("/")
        )
    )
