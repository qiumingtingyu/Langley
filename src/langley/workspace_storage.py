"""Concrete managed Workspace IO. No business ownership or database access."""

import asyncio
import difflib
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path, PureWindowsPath

from langley.settings import Settings


class WorkspaceFileError(ValueError):
    def __init__(self, code: str, **hints: object) -> None:
        super().__init__(code)
        self.code = code
        self.hints = hints


class WorkspaceStorage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.workspace_storage_root.absolute()
        self._locks: dict[str, asyncio.Lock] = {}

    def execution_lock(self, key: str) -> asyncio.Lock:
        """Serialize application-owned Runs against one physical directory."""
        self.workspace_root(key)
        return self._locks.setdefault(key, asyncio.Lock())

    def workspace_root(self, key: str) -> Path:
        if re.fullmatch(r"[a-f0-9]{32}", key) is None:
            raise WorkspaceFileError("WORKSPACE_UNAVAILABLE")
        root = self.root / key
        if root.is_symlink() or root.is_junction():
            raise WorkspaceFileError("PATH_OUTSIDE_WORKSPACE")
        return root

    def resolve(self, key: str, path: str = "") -> Path:
        # Windows drive/ADS/root syntax is refused on every host, including Linux.
        parts = path.split("/") if path else []
        if (
            PureWindowsPath(path).drive
            or path.startswith(("/", "\\"))
            or "\\" in path
            or ":" in path
            or "\x00" in path
            or any(p in {"", ".", ".."} or p.rstrip(" .") != p for p in parts)
        ):
            raise WorkspaceFileError("PATH_OUTSIDE_WORKSPACE")
        root = self.workspace_root(key)
        target = root
        for part in parts:
            if PureWindowsPath(part).is_reserved():
                raise WorkspaceFileError("PATH_OUTSIDE_WORKSPACE")
            target = target / part
            if target.is_symlink() or target.is_junction():
                raise WorkspaceFileError("PATH_OUTSIDE_WORKSPACE")
        if not target.resolve().is_relative_to(root.resolve()):
            raise WorkspaceFileError("PATH_OUTSIDE_WORKSPACE")
        return target

    def list_files(self, key: str, path: str = "") -> dict:
        target = self.resolve(key, path)
        if not target.exists():
            raise WorkspaceFileError("PATH_NOT_FOUND")
        if not target.is_dir():
            raise WorkspaceFileError("PATH_NOT_DIRECTORY")
        entries: list[dict[str, str]] = []
        # Do not enumerate an unbounded command-created directory into memory.
        with os.scandir(target) as children:
            for child in children:
                if len(entries) >= self.settings.workspace_max_entries:
                    raise WorkspaceFileError("DIRECTORY_TOO_LARGE")
                kind = (
                    "symlink"
                    if child.is_symlink() or Path(child.path).is_junction()
                    else (
                        "directory" if child.is_dir(follow_symlinks=False) else "file"
                    )
                )
                entries.append({"name": child.name, "type": kind})
        entries.sort(key=lambda e: (e["type"] != "directory", e["name"]))
        return {"path": path, "entries": entries}

    def _text(self, target: Path, bound: int) -> str:
        try:
            info = target.stat(follow_symlinks=False)
        except FileNotFoundError as error:
            raise WorkspaceFileError("FILE_NOT_FOUND") from error
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise WorkspaceFileError("TEXT_FILE_UNSUPPORTED")
        if info.st_size > bound:
            raise WorkspaceFileError("FILE_TOO_LARGE")
        with target.open("rb") as stream:
            raw = stream.read(bound + 1)
        if len(raw) > bound:
            raise WorkspaceFileError("FILE_TOO_LARGE")
        if raw.startswith((b"%PDF-", b"PK\x03\x04", b"GIF87a", b"GIF89a")):
            raise WorkspaceFileError("TEXT_FILE_UNSUPPORTED")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise WorkspaceFileError("TEXT_FILE_UNSUPPORTED") from error
        if "\x00" in text:
            raise WorkspaceFileError("TEXT_FILE_UNSUPPORTED")
        return text

    def read_file(self, key: str, path: str, offset: int = 1, limit: int = 200) -> dict:
        text = self._text(
            self.resolve(key, path), self.settings.workspace_max_text_read_bytes
        )
        if offset < 1 or limit < 1:
            raise WorkspaceFileError("INVALID_RANGE")
        lines = text.splitlines(keepends=True)
        result = {
            "path": path,
            "start_line": offset,
            "end_line": offset - 1,
            "total_lines": len(lines),
            "content": "",
        }
        # Bound the serialized Tool observation, including JSON escaping and
        # navigation metadata, while keeping each returned UTF-8 line intact.
        bound = self.settings.workspace_max_read_output_bytes
        if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > bound:
            raise WorkspaceFileError("READ_OUTPUT_TOO_LARGE")
        stop = offset - 1 + min(limit, self.settings.workspace_max_read_lines)
        content = ""
        for index, line in enumerate(lines[offset - 1 : stop], start=offset):
            candidate = {
                **result,
                "end_line": index,
                "content": content + line,
            }
            if len(json.dumps(candidate, ensure_ascii=False).encode("utf-8")) > bound:
                if index == offset:
                    raise WorkspaceFileError("READ_OUTPUT_TOO_LARGE")
                break
            result = candidate
            content += line
        return result

    def _replace(self, target: Path, content: str) -> None:
        raw = content.encode("utf-8")
        if len(raw) > self.settings.workspace_max_text_mutation_bytes:
            raise WorkspaceFileError("FILE_TOO_LARGE")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".langley-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, target)
        finally:
            Path(name).unlink(missing_ok=True)

    def _diff(self, path: str, before: str, after: str) -> dict:
        if len(before) + len(after) > 131072:
            return {"diff": "", "diff_truncated": True}
        value = "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile=path,
                tofile=path,
            )
        )
        bound = self.settings.workspace_max_diff_chars
        return {"diff": value[:bound], "diff_truncated": len(value) > bound}

    def write_file(
        self, key: str, path: str, content: str, overwrite: bool = False
    ) -> dict:
        if len(content.encode("utf-8")) > self.settings.workspace_max_write_bytes:
            raise WorkspaceFileError("FILE_TOO_LARGE")
        target = self.resolve(key, path)
        exists = target.exists()
        if exists and not overwrite:
            raise WorkspaceFileError("FILE_ALREADY_EXISTS")
        old: str | None = ""
        if exists:
            info = target.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise WorkspaceFileError("TEXT_FILE_UNSUPPORTED")
            try:
                old = self._text(
                    target, self.settings.workspace_max_text_mutation_bytes
                )
            except WorkspaceFileError as error:
                if error.code != "TEXT_FILE_UNSUPPORTED":
                    raise
                old = None
        self._replace(target, content)
        return {
            "path": path,
            "status": "overwritten" if exists else "created",
            **(
                self._diff(path, old, content)
                if old is not None
                else {
                    "diff": "",
                    "diff_truncated": True,
                }
            ),
        }

    def edit_file(self, key: str, path: str, edits: list[dict[str, str]]) -> dict:
        target = self.resolve(key, path)
        original = self._text(target, self.settings.workspace_max_text_mutation_bytes)
        spans: list[tuple[int, int, str]] = []
        if not edits or len(edits) > 100:
            raise WorkspaceFileError("INVALID_EDIT")
        for edit in edits:
            old, new = edit["old_text"], edit["new_text"]
            if not old:
                raise WorkspaceFileError("INVALID_EDIT")
            positions = []
            start = original.find(old)
            while start >= 0:
                positions.append(start)
                start = original.find(old, start + 1)
            if not positions:
                raise WorkspaceFileError("OLD_TEXT_NOT_FOUND")
            if len(positions) != 1:
                hints: dict[str, object] = {"match_count": len(positions)}
                if len(positions) <= 5:
                    hints["candidate_lines"] = [
                        original.count("\n", 0, p) + 1 for p in positions
                    ]
                else:
                    hints["hint"] = "Anchor is too broad; use a longer exact anchor."
                raise WorkspaceFileError("OLD_TEXT_NOT_UNIQUE", **hints)
            spans.append((positions[0], positions[0] + len(old), new))
        spans.sort()
        if any(right[0] < left[1] for left, right in zip(spans, spans[1:])):
            raise WorkspaceFileError("OVERLAPPING_EDITS")
        result = original
        for start, end, new in reversed(spans):
            result = result[:start] + new + result[end:]
        self._replace(target, result)
        return {
            "path": path,
            "changed_count": len(spans),
            **self._diff(path, original, result),
        }

    def manifest(self, key: str) -> dict:
        root = self.workspace_root(key)
        entries: dict[str, tuple] = {}
        total = 0
        visited = 0
        pending = [root]
        try:
            while pending:
                with os.scandir(pending.pop()) as children:
                    for child in children:
                        visited += 1
                        if visited > self.settings.workspace_manifest_max_files * 2:
                            return {"entries": entries, "complete": False}
                        target = Path(child.path)
                        info = child.stat(follow_symlinks=False)
                        if (
                            child.is_dir(follow_symlinks=False)
                            and not target.is_junction()
                        ):
                            pending.append(target)
                            continue
                        path = target.relative_to(root).as_posix()
                        if len(entries) >= self.settings.workspace_manifest_max_files:
                            return {"entries": entries, "complete": False}
                        if not stat.S_ISREG(info.st_mode):
                            entries[path] = ("special", info.st_mode, info.st_size)
                            continue
                        total += info.st_size
                        if total > self.settings.workspace_manifest_max_bytes:
                            return {"entries": entries, "complete": False}
                        digest = hashlib.sha256()
                        with target.open("rb") as stream:
                            for chunk in iter(lambda: stream.read(65536), b""):
                                digest.update(chunk)
                                if stream.tell() > info.st_size:
                                    return {"entries": entries, "complete": False}
                        entries[path] = ("file", info.st_size, digest.hexdigest())
        except OSError:
            # A removed/unreadable file makes this display snapshot incomplete.
            return {"entries": entries, "complete": False}
        return {"entries": entries, "complete": True}

    @staticmethod
    def changes(before: dict, after: dict) -> dict:
        a, b = before["entries"], after["entries"]
        complete = before["complete"] and after["complete"]
        return {
            "added": sorted(b.keys() - a.keys()) if complete else [],
            "modified": sorted(p for p in a.keys() & b.keys() if a[p] != b[p]),
            "deleted": sorted(a.keys() - b.keys()) if complete else [],
            "complete": complete,
        }


def import_exclusion(path: str) -> str | None:
    parts = path.lower().split("/")
    if any(
        p in {".git", "node_modules", ".venv", "venv", "__pycache__"} for p in parts
    ):
        return "cache_or_repository"
    name = parts[-1]
    if (
        name == ".env"
        or name.startswith(".env.")
        or name.endswith((".pem", ".key"))
        or name in {"id_rsa", "id_ed25519"}
    ):
        return "secret_filename"
    return None
