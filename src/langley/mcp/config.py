"""Explicit startup snapshot of local credentials; never a Web user fallback."""

import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from langley.mcp.contracts import Identifier, StrictModel


class Credential(StrictModel):
    principal_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    token_digest: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)
    user_id: Identifier
    allowed_kb_ids: list[Identifier] = Field(min_length=1, max_length=128)
    enabled: bool = True

    @field_validator("allowed_kb_ids")
    @classmethod
    def unique(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("INVALID_CONFIG")
        return value


class ConfigFile(StrictModel):
    port: int = Field(default=8766, ge=1024, le=65535)
    credentials: list[Credential] = Field(min_length=1, max_length=64, repr=False)

    @model_validator(mode="after")
    def duplicates(self) -> "ConfigFile":
        if len({c.token_digest for c in self.credentials}) != len(self.credentials):
            raise ValueError("INVALID_CONFIG")
        owners: dict[str, int] = {}
        for c in self.credentials:
            if owners.setdefault(c.principal_id, c.user_id) != c.user_id:
                raise ValueError("INVALID_CONFIG")
        return self


@dataclass(frozen=True)
class Principal:
    principal_id: str
    user_id: int
    allowed_kb_ids: frozenset[int]


@dataclass(frozen=True)
class MCPConfig:
    port: int
    credentials: tuple[tuple[str, Principal], ...] = field(repr=False)

    @classmethod
    def parse(cls, value: object) -> "MCPConfig":
        parsed = ConfigFile.model_validate(value)
        return cls(
            parsed.port,
            tuple(
                (
                    c.token_digest,
                    Principal(c.principal_id, c.user_id, frozenset(c.allowed_kb_ids)),
                )
                for c in parsed.credentials
                if c.enabled
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "MCPConfig":
        with path.open("rb") as stream:
            data = stream.read(65537)
        if len(data) > 65536:
            raise ValueError("INVALID_CONFIG")
        return cls.parse(json.loads(data))

    def authenticate(self, authorization: str) -> Principal | None:
        if not re.fullmatch(r"Bearer [A-Za-z0-9_-]{43,128}", authorization):
            return None
        digest = hashlib.sha256(authorization[7:].encode("ascii")).hexdigest()
        found = None
        for expected, principal in self.credentials:
            if hmac.compare_digest(expected, digest):
                found = principal
        return found

    @property
    def hosts(self) -> list[str]:
        return [f"127.0.0.1:{self.port}", f"localhost:{self.port}"]
