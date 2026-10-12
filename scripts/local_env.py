"""Explicit local dotenv launcher; never edits Windows or the parent environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from dotenv.parser import parse_stream
from pydantic import ValidationError
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from langley.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {
    "LANGLEY_LLM_PROVIDER",
    "LANGLEY_LLM_MODEL",
    "LANGLEY_CONVERSATION_COMPACTOR_MODEL",
    "LANGLEY_MEMORY_POLICY_MODEL",
    "LANGLEY_MEMORY_POLICY_ESTIMATED_TOKEN_BUDGET",
}
VENDOR_NAMES = {
    vendor: {f"LANGLEY_{vendor.upper()}_API_KEY", f"LANGLEY_{vendor.upper()}_BASE_URL"}
    for vendor in ("qwen", "deepseek")
}
SDK_NAMES = {
    "TAVILY_API_KEY",
    "TAVILY_PROJECT",
    "TAVILY_ORG_ID",
    "TAVILY_HTTP_PROXY",
    "TAVILY_HTTPS_PROXY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_WORKSPACE_ID",
}
ISOLATED_PREFIXES = (
    "LANGLEY_",
    "LANGSMITH_",
    "LANGCHAIN_",
    "TAVILY_",
    "C0_EVAL_",
    "DASHSCOPE_",
    "QWEN_",
    "DEEPSEEK_",
    "OPENAI_",
    "UVICORN_",
)
ISOLATED_NAMES = {
    "UV_ENV_FILE",
    "UV_NO_ENV_FILE",
    "HOSTED_LANGSERVE_PROJECT_NAME",
    "RUN_SANDBOX_IT",
    "WEB_CONCURRENCY",
    "FORWARDED_ALLOW_IPS",
    "TAVILY_API_KEY",
}
ERROR_CODES = frozenset(
    {
        "DOTENV_SYNTAX_INVALID",
        "DOTENV_FIELD_NOT_ALLOWED",
        "DOTENV_FIELD_DUPLICATED",
        "DOTENV_ASSIGNMENT_REQUIRED",
        "DOTENV_REFERENCES_NOT_SUPPORTED",
        "DOTENV_FILE_UNREADABLE",
        "TEST_DATABASE_REQUIRED",
        "TAVILY_ALIAS_CONFLICT",
        "PROVIDER_MODELS_REQUIRED",
        "TEST_DATABASE_MUST_BE_LANGLEY_TEST",
        "PROVIDER_SELECTION_MISMATCH",
        "PROVIDER_API_KEY_REQUIRED",
        "PROVIDER_BASE_URL_INVALID",
        "PROVIDER_MODEL_EMPTY",
        "MEMORY_POLICY_MODEL_EMPTY",
        "DEVELOPMENT_DATABASE_AND_USER_REQUIRED",
        "DEVELOPMENT_DATABASE_INVALID",
        "DEV_COMMAND_NOT_SUPPORTED",
        "TEST_COMMAND_MUST_BE_PYTEST",
        "CHILD_PROCESS_UNAVAILABLE",
        "UV_CONFIG_CHECK_FAILED",
        "CONFIGURATION_ERROR",
    }
)
VALIDATION_FIELDS = frozenset(Settings.model_fields) | {
    "settings",
    "TAVILY_API_KEY",
    "LANGLEY_TAVILY_API_KEY",
}
DEV_PASS = "PASS: effective configuration only; no service requests made"
TEST_PASS = "PASS: test environment; development/model credentials excluded"
VALIDATION_PREFIX = "FAIL: SETTINGS_INVALID; fields="


def report_names(provider: str) -> tuple[str, ...]:
    return (
        "LANGLEY_LLM_PROVIDER",
        f"LANGLEY_{provider.upper()}_API_KEY",
        f"LANGLEY_{provider.upper()}_BASE_URL",
        "LANGLEY_LLM_MODEL",
        "LANGLEY_CONVERSATION_COMPACTOR_MODEL",
        "LANGLEY_DATABASE_URL",
        "LANGLEY_LOCAL_USER_ID",
        "LANGLEY_MEMORY_POLICY_MODEL",
        "LANGLEY_MEMORY_POLICY_ESTIMATED_TOKEN_BUDGET",
        "LANGLEY_WEB_SEARCH_ENABLED",
        "LANGLEY_TRACING_ENABLED",
        "LANGLEY_TRACE_CONTENT_ENABLED",
        "LANGLEY_LOCAL_RUN_DIAGNOSTICS_INCLUDE_CONTENT",
    )


def forward_precheck(result, provider: str | None, origins: Mapping[str, str]) -> None:
    """Reconstruct only finite protocol messages; never relay raw subprocess text."""
    sources = {"common_file", "provider_file", "default"}
    allowed_lines = {DEV_PASS if provider else TEST_PASS}
    if provider:
        allowed_lines |= {
            f"{name}: source={origins.get(name, 'default')}"
            for name in report_names(provider)
            if origins.get(name, "default") in sources
        }
    for line in (getattr(result, "stdout", None) or "").splitlines():
        if line in allowed_lines:
            print(line)
        elif result.returncode and line.startswith(VALIDATION_PREFIX):
            fields = line.removeprefix(VALIDATION_PREFIX).split(",")
            if fields and all(field in VALIDATION_FIELDS for field in fields):
                print(VALIDATION_PREFIX + ",".join(sorted(set(fields))))
    if result.returncode:
        allowed_errors = {"FAIL: " + code for code in ERROR_CODES}
        error = next(
            (
                line
                for line in (getattr(result, "stderr", None) or "").splitlines()
                if line in allowed_errors
            ),
            "FAIL: UV_CONFIG_CHECK_FAILED",
        )
        print(error, file=sys.stderr)


class ConfigurationError(Exception):
    """Carry only a safe category, never a raw input or credential."""


def child_environment(parent: Mapping[str, str]) -> dict[str, str]:
    """Isolate application/remote SDK variables, preserving proxy and TLS settings."""
    return {
        name: value
        for name, value in parent.items()
        if not name.upper().startswith(ISOLATED_PREFIXES)
        and name.upper() not in ISOLATED_NAMES
    }


def file_names(path: Path, allowed: set[str]) -> set[str]:
    """Check dotenv structure without printing values or resolving references."""
    names: set[str] = set()
    try:
        with path.open(encoding="utf-8-sig") as source:
            for binding in parse_stream(source):
                if binding.error:
                    raise ConfigurationError("DOTENV_SYNTAX_INVALID")
                if binding.key is None:
                    continue
                if binding.key not in allowed:
                    raise ConfigurationError("DOTENV_FIELD_NOT_ALLOWED")
                if binding.key in names:
                    raise ConfigurationError("DOTENV_FIELD_DUPLICATED")
                if binding.value is None:
                    raise ConfigurationError("DOTENV_ASSIGNMENT_REQUIRED")
                literal = binding.original.string.split("=", 1)[1].lstrip()
                if "$" in binding.value and not literal.startswith("'"):
                    raise ConfigurationError("DOTENV_REFERENCES_NOT_SUPPORTED")
                names.add(binding.key)
    except (OSError, UnicodeError) as error:
        raise ConfigurationError("DOTENV_FILE_UNREADABLE") from error
    return names


def configuration_files(root: Path, provider: str | None, integration: bool):
    if provider is None:
        if not integration:
            return (), {}
        path = root / ".env.test.local"
        names = file_names(path, {"LANGLEY_TEST_DATABASE_URL", "LANGLEY_QDRANT_URL"})
        if "LANGLEY_TEST_DATABASE_URL" not in names:
            raise ConfigurationError("TEST_DATABASE_REQUIRED")
        return (path,), {name: "test_file" for name in names}
    common = root / ".env.common.local"
    vendor = root / f".env.{provider}.local"
    allowed = {"LANGLEY_" + name.upper() for name in Settings.model_fields} | SDK_NAMES
    allowed -= MODEL_NAMES | set.union(*VENDOR_NAMES.values())
    allowed -= {"LANGLEY_TEST_DATABASE_URL"}
    common_names = file_names(common, allowed)
    if len(common_names & {"TAVILY_API_KEY", "LANGLEY_TAVILY_API_KEY"}) > 1:
        raise ConfigurationError("TAVILY_ALIAS_CONFLICT")
    vendor_names = file_names(vendor, MODEL_NAMES | VENDOR_NAMES[provider])
    required = {
        "LANGLEY_LLM_PROVIDER",
        "LANGLEY_LLM_MODEL",
        "LANGLEY_CONVERSATION_COMPACTOR_MODEL",
    }
    if not required <= vendor_names:
        raise ConfigurationError("PROVIDER_MODELS_REQUIRED")
    return (common, vendor), {
        **{name: "common_file" for name in common_names},
        **{name: "provider_file" for name in vendor_names},
    }


def meaningful(value: str | None) -> bool:
    return bool(
        value
        and value.strip()
        and not value.startswith("replace-")
        and value not in {"...", "<fill-locally>"}
    )


def check_effective(root: Path, provider: str | None, integration: bool) -> int:
    """Inspect Settings in the uv-loaded child without creating the application."""
    _, origins = configuration_files(root, provider, integration)
    try:
        settings = Settings(_env_file=None)
    except ValidationError as error:
        fields = sorted(
            {
                (
                    str(item["loc"][0])
                    if item["loc"] and str(item["loc"][0]) in VALIDATION_FIELDS
                    else "settings"
                )
                for item in error.errors(include_input=False)
            }
        )
        print(VALIDATION_PREFIX + ",".join(fields))
        return 2
    if provider is None:
        if integration:
            try:
                url = make_url(settings.test_database_url or "")
                if url.get_backend_name() != "mysql" or url.database != "langley_test":
                    raise ValueError
            except (ArgumentError, ValueError, TypeError) as error:
                raise ConfigurationError(
                    "TEST_DATABASE_MUST_BE_LANGLEY_TEST"
                ) from error
        print(TEST_PASS)
        return 0
    if settings.llm_provider != provider:
        raise ConfigurationError("PROVIDER_SELECTION_MISMATCH")
    key = getattr(settings, provider + "_api_key")
    if key is None or not meaningful(key.get_secret_value()):
        raise ConfigurationError("PROVIDER_API_KEY_REQUIRED")
    try:
        parsed = urlsplit(getattr(settings, provider + "_base_url") or "")
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        _ = parsed.port
    except ValueError as error:
        raise ConfigurationError("PROVIDER_BASE_URL_INVALID") from error
    if (
        not settings.llm_model.strip()
        or not settings.conversation_compactor_model.strip()
    ):
        raise ConfigurationError("PROVIDER_MODEL_EMPTY")
    if (
        settings.memory_policy_model is not None
        and not settings.memory_policy_model.strip()
    ):
        raise ConfigurationError("MEMORY_POLICY_MODEL_EMPTY")
    if not meaningful(settings.database_url) or settings.local_user_id is None:
        raise ConfigurationError("DEVELOPMENT_DATABASE_AND_USER_REQUIRED")
    try:
        url = make_url(settings.database_url)
        if (
            url.get_backend_name() != "mysql"
            or not url.database
            or url.database == "langley_test"
        ):
            raise ValueError
    except (ArgumentError, ValueError, TypeError) as error:
        raise ConfigurationError("DEVELOPMENT_DATABASE_INVALID") from error
    for name in report_names(provider):
        print(f"{name}: source={origins.get(name, 'default')}")
    print(DEV_PASS)
    return 0


def run(args: argparse.Namespace, root: Path = ROOT) -> int:
    provider = args.provider if args.mode == "dev" else None
    integration = getattr(args, "integration", False)
    files, origins = configuration_files(root, provider, integration)
    environment = child_environment(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if provider is None:
        environment["LANGSMITH_TRACING"] = "false"
        environment["LANGSMITH_TEST_TRACKING"] = "false"
        if getattr(args, "sandbox", False):
            environment["RUN_SANDBOX_IT"] = "true"
    base = ["uv", "run", "--locked", "--no-sync"]
    if files:
        for path in files:
            # cwd is root; fixed relative names avoid uv's Windows path parsing.
            base.extend(("--env-file", path.name))
    else:
        base.append("--no-env-file")
    check = base + ["python", "-B", str(root / "scripts/local_env.py"), "_check"]
    if provider:
        check.extend(("--provider", provider))
    if integration:
        check.append("--integration")
    result = subprocess.run(
        check, cwd=root, env=environment, check=False, capture_output=True, text=True
    )
    forward_precheck(result, provider, origins)
    if result.returncode or args.check:
        return result.returncode
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        command = (
            ["uvicorn", "langley.main:create_app", "--factory", "--reload"]
            if provider
            else [
                "pytest",
                *(
                    ["tests/integration"]
                    if integration
                    else ["tests/integration/test_workspace_sandbox.py"]
                    if getattr(args, "sandbox", False)
                    else []
                ),
            ]
        )
    if provider and command[0] not in {"uvicorn", "alembic", "python"}:
        raise ConfigurationError("DEV_COMMAND_NOT_SUPPORTED")
    if not provider and command[0] != "pytest":
        raise ConfigurationError("TEST_COMMAND_MUST_BE_PYTEST")
    return subprocess.run(
        base + command, cwd=root, env=environment, check=False
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("dev", "test", "_check"):
        command = modes.add_parser(mode)
        if mode != "test":
            command.add_argument(
                "--provider", choices=("qwen", "deepseek"), required=mode == "dev"
            )
        if mode != "dev":
            command.add_argument("--integration", action="store_true")
        if mode == "test":
            command.add_argument(
                "--sandbox",
                action="store_true",
                help="Explicitly opt in to real Docker tests",
            )
        if mode != "_check":
            command.add_argument("--check", action="store_true")
            command.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.mode == "_check":
            return check_effective(ROOT, args.provider, args.integration)
        return run(args)
    except ConfigurationError as error:
        code = str(error) if str(error) in ERROR_CODES else "CONFIGURATION_ERROR"
        print("FAIL: " + code, file=sys.stderr)
        return 2
    except OSError:
        print("FAIL: CHILD_PROCESS_UNAVAILABLE", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
