"""Local launcher boundaries, using only synthetic credentials and dotenv files."""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/local_env.py"
spec = importlib.util.spec_from_file_location("local_env", SCRIPT)
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def _files(root, provider="deepseek", extra=""):
    (root / ".env.common.local").write_text(
        "LANGLEY_DATABASE_URL=mysql+asyncmy://test:synthetic@localhost/langley\n"
        "LANGLEY_LOCAL_USER_ID=1\n" + extra,
        encoding="utf-8",
    )
    (root / f".env.{provider}.local").write_text(
        f"LANGLEY_LLM_PROVIDER={provider}\n"
        f"LANGLEY_{provider.upper()}_API_KEY='synthetic-provider-secret$literal'\n"
        f"LANGLEY_{provider.upper()}_BASE_URL=https://example.test/v1\n"
        "LANGLEY_LLM_MODEL=answer-test\n"
        "LANGLEY_CONVERSATION_COMPACTOR_MODEL=compact-test\n",
        encoding="utf-8",
    )
    return launcher.configuration_files(root, provider, False)


def test_child_isolation_preserves_parent_proxy_tls_and_cache():
    parent = {
        "LANGLEY_LLM_MODEL": "old-model",
        "langley_memory_policy_model": "old-memory",
        "DEEPSEEK_API_KEY": "synthetic",
        "LANGSMITH_API_KEY": "synthetic",
        "LANGCHAIN_TRACING_V2": "true",
        "TAVILY_API_KEY": "synthetic",
        "DASHSCOPE_API_KEY": "synthetic",
        "OPENAI_API_KEY": "synthetic",
        "C0_EVAL_DATABASE_URL": "synthetic",
        "UV_ENV_FILE": "unexpected.env",
        "UVICORN_WORKERS": "8",
        "RUN_SANDBOX_IT": "true",
        "HTTPS_PROXY": "proxy",
        "SSL_CERT_FILE": "cert",
        "HF_HOME": "cache",
        "PATH": "tools",
    }
    before = dict(parent)
    assert launcher.child_environment(parent) == {
        "HTTPS_PROXY": "proxy",
        "SSL_CERT_FILE": "cert",
        "HF_HOME": "cache",
        "PATH": "tools",
    }
    assert parent == before


@pytest.mark.parametrize(
    "extra,code",
    [
        ("LANGLEY_LLM_MODEL=wrong\n", "DOTENV_FIELD_NOT_ALLOWED"),
        ("LANGLEY_TEST_DATABASE_URL=synthetic\n", "DOTENV_FIELD_NOT_ALLOWED"),
        ("LANGLEY_LOCAL_USER_ID=2\n", "DOTENV_FIELD_DUPLICATED"),
        ("LANGLEY_LANGSMITH_PROJECT=${OTHER}\n", "DOTENV_REFERENCES_NOT_SUPPORTED"),
        ("TAVILY_API_KEY=one\nLANGLEY_TAVILY_API_KEY=two\n", "TAVILY_ALIAS_CONFLICT"),
    ],
)
def test_file_boundaries(tmp_path, extra, code):
    with pytest.raises(launcher.ConfigurationError, match=code):
        _files(tmp_path, extra=extra)


def test_test_mode_ignores_dev_files_and_propagates_exit(tmp_path, monkeypatch):
    _files(tmp_path)
    calls = []
    monkeypatch.setenv("LANGLEY_DATABASE_URL", "synthetic-dev")
    monkeypatch.setenv("LANGSMITH_API_KEY", "synthetic-sdk")
    monkeypatch.setenv("RUN_SANDBOX_IT", "true")

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0 if len(calls) == 1 else 7)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    args = argparse.Namespace(
        mode="test",
        integration=False,
        check=False,
        command=["--", "pytest", "tests/fast/test_settings.py"],
    )
    assert launcher.run(args, tmp_path) == 7
    for command, kwargs in calls:
        assert "--no-env-file" in command
        assert "--env-file" not in command
        assert "LANGLEY_DATABASE_URL" not in kwargs["env"]
        assert "LANGSMITH_API_KEY" not in kwargs["env"]
        assert kwargs["env"]["LANGSMITH_TRACING"] == "false"
        assert "RUN_SANDBOX_IT" not in kwargs["env"]


def test_dev_uses_only_common_and_selected_provider(tmp_path, monkeypatch):
    _files(tmp_path, "qwen")
    _files(tmp_path, "deepseek")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    args = argparse.Namespace(mode="dev", provider="deepseek", check=True, command=[])
    assert launcher.run(args, tmp_path) == 0
    assert len(calls) == 1
    command = calls[0]
    assert command.count("--env-file") == 2
    assert ".env.deepseek.local" in command
    assert ".env.qwen.local" not in command


def test_uv_diagnostics_do_not_echo_raw_configuration(tmp_path, monkeypatch, capsys):
    _files(tmp_path)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 2, stdout="synthetic-secret-raw\n", stderr="synthetic-secret-raw\n"
        ),
    )
    args = argparse.Namespace(mode="dev", provider="deepseek", check=True, command=[])
    assert launcher.run(args, tmp_path) == 2
    output = capsys.readouterr()
    assert "UV_CONFIG_CHECK_FAILED" in output.err
    assert "synthetic-secret-raw" not in output.out + output.err


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize(
    "payload",
    [
        "FAIL: SYNTHETIC_SENSITIVE_VALUE",
        "LANGLEY_LLM_MODEL: source=SYNTHETIC_SENSITIVE_VALUE",
        "LANGLEY_LLM_MODEL: source=provider_file SYNTHETIC_SENSITIVE_VALUE",
        "LANGLEY_SYNTHETIC_SENSITIVE_VALUE: source=provider_file",
        "PASS: SYNTHETIC_SENSITIVE_VALUE",
        "FAIL: SETTINGS_INVALID; fields=llm_model,SYNTHETIC_SENSITIVE_VALUE",
        "FAIL: PROVIDER_API_KEY_REQUIRED SYNTHETIC_SENSITIVE_VALUE",
        "FAIL: SETTINGS_INVALID; fields=llm_model\x1b[0mSYNTHETIC_SENSITIVE_VALUE",
    ],
)
def test_precheck_rejects_malicious_protocol_shapes(capsys, stream, payload):
    result = subprocess.CompletedProcess(["synthetic"], 2, stdout="", stderr="")
    setattr(result, stream, payload)
    launcher.forward_precheck(result, "deepseek", {})
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "FAIL: UV_CONFIG_CHECK_FAILED\n"
    assert "SYNTHETIC_SENSITIVE_VALUE" not in output.out + output.err


def test_precheck_accepts_only_defined_fields_sources_and_codes(capsys):
    result = subprocess.CompletedProcess(
        ["synthetic"],
        2,
        stdout="LANGLEY_LLM_MODEL: source=provider_file\n"
        "FAIL: SETTINGS_INVALID; fields=llm_model,settings,llm_model\n",
        stderr="FAIL: PROVIDER_API_KEY_REQUIRED\n",
    )
    launcher.forward_precheck(
        result, "deepseek", {"LANGLEY_LLM_MODEL": "provider_file"}
    )
    output = capsys.readouterr()
    assert output.out == "LANGLEY_LLM_MODEL: source=provider_file\n" + (
        "FAIL: SETTINGS_INVALID; fields=llm_model,settings\n"
    )
    assert output.err == "FAIL: PROVIDER_API_KEY_REQUIRED\n"


def test_sandbox_opt_in_sets_only_child_flag_without_loading_dev(tmp_path, monkeypatch):
    _files(tmp_path)
    monkeypatch.delenv("RUN_SANDBOX_IT", raising=False)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    args = argparse.Namespace(
        mode="test", integration=False, sandbox=True, check=False, command=[]
    )
    assert launcher.run(args, tmp_path) == 0
    assert "RUN_SANDBOX_IT" not in os.environ
    assert calls[-1][0][-2:] == [
        "pytest",
        "tests/integration/test_workspace_sandbox.py",
    ]
    for command, kwargs in calls:
        assert kwargs["env"]["RUN_SANDBOX_IT"] == "true"
        assert "LANGLEY_DATABASE_URL" not in kwargs["env"]
        assert "--no-env-file" in command
        assert "--env-file" not in command


@pytest.mark.parametrize("database", ["langley", "", "not-a-url"])
def test_integration_guard_rejects_wrong_database_without_echo(
    tmp_path,
    monkeypatch,
    capsys,
    database,
):
    (tmp_path / ".env.test.local").write_text(
        "LANGLEY_TEST_DATABASE_URL=synthetic\n",
        encoding="utf-8",
    )
    for name in list(os.environ):
        if name not in launcher.child_environment(os.environ):
            monkeypatch.delenv(name, raising=False)
    url = (
        f"mysql+asyncmy://test:synthetic-secret@localhost/{database}"
        if database != "not-a-url"
        else "synthetic-secret-invalid-url"
    )
    monkeypatch.setenv("LANGLEY_TEST_DATABASE_URL", url)
    with pytest.raises(launcher.ConfigurationError, match="TEST_DATABASE_MUST"):
        launcher.check_effective(tmp_path, None, True)
    output = capsys.readouterr()
    assert "synthetic-secret" not in output.out + output.err


def test_real_uv_loading_and_safe_validation_errors(tmp_path):
    """Exercise uv's dotenv precedence in a real isolated child, without any HTTP."""
    tmp_path = tmp_path / "configuration with spaces"
    tmp_path.mkdir()
    _files(tmp_path, extra="LANGLEY_LOCAL_RUN_DIAGNOSTICS_INCLUDE_CONTENT=false\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/local_env.py").write_bytes(SCRIPT.read_bytes())
    env = dict(os.environ)
    env.update(
        {
            "LANGLEY_LLM_MODEL": "inherited-qwen",
            "UV_ENV_FILE": "missing.env",
            "LANGLEY_MEMORY_POLICY_MODEL": "old-memory",
            "LANGLEY_MEMORY_POLICY_ESTIMATED_TOKEN_BUDGET": "invalid-inherited",
        }
    )
    command = [
        sys.executable,
        "-B",
        str(tmp_path / "scripts/local_env.py"),
        "dev",
        "--provider",
        "deepseek",
        "--check",
    ]
    result = subprocess.run(
        command,
        cwd=SCRIPT.parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "LANGLEY_LLM_MODEL: source=provider_file" in result.stdout
    assert "synthetic-provider-secret" not in result.stdout + result.stderr
    assert "inherited-qwen" not in result.stdout + result.stderr
    path = tmp_path / ".env.common.local"
    path.write_text(
        path.read_text()
        + "LANGLEY_MEMORY_ESTIMATED_TOKEN_BUDGET=synthetic-secret-invalid\n"
    )
    result = subprocess.run(
        command,
        cwd=SCRIPT.parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "SETTINGS_INVALID" in result.stdout
    assert "synthetic-secret-invalid" not in result.stdout + result.stderr
