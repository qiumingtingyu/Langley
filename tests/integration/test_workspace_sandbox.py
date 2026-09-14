"""Opt-in real Docker acceptance. Never substitutes a mocked CLI for isolation."""

import asyncio
import json
import os

import pytest

from langley.sandbox import SandboxRuntime
from langley.settings import Settings

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SANDBOX_IT") != "true", reason="Opt-in real Docker"
)


def test_internal_supervisor_ignores_workspace_python_shadowing(tmp_path):
    (tmp_path / "subprocess.py").write_text(
        "PROJECT_MARKER = 'project-local'\n", encoding="utf-8"
    )
    (tmp_path / "project.py").write_text(
        "import subprocess\nprint(subprocess.PROJECT_MARKER)\n", encoding="utf-8"
    )

    async def check():
        runtime = SandboxRuntime(tmp_path, Settings())
        try:
            echo = await runtime.execute("echo ok")
            assert echo["exit_code"] == 0 and echo["output"] == "ok\n"
            project = await runtime.execute("python project.py")
            assert project["exit_code"] == 0
            assert project["output"] == "project-local\n"
        finally:
            await runtime.close()

    asyncio.run(check())


def test_real_sandbox_lifecycle_boundary_and_aftermath(tmp_path):
    async def check():
        runtime = SandboxRuntime(tmp_path, Settings(sandbox_max_output_bytes=1024))
        assert runtime.container is None
        try:
            result = await runtime.execute(
                "printf first > artifact; printf scratch > /tmp/state; "
                "export ONLY_THIS_SHELL=yes; exit 7"
            )
            assert result["exit_code"] == 7 and not result["timed_out"]
            name = runtime.container
            result = await runtime.execute(
                'test -f /tmp/state && test -z "$ONLY_THIS_SHELL" && cat artifact'
            )
            assert result["output"] == "first" and result["exit_code"] == 0
            assert runtime.container == name
            runtime.settings.sandbox_max_output_bytes = 16384
            code, inspect, _ = await runtime._docker(
                "inspect", name, "--format={{json .HostConfig}}"
            )
            config = json.loads(inspect)
            runtime.settings.sandbox_max_output_bytes = 1024
            assert config["NetworkMode"] == "none" and config["ReadonlyRootfs"]
            assert config["CapDrop"] == ["ALL"] and config["PidsLimit"] == 64
            assert not config["Privileged"] and len(config["Mounts"]) == 1
            boundary = await runtime.execute(
                "python -c 'import os,socket; assert os.getuid()!=0; "
                'assert not os.path.exists("/var/run/docker.sock"); '
                'assert not os.path.exists("/root/.ssh"); '
                "s=socket.socket(); s.settimeout(1); "
                'assert s.connect_ex(("1.1.1.1",443))!=0; print("isolated")\''
            )
            assert boundary["exit_code"] == 0
            output = await runtime.execute(
                'python -c \'print("a"*4000);print("TAIL")\''
            )
            assert output["truncated"] and output["output"].endswith("TAIL\n")
            assert len(output["output"].encode()) <= 1024
            timeout = await runtime.execute("printf kept > aftermath; sleep 30", 1)
            assert timeout["timed_out"] and runtime.container is None
            assert (tmp_path / "aftermath").read_text() == "kept"
            fresh = await runtime.execute("test ! -f /tmp/state && cat aftermath")
            assert fresh["exit_code"] == 0 and fresh["output"] == "kept"
            assert runtime.container != name
            task = asyncio.create_task(
                runtime.execute("printf cancelled > cancelled; sleep 30")
            )
            for _ in range(100):
                if (tmp_path / "cancelled").exists():
                    break
                await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert runtime.container is None and (tmp_path / "cancelled").exists()
        finally:
            await runtime.close()

    asyncio.run(check())
