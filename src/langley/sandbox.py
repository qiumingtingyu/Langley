"""Run-local Docker fault containment; no host shell execution or automatic retry."""

import asyncio
from pathlib import Path
from uuid import uuid4

from langley.settings import Settings

# Kill foreground shell descendants after each command, including on normal exit.
_COMMAND = """import os, signal, subprocess, sys
p = subprocess.Popen(['bash', '-lc', sys.argv[1]], start_new_session=True)
try:
    code = p.wait()
finally:
    try: os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError: pass
sys.exit(code)
"""


class _DockerTimeout(TimeoutError):
    def __init__(self, output: str, truncated: bool) -> None:
        self.output = output
        self.truncated = truncated


class SandboxRuntime:
    def __init__(self, root: Path, settings: Settings) -> None:
        self.root = root.resolve()
        self.settings = settings
        self.container: str | None = None

    async def _docker(self, *args: str, timeout: float = 30) -> tuple[int, str, bool]:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )
        tail = bytearray()
        truncated = False
        assert process.stdout is not None
        try:
            async with asyncio.timeout(timeout):
                while chunk := await process.stdout.read(8192):
                    tail.extend(chunk)
                    if len(tail) > self.settings.sandbox_max_output_bytes:
                        truncated = True
                        del tail[: -self.settings.sandbox_max_output_bytes]
                code = await process.wait()
        except TimeoutError as error:
            raise _DockerTimeout(
                tail.decode("utf-8", errors="replace"), truncated
            ) from error
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        return code, tail.decode("utf-8", errors="replace"), truncated

    async def _start(self) -> None:
        if self.container is not None:
            return
        self.container = "langley-workspace-" + uuid4().hex
        code, _, _ = await self._docker(
            "run",
            "--detach",
            "--pull=never",
            "--name",
            self.container,
            "--label",
            "langley.workspace-sandbox=v0",
            "--network=none",
            "--read-only",
            "--user=10001:10001",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--cpus",
            str(self.settings.sandbox_cpus),
            "--memory",
            str(self.settings.sandbox_memory_mb) + "m",
            "--memory-swap",
            str(self.settings.sandbox_memory_mb) + "m",
            "--pids-limit",
            str(self.settings.sandbox_pids_limit),
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=67108864,mode=1777",
            "--mount",
            f"type=bind,source={self.root},target=/workspace",
            "--workdir=/workspace",
            "--env=HOME=/tmp",
            "--env=PYTHONDONTWRITEBYTECODE=1",
            self.settings.sandbox_image,
            "sleep",
            "infinity",
        )
        if code:
            raise RuntimeError("SANDBOX_START_FAILED")

    async def execute(self, command: str, timeout_seconds: int | None = None) -> dict:
        timeout = min(
            timeout_seconds or self.settings.sandbox_command_timeout_seconds,
            self.settings.sandbox_max_timeout_seconds,
        )
        try:
            await self._start()
            assert self.container is not None
            code, output, truncated = await self._docker(
                "exec",
                self.container,
                "python",
                "-I",
                "-c",
                _COMMAND,
                command,
                timeout=timeout,
            )
            return {
                "exit_code": code,
                "output": output,
                "timed_out": False,
                "truncated": truncated,
            }
        except _DockerTimeout as error:
            await self.reset()
            return {
                "exit_code": None,
                "output": error.output,
                "timed_out": True,
                "truncated": error.truncated,
            }
        except BaseException:
            await self.reset()
            raise

    async def reset(self) -> None:
        if self.container is None:
            return
        name = self.container
        code, output, _ = await self._docker("rm", "--force", name)
        if code and "No such container" not in output:
            raise RuntimeError("SANDBOX_CLEANUP_FAILED")
        self.container = None

    async def close(self) -> None:
        await self.reset()
