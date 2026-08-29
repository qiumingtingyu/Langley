"""Controlled JSONL worker for PDF process-lifecycle and diagnostic tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def emit(value: dict[str, object]) -> None:
    print(json.dumps(value, separators=(",", ":")), flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--behavior-path", type=Path, required=True)
parser.add_argument("--pid-log-path", type=Path, required=True)
parser.add_argument("--done-path", type=Path, required=True)
arguments = parser.parse_args()

emit({"event": "ready", "pid": os.getpid()})
for line in sys.stdin:
    command = json.loads(line)
    if command == {"command": "shutdown"}:
        raise SystemExit(0)
    identity = {
        "job_id": command["job_id"],
        "attempt_no": command["attempt_no"],
        "document_version_id": command["document_version_id"],
        "pid": os.getpid(),
    }
    with arguments.pid_log_path.open("a", encoding="ascii", newline="\n") as stream:
        stream.write(f"{os.getpid()}\n")
    emit({"event": "stage", "stage": "PARSING", **identity})
    behavior = arguments.behavior_path.read_text(encoding="ascii").strip()
    if behavior == "timeout":
        time.sleep(30)
        arguments.done_path.write_text("unexpected", encoding="ascii")
        continue
    if behavior == "crash":
        os._exit(17)
    if behavior == "failure":
        try:
            raise RuntimeError("controlled failure")
        except RuntimeError as error:
            print(
                json.dumps(
                    {
                        "event": "worker_exception",
                        "stage": "PARSING",
                        "exception_type": type(error).__name__,
                        **identity,
                    },
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            emit({"event": "error", "error_code": "PDF_PARSE_FAILED", **identity})
        continue
    if behavior != "normal":
        raise RuntimeError("unknown controlled behavior")
    emit({"event": "stage", "stage": "CHUNKING", **identity})
    Path(command["staging_path"]).write_bytes(b"x" * 100_000)
    emit(
        {
            "event": "completed",
            "page_count": 1,
            "chunk_count": 1,
            "parse_ms": 10.0,
            "chunk_ms": 5.0,
            "total_ms": 15.0,
            **identity,
        }
    )
