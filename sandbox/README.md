# Workspace / Sandbox v0

Run the backend with Docker Desktop's Linux engine available to the backend
process. Build the fixed local image once (build needs network; execution does not):

```powershell
docker build -t langley-sandbox:v0 sandbox
```

Use the existing database configuration and apply migration `0015_workspaces`
before starting the backend. Development and test database URLs are separate:

```powershell
# Development only: explicitly configure LANGLEY_DATABASE_URL first.
uv run alembic upgrade head
uv run uvicorn langley.main:create_app --factory --host 127.0.0.1
# Another terminal, frontend/:
npm run dev -- --host 127.0.0.1
```

Keep the existing configured local user and model credentials. Workspace files
are copied into `data/workspaces/<server-generated-key>` by default. Never point
that root at source code or a secret directory. Workspace binding is immutable;
the ordinary New Conversation action continues creating standalone chats.

## Acceptance

1. Open **工作区 → 新建工作区**. Ask: “Create `hello.py` that prints hello,
   read it back, then run it and report the result.” Confirm the first chat opens,
   file preview works and the final summary lists `+ hello.py`.
2. Import a folder with `calc.py` containing `def add(a,b): return a-b` and
   `test_calc.py` asserting `add(2,3) == 5`. Ask the Agent to run pytest, repair
   the failure and rerun. Each mutation/command must be followed by observation.
3. Include a nested text file, a binary image, `.env` and `node_modules/cache` in
   an import. Confirm relative layout, safe binary-preview refusal and skipped
   counts. Confirm the original folder is unchanged.
4. Ask for `printf kept > aftermath.txt; sleep 30` with a one-second timeout,
   then `cat aftermath.txt`. Confirm compute resets but the file survives.

Automated acceptance uses scripted model decisions (not model-quality evidence),
the production API/Workflow, real MySQL and real Docker:

```powershell
# LANGLEY_TEST_DATABASE_URL must explicitly name langley_test.
# The API integration test resets ONLY that dedicated test database.
New-Item -ItemType Directory -Force .review/workspace-v0 | Out-Null
$env:RUN_SANDBOX_IT = 'true'
uv run pytest tests/integration/test_workspaces_api.py tests/integration/test_workspace_sandbox.py -q -p no:cacheprovider --basetemp=.review/workspace-v0/acceptance
```

The API test exercises a real pytest failure → exact edit → pytest pass, failed
Run with surviving files, and a new Retry Run reading current files. It also
checks ownership, first/second conversations and failed-import publication.

## Local Run diagnostics

The existing execution tracing seam also writes append-only UTF-8 JSONL to
`.runtime/traces/run-<run_id>.jsonl`. It records Run start/outcome, context
compaction, completed/failed LLM rounds, and Tool results including validation
rejections. `run.success` reflects workflow completion, not a database receipt.

`LANGLEY_LOCAL_RUN_DIAGNOSTICS_ENABLED` and
`LANGLEY_LOCAL_RUN_DIAGNOSTICS_INCLUDE_CONTENT` independently default to `true`
when `LANGLEY_ENVIRONMENT=development`, and `false` otherwise. Explicit `true` or
`false` overrides either default. `LANGLEY_LOCAL_RUN_DIAGNOSTICS_ROOT` defaults
to `.runtime/traces`. Disabling diagnostics prevents local trace writes even
when its content flag is true.

Local content includes normalized requests (including context), model-proposed
Tool arguments and Tool results. `LANGLEY_TRACE_CONTENT_ENABLED` continues to
control only external/LangSmith content; local content does not enable external
content. `.runtime/` is Git-ignored. There is no automatic retention or cleanup.
Writes fail open and stop for that Run after a write error; traces are best-effort
diagnostics, not durable audit or Run recovery state.

Strict Workspace argument errors return at most five sanitized field issues.
For example, `timeout_seconds="1"` is rejected; use integer `1` or omit/null for
the default. The Agent can plan a new corrected call after observing the error;
the executor neither coerces values nor automatically retries the Tool.

## Limits and aftermath

Defaults (all server settings use `LANGLEY_` prefix):

| Setting | Default |
| --- | --- |
| `WORKSPACE_MAX_IMPORT_FILE_BYTES` | 10 MiB |
| `WORKSPACE_MAX_IMPORT_TOTAL_BYTES` | 50 MiB |
| `WORKSPACE_MAX_IMPORT_FILES` | 1000 |
| `WORKSPACE_MAX_TEXT_READ_BYTES` / `WORKSPACE_MAX_TEXT_MUTATION_BYTES` / `WORKSPACE_MAX_WRITE_BYTES` | 1 MiB each |
| `WORKSPACE_MAX_READ_OUTPUT_BYTES` | 16 KiB serialized read observation, including JSON metadata |
| `WORKSPACE_MAX_ENTRIES` / `WORKSPACE_MAX_READ_LINES` | 1000 / 500 |
| `WORKSPACE_MANIFEST_MAX_FILES` / `WORKSPACE_MANIFEST_MAX_BYTES` | 5000 / 100 MiB |
| `WORKSPACE_MAX_LLM_ROUNDS` / `WORKSPACE_MAX_TOOL_CALLS` | 8 / 10 |
| `SANDBOX_COMMAND_TIMEOUT_SECONDS` / `SANDBOX_MAX_TIMEOUT_SECONDS` | 30 / 60 |
| `SANDBOX_MAX_OUTPUT_BYTES` | 16 KiB tail |
| `SANDBOX_CPUS` / `SANDBOX_MEMORY_MB` / `SANDBOX_PIDS_LIMIT` | 1 / 256 / 64 |

Overall workflow deadline stays 180 seconds. The image runs as UID/GID 10001,
network none, read-only root, dropped capabilities and no-new-privileges.
Only `/workspace` is persistently writable; `/tmp` is a 64 MiB tmpfs. Docker
provides its standard ephemeral `/dev`, `/dev/shm` and pseudo-filesystems.
No Docker socket, backend environment, home or credentials are mounted.
On a native Linux host, provision the managed directory's ownership/permissions
for UID/GID 10001 and the backend user before use; this delivery was accepted on
Windows Docker Desktop, not native Linux host permissions.

One container is created lazily per Run, reused with a fresh shell for each
command, and removed on completion/failure/cancel/timeout. There is no transparent
Tool retry. Workspace changes survive failed/cancelled Runs. Retry is a new Run
that observes current files. No snapshots, rollback or persisted Tool transcripts.
The internal Python supervisor uses isolated mode (`python -I`); user commands
retain ordinary shell and project-local Python import behavior.

Structured edits validate all unique anchors against one original snapshot and
atomically replace the file once. Reads return only complete lines within the
output byte budget; an oversized first requested line returns recoverable
`READ_OUTPUT_TOO_LARGE`. Same-Workspace Runs and file list/preview API reads share
one lock in this single backend process. API reads release database scope before
waiting for that lock. Arbitrary host writers/multiple backend processes are not
supported synchronization participants.

Successful summaries compare bounded SHA-256 manifests. Incomplete scans say so
and suppress uncertain additions/deletions. Diffs are bounded to 8000 characters
and omitted above a 128 Ki-character combined input budget. The last 100 successful
summaries are process-local display data, recoverable through owned Run reads;
they disappear on restart/eviction. Failure/cancel summaries and user-facing
per-Tool diff/status rendering are not implemented. Refresh files to inspect
failure aftermath; Agent observations include command outcomes and exact diffs.

This is single-user fault containment, not a hostile multi-tenant OS sandbox.
There is no Workspace disk quota or crash-atomic transaction spanning MySQL and
the filesystem. A hard backend kill/daemon outage can leave containers or
unpublished storage. Publication failures before commit clean up storage; once
commit starts, an uncertain outcome preserves the final directory so a committed
Workspace cannot lose its files. This may leave unreferenced storage. For recovery, inspect
`docker ps -a --filter label=langley.workspace-sandbox=v0`, verify no live Run owns
the container, then remove that exact container. Exclusions are filename rules,
not a complete secret scanner. Browser upload transports regular file bytes and
does not expose original filesystem symlink metadata; it cannot certify source
symlink provenance. Text Tools reject symlinks/junctions in managed storage.
