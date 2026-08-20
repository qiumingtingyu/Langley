# Langley

Langley provides a bounded Learning Assistant Agent and Personal Context
Memory over a durable, process-local streaming execution shell. MySQL remains
authoritative for Conversation, Message, Run, and Memory facts; LangGraph
state, Tool calls/results, and streamed partial text remain transient. Only the
canonical completed ASSISTANT message is persisted.

## Prerequisites

- Windows PowerShell for the commands below
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Node.js 24 and npm (`.node-version` pins `24`; `frontend/package.json`
  requires `>=24 <25`)
- Docker Desktop with Docker Compose and MySQL 8.4

## Environment configuration

Settings are read from `LANGLEY_` environment variables. The application does
not automatically load `.env`; use `.env.example` only as a safe reference and
set values in the PowerShell session that runs a command.

The current settings are:

- `LANGLEY_ENVIRONMENT` (defaults to `development`)
- `LANGLEY_LOG_LEVEL` (defaults to `INFO`)
- `LANGLEY_LOG_FORMAT` (defaults to `console`)
- `LANGLEY_DATABASE_URL` (development database)
- `LANGLEY_TEST_DATABASE_URL` (dedicated integration database only)
- `LANGLEY_LOCAL_USER_ID` (explicit local/demo user for conversation APIs)
- `LANGLEY_LLM_PROVIDER` (currently only `qwen`; defaults to `qwen`)
- `LANGLEY_QWEN_API_KEY` (real-service opt-in credential; never commit it)
- `LANGLEY_QWEN_BASE_URL` (provider-issued OpenAI-compatible API base URL)
- `LANGLEY_LLM_MODEL` (defaults to `qwen3.7-plus-2026-05-26`)
- `LANGLEY_HISTORY_ESTIMATED_TOKEN_BUDGET` (defaults to `16000`)
- `LANGLEY_MEMORY_ESTIMATED_TOKEN_BUDGET` (defaults to `8192`)
- `LANGLEY_MEMORY_POLICY_MODEL` (enables the Memory Policy when configured)
- `LANGLEY_MEMORY_POLICY_ESTIMATED_TOKEN_BUDGET` (required with the policy model)
- `LANGLEY_MAX_LLM_ROUNDS` (defaults to `4`)
- `LANGLEY_MAX_TOOL_CALLS` (defaults to `3`)
- `LANGLEY_OVERALL_WORKFLOW_DEADLINE_SECONDS` (defaults to `180`)
- `LANGLEY_TRACING_ENABLED` (defaults to `false`)
- `LANGLEY_TRACE_CONTENT_ENABLED` (defaults to `false`)
- `LANGLEY_LANGSMITH_PROJECT` (optional LangSmith project name)

Start the local MySQL 8.4 service and wait for it to become healthy:

```powershell
docker compose up -d mysql
docker compose ps mysql
docker inspect --format '{{.State.Health.Status}}' (docker compose ps -q mysql)
```

The final command must print `healthy`. Compose binds MySQL only to
`127.0.0.1:3306`.

For the local Compose service, set distinct development and test URLs. The
password below is the intentional local-only default in `compose.yaml`; do not
reuse it outside local development.

```powershell
$env:LANGLEY_DATABASE_URL = 'mysql+asyncmy://root:langley-local-root-password@127.0.0.1:3306/langley'
$env:LANGLEY_TEST_DATABASE_URL = 'mysql+asyncmy://root:langley-local-root-password@127.0.0.1:3306/langley_test'
$env:LANGLEY_LOCAL_USER_ID = '1'
```

`LANGLEY_TEST_DATABASE_URL` must target exactly `langley_test`. Integration
tests reset only that dedicated database and never fall back to the development
database.

## Backend and database

From the repository root, install the locked Python environment and apply the
migrations to the development database:

```powershell
uv sync --locked
uv run alembic upgrade head
uv run alembic current
```

Create the local/demo user explicitly after migration. This command is
idempotent; application startup and request handling do not seed users.

```powershell
uv run python -m langley.bootstrap
```

Start the backend in a separate PowerShell window:

```powershell
uv run uvicorn langley.main:create_app --factory --reload
```

### Real Qwen and LangSmith opt-in

The production composition root uses Qwen only when both its API key and base
URL are explicitly present. Without them, answer attempts fail safely with a
stable provider error; application startup makes no model request. Automated
tests inject `FakeProvider`, keep the network off, and never export LangSmith
traces.

For an explicit real-service acceptance session, set the credential and the
OpenAI-compatible base URL issued for the intended Qwen deployment in the same
PowerShell window. Do not put a real key in `.env.example`, Git, logs, or test
fixtures.

```powershell
$env:LANGLEY_QWEN_API_KEY = 'set-only-in-your-shell'
$env:LANGLEY_QWEN_BASE_URL = 'provider-issued-openai-compatible-base-url'
$env:LANGLEY_LLM_MODEL = 'qwen3.7-plus-2026-05-26'
$env:LANGLEY_TRACING_ENABLED = 'false'
$env:LANGLEY_TRACE_CONTENT_ENABLED = 'false'
uv run uvicorn langley.main:create_app --factory --reload
```

For the required content-off LangSmith smoke, opt in separately. `LANGSMITH_API_KEY`
is consumed by the LangSmith SDK, while the Langley settings remain prefixed
with `LANGLEY_`. Keep `LANGLEY_TRACE_CONTENT_ENABLED=false`: the exporter then
receives correlation metadata and activity shape only, never raw
System/User/Assistant/Tool content. Tracing is fail-open and cannot affect the
answer result.

```powershell
$env:LANGSMITH_API_KEY = 'set-only-in-your-shell'
$env:LANGLEY_TRACING_ENABLED = 'true'
$env:LANGLEY_LANGSMITH_PROJECT = 'langley-slice4-acceptance'
$env:LANGLEY_TRACE_CONTENT_ENABLED = 'false'
```

Real Qwen and LangSmith smoke/eval are manual acceptance activities: they
are never default CI commands and should run only against synthetic,
non-sensitive prompts.

The current application supports exactly one active FastAPI application worker
because execution and observation runtime remains process-local. Do not use
`uvicorn --workers > 1`. A reload supervisor is acceptable only when it leaves
one serving/executing application worker active at a time.

The health endpoint remains available at `http://127.0.0.1:8000/health` and
returns a server-generated `X-Request-ID` header.

## Frontend and chat workflow

In a second PowerShell window, start the Vite development server:

```powershell
Set-Location frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open the URL printed by Vite (normally `http://127.0.0.1:5173`). The Vite
development proxy forwards `/api` and `/health` to FastAPI, so no local CORS
configuration is needed.

The chat workflow is acceptance-based:

1. Create and select a conversation.
2. Send a question. A `202 Accepted` command response means the USER and an
   active Run are durable; it does not mean generation has succeeded. An active
   replay also returns `202`; a terminal replay returns `200`.
3. The UI observes the active Run through transient SSE events at
   `GET /api/runs/{run_id}/events`. Streamed text is display-only and is never
   treated as persisted chat history.
4. On success, stream completion is reconciled through `GET /api/runs/{run_id}`
   and persisted Conversation facts. The final ASSISTANT message appears only
   after that authoritative reconciliation.
5. Select **Stop** to request `POST /api/runs/{run_id}/cancel`. Cancellation is
   reconciled through the authoritative Run result; it does not persist a
   partial ASSISTANT message.
6. A refresh or transient SSE loss queries the Run. If it remains active, the
   UI clears its old transient prefix and subscribes again; SSE closure alone
   never implies a terminal Run.

On startup or graceful process interruption, residual PENDING/RUNNING Runs are
marked `FAILED / PROCESS_INTERRUPTED`. They are not treated as user
cancellations and can be retried through normal business semantics.

## Verification

Backend fast checks do not require MySQL:

```powershell
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run mypy src/langley
uv run pytest
```

With healthy MySQL and both database URLs explicitly set, run real MySQL
integration tests from the repository root:

```powershell
uv run pytest tests/integration
```

The integration suite uses `asyncmy`, resets the dedicated `langley_test`
database, runs migrations from empty, and covers command admission, independent
execution, terminal races, Run query/cancel, SSE observation, lifecycle repair,
transactions, locks, concurrency, Retry, Regenerate, and deterministic Personal
Context persistence, ordered / NO-HOLE processing, barriers, Memory API, and
outcome semantics.

Before frontend checks on Windows, stop any Langley Vite server started above;
otherwise its native binding can be locked.

```powershell
Set-Location frontend
npm ci
npm run lint
npm run test
npm run build
Set-Location ..
```

Manual browser acceptance remains a separate step. With the explicit real Qwen
opt-in, exercise a direct answer, `东京现在几点？` Tool loop, Stop after
generation begins, refresh/SSE recovery while active, Retry, Regenerate,
A-to-B-to-A stale callbacks, Rename, logical Delete, and terminal replay before
declaring browser acceptance complete. The Tool protocol must never appear in
the UI or MySQL Message facts.

Also smoke the Personal Context UI: load current Memory items and settings,
toggle auto-memory, directly add/edit/forget an item, inspect a source, and
observe Memory outcome feedback. This UI smoke does not replace the deferred
T10 real-model semantic evaluation.

## Current boundaries

- one active FastAPI application worker only; no multi-worker execution
- no Redis, durable queue, worker service, lease, heartbeat, or execution owner
- no durable SSE event log or `Last-Event-ID` replay
- no persisted streaming partials
- no persistent LangGraph checkpointer, RunStep, ToolInvocation, or ToolMessage
- no RAG/Knowledge, web search, model router, reflection, or multi-agent
- Personal Context Memory is current-state MySQL data with detached Policy
  execution; it has no vector index, semantic retrieval, revision/tombstone
  history, durable worker, queue, or multi-worker coordination
- Qwen and LangSmith network access are explicit manual opt-ins, never CI defaults
