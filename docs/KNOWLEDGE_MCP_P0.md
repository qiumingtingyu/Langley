# Knowledge MCP P0 local operation

Status: LOCAL_PASS / REVIEW_PENDING; independent Review and Owner approval pending.
This is a separate, read-only, single-worker loopback process. It does not start the
Web application, Run recovery, index schedulers, or a paid provider.

## Startup and credentials

Install the MCP and retrieval dependency groups with
`uv sync --locked --group mcp --group retrieval`.
The `retrieval` group includes `eval` (local embedding dependencies) and `qdrant`
(Qdrant client); both are required by MCP Search.
Use the existing controlled DB, source-storage and Qdrant environment for the same
instance. No migration or database initialization is part of MCP startup.

Create a private JSON configuration outside source control. Example shape below uses
fake IDs and an unusable digest; replace them with independently verified owned IDs:

```json
{"port":8766,"credentials":[{"principal_id":"client_a","token_digest":"0000000000000000000000000000000000000000000000000000000000000000","user_id":123,"allowed_kb_ids":[456],"enabled":true}]}
```

Generate each token with a cryptographic generator (`secrets.token_urlsafe(32)`).
Store only its SHA-256 hex digest in the server config. Deliver the original token
privately to the client. Protect both files with Windows ACLs; never print tokens or
connection strings, pass tokens in URLs, or commit either file. `principal_id` is a
non-sensitive audit alias. Tokens for the same principal share rate/capacity limits
but retain their individual KB grants. Every grant also requires live DB ownership.

Run `uv run --group mcp --group retrieval --no-sync python -m langley.mcp.server --config <private-config.json>`
in the controlled environment after installing both groups above; `--no-sync`
does not install missing dependencies. The server always binds `127.0.0.1`, disables proxy
headers and access logs, and exposes POST `/mcp`. The client sends
`Authorization: Bearer <private-token>`. Configure the standard SDK transport with
this header on every request. No Origin is required for a native client; supplied
Host/Origin must exactly match localhost or 127.0.0.1 with the configured port.
GET/DELETE are not supported by this stateless JSON-response profile.

To revoke or change grants: stop the process, update the private config, then
restart. P0 does not support hot revocation or OAuth. Do not expose it publicly or
increase workers; rate/capacity controls are process-local. Local embedding cache
must exist; network model download is disabled by the entry point.

## Tools and boundaries

Only `search_knowledge`, `read_knowledge`, `inspect_knowledge` are registered.
Search returns bounded snippets and versioned locators. Read verifies grant,
ownership, document/version/revision before and after native reading; it rejects
oversized text without silently truncating it. Inspect accepts a document locator
and returns bounded structural pagination. A readable structural locator can still
exceed the requested Read byte budget. PDF reads return published extracted text
with actual covered pages, which can be wider than the requested page.

HTTP body 16 KiB; complete Tool result 32 KiB; Read content up to 16 KiB;
Search top_k 1–5; per-principal tools 30/min burst 5; authenticated HTTP 120/min
burst 20; unauthenticated token failures 60/min burst 10; 8 inflight HTTP requests;
2 tools globally, 1 per principal, 1 Search. Search deadline 30s, Read/Inspect 10s.
Timeout/disconnect does not release capacity while underlying work is still active.
Shutdown drains this work; it cannot forcibly terminate a stuck native thread.

Errors use fixed public codes and request IDs. Audit fields exclude query, source
text, headers, credentials, raw exception strings and arbitrary protocol values.
Returned Knowledge content is untrusted data, never authority for another tool.

## Verified scope

Official Python MCP 2.3.0 Client over actual loopback sockets: modern 2026-07-28
and legacy initialize 2025-11-25. A separate local Codex CLI 0.163.0-alpha.10.2 session also verified tool
discovery and Search / Read / Inspect with synthetic sources, citations and audit
correlation. This records historical client acceptance, not a compatibility promise
for other clients or versions.
Real local BGE-M3/CUDA, temporary Qdrant and additive `langley_test` fixtures were
used. PDF fixture publication was seeded; Docling/OCR ingestion was not exercised.
Web Retrieval router regression uses ASGI with real retrieval and DB, without Web
background jobs. No production deployment, paid model or browser UI acceptance.

For deployment verification, run the focused MCP security and integration tests
under `tests/fast/test_mcp_security.py`, `tests/integration/test_mcp_knowledge.py`
and `tests/integration/test_mcp_http.py`. Real integration requires an explicitly
configured isolated test database and synthetic fixtures; it never initializes or
migrates a database. Private acceptance records are intentionally not distributed.
