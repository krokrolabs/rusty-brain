# MCP compatibility: initial verified v1

## Transport, revisions, and local identity

`rusty-brain serve` is a trusted single-user MCP server over stdio using rmcp
3.9. stdout carries JSON-RPC protocol frames only; diagnostics, if any, use
stderr. The SDK recognizes protocol revisions `2024-11-05`, `2025-03-26`,
`2025-06-18`, `2025-11-25`, and `2026-07-28`. The real child-process suite uses
the SDK negotiation and a raw `2025-06-18` initialization.

There is no remote canonical URI. Under the feature's CT-3D exception, the local
identity tuple is **`(serverName, project, cwd)`**, with `serverName =
rusty-brain`. `RUSTY_BRAIN_HOME` selects the data root and
`RUSTY_BRAIN_PROJECT` becomes the global `--project` argument before `serve`.
The dsh overlay deliberately retains `!!js` expressions in `--dump-config`;
structural checks compare those literal strings without executing them. A live
run proves their resolved identity through `data_path`, session `meta.cwd`, and
read-only inspection of the isolated database.

The required scope list is empty because the operating-system user and explicit
project/data roots are the complete trust boundary. No credential, OAuth scope,
tenant, or hosted provider crosses the stdio boundary.

## Contract and bounds

The versioned source of truth is [`contracts/mcp-tools.json`](../contracts/mcp-tools.json).
Its five entries are sorted by raw name. dsh exposes each as
`mcp__rusty-brain__<name>`.

Every input schema has `additionalProperties: false`. Wire-shape/unknown-field
failures are MCP invalid-params errors. Semantically invalid values are bounded
`isError: true` tool results with no diagnostic chain, backtrace, or marker
credential. Error text is at most 4,096 UTF-8 bytes.

Every success has an `outputSchema`, matching `structuredContent`, and a text
fallback. Recall's fallback is the JSON hit array; in particular, limit zero
returns structured `{"hits":[],"truncated":false}` and text `[]`.

The success cap is exactly 1,000,000 compact UTF-8 bytes from
`serde_json::to_vec(CallToolResult)` after content, structured content, result
type, error flag, and metadata are populated. It excludes the JSON-RPC
envelope/id and stdio framing. A result of 1,000,001 bytes or more is replaced,
not truncated, by an error telling the caller to lower the limit or narrow the
query.

| Raw name | Shared operation | Effect and risk | Confirmation / idempotency | Tool-specific limits |
|---|---|---|---|---|
| `remember` | `Brain::remember` | Creates durable local content and its local embedding | Explicit content is the request; not idempotent, repeated calls create distinct UUIDs | Trimmed non-empty content, at most 100,000 UTF-8 bytes; optional comma-separated tags |
| `recall` | `Brain::recall` | Reads local project content through hybrid semantic and exact-keyword search | No confirmation; idempotent | Trimmed non-empty query; escaped FTS operators; limit 0–50, default 5; never reports truncation |
| `forget` | `Brain::forget` | Recoverable local soft deletion | Explicit user-supplied UUID under CT-4C; idempotent, unknown/already-forgotten UUIDs are no-ops | Canonical hyphenated lowercase UUID |
| `stats` | `Brain::stats` | Reads active count, pinned model, schema version, DB path and size | No confirmation; idempotent | Empty input; one fixed object |
| `data_path` | `Config` home/project/`db_path` | Reads selected local paths and pinned model | No confirmation; idempotent | Empty input; one fixed object |

All five have `openWorldHint: false`. `remember` has `(readOnlyHint=false,
destructiveHint=false, idempotentHint=false)`; `forget` has `(false, true,
true)`; `recall`, `stats`, and `data_path` have `(true, false, true)`.
Annotations are descriptive, not authorization controls.

The catalog records each description, input/output schema, annotations, effect,
risk, local scope, required scopes, confirmation, idempotency, output cap,
shared operation, limits, and lowercase SHA-256 hashes. Descriptions hash their
exact UTF-8 bytes. Schemas and annotations are recursively key-sorted compact
JSON with array order preserved and null-valued keys rejected. The unit suite
checks the fixed known-answer vectors independently through Python and rejects
description/schema/annotation/value drift and mutated-null controls.

## Initial corrections and future compatibility

This catalog is the deliberately corrected initial verified v1 contract under
CT-4E/FR-025. Sorted listing and additive catalog metadata are nonbreaking.
Refusing formerly accepted unknown fields or limits above 50, requiring output
schemas/structured results, and converting application failures to `isError`
are breaking for early source-install clients that relied on the unverified
behavior. Those corrections land together with the existing five names and no
alias/deprecation layer. Future breaking changes require a new tool name and a
deprecation window.

There are no cursors, filters, idempotency-key arguments, or separate
confirmation handles under CT-4B/CT-4C/CT-4D. Repeated `remember` calls are
intentional new facts; `forget` is idempotent by UUID. Recall is deterministic
bounded top-k, while stats/data_path are singleton reads.

## Verification surface

Ordinary tests cover schemas, annotations, hashes, null/drift controls, error
bounds, and exact byte boundaries without network. After priming an isolated
model cache, run:

```bash
RUSTY_BRAIN_E2E=1 RUSTY_BRAIN_HOME="$RUSTY_BRAIN_HOME" \
  RUSTY_BRAIN_PROJECT="$RUSTY_BRAIN_PROJECT" \
  cargo test --offline --test mcp_stdio -- --nocapture
```

The gated suite spawns the actual binary with rmcp's child-process transport,
checks all tools and error/lifecycle/cap paths, and captures stderr for marker
and backtrace absence. It first proves that
`examples/stdout_contaminating_server.rs` is rejected in startup,
successful-call, and handled-error modes, then accepts the real server's raw
stdout as protocol-only. `scripts/verify-dsh.sh` separately performs structural
composition and the real local dsh/Ollama journey; a dump-only run is never a
live pass.

## Provenance, recall scoping and model switching (schema 3)

- Every memory carries `agent` (which harness/CLI wrote it) and `session` (when
  the harness provided one). Brains created before provenance are migrated
  additively on open; their rows read as empty strings.
- `recall` accepts an optional `agent` argument: `all` (default), `self`,
  `others`, or an exact agent name. Both the semantic and the keyword legs are
  filtered, so `others` never leaks the caller's own rows.
- `stats` adds `by_agent`: per-agent active-memory counts, sorted by agent.
- Schema 3 keeps content in `memories` and vectors in
  `memory_embeddings (memory_id, model, dim, embedding)`; the active model is the
  `meta.current_model` pointer. Re-embedding adds vectors and flips the pointer —
  memories are never rewritten, and old models' vectors stay unless pruned.
  `reembed --prune` deletes the previous model's vectors; `--prune-all` deletes
  every older model's. Unknown model ids are refused.
- `stats` adds `models` (every model with vectors present). Exported snapshots
  join each memory with its CURRENT model's vector only (384 or 768 wide) and
  include `agent` and `session`.
