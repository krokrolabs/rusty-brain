# Verification evidence

This tracked record contains the feature bases, commands, observed results,
fixed defects, hypotheses, initial evaluation, and limitations. Exact final
tree identity, dirty file state, and post-commit identity/results are detached
worker/PR evidence, not properties written back into this file.

## Bases and environment

- Original feature/base-fixture SHA: `f586d4e6683842085165d808df8a5021f2f67210`.
- Current base SHA: `0a32ad382e09bd83cff0f88ec6eaba92093891c5` (the `feat/embeddings-table-v3` layer audited on 2026-11-06).
- Sanitized retained fixtures: [`tests/fixtures/base-f586d4e/README.md`](../tests/fixtures/base-f586d4e/README.md) records source commands, expected rows, and SHA-256 provenance.
- Observed acceptance matrix: macOS 27.0.1 (Darwin 27), arm64; rustc/cargo 1.97.1 with package MSRV 1.88; uv 0.11.3; DuckDB 2.0.0 via uv; dsh 0.1.1-rc.2 `headless`; Ollama 0.40.15; local tool-capable `qwen3.8:27b-mlx`.
- All mutable brains, model caches, installs, sessions, and logs used isolated `/tmp/rusty-brain-verify.*` roots. No real user brain, persistent dsh profile, deepseek-harness source, or real credential was used.

## Commands and observed results

The audit-fix layer reran every command below on the current files on
2026-11-06. The isolated evidence root was `/tmp/rusty-brain-verify.cValfJ`;
the root name is evidence only, not a reusable fixture or credential.

| Command | Observed result on the implementation base |
|---|---|
| `cargo fmt --check` | PASS, exit 0. |
| `cargo clippy --all-targets --all-features -- -D warnings` | PASS, exit 0; no warnings. |
| `cargo test --offline` | PASS, exit 0; 53 tests passed (6 lib + 3 agent + 12 audit + 10 brain + 6 CLI + 3 export + 9 MLX + 4 stdio), 0 failed. Model-backed stdio checks remain explicitly gated in the ordinary suite; the MLX protocol tests run a deterministic fake sidecar with the system python, no MLX install needed. |
| `cargo build --offline` | PASS, exit 0. |
| `VERIFY_ROOT="$VERIFY_ROOT" scripts/verify-local.sh` | PASS, exit 0. Exact A–J lifecycle passed with semantic A rank 1 and keyword B rank 1; six offline fresh processes, network/probe controls, base compatibility, populated/rapid/zero/base Parquet, all vectors/counts, and write/close/hard-link/post-link cleanup faults passed. A real Arctic re-embed exported nine 768-wide vectors through DuckDB; incomplete `use-model` was refused and fully covered BGE↔Arctic pointer switches passed. |
| `RUSTY_BRAIN_E2E=1 RUSTY_BRAIN_HOME="$RUSTY_BRAIN_HOME" RUSTY_BRAIN_PROJECT="$RUSTY_BRAIN_PROJECT" cargo test --offline --test mcp_stdio -- --nocapture` | PASS, exit 0 on a primed isolated home: 4 passed, 0 failed. Real stdio catalog/results/errors/lifecycle/cap checks and all three stdout-contamination controls passed. |
| `VERIFY_ROOT="$VERIFY_ROOT" scripts/verify-dsh.sh` | PASS, exit 0. Safe structural config, one real local dsh session, ordered six-call lifecycle, final absence, token usage, and marker/chain/stack controls passed. Independent SQLite inspection verified the remembered row stored `agent=dsh` and `session=dsh-headless`. |
| `uv run --isolated scripts/verify_docs.py --self-test` | PASS, exit 0: local links/topics passed; one command-removal and eighteen troubleshooting-field-removal controls were all rejected. |
| `RUSTY_BRAIN_MLX_E2E=1 scripts/verify-mlx.sh` | PASS, exit 0 on the M4 Max: `setup-mlx` wrote the sidecar, built the pinned venv (`mlx 0.32.4`, `transformers 5.22.2`), and downloaded/size-verified the 4-bit weights into the isolated root; three remembers embedded through the live Metal sidecar, self/all recall scoping passed, `reembed BGESmallENV15Q` + `use-model mlx-modernbert-4bit` round-tripped with both vector sets retained, and the 768-wide snapshot read back through DuckDB 2.0.0. |
| `git diff --check` | PASS on the implementation base; repeated for the intended documentation tree. |

The exact public walkthrough, including explicit installation, cleanup, direct
commands, and the full gated stdio command, is in
[`README.md`](../README.md). Schema/publication details are in
[`parquet-schema.md`](parquet-schema.md); negotiated revisions, annotations,
limits, identity, and catalog behavior are in
[`mcp-compatibility.md`](mcp-compatibility.md).

## H1 — cached local embeddings with networking denied

**Confirmed.** `scripts/verify-local.sh` proved the outside HTTPS control exited
0 while the sandboxed control was denied. Six fresh-process embedding operations
then succeeded under `tests/fixtures/no-network.sb`: two remembers, four recalls,
and correct semantic/keyword results. The loopback probe observed zero
connections and zero requests with a ready cache; its empty-cache positive
control observed at least one connection and request and failed with actionable
first-download guidance without creating a brain. This is process-level denial
plus observed attempts, not an environment-flag-only offline claim.

The exact fixture passed: semantic A ranked within 3, keyword B ranked within 5
with `matched_text: true`, operator text produced no injected lexical match,
recent contained A–J before forgetting F and stats reported 9 afterward, and
explicit projects remained isolated. Active-row controls made the absence
predicates fail before forget and pass afterward. Unit tests additionally cover
ordering after reopen, UTF-8 byte boundaries, limits, UUID semantics,
model-metadata pre-mutation refusal, and second-statement transaction rollback.

## H2 — portable, atomically published Parquet

**Confirmed.** DuckDB 2.0.0 read populated, immediate-successive, zero-row,
base-revision, and post-Arctic-migration snapshots. Every active row and every
one of the 384 or 768 float32 values matched the raw source vectors; forgotten
rows were absent. Zero/base schemas retained `FixedSizeList<Float32>(384)` and
the real Arctic export used `FixedSizeList<Float32>(768)` (DuckDB `FLOAT[]`).
The selected row set supplied the reported count.

Two rapid exports had distinct final paths and preserved the first hash. The
source and retained base files remained unchanged, with mutated-copy and
planted-final controls seen to fail. Injected write, close, hard-link, and
post-link cleanup faults all exited nonzero with stage-specific stderr, no
success stdout/new final/temp artifact, and unchanged source/prior snapshots.
Post-link cleanup failure rolled back the just-linked final. See
[`parquet-schema.md`](parquet-schema.md).

## H3 and Initial live evaluation

**Confirmed.** On dsh 0.1.1-rc.2 and Ollama 0.40.15 with
`qwen3.8:27b-mlx`, one clean local scenario passed with no hosted fallback. The
ordered public calls were remember, recall, stats, data_path, forget, recall:
six total calls, with recall called twice and every other named tool once.
`data_path`, session `meta.cwd`, and independent database inspection proved the
isolated root/project/cwd and the actual BGE model, models directory, and project
directory. The remembered row stored `agent=dsh` and `session=dsh-headless`; its
token had non-null `deleted_at` after forget, active count after forget was 0,
and the final recall was absent.

This current run recorded usage from six `assistant/message` records: 10,932
input tokens, 1,583 output tokens, 12,515 derived total tokens, and 58,198
cache-read tokens. The client did not expose a separate `totalTokens` field, so
the total was input plus output. This remains the CT-5A **initial one-scenario
baseline**, not a multi-task eval suite and not a comparison with a prior
baseline. The generated live token/UUID are intentionally not copied into
tracked documentation.

Safe config validation retained literal `!!js` expressions and rejected nulls;
live evidence, not dump-config, proved resolved paths. The secrecy scanner
rejected a deliberately leaking marker/chain/stack control before accepting
actual config/stdout/stderr/JSONL/evidence. Real stdio tests likewise rejected
startup, successful-call, and handled-error stdout contamination before
accepting protocol-only server stdout.

## Defects fixed

- Memory insert + FTS insert and soft-delete + FTS delete now each commit or roll back in one libsql transaction.
- Existing incompatible `meta.embedding_model` is refused before any database/schema/metadata mutation.
- Recall ordering is exactly `score DESC, created_at ASC, id ASC`; content/query/UUID/limit validation and escaped operator behavior match the published contract.
- CLI invalid-input and incompatible-model failures now give actionable top-level diagnostics without partial success output, marker credentials, diagnostic chains, or stack traces.
- Export count comes from `rows.len()` for the exact selected/written set rather than a later stats query.
- Export writes/closes a unique temporary sibling and atomically publishes no-overwrite; write/close/hard-link/post-link cleanup failures have explicit cleanup and rollback semantics.
- MCP now has deterministic five-tool listing, closed inputs, output schemas with structured content and text fallback, truthful annotations, bounded `isError` failures, exact compact CallToolResult byte caps, and protocol-only stdout.
- The versioned catalog rejects nulls and detects description/schema/annotation/value drift with independently checked canonical hash vectors.
- Shipped dsh overlays and verification preserve literal structural expressions while checking live root/project/cwd and ordered final state.
- The first documentation-only walkthrough caught a standalone structural-check project mismatch: the documented commands inherited `rb-local-e2e` while the checker requires the live `rb-dsh-e2e` override. The walkthrough now exports `RUSTY_BRAIN_PROJECT="rb-dsh-e2e"` before those commands, and the docs checker requires that correction.
- The 2026-11-06 audit fixes reread `meta.current_model` for each Brain operation and reload cached store/embedder state by model id. Synthetic same-width BGE↔MiniLM tests cover live remember/recall/stats/export, and a remember blocked during migration retries with the new model.
- Re-embed holds one `BEGIN IMMEDIATE` transaction across its row snapshot, blocking-pool inference, pre-write width/finiteness checks, persisted-vector validation, pruning, and pointer flip. Trigger-injected prune/pointer faults roll back completely; old-model recall/export remains usable.
- Empty interrupted initialization now resumes, while nonempty memories without model metadata remain refused. Agent/session migration steps resume independently.
- `data_path` reports the active model plus model-cache/project directories; its schema/catalog and the gated BGE/schema-3 assertions were resynced. Both RRF source rankings use stable id tie-breakers.
- Query/tags/agent/session byte limits are enforced before inference/writes, with schema maxima where applicable; protocol invalid-params text uses the same 4,096-byte cap. `--prune` conflicts with `--prune-all`, and `use-model` provides covered pointer-only switches.
- Unix data/project/export directories are `0700`, and brain/Parquet files are `0600`. The dsh overlay and live verifier require dsh agent/session provenance. Agent/project remain trusted-local conveniences, not isolation boundaries.
- The 2026-11-10 MLX layer fixed a pre-existing CLI scoping defect caught by the new e2e: `recall --agent self|others` collided with the global `--agent` provenance argument (same flag; clap globals collect the value too), so the scope word reached `Config` as the recorded identity and overrode `RUSTY_BRAIN_AGENT` (`filter=Own("self")`, empty results). Scope words are now discarded from the identity so `self`/`others` resolve the calling agent (`mlx.rs`-independent; store-level SQL was verified correct).

Initial MCP corrections are classified under CT-4E/FR-025: sorted listing and
additive metadata are nonbreaking; stricter unknown-field/limit refusal,
required outputs, and application `isError` conversion are breaking for clients
that relied on the previously unverified behavior. They land together as the
initial verified v1 with existing names and no alias layer; future breaking
changes require a new name and deprecation window.

## Limitations and evidence boundary

- Only the exact matrix above is claimed. There is no broader OS/toolchain, remote-hosting, multi-user, additional-model, or additional-harness claim.
- The model download and initial dependency installation need network access. Offline claims begin only after a complete cache exists.
- H3 is one realistic six-call scenario. No multi-task pass rate, historical comparison, latency benchmark, or throughput benchmark is claimed.
- The MLX backend is optional and gated: the e2e was verified on one machine (M4 Max, Metal) with the pinned sidecar versions. No Metal-less-host, non-arm64, or other-MLX-model claim is made; the fastembed arms remain the CPU-only path.
- Retained fixtures are sanitized read-only compatibility inputs. Regenerating them with `git archive` is maintainer-only in a full clone containing the base object; a materialized docs-only tree must not attempt it.
- No Dockerfile/image/registry, hosted CI, remote telemetry/health endpoint, remote transport/OAuth/tenant, cursors, idempotency-key store, separate confirmation handle, or multi-replica qualification is added. Recall's documented agent filter is supported. The feature plan's CT-1A through CT-5C individually record these local-only exceptions; they are not unqualified constitution compliance.
- **Known limitation:** Parquet export still buffers the selected brain rows and a complete Arrow record batch in memory. Streaming export for very large brains is outside this audit-fix layer; no large-brain memory/throughput claim is made.
- The **final tree OID**, **dirty disclosure**, exact frozen-tree walkthrough result, and **post-commit SHA** with clean post-commit results are recorded only in **detached** worker/PR evidence. They must never be appended to this tracked file. Any docs correction regenerates this tracked command/result evidence, reruns all automated gates, creates a new detached tree record, and repeats the complete walkthrough before a new commit.

## Deterministic verification fixture (A–J)

The exact memories used by `scripts/verify-local.sh`:

| Key | Content | Tags |
|---|---|---|
| A | `DuckDB instances are reused per customer` | `decision,duckdb` |
| B | `Parquet exports are derived, write-once snapshots` | `principle,parquet` |
| C | `The local brain stores data under the project directory` | `storage` |
| D | `PostgreSQL connection pooling is not used by rusty-brain` | `negative` |
| E | `The MCP server communicates over standard input and output` | `mcp` |
| F | `Forgetting a memory is a soft deletion` | `lifecycle` |
| G | `The embedding model produces 384 values` | `embedding` |
| H | `Use uv to run the DuckDB verification script` | `verification` |
| I | `Ollama serves the local agent model` | `ollama` |
| J | `deepseek-harness mounts memory through a patch overlay` | `dsh` |
