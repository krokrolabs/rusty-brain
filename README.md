# rusty-brain

Local-first memory for AI coding agents. One native binary keeps one SQLite
(libsql) file per project, computes embeddings on your machine with fastembed /
ONNX Runtime, serves MCP over stdio, and exports derived Parquet snapshots. It
needs no account, API key, telemetry endpoint, or remote inference.

## Supported verification matrix

The commands and results in this repository were verified on **macOS 27.0.1,
arm64 / Darwin 27**, with **rustc/cargo 1.97.1** (the package retains **MSRV 1.88**), **uv 0.11.3**, **DuckDB 2.0.0** through uv, **dsh 0.1.1-rc.2** with
its `headless` profile, **Ollama 0.40.14**, and the already-local tool-capable
model **`qwen3.8:27b-mlx`**. `sandbox-exec`, `curl`, and normal macOS shell
utilities are also required. No broader platform or toolchain claim is made.

Network access is needed only for initial Rust/Python dependency resolution and
the first **`AllMiniLML6V2Q`** model download. The cached model has 384 float32
values per embedding. Do not use a real user brain, persistent dsh profile, or
real credential for verification.

## Install

From a full source checkout, the ordinary installation command is:

```bash
cargo install --path .
```

For the isolated acceptance walkthrough below, start at the repository root and
use only a fresh temporary root. The offline Cargo command reuses the already
populated Cargo cache; use the ordinary install command first if dependencies
are not cached.

```bash
export RB="$PWD"
export VERIFY_ROOT="$(mktemp -d /tmp/rusty-brain-verify.XXXXXX)"
export RUSTY_BRAIN_HOME="$VERIFY_ROOT/home"
export RUSTY_BRAIN_PROJECT="rb-local-e2e"
export RUSTY_BRAIN_VERIFY_ROOT="$VERIFY_ROOT"
mkdir -p "$VERIFY_ROOT/work/project" "$VERIFY_ROOT/install"
cargo install --offline --debug --path "$RB" --root "$VERIFY_ROOT/install"
export BIN="$VERIFY_ROOT/install/bin/rusty-brain"
test -x "$BIN"
```

`RUSTY_BRAIN_HOME` selects the data root. `RUSTY_BRAIN_PROJECT` selects an
explicit project for the MCP/dsh child. Direct commands accept the equivalent
global `--data-dir <path>` and `--project <name>` options before the subcommand.
Without an explicit project, the current working directory determines project
identity. `RUSTY_BRAIN_VERIFY_ROOT` is verification-only: it locates isolated
dsh sessions and evidence.

## Verify from a clean temporary root

Run these commands from `$RB` in order. Each script installs its own copy of the
current binary into `$VERIFY_ROOT/install`; none reads or writes a real user
brain or persistent dsh profile.

### 1. Static and network-free Rust gates

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --offline
cargo build --offline
```

Expected: every command exits 0. Ordinary tests use synthetic embeddings and do
not download a model. The model-backed stdio tests are skipped unless explicitly
enabled in step 4.

### 2. Local CLI, offline operation, and exports

```bash
cd "$RB"
VERIFY_ROOT="$VERIFY_ROOT" scripts/verify-local.sh
```

This primes the model once, installs the binary, creates the exact fixture
below, and verifies the complete lifecycle in fresh processes. Expected output
includes semantic A at rank 3 or better, keyword B at rank 5 or better with
`matched_text: true`, exactly ten rows before forget and nine after, six
successful sandboxed embedding operations, zero ready-cache probe connections
and requests, at least one empty-cache control connection/request, and `H1
confirmed` / `H2 confirmed`.

The script also proves the outside-network success/inside-network denial
control, missing-cache first-download guidance without a brain write, project
isolation, canonical/unknown/already-forgotten UUID behavior, read-only retained
base fixtures, populated/rapid/zero/base Parquet readback, exact selected
counts, and write/close/hard-link/post-link cleanup failures. Absence and hash
checks have mutated or planted positive controls.

### 3. Direct command reference

All commands use the same selected project. For a manual, credential-isolated
session after step 2, define this helper and run from the temporary project:

```bash
cd "$VERIFY_ROOT/work/project"
rb() {
  env -i HOME="$HOME" \
    PATH="$VERIFY_ROOT/install/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
    TMPDIR="${TMPDIR:-/tmp}" LANG=en_US.UTF-8 \
    RUSTY_BRAIN_HOME="$RUSTY_BRAIN_HOME" \
    "$BIN" --project rb-docs-manual "$@"
}
```

Then each workflow is directly reproducible:

```bash
MEMORY_ID="$(rb remember "Documentation walkthrough memory" --tags docs,local)"
rb recall "documentation walkthrough" --limit 5
rb recent --limit 10
rb stats
rb path
rb recall "documentation walkthrough" --limit 0   # []
rb forget "$MEMORY_ID"                            # forgotten <uuid>
rb forget "$MEMORY_ID"                            # no-op: no such memory
rb export                                         # exported 0 memories to <new path>
```

The equivalent installed-binary forms are `rusty-brain remember <content>
[--tags <csv>]`, `rusty-brain recall <query> [--limit <0–50>]`, `rusty-brain recent [--limit <0–50>]`, `rusty-brain stats`, `rusty-brain path`, `rusty-brain
forget <canonical-uuid>`, `rusty-brain export`, and `rusty-brain serve`.
Content is trimmed non-empty and at most 100,000 UTF-8 bytes; 100,001 bytes are
rejected without insertion. Recall/recent limits are 0–50 inclusive: zero
returns `[]`, and 51 fails. Recall requires a trimmed non-empty query, escapes
FTS operators, and orders ties exactly by `score DESC, created_at ASC, id ASC`.
Forget accepts only a canonical hyphenated UUID; unknown and already-forgotten
canonical UUIDs are successful no-ops.

Remember's memory row and FTS insert are one transaction. Forget's soft-delete and FTS delete are one transaction. A brain whose metadata names a different
embedding model is refused before any database, schema, or metadata write.

### 4. Explicitly gated real MCP stdio suite

Use the home primed in step 2. This is a separate required gate; an ordinary
`cargo test --offline` does not enable it.

```bash
cd "$RB"
RUSTY_BRAIN_E2E=1 RUSTY_BRAIN_HOME="$RUSTY_BRAIN_HOME" \
  RUSTY_BRAIN_PROJECT="$RUSTY_BRAIN_PROJECT" \
  cargo test --offline --test mcp_stdio -- --nocapture
```

Expected: the real child initializes and lists exactly `data_path`, `forget`,
`recall`, `remember`, `stats`; lifecycle, schemas, annotations, errors, and byte
caps pass. The suite rejects the startup, successful-call, and handled-error
stdout contamination controls before accepting the real server's protocol-only
stdout. Application error text is at most 4,096 UTF-8 bytes. A successful MCP
`CallToolResult` is at most 1,000,000 compact UTF-8 JSON bytes; 1,000,001 bytes
becomes an actionable `isError` result. See [MCP compatibility and tool
bounds](docs/mcp-compatibility.md).

### 5. Real dsh + Ollama integration

Keep Ollama running with `qwen3.8:27b-mlx` available. Then run:

```bash
cd "$RB"
export PATH="$VERIFY_ROOT/install/bin:$PATH"
VERIFY_ROOT="$VERIFY_ROOT" scripts/verify-dsh.sh
```

The script uses the shipped [Ollama overlay](examples/deepseek-harness/ollama-local.cordis.yml)
and [rusty-brain stdio overlay](examples/deepseek-harness/rusty-brain.cordis.yml)
with `dsh --profile headless --patch ...`; deepseek-harness itself is never
edited. Its structural phase runs this composition and validates the dumped YAML
without executing `!!js`:

```bash
cd "$VERIFY_ROOT/work/project"
export RUSTY_BRAIN_PROJECT="rb-dsh-e2e"
dsh --profile headless \
  --patch "$RB/examples/deepseek-harness/ollama-local.cordis.yml" \
  --patch "$RB/examples/deepseek-harness/rusty-brain.cordis.yml" \
  --dump-config > "$VERIFY_ROOT/dsh-config.yml"
uv run --isolated "$RB/scripts/verify_dsh_config.py" \
  --repo "$RB" --dump "$VERIFY_ROOT/dsh-config.yml" \
  --root "$VERIFY_ROOT" --project-cwd "$VERIFY_ROOT/work/project"
```

The live phase uses `env -i` with an explicit allowlist, marker-only credential
variables, disabled hosted provider/web/telemetry, and an initially empty
session root. It requires exactly one new `session*.jsonl`, ordered calls to
remember → recall → stats → data_path → forget → recall, independently checked
final database absence, matching root/project/cwd, and every available
`assistant/message` / `data.usage` token record. Its final line is `H3 PASS`.
The exact task and session field paths are implemented in
[`scripts/verify-dsh.sh`](scripts/verify-dsh.sh) and
[`scripts/verify_dsh_session.py`](scripts/verify_dsh_session.py).

### 6. Documentation contract and positive controls

```bash
cd "$RB"
uv run --isolated scripts/verify_docs.py --self-test
```

Expected: local links and required topics pass, all six troubleshooting
categories have all three fields, and one required-command removal plus each of
the eighteen symptom/cause/remedy removals is rejected. Use
`uv run --isolated scripts/verify_docs.py` for the static check alone.

## Deterministic lifecycle fixture

`scripts/verify-local.sh` inserts these exact rows in order:

| Key | Exact content | Tags |
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

Semantic query `customer-specific database instances stay warm`, limit 5, must
place A at rank 3 or better. Keyword query `write-once`, limit 5, must place B at
rank 5 or better with `matched_text: true`. Operator query `" OR * : ( )`, limit
5, must succeed without an injected lexical match. `recent --limit 10` contains
exactly A–J before F is forgotten and nine active rows afterward.

## Storage and local data

Everything lives under `~/.rusty-brain` by default, or under the selected
`RUSTY_BRAIN_HOME` / `--data-dir`:

```text
<home>/
  models/                          # AllMiniLML6V2Q, downloaded once
  projects/<project>/              # explicit name or cwd-derived name+hash
    brain.db                       # authoritative libsql/SQLite store
    exports/memories-*.parquet     # derived write-once snapshots
```

`rusty-brain path` prints home, project, brain, model, and export paths without
initializing a database or downloading a model. Two cwd-derived projects with
the same basename remain isolated by path hash. Forgetting is a recoverable
soft deletion: the source row retains `deleted_at`, while recall, recent, stats,
and export exclude it.

## Parquet and DuckDB

`rusty-brain export` selects all and only active memories and reports the exact
selected row count. Every file, including zero rows, contains `id`, `content`,
`tags`, `embedding_model`, `created_at`, and a 384-element float32 `embedding`.
It writes a unique `create_new(true)` temporary sibling, closes the writer, and
publishes through same-filesystem atomic no-overwrite `hard_link` publication.
Failures clean up without a success claim, new final artifact, source mutation,
or prior-snapshot mutation. See [the complete Parquet schema and publication
contract](docs/parquet-schema.md).

Read any snapshot independently with DuckDB 2.0.0:

```sql
SELECT id, content, tags, embedding_model, created_at, embedding
FROM read_parquet('/absolute/path/to/memories.parquet');
```

Or compare every active source row and all vector values:

```bash
uv run --isolated "$RB/scripts/verify_parquet.py" \
  --db "$RUSTY_BRAIN_HOME/projects/rb-local-e2e/brain.db" \
  --parquet "$(awk '{print $NF}' "$VERIFY_ROOT/export-1.stdout")" \
  --reported-count 9
```

The retained sanitized base artifacts are read-only compatibility inputs:

```bash
cd "$RB"
uv run --isolated scripts/verify_parquet.py \
  --db tests/fixtures/base-f586d4e/brain.db \
  --parquet tests/fixtures/base-f586d4e/memories.parquet \
  --reported-count 2
```

Expected: two active rows and exact vectors; the forgotten base row remains
excluded. `scripts/generate_base_fixture.sh` is **maintainer-only** in a full Git
clone containing the base object. An exported documentation-only acceptance
tree has no `.git`; do not regenerate fixtures there.

## Troubleshooting

### 1. Missing or incomplete model cache

- **Observable symptom:** An offline remember/recall exits nonzero, MCP returns `isError: true`, and the diagnostic says the first run must download the model; no brain should be created for a new project.
- **Likely cause:** `AllMiniLML6V2Q` is absent or incomplete under `$RUSTY_BRAIN_HOME/models`, or the command selected a different home.
- **Corrective action:** With network enabled, rerun `scripts/verify-local.sh` against the same `$VERIFY_ROOT`; confirm a non-empty file exists under `$RUSTY_BRAIN_HOME/models`, then rerun the offline and probe checks.

### 2. Unavailable dsh, Ollama, or local model

- **Observable symptom:** `dsh --version`, `ollama --version`, or `ollama show qwen3.8:27b-mlx` fails, the model lacks `tools`, or `verify-dsh.sh` exits before `H3 PASS`.
- **Likely cause:** dsh/Ollama is not installed or on `PATH`, Ollama is not running, or the pinned local model is absent or not tool-capable.
- **Corrective action:** Install/select dsh 0.1.1-rc.2 and Ollama 0.40.14, start Ollama, make `qwen3.8:27b-mlx` locally available, verify `ollama show` reports tools, and rerun the live script without enabling hosted fallback.

### 3. Missing rusty-brain executable

- **Observable symptom:** `test -x "$BIN"` fails, `rusty-brain` is not found, or the MCP child cannot start.
- **Likely cause:** Installation did not complete, or `$VERIFY_ROOT/install/bin` is absent from the client `PATH`.
- **Corrective action:** Run the documented offline isolated install, verify `test -x "$VERIFY_ROOT/install/bin/rusty-brain"`, prepend that directory to `PATH`, and rerun composition and the live check.

### 4. Invalid mount, config, cwd, or override

- **Observable symptom:** `verify_dsh_config.py` rejects literal `!!js` fields, tools do not appear, or live `data_path`, session `meta.cwd`, and the isolated database disagree.
- **Likely cause:** The profile/patch order is wrong, `RUSTY_BRAIN_HOME`/`RUSTY_BRAIN_PROJECT` is missing or inconsistent, or dsh was launched outside `$VERIFY_ROOT/work/project`.
- **Corrective action:** Use both shipped overlays in the documented order, preserve the literal expressions, export both overrides, launch from the documented temporary cwd, and rerun structural plus live checks; never resolve the expressions by editing the YAML.

### 5. Failed Parquet or DuckDB readback

- **Observable symptom:** `verify_parquet.py` reports a column/type, row-count, metadata, or float32 mismatch, or DuckDB cannot read a snapshot.
- **Likely cause:** The database and Parquet paths belong to different projects/snapshots, the reported count is not the selected row count, or the reader is not DuckDB 2.0.0.
- **Corrective action:** Use the matching `brain.db` and export printed by the same run, pass its printed count, run through `uv run --isolated scripts/verify_parquet.py`, and inspect the six-column/384-value schema; do not repair or overwrite the source or prior snapshot.

### 6. Failed network-denial positive control

- **Observable symptom:** The outside curl does not succeed, the sandboxed curl succeeds, the ready-cache probe observes a connection/request, or the empty-cache control observes none.
- **Likely cause:** External network is unavailable for the outside control, `sandbox-exec` is not enforcing `tests/fixtures/no-network.sb`, the model cache is not ready, or the probe endpoint/state is stale.
- **Corrective action:** Restore initial-download connectivity, use a fresh `$VERIFY_ROOT`, rerun `verify-local.sh`, and require outside=0/inside nonzero plus zero ready-cache and at least one empty-cache observation before trusting offline evidence.

## Evidence and limitations

[`docs/verification.md`](docs/verification.md) records the original and current
base SHA, exact commands, observed results, fixed defects, H1–H3, the initial
one-scenario evaluation, and limitations. It intentionally does not identify
its own final tree OID, dirty disclosure, or post-commit SHA: those belong only
in detached worker/PR evidence so the tracked evidence does not rewrite itself.
Any documentation correction requires regenerating tracked results, rerunning
all automated gates, freezing a new detached tree, and repeating the complete
clean-tree walkthrough before commit. After commit, all gates and the complete
walkthrough run again and their results map to that commit only in detached
evidence.

## Cleanup

Retain `$VERIFY_ROOT` while inspecting evidence. When finished, remove only the
run-specific guarded temporary root; do not delete a real home, checkout, or dsh
profile:

```bash
case "$VERIFY_ROOT" in
  /tmp/rusty-brain-verify.*) rm -rf -- "$VERIFY_ROOT" ;;
  *) printf 'refusing unexpected VERIFY_ROOT: %s\n' "$VERIFY_ROOT" >&2; exit 2 ;;
esac
```

For normal use, delete a whole project only after confirming the exact path from
`rusty-brain path`; individual memories should normally be soft-deleted with
`rusty-brain forget <id>`.
