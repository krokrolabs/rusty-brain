# rusty-brain

Local-first memory for AI coding agents. One binary, one SQLite (libsql) file per project,
embeddings computed **on your machine** by [fastembed](https://github.com/Anush008/fastembed-rs)
(ONNX Runtime). No account, no API keys, no telemetry, no network at embed time.

```bash
cargo install --path .

rusty-brain remember "DuckDB instances are reused per customer" --tags decision,duckdb
rusty-brain recall "how do we route duckdb jobs"
rusty-brain path            # shows exactly where every byte lives
rusty-brain stats
rusty-brain export          # new Parquet file, readable by DuckDB/pandas/Spark
rusty-brain serve           # MCP server on stdin/stdout (how harnesses mount it)
```

## Where your data is (the transparency contract)

Everything lives under one directory — **`~/.rusty-brain`**, or `$RUSTY_BRAIN_HOME`, or `--data-dir`:

```
~/.rusty-brain/
  models/                          # ONNX weights, downloaded once, cached here
  projects/<name>-<path-hash>/     # one directory per project (cwd-derived)
    brain.db                       # the ONLY source of truth (libsql/SQLite)
    exports/memories-*.parquet     # derived snapshots, new file per export
```

- **Project scoping:** the project is derived from the current working directory
  (`myrepo-a1b2c3d4`). Two checkouts of the same repo name never share a brain — the slug
  includes a hash of the absolute path. Override with `--project <name>`.
- **Inspect it yourself:** `brain.db` is a normal SQLite file. `sqlite3 brain.db
  'SELECT content FROM memories'` works; nothing is encrypted or hidden.
- **Delete it yourself:** forgetting is per-id (`rusty-brain forget <id>`); deleting the whole
  project is `rm -rf` on its directory. There is nothing else, anywhere else.
- **Parquet exports are derived, write-once artifacts.** The brain file stays the truth; an export
  is a new file per run, re-derivable at any time, safe to feed to analytics (DuckDB reads them
  directly, embeddings included as `FLOAT[]`).

## Embeddings: local, pinned, never mixed

- Model: **`AllMiniLML6V2Q`** (384-dim, quantized ONNX, CPU). The weights download once from
  Hugging Face into `models/`; after that, embedding is fully offline. Prove it: unplug and recall
  still works.
- The model id is recorded **per row** (`embedding_model`) and pinned per brain: opening a brain
  created with a different model is refused. Comparing vectors across embedding spaces returns
  plausible-looking garbage, so rusty-brain makes that impossible instead of convenient.
- Recall is **hybrid**: cosine similarity over the embeddings plus SQLite FTS5 keyword match,
  merged with reciprocal rank fusion. FTS input is escaped — user text cannot inject FTS operators.

## Mount it in deepseek-harness (or any MCP client)

Same pattern as the harness's `apps/cli/config/examples/mcp-memory/` references — add this fragment
to your `cordis.yml` (see `examples/deepseek-harness/rusty-brain.cordis.yml`):

```yaml
- insert:
    - id: memory-rusty-brain
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: rusty-brain
        transport: stdio
        command: rusty-brain
        args: [serve]
        cwd: !!js process.cwd()   # the brain is scoped to this project directory
```

Tools exposed to the agent: `remember`, `recall`, `forget`, `stats`, `data_path`.
The agent can always answer "where is my data?" by calling `data_path`.

## Development

```bash
cargo test        # store tests use synthetic embeddings; no model download needed
cargo build       # debug binary at target/debug/rusty-brain
```

Stack: `libsql` (local SQLite), `fastembed` (local embeddings), `rmcp` (official MCP Rust SDK),
`arrow`/`parquet` (exports), `clap` (CLI). MSRV 1.88.
