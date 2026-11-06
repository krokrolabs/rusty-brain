# Parquet snapshot schema and publication

`rusty-brain export` writes a derived snapshot of all and only the active rows in
the current project brain. Each row carries **only the current model's embedding**
(from `memory_embeddings`, filtered on `meta.current_model`); older models' vectors
are never exported. The selection is ordered by `created_at ASC`; the
reported count is `rows.len()` from that exact selected row set, not a later
`stats` query. The SQLite/libsql brain remains the source of truth.

## Schema

Every snapshot, including a zero-active-memory snapshot, has these eight columns
in this order (the legacy six-column schema without provenance is still readable):

| Column | Arrow type | Meaning |
|---|---|---|
| `id` | required `Utf8` | Canonical memory UUID |
| `content` | required `Utf8` | Exact memory text, including non-ASCII UTF-8 |
| `tags` | required `Utf8` | Comma-separated tags; the empty string is valid |
| `embedding_model` | required `Utf8` | The brain's model (`BGESmallENV15Q` by default, `SnowflakeArcticEmbedMQ` after re-embed) |
| `created_at` | required `Utf8` | UTC RFC 3339 creation timestamp |
| `agent` | required `Utf8` | Which agent wrote the memory (`claude-code`, `dsh`, `cli`, …); empty for legacy rows |
| `session` | required `Utf8` | Session id/name when the harness provided one; empty otherwise |
| `embedding` | required `FixedSizeList<Float32>(dim)` | Exact float32 values; `dim` is the brain's model width (384 or 768) |

DuckDB 2.0.0 maps the last column to `FLOAT[]`; every returned array has exactly
the brain's model width (384 for MiniLM/BGE, 768 for Arctic). The Rust contract tests assert the Arrow fixed-size list
declaration even for zero rows. `scripts/verify_parquet.py` uses an isolated
`duckdb==2.0.0` environment and compares every exported row and every float32
value with the source database's raw little-endian vector bytes.

```bash
uv run --isolated scripts/verify_parquet.py \
  --db /path/to/brain.db \
  --parquet /path/to/memories.parquet \
  --reported-count 9
```

## Atomic, write-once publication

Final files are named
`exports/memories-<high-resolution-UTC-time>-<uuid>.parquet`, so successive
exports do not collide and no delay is used.

1. Build the complete Arrow record batch and create a unique temporary sibling
   with `create_new(true)`.
2. Write and successfully close the Parquet writer on that temporary file.
3. Publish with same-filesystem `hard_link(temp, final)`, which atomically
   creates the final name without overwriting an existing path.
4. Remove the temporary name and only then return the final path and selected
   row count.

A write or close failure removes only the temporary sibling. A hard-link
failure likewise removes the temporary sibling and leaves no new final path.
If temporary-name removal fails after the link, export deletes the just-linked
final path to roll back publication, makes a best-effort temporary cleanup, and
returns a post-publication cleanup error. No failure returns a success message
or count, and prior snapshots and the source brain are not modified.

## Verification and retained base fixture

`scripts/verify-local.sh` exercises populated, immediate successive, zero-row,
and base-revision exports. It checks source and prior-file hashes, includes a
mutated-copy and planted-final positive control, and uses debug-binary fault
controls for write, close, hard-link, and post-link cleanup stages.

`tests/fixtures/base-f586d4e/` retains a sanitized database and Parquet export
generated from `f586d4e6683842085165d808df8a5021f2f67210`; its README records
the expected rows, source commands, and SHA-256 provenance. Verification opens
a copy of the database and reads the retained Parquet file without changing
either retained artifact.
