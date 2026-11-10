# Embedding models: choice and licenses

Default for NEW brains on Apple-silicon Macs: **`mlx-modernbert-4bit`**
(MLX ModernBERT-embed-base 4-bit, 768-dim, ~80 MB); requires
[`rusty-brain setup-mlx`](../README.md#metal-acceleration-optional) once.

| Model id | Upstream | Dim | Runtime | License (verified 2026-11-10) | Commercial use |
|---|---|---|---|---|---|
| `mlx-modernbert-4bit` | [mlx-community/nomicai-modernbert-embed-base-4bit](https://huggingface.co/mlx-community/nomicai-modernbert-embed-base-4bit) | 768 | MLX sidecar (Metal/CPU) | **Apache-2.0** | yes |
| `BGESmallENV15Q` | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 384 | fastembed (CPU) | **MIT** | yes |
| `AllMiniLML6V2Q` | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384 | fastembed (CPU) | **Apache-2.0** | yes |
| `SnowflakeArcticEmbedMQ` | [Snowflake/snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) | 768 | fastembed (CPU) | **Apache-2.0** | yes |

Verified from each model's Hugging Face metadata (`cardData.license` / `license:`
tag).

Retrieval prefixes follow each model's convention and are applied by the
caller at embed time: ModernBERT uses `search_query: ` on queries and
`search_document: ` on documents; BGE/Arctic use
`Represent this sentence for searching relevant passages: ` on queries only;
documents are always raw on the fastembed arms.

## The MLX (Metal) backend

`mlx-modernbert-4bit` was measured on the M4 Max (2026-11-09/10 validation
harness, 20-memory/15-query recall fixture — identical or better Recall@5,
MRR@5 and top-1 than `BGESmallENV15Q`):

- Warm embed latency, batch of 1: Metal **p50 2.4 ms / p95 2.8-4.6 ms**
  (graph-compile warmup covers the 32-512 token buckets; no single call pays
  the compile tail) — incumbent-class, vs BGE's 2.4/3.0;
  CPU p50 14.1 ms / p95 14.7 ms.
- RSS 249 MB; download ~80 MB (4-bit safetensors, single file).
- Architecture: the sidecar (`sidecar/mlx_serve.py`) is self-owned MIT code
  driving `mlx.core` and a transformers tokenizer. **mlx-embeddings is NOT
  used** (GPL-3.0); mlx itself is MIT and transformers is Apache-2.0, both
  pinned in setup (`mlx==0.32.4`, `transformers==5.22.2`). The MLX backend is
  gated on one-time setup; without it, first use fails with
  `run \`rusty-brain setup-mlx\` to install the MLX sidecar venv and model`.

CPU-only machines should set `RUSTY_BRAIN_MODEL=BGESmallENV15Q` (the
previous default) before the first remember of a new brain; existing brains
keep whatever model they recorded, and `reembed`/`use-model` move them
between arms regardless of the default.
