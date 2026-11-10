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

## Evaluation history: the two MLX candidates measured on this machine

The adoption gate (beat the incumbent on the local recall fixture AND warm
p95 < 100 ms) was run empirically per candidate, on an M4 Max with a
20-memory/15-query fixture plus off-task negatives.

**ModernBERT-embed-base 4-bit — ADOPTED** (sidecar is self-owned MIT code
driving `mlx.core`; `mlx-embeddings` (GPL-3.0) is NOT used):

- Warm embed, batch of 1: Metal **p50 2.4 ms / p95 2.8-4.6 ms** (bucketed
  graph-compile warmup covers 32-512-token shapes; no call pays the compile
  tail) — incumbent-class, vs BGE's 2.4/3.0; CPU-only **p50 14.1 ms /
  p95 14.7 ms**.
- Recall@5 **1.000 / MRR@5 0.9467 / top-1 0.9333** — better on every metric
  than BGE-small (0.9667 / 0.9000 / 0.800); rankings reproduced by the
  self-owned encoder exactly (all 15 fixture queries).
- RSS 249 MB; download ~80 MB (single 4-bit safetensors).
- mlx MIT and transformers Apache-2.0, pinned in setup
  (`mlx==0.32.4`, `transformers==5.22.2`); first use without setup fails with
  `run rusty-brain setup-mlx to install the MLX sidecar venv and model`.
- CPU-only machines can set `RUSTY_BRAIN_MODEL=BGESmallENV15Q` before the
  first remember of a new brain; existing brains keep their recorded model,
  and `reembed`/`use-model` move them between arms regardless of the default.

**Qwen3-Embedding-4B-mxfp8 — evaluated and HELD (2026-11-09)** (2560-dim,
~4.15 GB, Apache-2.0, via the `mlx-embeddings` Qwen3 loader; its recall was
below the incumbent, so it was not adopted as the backend while ModernBERT
was):

| Metric | BGE-small (fastembed ONNX, CPU) | Qwen3-4B-mxfp8 (MLX, Metal) |
|---|---|---|
| Recall@5 (20 mem / 15 queries) | **0.9667** | 0.9333 |
| MRR@5 | **0.9000** | 0.8667 |
| Top-1 | 0.800 | 0.800 |
| Warm embed p50 / p95 / max | **2.4 / 3.0 / 3.6 ms** | 57.6 / 61.6 / 92.0 ms |
| RSS of the embed process | 234 MB | 3.34 GB |
| Download | 66 MB | 4.15 GB |
| Negative separation (max off-task cos; lower better) | 0.375–0.477 | **0.196–0.267** |

Qwen's cleaner negative separation did not convert into wins on short
code-mixed memories; its one full loss was a paraphrase of "model switching".
Also measured: MLX on CPU (no Metal) is unusable for a 4B model (~16.3 s per
embed) while the 149M ModernBERT runs fine on CPU (14.7 ms p95).
