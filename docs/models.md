# Embedding models: choice and licenses

Default for CPU-only use: **`BGESmallENV15Q`** (BGE-small-en-v1.5, 384-dim, ~66 MB).

| Model id (fastembed) | Upstream | Dim | License (verified 2026-11-06) | Commercial use |
|---|---|---|---|---|
| `BGESmallENV15Q` | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 384 | **MIT** | yes |
| `AllMiniLML6V2Q` | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384 | **Apache-2.0** | yes |
| `SnowflakeArcticEmbedMQ` | [Snowflake/snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) | 768 | **Apache-2.0** | yes |

Verified from each model's Hugging Face metadata (`cardData.license` / `license:` tag).

Why BGE-small: +9.7 MTEB Retrieval over MiniLM at the same 384-dim width and a
modest download; Arctic-M is the higher-quality 768-dim alternative. Query
prefixes follow each model's retrieval convention (BGE/Arctic: instruction on
queries only, documents raw).

## Held for later: the MLX (Metal) option

Not implemented; evaluation on hold per owner decision. If adopted, the best
candidate is **Qwen3-Embedding-4B-mxfp8**
([mlx-community](https://huggingface.co/mlx-community/Qwen3-Embedding-4B-mxfp8),
2560-dim, ~4.15 GB, **Apache-2.0**), behind an optional Python sidecar
(`mlx-embeddings`; **GPL-3.0**, so bundling needs review). Not `mlx-rs`: it has no
embedding architectures. Adopt only if it beats BGE-small on a local recall
fixture with warm p95 < 100 ms. Quantization: prefer MXFP8 over 4-bit DWQ
(retrieval fidelity of the DWQ artifact is unvalidated).

## Measured: the MLX (Metal) option — evaluated and HELD (2026-11-09)

The gate (beat BGE-small on the local recall fixture AND warm p95 < 100 ms)
was run on this machine (M4 Max, 36 GB) against
`mlx-community/Qwen3-Embedding-4B-mxfp8` (2560-dim, Apache-2.0, via the
`mlx-embeddings` Qwen3 loader, query-instructed; documents raw). Verdict:
**HOLD — the gate fails on quality.**

| Metric | BGE-small (fastembed ONNX, CPU) | Qwen3-4B-mxfp8 (MLX, Metal) |
|---|---|---|
| Recall@5 (20 mem / 15 queries) | **0.9667** | 0.9333 |
| MRR@5 | **0.9000** | 0.8667 |
| Top-1 | 0.800 | 0.800 |
| Warm embed p50 / p95 / max | **2.4 / 3.0 / 3.6 ms** | 57.6 / 61.6 / 92.0 ms |
| RSS of the embed process | 234 MB | 3.34 GB |
| Download | 66 MB | 4.15 GB |
| Negative separation (max off-task cos; lower better) | 0.375–0.477 | **0.196–0.267** |

MLX on CPU (no Metal) is unusable for interactive recall: ~16.3 s per embed.
Vector storage at 2560-dim is ~6.7× the 384-dim default.

Quality detail: Qwen's cleaner negative separation did not convert into wins
on short code-mixed memories; its one full loss was a paraphrase of "model
switching". Switching back with `use-model` is unaffected — adoption only
changes HOW vectors are produced, and `reembed` already migrates safely.

Would reopen the case: (1) a larger/harder fixture showing Qwen's negative
separation converting at realistic corpus sizes, or (2) a mid-size (0.6B-class)
MLX embedder holding p95 well under budget. Until then: fastembed remains.
