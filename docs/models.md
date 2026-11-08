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
