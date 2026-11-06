//! Local embeddings via fastembed (ONNX Runtime). No provider, no key, no
//! network at embed time; the only download ever made is the model weights,
//! cached under `<home>/models` on first use.

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::path::Path;

pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    pub fn new(models_dir: &Path) -> Result<Self> {
        let model = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::AllMiniLML6V2Q)
                .with_cache_dir(models_dir.to_path_buf())
                .with_show_download_progress(true),
        )
        .context("loading embedding model (first run downloads weights to the models dir)")?;
        Ok(Self { model })
    }

    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut out = self
            .model
            .embed(vec![text], None)
            .context("embedding text")?;
        out.pop().context("empty embedding result")
    }
}

/// `vector32()` in libsql parses this JSON-ish form. At brain scale (thousands
/// of rows) the parse cost is noise; correctness first.
pub fn to_vector_literal(v: &[f32]) -> String {
    let mut s = String::with_capacity(v.len() * 12);
    s.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{x}"));
    }
    s.push(']');
    s
}
