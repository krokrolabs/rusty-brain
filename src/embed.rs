//! Local embeddings via fastembed (ONNX Runtime). No provider, no key, no
//! network at embed time; the only download ever made is the model weights,
//! cached under `<home>/models` on first use.

use anyhow::{Context, Result};
use fastembed::{TextEmbedding, TextInitOptions};
use std::path::Path;

use crate::config::ModelSpec;

pub struct Embedder {
    model: TextEmbedding,
    spec: &'static ModelSpec,
}

impl Embedder {
    pub fn new(models_dir: &Path, spec: &'static ModelSpec) -> Result<Self> {
        std::fs::create_dir_all(models_dir)
            .with_context(|| format!("creating model cache at {}", models_dir.display()))?;
        let model = TextEmbedding::try_new(
            TextInitOptions::new(spec.variant.clone())
                .with_cache_dir(models_dir.to_path_buf())
                .with_show_download_progress(true),
        )
        .with_context(|| {
            format!(
                "loading embedding model {} (first run downloads weights to the models dir)",
                spec.id
            )
        })?;
        Ok(Self { model, spec })
    }

    /// Document embedding (memories). No retrieval instruction is ever added.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut out = self
            .model
            .embed(vec![text], None)
            .context("embedding text")?;
        out.pop().context("empty embedding result")
    }

    /// Query embedding: applies the model's retrieval instruction if it has one.
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        if self.spec.query_prefix.is_empty() {
            self.embed(text)
        } else {
            self.embed(&format!("{}{text}", self.spec.query_prefix))
        }
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
