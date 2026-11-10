//! Local embeddings via fastembed (ONNX Runtime). No provider, no key, no
//! network at embed time; the only download ever made is the model weights,
//! cached under `<home>/models` on first use.

use anyhow::{Context, Result};
use fastembed::{TextEmbedding, TextInitOptions};
use std::path::Path;

use crate::config::{Backend, ModelSpec};

enum EmbedderModel {
    FastEmbed(Box<TextEmbedding>),
    Mlx(Box<crate::mlx::MlxEmbedder>),
    Synthetic,
}

pub struct Embedder {
    model: EmbedderModel,
    spec: &'static ModelSpec,
}

impl Embedder {
    pub fn new(models_dir: &Path, spec: &'static ModelSpec) -> Result<Self> {
        match spec.backend {
            Backend::Mlx => {
                let sidecar = crate::mlx::MlxEmbedder::new(models_dir, spec)?;
                Ok(Self {
                    model: EmbedderModel::Mlx(Box::new(sidecar)),
                    spec,
                })
            }
            Backend::Fastembed => {
                crate::config::ensure_private_dir(models_dir)?;
                let variant = spec.fastembed_variant.clone().with_context(|| {
                    format!("model {} has no fastembed registry entry", spec.id)
                })?;
                let model = TextEmbedding::try_new(
                    TextInitOptions::new(variant)
                        .with_cache_dir(models_dir.to_path_buf())
                        .with_show_download_progress(true),
                )
                .with_context(|| {
                    format!(
                        "loading embedding model {} (first run downloads weights to the models dir)",
                        spec.id
                    )
                })?;
                Ok(Self {
                    model: EmbedderModel::FastEmbed(Box::new(model)),
                    spec,
                })
            }
        }
    }

    #[doc(hidden)]
    pub fn synthetic(spec: &'static ModelSpec) -> Self {
        Self {
            model: EmbedderModel::Synthetic,
            spec,
        }
    }

    /// Document embedding (memories). The fastembed arms apply no retrieval
    /// instruction; the MLX arm applies its document prefix in the sidecar.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        match &mut self.model {
            EmbedderModel::FastEmbed(model) => {
                let mut out = model.embed(vec![text], None).context("embedding text")?;
                out.pop().context("empty embedding result")
            }
            EmbedderModel::Mlx(sidecar) => sidecar.embed(text),
            EmbedderModel::Synthetic => {
                let mut vector = vec![0.0; self.spec.dim as usize];
                let model_index = crate::config::MODELS
                    .iter()
                    .position(|candidate| candidate.id == self.spec.id)
                    .context("synthetic model is not registered")?;
                vector[model_index] = 1.0;
                // Keep text relevant without conflating model identity and width.
                let text_index = model_index + 1 + text.len() % (vector.len() - 2);
                vector[text_index] = 0.25;
                Ok(vector)
            }
        }
    }

    /// Query embedding: applies the model's retrieval instruction if it has one.
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        if let EmbedderModel::Mlx(sidecar) = &mut self.model {
            return sidecar.embed_query(text);
        }
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
