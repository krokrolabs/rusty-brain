//! Data locations: one directory, everything visible, nothing hidden.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Default model for NEW brains: ModernBERT-embed on MLX/Metal (owner decision,
/// 2026-11-10). Existing brains keep their recorded model; the model only works
/// after `rusty-brain setup-mlx`. The three fastembed arms stay selectable with
/// `$RUSTY_BRAIN_MODEL` for CPU-only machines and via reembed/use-model.
pub const MODEL_ID: &str = "mlx-modernbert-4bit";
pub const MODEL_DIM: i64 = 768;

/// Which runtime computes a model's embeddings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// In-process fastembed (ONNX Runtime, CPU only).
    Fastembed,
    /// Out-of-process MLX sidecar (`rusty-brain setup-mlx`); Metal when
    /// present, CPU otherwise.
    Mlx,
}

/// A supported embedding model. The id is recorded per brain and per row;
/// switching models means re-embedding, never reusing old vectors.
#[derive(Debug)]
pub struct ModelSpec {
    pub id: &'static str,
    pub dim: i64,
    pub backend: Backend,
    /// Prefix prepended to *documents* (ModernBERT convention; the fastembed
    /// arms embed documents raw).
    pub doc_prefix: &'static str,
    /// Prefix prepended to *queries only*, never to documents (BGE/Arctic
    /// convention; MiniLM and ModernBERT document arms differ).
    pub query_prefix: &'static str,
    /// fastembed registry entry; `None` for models fastembed cannot serve.
    pub fastembed_variant: Option<fastembed::EmbeddingModel>,
    /// Upstream Hugging Face repo (or path) of the model artifact.
    pub source: &'static str,
}

pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "AllMiniLML6V2Q",
        dim: 384,
        backend: Backend::Fastembed,
        doc_prefix: "",
        query_prefix: "",
        fastembed_variant: Some(fastembed::EmbeddingModel::AllMiniLML6V2Q),
        // Upstream model card: Apache-2.0.
        source: "sentence-transformers/all-MiniLM-L6-v2",
    },
    ModelSpec {
        id: "BGESmallENV15Q",
        dim: 384,
        backend: Backend::Fastembed,
        doc_prefix: "",
        query_prefix: "Represent this sentence for searching relevant passages: ",
        fastembed_variant: Some(fastembed::EmbeddingModel::BGESmallENV15Q),
        // Upstream model card: MIT.
        source: "BAAI/bge-small-en-v1.5",
    },
    ModelSpec {
        id: "SnowflakeArcticEmbedMQ",
        dim: 768,
        backend: Backend::Fastembed,
        doc_prefix: "",
        query_prefix: "Represent this sentence for searching relevant passages: ",
        fastembed_variant: Some(fastembed::EmbeddingModel::SnowflakeArcticEmbedMQ),
        // Upstream model card: Apache-2.0.
        source: "Snowflake/snowflake-arctic-embed-m",
    },
    ModelSpec {
        id: "mlx-modernbert-4bit",
        dim: 768,
        backend: Backend::Mlx,
        doc_prefix: "search_document: ",
        query_prefix: "search_query: ",
        fastembed_variant: None,
        // Apache-2.0 (nomic-ai ModernBERT-embed-base; the mlx-community 4-bit
        // conversion inherits the license). Served by the self-owned MIT
        // sidecar; mlx is MIT and transformers is Apache-2.0 — the GPL-3.0
        // mlx-embeddings package is deliberately NOT used anywhere.
        source: "mlx-community/nomicai-modernbert-embed-base-4bit",
    },
];

pub fn model_by_id(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == id)
}

pub fn default_model() -> &'static ModelSpec {
    model_by_id(MODEL_ID).expect("default model is registered")
}

/// Model for NEW brains: `$RUSTY_BRAIN_MODEL` when set (validated against the
/// registry), else the default. Existing brains always keep their recorded
/// model regardless of the variable.
pub fn model_from_env() -> Result<&'static ModelSpec> {
    let requested = std::env::var("RUSTY_BRAIN_MODEL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    match requested {
        None => Ok(default_model()),
        Some(id) => model_by_id(&id).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown RUSTY_BRAIN_MODEL '{id}' (known models: {})",
                MODELS.iter().map(|m| m.id).collect::<Vec<_>>().join(", ")
            )
        }),
    }
}

/// Which agent wrote a memory: the harness's mount config sets this env var
/// (`claude-code`, `dsh`, …). Manual CLI use records `cli`.
pub fn agent_from_env() -> String {
    std::env::var("RUSTY_BRAIN_AGENT")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "cli".to_string())
}

/// The recall-filter control words. They are reserved for `--agent <scope>`
/// arguments and are never valid recorded agent names: the CLI's global
/// `--agent` argument and the `recall --agent` filter argument share one
/// `--agent` flag name, so a scope word can reach `Config` as an override and
/// must be discarded there instead of poisoning the recorded identity.
const AGENT_SCOPE_WORDS: [&str; 3] = ["self", "others", "all"];

fn is_agent_scope_word(value: &str) -> bool {
    AGENT_SCOPE_WORDS.contains(&value)
}

#[derive(Clone, Debug)]
pub struct Config {
    /// `~/.rusty-brain`, or `$RUSTY_BRAIN_HOME`, or `--data-dir`.
    pub home: PathBuf,
    /// Slug of the current project (a directory under `<home>/projects/`).
    pub project: String,
    /// Which agent is writing (`--agent`, or `$RUSTY_BRAIN_AGENT`, or `cli`).
    pub agent: String,
    /// Optional session identifier/name, when the harness provides one
    /// (`--session`, or `$RUSTY_BRAIN_SESSION`).
    pub session: String,
    /// Model used when creating a NEW brain (`$RUSTY_BRAIN_MODEL` or default).
    pub model: &'static ModelSpec,
}

impl Config {
    pub fn new(
        data_dir: Option<PathBuf>,
        project: Option<String>,
        agent: Option<String>,
        session: Option<String>,
    ) -> Result<Self> {
        let home = match data_dir {
            Some(d) => d,
            None => match std::env::var_os("RUSTY_BRAIN_HOME") {
                Some(d) => PathBuf::from(d),
                None => default_home()?,
            },
        };
        let project = match project {
            Some(p) => validate_project(&p)?,
            None => project_from_cwd()?,
        };
        // Discard recall-scope words (`--agent self` reaching the global arg);
        // a real agent identity then comes from `RUSTY_BRAIN_AGENT`.
        let agent = agent
            .filter(|value| !is_agent_scope_word(value))
            .unwrap_or_else(agent_from_env);
        let session = session
            .or_else(|| std::env::var("RUSTY_BRAIN_SESSION").ok())
            .unwrap_or_default();
        crate::store::validate_provenance("agent", &agent)?;
        crate::store::validate_provenance("session", &session)?;
        Ok(Self {
            home,
            project,
            agent,
            session,
            model: model_from_env()?,
        })
    }

    pub fn project_dir(&self) -> PathBuf {
        self.home.join("projects").join(&self.project)
    }

    pub fn db_path(&self) -> PathBuf {
        self.project_dir().join("brain.db")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.home.join("models")
    }

    pub fn exports_dir(&self) -> PathBuf {
        self.project_dir().join("exports")
    }

    pub fn ensure_dirs(&self) -> Result<()> {
        ensure_private_dir(&self.home)?;
        ensure_private_dir(&self.home.join("projects"))?;
        ensure_private_dir(&self.project_dir())?;
        ensure_private_dir(&self.models_dir())?;
        Ok(())
    }
}

pub fn ensure_private_dir(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path).with_context(|| format!("creating {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .with_context(|| format!("securing directory {}", path.display()))?;
    }
    Ok(())
}

pub fn set_private_file(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("securing file {}", path.display()))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn default_home() -> Result<PathBuf> {
    let base = directories::BaseDirs::new().context("no home directory available")?;
    Ok(base.home_dir().join(".rusty-brain"))
}

/// `myrepo-1a2b3c4d`: readable name plus a hash of the absolute path, so two
/// checkouts with the same directory name never share a brain.
pub fn project_from_path(path: &Path) -> Result<String> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("resolving {}", path.display()))?;
    let name = canonical
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("root")
        .to_string();
    let digest = Sha256::digest(canonical.to_string_lossy().as_bytes());
    let short = digest[..4]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    Ok(format!("{name}-{short}"))
}

fn project_from_cwd() -> Result<String> {
    project_from_path(&std::env::current_dir().context("reading current directory")?)
}

fn validate_project(p: &str) -> Result<String> {
    anyhow::ensure!(
        !p.is_empty()
            && p.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
            && !p.starts_with('.'),
        "project name must be non-empty, not start with '.', and use only [A-Za-z0-9._-]"
    );
    Ok(p.to_string())
}
