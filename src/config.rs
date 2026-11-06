//! Data locations: one directory, everything visible, nothing hidden.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Default model: small, quantized, runs on CPU with no network at embed time.
pub const MODEL_ID: &str = "AllMiniLML6V2Q";
pub const MODEL_DIM: i64 = 384;

#[derive(Clone, Debug)]
pub struct Config {
    /// `~/.rusty-brain`, or `$RUSTY_BRAIN_HOME`, or `--data-dir`.
    pub home: PathBuf,
    /// Slug of the current project (a directory under `<home>/projects/`).
    pub project: String,
}

impl Config {
    pub fn new(data_dir: Option<PathBuf>, project: Option<String>) -> Result<Self> {
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
        Ok(Self { home, project })
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
        std::fs::create_dir_all(self.project_dir())
            .with_context(|| format!("creating {}", self.project_dir().display()))?;
        std::fs::create_dir_all(self.models_dir())
            .with_context(|| format!("creating {}", self.models_dir().display()))?;
        Ok(())
    }
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
