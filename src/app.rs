//! The application core shared by the CLI and the MCP server.

use anyhow::{Result, bail};
use std::sync::{Arc, Mutex};

use crate::config::Config;
use crate::embed::Embedder;
use crate::store::{Hit, Memory, Stats, Store};

const MAX_CONTENT_BYTES: usize = 100_000;

pub struct Brain {
    pub cfg: Config,
    pub store: Store,
    embedder: Arc<Mutex<Embedder>>,
}

impl Brain {
    pub async fn open(cfg: Config) -> Result<Self> {
        cfg.ensure_dirs()?;
        let store = Store::open(&cfg.db_path()).await?;
        let models = cfg.models_dir();
        // Model load (and possibly the one-time weights download) is blocking.
        let embedder = tokio::task::spawn_blocking(move || Embedder::new(&models)).await??;
        Ok(Self {
            cfg,
            store,
            embedder: Arc::new(Mutex::new(embedder)),
        })
    }

    fn embedder(&self) -> Arc<Mutex<Embedder>> {
        self.embedder.clone()
    }

    async fn embed(&self, text: String) -> Result<Vec<f32>> {
        let embedder = self.embedder();
        tokio::task::spawn_blocking(move || {
            let mut guard = embedder
                .lock()
                .map_err(|_| anyhow::anyhow!("embedding model lock poisoned"))?;
            guard.embed(&text)
        })
        .await?
    }

    pub async fn remember(&self, content: &str, tags: &str) -> Result<String> {
        if content.trim().is_empty() {
            bail!("memory content must not be empty");
        }
        if content.len() > MAX_CONTENT_BYTES {
            bail!("memory content exceeds {MAX_CONTENT_BYTES} bytes");
        }
        let embedding = self.embed(content.to_string()).await?;
        self.store.remember(content, tags, &embedding).await
    }

    pub async fn recall(&self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        if query.trim().is_empty() {
            bail!("recall query must not be empty");
        }
        let embedding = self.embed(query.to_string()).await?;
        self.store.recall(query, limit, &embedding).await
    }

    pub async fn forget(&self, id: &str) -> Result<bool> {
        self.store.forget(id).await
    }

    pub async fn recent(&self, limit: usize) -> Result<Vec<Memory>> {
        self.store.recent(limit).await
    }

    pub async fn stats(&self) -> Result<Stats> {
        self.store.stats(&self.cfg.db_path()).await
    }

    pub async fn export_parquet(&self) -> Result<(std::path::PathBuf, usize)> {
        let rows = self.store.all_for_export().await?;
        let dir = self.cfg.exports_dir();
        let path =
            tokio::task::spawn_blocking(move || crate::export::export(&dir, &rows)).await??;
        let count = self.store.stats(&self.cfg.db_path()).await?.memories;
        Ok((path, count as usize))
    }
}
