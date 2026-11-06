//! The application core shared by the CLI and the MCP server.

use anyhow::Result;
use std::sync::{Arc, Mutex};
use tokio::sync::OnceCell;

use crate::config::Config;
use crate::embed::Embedder;
use crate::store::{
    Hit, Memory, Stats, Store, validate_content, validate_limit, validate_memory_id, validate_query,
};

pub struct Brain {
    pub cfg: Config,
    store: OnceCell<Store>,
    embedder: OnceCell<Arc<Mutex<Embedder>>>,
}

impl Brain {
    pub async fn open(cfg: Config) -> Result<Self> {
        Ok(Self {
            cfg,
            store: OnceCell::new(),
            embedder: OnceCell::new(),
        })
    }

    async fn store(&self) -> Result<&Store> {
        self.store
            .get_or_try_init(|| async {
                self.cfg.ensure_dirs()?;
                Store::open(&self.cfg.db_path()).await
            })
            .await
    }

    async fn embedder(&self) -> Result<Arc<Mutex<Embedder>>> {
        let embedder = self
            .embedder
            .get_or_try_init(|| async {
                let models = self.cfg.models_dir();
                let embedder =
                    tokio::task::spawn_blocking(move || Embedder::new(&models)).await??;
                Ok::<_, anyhow::Error>(Arc::new(Mutex::new(embedder)))
            })
            .await?;
        Ok(embedder.clone())
    }

    async fn embed(&self, text: String) -> Result<Vec<f32>> {
        // Existing incompatible brains are rejected before model/cache writes.
        // For a new brain, model readiness precedes Store::open, so an offline
        // empty cache cannot create the database/schema/meta tables.
        Store::validate_existing_model(&self.cfg.db_path()).await?;
        let embedder = self.embedder().await?;
        tokio::task::spawn_blocking(move || {
            let mut guard = embedder
                .lock()
                .map_err(|_| anyhow::anyhow!("embedding model lock poisoned"))?;
            guard.embed(&text)
        })
        .await?
    }

    pub async fn remember(&self, content: &str, tags: &str) -> Result<String> {
        validate_content(content)?;
        let embedding = self.embed(content.to_string()).await?;
        self.store()
            .await?
            .remember(content, tags, &embedding)
            .await
    }

    pub async fn recall(&self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        validate_query(query)?;
        validate_limit(limit)?;
        Store::validate_existing_model(&self.cfg.db_path()).await?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let embedding = self.embed(query.to_string()).await?;
        self.store().await?.recall(query, limit, &embedding).await
    }

    pub async fn forget(&self, id: &str) -> Result<bool> {
        validate_memory_id(id)?;
        self.store().await?.forget(id).await
    }

    pub async fn recent(&self, limit: usize) -> Result<Vec<Memory>> {
        validate_limit(limit)?;
        Store::validate_existing_model(&self.cfg.db_path()).await?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        self.store().await?.recent(limit).await
    }

    pub async fn stats(&self) -> Result<Stats> {
        self.store().await?.stats(&self.cfg.db_path()).await
    }

    pub async fn export_parquet(&self) -> Result<(std::path::PathBuf, usize)> {
        let rows = self.store().await?.all_for_export().await?;
        let dir = self.cfg.exports_dir();
        tokio::task::spawn_blocking(move || crate::export::export(&dir, &rows)).await?
    }
}
