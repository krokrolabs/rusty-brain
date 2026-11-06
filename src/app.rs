//! The application core shared by the CLI and the MCP server.

use anyhow::{Context, Result, bail};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::OnceCell;

use crate::config::{Config, MODEL_ID, ModelSpec, model_by_id};
use crate::embed::Embedder;
use crate::store::{
    AgentFilter, Hit, Memory, Stats, Store, validate_content, validate_limit, validate_memory_id,
    validate_query,
};

pub struct Brain {
    pub cfg: Config,
    store: OnceCell<Store>,
    embedder: OnceCell<Arc<Mutex<Embedder>>>,
}

#[derive(Debug)]
pub struct ReembedReport {
    pub from_model: String,
    pub to_model: String,
    pub memories: usize,
    pub pruned_models: Vec<String>,
}

/// What to do with older models' vectors after a successful re-embed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Prune {
    /// Keep every model's vectors (default; switch-back is a pointer flip).
    Keep,
    /// Delete only the previous model's vectors.
    Old,
    /// Delete every model's vectors except the new one.
    All,
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

    /// The model this brain speaks: the recorded one for an existing brain,
    /// the binary default for a new one.
    fn model_spec(&self, recorded: Option<String>) -> &'static ModelSpec {
        recorded
            .as_deref()
            .and_then(model_by_id)
            .unwrap_or_else(|| crate::config::default_model())
    }

    async fn embedder(&self) -> Result<Arc<Mutex<Embedder>>> {
        let embedder = self
            .embedder
            .get_or_try_init(|| async {
                let spec = self.model_spec(Store::model_of(&self.cfg.db_path()).await?);
                let models = self.cfg.models_dir();
                let embedder =
                    tokio::task::spawn_blocking(move || Embedder::new(&models, spec)).await??;
                Ok::<_, anyhow::Error>(Arc::new(Mutex::new(embedder)))
            })
            .await?;
        Ok(embedder.clone())
    }

    async fn embed(&self, text: String) -> Result<Vec<f32>> {
        // Existing incompatible brains are rejected before model/cache writes.
        Store::model_of(&self.cfg.db_path()).await?;
        let embedder = self.embedder().await?;
        tokio::task::spawn_blocking(move || {
            let mut guard = embedder
                .lock()
                .map_err(|_| anyhow::anyhow!("embedding model lock poisoned"))?;
            guard.embed(&text)
        })
        .await?
    }

    async fn embed_query(&self, text: String) -> Result<Vec<f32>> {
        Store::model_of(&self.cfg.db_path()).await?;
        let embedder = self.embedder().await?;
        tokio::task::spawn_blocking(move || {
            let mut guard = embedder
                .lock()
                .map_err(|_| anyhow::anyhow!("embedding model lock poisoned"))?;
            guard.embed_query(&text)
        })
        .await?
    }

    pub async fn remember(&self, content: &str, tags: &str) -> Result<String> {
        validate_content(content)?;
        let embedding = self.embed(content.to_string()).await?;
        let agent = self.cfg.agent.clone();
        let session = self.cfg.session.clone();
        self.store()
            .await?
            .remember(content, tags, &agent, &session, &embedding)
            .await
    }

    pub async fn recall(&self, query: &str, limit: usize, filter: AgentFilter) -> Result<Vec<Hit>> {
        validate_query(query)?;
        validate_limit(limit)?;
        Store::model_of(&self.cfg.db_path()).await?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let embedding = self.embed_query(query.to_string()).await?;
        self.store()
            .await?
            .recall(query, limit, &filter, &embedding)
            .await
    }

    pub fn agent_filter(&self, arg: Option<&str>) -> AgentFilter {
        match arg {
            None | Some("all") => AgentFilter::All,
            Some("self") => AgentFilter::Own(self.cfg.agent.clone()),
            Some("others") => AgentFilter::Others(self.cfg.agent.clone()),
            Some(name) => AgentFilter::Named(name.to_string()),
        }
    }

    pub async fn forget(&self, id: &str) -> Result<bool> {
        validate_memory_id(id)?;
        self.store().await?.forget(id).await
    }

    pub async fn recent(&self, limit: usize) -> Result<Vec<Memory>> {
        validate_limit(limit)?;
        Store::model_of(&self.cfg.db_path()).await?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        self.store().await?.recent(limit).await
    }

    pub async fn stats(&self) -> Result<Stats> {
        self.store().await?.stats(&self.cfg.db_path()).await
    }

    pub async fn export_parquet(&self) -> Result<(PathBuf, usize)> {
        let store = self.store().await?;
        let dim = store.dim() as usize;
        let rows = store.all_for_export().await?;
        let dir = self.cfg.exports_dir();
        tokio::task::spawn_blocking(move || {
            crate::export::export_with_dim(&dir, &rows, dim, crate::export::injected_fault()?)
        })
        .await?
    }

    /// Re-embed the whole brain with another model. Ids, tags, timestamps,
    /// provenance and soft-deletion state are preserved; vectors are always
    /// regenerated, never reused. Old vectors stay in `memory_embeddings`, so
    /// switching back is a pointer flip — nothing is deleted.
    pub async fn reembed(&self, new_model_id: &str, prune: Prune) -> Result<ReembedReport> {
        let spec =
            model_by_id(new_model_id).with_context(|| format!("unknown model {new_model_id}"))?;
        let models = self.cfg.models_dir();
        let embedder = tokio::task::spawn_blocking(move || Embedder::new(&models, spec)).await??;
        let embedder = Mutex::new(embedder);
        self.reembed_with(spec, prune, move |text| {
            let mut guard = embedder
                .lock()
                .map_err(|_| anyhow::anyhow!("embedding model lock poisoned"))?;
            guard.embed(&text)
        })
        .await
    }

    /// The migration core, with the embedding function injected (tests use
    /// synthetic embeddings, so no model download is needed).
    pub async fn reembed_with(
        &self,
        spec: &'static ModelSpec,
        prune: Prune,
        embed: impl Fn(String) -> Result<Vec<f32>>,
    ) -> Result<ReembedReport> {
        let db_path = self.cfg.db_path();
        let from_model = Store::model_of(&db_path)
            .await?
            .unwrap_or_else(|| MODEL_ID.to_string());
        if from_model == spec.id {
            bail!("this brain already uses model {}", spec.id);
        }
        let store = self.store().await?;
        let rows = store.raw_rows().await?;
        let total = rows.len();

        for row in &rows {
            let embedding = embed(row.content.clone())?;
            store
                .insert_embedding(&row.id, spec.id, spec.dim, &embedding)
                .await?;
        }
        let written = store.count_embeddings(spec.id).await?;
        anyhow::ensure!(
            written as usize == total,
            "re-embed count mismatch: {total} rows in, {written} out"
        );
        let mut pruned_models = Vec::new();
        match prune {
            Prune::Keep => {}
            Prune::Old => {
                store.delete_embeddings(&from_model).await?;
                pruned_models.push(from_model.clone());
            }
            Prune::All => {
                store.delete_other_embeddings(spec.id).await?;
                if from_model != spec.id {
                    pruned_models.push(from_model.clone());
                }
            }
        }
        store.set_current_model(spec.id).await?;
        Ok(ReembedReport {
            from_model,
            to_model: spec.id.to_string(),
            memories: total,
            pruned_models,
        })
    }
}
