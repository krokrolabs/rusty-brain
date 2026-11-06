//! The application core shared by the CLI and the MCP server.

use anyhow::{Context, Result, bail};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::Mutex as AsyncMutex;

use crate::config::{Config, MODEL_ID, ModelSpec, model_by_id};
use crate::embed::Embedder;
use crate::store::{
    AgentFilter, Hit, Memory, ModelChanged, Stats, Store, validate_content, validate_limit,
    validate_memory_id, validate_provenance, validate_query, validate_tags,
};

struct CachedEmbedder {
    model_id: String,
    embedder: Arc<Mutex<Embedder>>,
}

pub struct Brain {
    pub cfg: Config,
    store: AsyncMutex<Option<Store>>,
    embedder: AsyncMutex<Option<CachedEmbedder>>,
    synthetic_embeddings: bool,
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
    /// Keep every model's vectors (default; `use-model` can switch back).
    Keep,
    /// Delete only the previous model's vectors.
    Old,
    /// Delete every model's vectors except the new one.
    All,
}

impl Brain {
    pub async fn open(cfg: Config) -> Result<Self> {
        Self::open_inner(cfg, false).await
    }

    /// Deterministic local vectors for integration tests; never used by the CLI.
    #[doc(hidden)]
    pub async fn open_synthetic(cfg: Config) -> Result<Self> {
        Self::open_inner(cfg, true).await
    }

    async fn open_inner(cfg: Config, synthetic_embeddings: bool) -> Result<Self> {
        Ok(Self {
            cfg,
            store: AsyncMutex::new(None),
            embedder: AsyncMutex::new(None),
            synthetic_embeddings,
        })
    }

    /// Read the pointer on every public operation so a long-lived server follows
    /// external re-embed/use-model processes instead of retaining stale state.
    pub async fn current_model(&self) -> Result<String> {
        Ok(Store::model_of(&self.cfg.db_path())
            .await?
            .unwrap_or_else(|| MODEL_ID.to_string()))
    }

    async fn store_for(&self, model_id: &str) -> Result<Store> {
        let mut cached = self.store.lock().await;
        if let Some(store) = cached.as_ref()
            && store.model_id() == model_id
        {
            return Ok(store.clone());
        }
        self.cfg.ensure_dirs()?;
        let spec = model_by_id(model_id).context("unknown current model")?;
        let store = Store::open_with_model(&self.cfg.db_path(), Some(spec)).await?;
        if store.model_id() != model_id {
            return Err(anyhow::Error::new(ModelChanged {
                expected: model_id.to_string(),
                actual: store.model_id().to_string(),
            }));
        }
        *cached = Some(store.clone());
        Ok(store)
    }

    async fn embedder_for(&self, model_id: &str) -> Result<Arc<Mutex<Embedder>>> {
        let mut cached = self.embedder.lock().await;
        if let Some(cached) = cached.as_ref()
            && cached.model_id == model_id
        {
            return Ok(cached.embedder.clone());
        }
        let spec = model_by_id(model_id).context("unknown current model")?;
        let models = self.cfg.models_dir();
        let synthetic = self.synthetic_embeddings;
        let embedder = tokio::task::spawn_blocking(move || {
            if synthetic {
                Ok(Embedder::synthetic(spec))
            } else {
                Embedder::new(&models, spec)
            }
        })
        .await??;
        let embedder = Arc::new(Mutex::new(embedder));
        *cached = Some(CachedEmbedder {
            model_id: model_id.to_string(),
            embedder: embedder.clone(),
        });
        Ok(embedder)
    }

    async fn embed(&self, model_id: &str, text: String, query: bool) -> Result<Vec<f32>> {
        let embedder = self.embedder_for(model_id).await?;
        tokio::task::spawn_blocking(move || {
            let mut guard = embedder
                .lock()
                .map_err(|_| anyhow::anyhow!("embedding model lock poisoned"))?;
            if query {
                guard.embed_query(&text)
            } else {
                guard.embed(&text)
            }
        })
        .await?
    }

    fn model_changed(error: &anyhow::Error) -> bool {
        error.downcast_ref::<ModelChanged>().is_some()
    }

    pub async fn remember(&self, content: &str, tags: &str) -> Result<String> {
        validate_content(content)?;
        validate_tags(tags)?;
        validate_provenance("agent", &self.cfg.agent)?;
        validate_provenance("session", &self.cfg.session)?;
        for _ in 0..4 {
            let model = self.current_model().await?;
            let embedding = self.embed(&model, content.to_string(), false).await?;
            let store = match self.store_for(&model).await {
                Ok(store) => store,
                Err(error) if Self::model_changed(&error) => continue,
                Err(error) => return Err(error),
            };
            match store
                .remember(
                    content,
                    tags,
                    &self.cfg.agent,
                    &self.cfg.session,
                    &embedding,
                )
                .await
            {
                Err(error) if Self::model_changed(&error) => continue,
                result => return result,
            }
        }
        bail!("model changed repeatedly; retry the remember operation")
    }

    pub async fn recall(&self, query: &str, limit: usize, filter: AgentFilter) -> Result<Vec<Hit>> {
        validate_query(query)?;
        validate_limit(limit)?;
        match &filter {
            AgentFilter::Own(agent) | AgentFilter::Others(agent) | AgentFilter::Named(agent) => {
                validate_provenance("agent", agent)?;
            }
            AgentFilter::All => {}
        }
        if limit == 0 {
            self.current_model().await?;
            return Ok(Vec::new());
        }
        for _ in 0..4 {
            let model = self.current_model().await?;
            let embedding = self.embed(&model, query.to_string(), true).await?;
            let store = match self.store_for(&model).await {
                Ok(store) => store,
                Err(error) if Self::model_changed(&error) => continue,
                Err(error) => return Err(error),
            };
            return store.recall(query, limit, &filter, &embedding).await;
        }
        bail!("model changed while preparing recall; retry the operation")
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
        let model = self.current_model().await?;
        self.store_for(&model).await?.forget(id).await
    }

    pub async fn recent(&self, limit: usize) -> Result<Vec<Memory>> {
        validate_limit(limit)?;
        let model = self.current_model().await?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        self.store_for(&model).await?.recent(limit).await
    }

    pub async fn stats(&self) -> Result<Stats> {
        let model = self.current_model().await?;
        self.store_for(&model)
            .await?
            .stats(&self.cfg.db_path())
            .await
    }

    pub async fn export_parquet(&self) -> Result<(PathBuf, usize)> {
        let model = self.current_model().await?;
        let store = self.store_for(&model).await?;
        let dim = store.dim() as usize;
        let rows = store.all_for_export().await?;
        let dir = self.cfg.exports_dir();
        tokio::task::spawn_blocking(move || {
            crate::export::export_with_dim(&dir, &rows, dim, crate::export::injected_fault()?)
        })
        .await?
    }

    /// Re-embed the whole brain with another model. Ids, tags, timestamps,
    /// provenance and soft-deletion state are preserved; vectors are regenerated.
    /// Retained vectors can be selected later with `use-model` without inference.
    pub async fn reembed(&self, new_model_id: &str, prune: Prune) -> Result<ReembedReport> {
        let spec =
            model_by_id(new_model_id).with_context(|| format!("unknown model {new_model_id}"))?;
        let models = self.cfg.models_dir();
        let synthetic = self.synthetic_embeddings;
        let embedder = tokio::task::spawn_blocking(move || {
            if synthetic {
                Ok(Embedder::synthetic(spec))
            } else {
                Embedder::new(&models, spec)
            }
        })
        .await??;
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
    /// synthetic embeddings, so no model download is needed). Inference runs on
    /// the blocking pool while the immediate transaction excludes all writers.
    pub async fn reembed_with(
        &self,
        spec: &'static ModelSpec,
        prune: Prune,
        embed: impl Fn(String) -> Result<Vec<f32>> + Send + Sync + 'static,
    ) -> Result<ReembedReport> {
        let model = self.current_model().await?;
        let store = self.store_for(&model).await?;
        let tx = store.begin_reembed().await?;
        let from_model = Store::reembed_current_model(&tx).await?;
        if from_model == spec.id {
            bail!("this brain already uses model {}", spec.id);
        }
        let rows = Store::reembed_rows(&tx).await?;
        let total = rows.len();
        let texts: Vec<String> = rows.iter().map(|row| row.content.clone()).collect();
        let embeddings = tokio::task::spawn_blocking(move || {
            texts.into_iter().map(embed).collect::<Result<Vec<_>>>()
        })
        .await??;
        Store::commit_reembed(tx, &from_model, spec, prune, &rows, &embeddings).await?;
        let pruned_models = match prune {
            Prune::Keep => Vec::new(),
            Prune::Old | Prune::All => vec![from_model.clone()],
        };
        Ok(ReembedReport {
            from_model,
            to_model: spec.id.to_string(),
            memories: total,
            pruned_models,
        })
    }

    /// Select an existing vector set without re-embedding. Every active memory
    /// must have a valid vector for the requested model.
    pub async fn use_model(&self, new_model_id: &str) -> Result<String> {
        let model = self.current_model().await?;
        self.store_for(&model).await?.use_model(new_model_id).await
    }
}
