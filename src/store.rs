//! The brain file: one libsql (SQLite) database per project. This is the only
//! source of truth; Parquet exports are derived artifacts and can be rebuilt.
//!
//! Schema 3: memories hold content and provenance; embeddings live in their own
//! table keyed by (memory_id, model). Switching models adds vectors and flips
//! `meta.current_model` — it never rewrites memories and never deletes the old
//! vectors, so a model switch (or switch-back) loses nothing.

use anyhow::{Context, Result, bail, ensure};
use libsql::{Connection, Database, params};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;

use crate::config::{MODEL_ID, default_model, model_by_id};
use crate::embed::to_vector_literal;

const SCHEMA_VERSION: i64 = 3;
const MAX_CONTENT_BYTES: usize = 100_000;
const MAX_LIMIT: usize = 50;

#[derive(Clone)]
pub struct Store {
    conn: Connection,
    model_id: String,
    dim: i64,
}

#[derive(Debug, Serialize)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub tags: String,
    pub embedding_model: String,
    pub created_at: String,
    /// Which agent wrote this memory (`claude-code`, `dsh`, `cli`, …).
    /// Empty string means a brain that predates provenance tracking.
    pub agent: String,
    /// Session id/name when the harness provided one, else empty.
    pub session: String,
}

#[derive(Debug, Serialize)]
pub struct Hit {
    pub memory: Memory,
    pub score: f64,
    pub distance: Option<f64>,
    pub matched_text: bool,
}

#[derive(Debug, Serialize)]
pub struct Stats {
    pub memories: i64,
    pub model: String,
    pub schema_version: i64,
    pub db_path: String,
    pub db_bytes: u64,
    pub by_agent: Vec<(String, i64)>,
    /// Every model that currently has vectors in this brain.
    pub models: Vec<String>,
}

/// Recall scoping: every memory, only the calling agent's, everyone else's,
/// or one named agent's.
#[derive(Clone, Debug)]
pub enum AgentFilter {
    All,
    Own(String),
    Others(String),
    Named(String),
}

/// A row as re-embedding needs it: identity and content, no vectors.
#[derive(Debug)]
pub struct RawRow {
    pub id: String,
    pub content: String,
    pub tags: String,
    pub created_at: String,
    pub deleted_at: Option<String>,
    pub agent: String,
    pub session: String,
}

pub fn validate_content(content: &str) -> Result<()> {
    ensure!(
        !content.trim().is_empty(),
        "memory content must not be empty"
    );
    ensure!(
        content.len() <= MAX_CONTENT_BYTES,
        "memory content exceeds {MAX_CONTENT_BYTES} bytes"
    );
    Ok(())
}

pub fn validate_query(query: &str) -> Result<()> {
    ensure!(!query.trim().is_empty(), "recall query must not be empty");
    Ok(())
}

pub fn validate_limit(limit: usize) -> Result<()> {
    ensure!(limit <= MAX_LIMIT, "limit must be between 0 and 50");
    Ok(())
}

pub fn validate_memory_id(id: &str) -> Result<()> {
    let parsed = uuid::Uuid::parse_str(id)
        .ok()
        .filter(|uuid| uuid.hyphenated().to_string() == id);
    ensure!(parsed.is_some(), "memory id must be a canonical UUID");
    Ok(())
}

impl Store {
    /// The model recorded in an existing brain, if the file exists. Returns
    /// None when there is no brain yet; errors on a brain with no metadata or
    /// an unregistered model (refusing to mix embedding spaces).
    pub async fn model_of(path: &Path) -> Result<Option<String>> {
        if !path
            .try_exists()
            .with_context(|| format!("checking {}", path.display()))?
        {
            return Ok(None);
        }
        let db = libsql::Builder::new_local(path)
            .build()
            .await
            .with_context(|| format!("opening existing brain at {}", path.display()))?;
        let conn = db.connect().context("connecting to existing brain")?;
        let recorded = Self::read_meta(&conn, "current_model")
            .await?
            .or(Self::read_meta(&conn, "embedding_model").await?);
        let Some(recorded) = recorded else {
            bail!("existing brain has no embedding model metadata; refusing to modify it");
        };
        if model_by_id(&recorded).is_none() {
            bail!(
                "this brain was created with unknown model {recorded}; \
                 refusing to mix embedding spaces (known models: {})",
                crate::config::MODELS
                    .iter()
                    .map(|m| m.id)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Ok(Some(recorded))
    }

    async fn read_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
        let mut rows = conn
            .query("SELECT value FROM meta WHERE key = ?1", params![key])
            .await?;
        match rows.next().await? {
            Some(row) => Ok(Some(row.get(0)?)),
            None => Ok(None),
        }
    }

    pub async fn open(path: &Path) -> Result<Self> {
        Self::open_with_model(path, None).await
    }

    /// Open (or create) a brain. `create_model` is only used for brand-new
    /// brains; an existing brain always opens with its recorded model.
    pub async fn open_with_model(
        path: &Path,
        create_model: Option<&'static crate::config::ModelSpec>,
    ) -> Result<Self> {
        let recorded = Self::model_of(path).await?;
        let spec = match recorded.as_deref() {
            Some(id) => model_by_id(id).expect("model_of validated the registry"),
            None => create_model.unwrap_or_else(|| default_model()),
        };
        let dim = spec.dim;
        let db: Database = libsql::Builder::new_local(path)
            .build()
            .await
            .with_context(|| format!("opening brain at {}", path.display()))?;
        let conn = db.connect().context("connecting to brain")?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                deleted_at TEXT,
                agent TEXT NOT NULL DEFAULT '',
                session TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT NOT NULL,
                model TEXT NOT NULL,
                dim INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                PRIMARY KEY (memory_id, model)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(id UNINDEXED, content);
            "#,
        )
        .await
        .context("creating schema")?;

        // Additive migration: schema 1 (no provenance) and schema 2 (vectors in
        // the memories table) both upgrade in place.
        let mut cols = conn.query("PRAGMA table_info(memories)", ()).await?;
        let mut has_agent = false;
        let mut has_embedding = false;
        while let Some(row) = cols.next().await? {
            let name: String = row.get(1)?;
            if name == "agent" {
                has_agent = true;
            }
            if name == "embedding" {
                has_embedding = true;
            }
        }
        if !has_agent {
            conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN agent TEXT NOT NULL DEFAULT '';
                 ALTER TABLE memories ADD COLUMN session TEXT NOT NULL DEFAULT '';",
            )
            .await
            .context("migrating provenance columns")?;
        }
        if has_embedding {
            // Schema 2 -> 3: move vectors into their own table, then drop the
            // old columns so there is exactly one home for embeddings.
            conn.execute(
                "INSERT OR IGNORE INTO memory_embeddings (memory_id, model, dim, embedding)
                 SELECT id, embedding_model, length(embedding) / 4, embedding FROM memories",
                (),
            )
            .await
            .context("moving embeddings to memory_embeddings")?;
            if Self::read_meta(&conn, "current_model").await?.is_none() {
                let legacy = Self::read_meta(&conn, "embedding_model")
                    .await?
                    .unwrap_or_else(|| MODEL_ID.to_string());
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('current_model', ?1)",
                    params![legacy],
                )
                .await?;
            }
            if let Err(error) = conn
                .execute_batch(
                    "ALTER TABLE memories DROP COLUMN embedding;
                     ALTER TABLE memories DROP COLUMN embedding_model;",
                )
                .await
            {
                // Older engines cannot drop columns: leave them, unused.
                eprintln!("note: legacy embedding columns left in place: {error}");
            }
        }

        if Self::read_meta(&conn, "current_model").await?.is_none() {
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('current_model', ?1)",
                params![spec.id],
            )
            .await?;
        }
        // Write only on an actual version change: a read-only open must not
        // mutate the brain file (hash-stable reopening is part of the contract).
        let current_version = Self::read_meta(&conn, "schema_version").await?;
        if current_version.as_deref() != Some(SCHEMA_VERSION.to_string().as_str()) {
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?1)",
                params![SCHEMA_VERSION],
            )
            .await?;
        }
        Ok(Self {
            conn,
            model_id: spec.id.to_string(),
            dim,
        })
    }

    pub async fn remember(
        &self,
        content: &str,
        tags: &str,
        agent: &str,
        session: &str,
        embedding: &[f32],
    ) -> Result<String> {
        validate_content(content)?;
        ensure!(
            embedding.len() == self.dim as usize && embedding.iter().all(|v| v.is_finite()),
            "embedding must contain exactly {} finite float32 values",
            self.dim
        );

        let id = uuid::Uuid::new_v4().to_string();
        let created_at = chrono::Utc::now().to_rfc3339();
        let vector = to_vector_literal(embedding);
        let tx = self
            .conn
            .transaction()
            .await
            .context("starting remember transaction")?;
        tx.execute(
            "INSERT INTO memories (id, content, tags, created_at, agent, session)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                id.as_str(),
                content,
                tags,
                created_at.as_str(),
                agent,
                session
            ],
        )
        .await
        .context("inserting memory")?;
        tx.execute(
            "INSERT INTO memory_embeddings (memory_id, model, dim, embedding)
             VALUES (?1, ?2, ?3, vector32(?4))",
            params![
                id.as_str(),
                self.model_id.as_str(),
                self.dim,
                vector.as_str()
            ],
        )
        .await
        .context("inserting embedding")?;
        tx.execute(
            "INSERT INTO memories_fts (id, content) VALUES (?1, ?2)",
            params![id.as_str(), content],
        )
        .await
        .context("inserting search index")?;
        tx.commit()
            .await
            .context("committing remember transaction")?;
        Ok(id)
    }

    /// Hybrid recall over the CURRENT model's vectors: cosine top-k plus FTS5
    /// matches, merged with reciprocal rank fusion (k = 60), scoped by agent.
    pub async fn recall(
        &self,
        query: &str,
        limit: usize,
        filter: &AgentFilter,
        embedding: &[f32],
    ) -> Result<Vec<Hit>> {
        validate_query(query)?;
        validate_limit(limit)?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let candidates = (limit * 4).max(20) as i64;
        let vector = to_vector_literal(embedding);
        let (clause, agent_arg) = match filter {
            AgentFilter::All => ("", None),
            AgentFilter::Own(a) | AgentFilter::Named(a) => ("AND m.agent = ?4", Some(a.clone())),
            AgentFilter::Others(a) => ("AND m.agent <> ?4", Some(a.clone())),
        };
        let semantic_sql = format!(
            "SELECT m.id, vector_distance_cos(e.embedding, vector32(?1)) AS d
             FROM memory_embeddings e JOIN memories m ON m.id = e.memory_id AND e.model = ?2
             WHERE m.deleted_at IS NULL {clause} ORDER BY d ASC LIMIT ?3"
        );
        let mut sem = self
            .conn
            .query(
                &semantic_sql,
                libsql::params_from_iter(
                    [
                        libsql::Value::from(vector.as_str()),
                        libsql::Value::from(self.model_id.as_str()),
                        libsql::Value::from(candidates),
                    ]
                    .into_iter()
                    .chain(agent_arg.as_ref().map(|a| libsql::Value::from(a.as_str()))),
                ),
            )
            .await?;
        let mut semantic_rank: Vec<(String, f64)> = Vec::new();
        while let Some(row) = sem.next().await? {
            semantic_rank.push((row.get(0)?, row.get(1)?));
        }

        let fts_query = fts_escape(query);
        let mut lexical_rank: Vec<String> = Vec::new();
        if !fts_query.is_empty() {
            let mut lex = self
                .conn
                .query(
                    &format!(
                        "SELECT f.id FROM memories_fts f JOIN memories m ON m.id = f.id
                         WHERE memories_fts MATCH ?1 AND m.deleted_at IS NULL {clause}
                         ORDER BY rank LIMIT ?2"
                    ),
                    libsql::params_from_iter(
                        [
                            libsql::Value::from(fts_query.as_str()),
                            libsql::Value::from(candidates),
                        ]
                        .into_iter()
                        .chain(agent_arg.as_ref().map(|a| libsql::Value::from(a.as_str()))),
                    ),
                )
                .await?;
            while let Some(row) = lex.next().await? {
                lexical_rank.push(row.get(0)?);
            }
        }

        #[derive(Default)]
        struct Merge {
            score: f64,
            distance: Option<f64>,
            matched_text: bool,
        }
        let mut merged: HashMap<String, Merge> = HashMap::new();
        for (rank, (id, distance)) in semantic_rank.into_iter().enumerate() {
            let entry = merged.entry(id).or_default();
            entry.score += 1.0 / (60.0 + rank as f64 + 1.0);
            entry.distance = Some(distance);
        }
        for (rank, id) in lexical_rank.into_iter().enumerate() {
            let entry = merged.entry(id).or_default();
            entry.score += 1.0 / (60.0 + rank as f64 + 1.0);
            entry.matched_text = true;
        }

        let mut hits = Vec::with_capacity(merged.len());
        for (id, merged) in merged {
            if let Some(memory) = self.get(&id).await? {
                hits.push(Hit {
                    memory,
                    score: merged.score,
                    distance: merged.distance,
                    matched_text: merged.matched_text,
                });
            }
        }
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.memory.created_at.cmp(&b.memory.created_at))
                .then_with(|| a.memory.id.cmp(&b.memory.id))
        });
        hits.truncate(limit);
        Ok(hits)
    }

    fn memory_from_row(&self, row: &libsql::Row) -> Result<Memory> {
        Ok(Memory {
            id: row.get(0)?,
            content: row.get(1)?,
            tags: row.get(2)?,
            embedding_model: self.model_id.clone(),
            created_at: row.get(3)?,
            agent: row.get(4)?,
            session: row.get(5)?,
        })
    }

    pub async fn get(&self, id: &str) -> Result<Option<Memory>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, content, tags, created_at, agent, session
                 FROM memories WHERE id = ?1 AND deleted_at IS NULL",
                params![id],
            )
            .await?;
        match rows.next().await? {
            Some(row) => Ok(Some(self.memory_from_row(&row)?)),
            None => Ok(None),
        }
    }

    pub async fn forget(&self, id: &str) -> Result<bool> {
        validate_memory_id(id)?;
        let now = chrono::Utc::now().to_rfc3339();
        let tx = self
            .conn
            .transaction()
            .await
            .context("starting forget transaction")?;
        let changed = tx
            .execute(
                "UPDATE memories SET deleted_at = ?2 WHERE id = ?1 AND deleted_at IS NULL",
                params![id, now.as_str()],
            )
            .await
            .context("soft-deleting memory")?;
        if changed > 0 {
            tx.execute("DELETE FROM memories_fts WHERE id = ?1", params![id])
                .await
                .context("deleting search index")?;
        }
        tx.commit().await.context("committing forget transaction")?;
        Ok(changed > 0)
    }

    pub async fn recent(&self, limit: usize) -> Result<Vec<Memory>> {
        validate_limit(limit)?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut rows = self
            .conn
            .query(
                "SELECT id, content, tags, created_at, agent, session FROM memories
                 WHERE deleted_at IS NULL ORDER BY created_at DESC LIMIT ?1",
                params![limit as i64],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(self.memory_from_row(&row)?);
        }
        Ok(out)
    }

    /// Export joins each active memory with its CURRENT model's vector only.
    pub async fn all_for_export(&self) -> Result<Vec<(Memory, Vec<f32>)>> {
        let mut rows = self
            .conn
            .query(
                "SELECT m.id, m.content, m.tags, m.created_at, m.agent, m.session, e.embedding
                 FROM memories m JOIN memory_embeddings e
                   ON e.memory_id = m.id AND e.model = ?1
                 WHERE m.deleted_at IS NULL ORDER BY m.created_at ASC",
                params![self.model_id.as_str()],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let memory = self.memory_from_row(&row)?;
            let blob: Vec<u8> = row.get(6)?;
            let floats = blob
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            out.push((memory, floats));
        }
        Ok(out)
    }

    pub async fn stats(&self, db_path: &Path) -> Result<Stats> {
        let mut rows = self
            .conn
            .query("SELECT count(*) FROM memories WHERE deleted_at IS NULL", ())
            .await?;
        let memories: i64 = rows.next().await?.context("stats row")?.get(0)?;
        let mut by_agent_rows = self
            .conn
            .query(
                "SELECT agent, count(*) FROM memories WHERE deleted_at IS NULL
                 GROUP BY agent ORDER BY agent",
                (),
            )
            .await?;
        let mut by_agent = Vec::new();
        while let Some(row) = by_agent_rows.next().await? {
            by_agent.push((row.get(0)?, row.get(1)?));
        }
        let mut model_rows = self
            .conn
            .query(
                "SELECT DISTINCT model FROM memory_embeddings ORDER BY model",
                (),
            )
            .await?;
        let mut models = Vec::new();
        while let Some(row) = model_rows.next().await? {
            models.push(row.get(0)?);
        }
        let db_bytes = std::fs::metadata(db_path).map(|m| m.len()).unwrap_or(0);
        Ok(Stats {
            memories,
            model: self.model_id.clone(),
            schema_version: SCHEMA_VERSION,
            db_path: db_path.display().to_string(),
            db_bytes,
            by_agent,
            models,
        })
    }

    /// The embedding width this brain currently uses.
    pub fn dim(&self) -> i64 {
        self.dim
    }

    /// Every row, including soft-deleted ones, without vectors: the input of a
    /// re-embedding migration. Ids, timestamps and provenance are preserved.
    pub async fn raw_rows(&self) -> Result<Vec<RawRow>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, content, tags, created_at, deleted_at, agent, session FROM memories
                 ORDER BY created_at ASC",
                (),
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(RawRow {
                id: row.get(0)?,
                content: row.get(1)?,
                tags: row.get(2)?,
                created_at: row.get(3)?,
                deleted_at: row.get(4)?,
                agent: row.get(5)?,
                session: row.get(6)?,
            });
        }
        Ok(out)
    }

    /// Insert or replace one memory's vector for one model. Memories are never
    /// touched: a model switch is additive.
    pub async fn insert_embedding(
        &self,
        memory_id: &str,
        model: &str,
        dim: i64,
        embedding: &[f32],
    ) -> Result<()> {
        let vector = to_vector_literal(embedding);
        self.conn
            .execute(
                "INSERT OR REPLACE INTO memory_embeddings (memory_id, model, dim, embedding)
                 VALUES (?1, ?2, ?3, vector32(?4))",
                params![memory_id, model, dim, vector.as_str()],
            )
            .await?;
        Ok(())
    }

    /// Flip the current-model pointer after a successful re-embed.
    pub async fn set_current_model(&self, model: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('current_model', ?1)",
                params![model],
            )
            .await?;
        Ok(())
    }

    /// Remove one model's vectors (after a successful re-embed, on request).
    pub async fn delete_embeddings(&self, model: &str) -> Result<u64> {
        let deleted = self
            .conn
            .execute(
                "DELETE FROM memory_embeddings WHERE model = ?1",
                params![model],
            )
            .await?;
        Ok(deleted)
    }

    /// Remove every vector that is not the current model's (--prune-all).
    pub async fn delete_other_embeddings(&self, keep_model: &str) -> Result<u64> {
        let deleted = self
            .conn
            .execute(
                "DELETE FROM memory_embeddings WHERE model <> ?1",
                params![keep_model],
            )
            .await?;
        Ok(deleted)
    }

    /// Count of rows that have vectors for one model (migration validation).
    pub async fn count_embeddings(&self, model: &str) -> Result<i64> {
        let mut rows = self
            .conn
            .query(
                "SELECT count(*) FROM memory_embeddings WHERE model = ?1",
                params![model],
            )
            .await?;
        Ok(rows.next().await?.context("count row")?.get(0)?)
    }
}

/// Every whitespace-separated token becomes a quoted term, OR-joined. User text
/// can never inject FTS operators.
fn fts_escape(query: &str) -> String {
    query
        .split_whitespace()
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// The default model id, re-exported for schema display in the MCP layer.
pub const DEFAULT_MODEL_ID: &str = MODEL_ID;
