//! The brain file: one libsql (SQLite) database per project. This is the only
//! source of truth; Parquet exports are derived artifacts and can be rebuilt.

use anyhow::{Context, Result, bail};
use libsql::{Connection, Database, params};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;

use crate::config::{MODEL_DIM, MODEL_ID};
use crate::embed::to_vector_literal;

const SCHEMA_VERSION: i64 = 1;

#[derive(Clone)]
pub struct Store {
    conn: Connection,
}

#[derive(Debug, Serialize)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub tags: String,
    pub embedding_model: String,
    pub created_at: String,
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
}

impl Store {
    pub async fn open(path: &Path) -> Result<Self> {
        let db: Database = libsql::Builder::new_local(path)
            .build()
            .await
            .with_context(|| format!("opening brain at {}", path.display()))?;
        let conn = db.connect().context("connecting to brain")?;
        conn.execute_batch(&format!(
            r#"
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                embedding F32_BLOB({MODEL_DIM}) NOT NULL,
                embedding_model TEXT NOT NULL,
                created_at TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(id UNINDEXED, content);
            "#
        ))
        .await
        .context("creating schema")?;

        // A brain is created by one model, forever. Mixing embedding spaces
        // would return plausible-looking garbage instead of an error.
        let mut rows = conn
            .query("SELECT value FROM meta WHERE key = 'embedding_model'", ())
            .await?;
        if let Some(row) = rows.next().await? {
            let recorded: String = row.get(0)?;
            if recorded != MODEL_ID {
                bail!(
                    "this brain was created with model {recorded}, but this binary uses {MODEL_ID}; \
                     refusing to mix embedding spaces"
                );
            }
        } else {
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('embedding_model', ?1)",
                params![MODEL_ID],
            )
            .await?;
        }
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?1)",
            params![SCHEMA_VERSION],
        )
        .await?;
        Ok(Self { conn })
    }

    pub async fn remember(&self, content: &str, tags: &str, embedding: &[f32]) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        let created_at = chrono::Utc::now().to_rfc3339();
        let vector = to_vector_literal(embedding);
        self.conn
            .execute(
                "INSERT INTO memories (id, content, tags, embedding, embedding_model, created_at)
                 VALUES (?1, ?2, ?3, vector32(?4), ?5, ?6)",
                params![
                    id.as_str(),
                    content,
                    tags,
                    vector.as_str(),
                    MODEL_ID,
                    created_at.as_str()
                ],
            )
            .await
            .context("inserting memory")?;
        self.conn
            .execute(
                "INSERT INTO memories_fts (id, content) VALUES (?1, ?2)",
                params![id.as_str(), content],
            )
            .await?;
        Ok(id)
    }

    /// Hybrid recall: cosine top-k plus FTS5 matches, merged with reciprocal
    /// rank fusion (k = 60).
    pub async fn recall(&self, query: &str, limit: usize, embedding: &[f32]) -> Result<Vec<Hit>> {
        let candidates = (limit * 4).max(20) as i64;
        let vector = to_vector_literal(embedding);

        let mut sem = self
            .conn
            .query(
                "SELECT id, vector_distance_cos(embedding, vector32(?1)) AS d
                 FROM memories WHERE deleted_at IS NULL ORDER BY d ASC LIMIT ?2",
                params![vector.as_str(), candidates],
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
                    "SELECT f.id FROM memories_fts f JOIN memories m ON m.id = f.id
                     WHERE memories_fts MATCH ?1 AND m.deleted_at IS NULL
                     ORDER BY rank LIMIT ?2",
                    params![fts_query.as_str(), candidates],
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

        let mut ordered: Vec<(String, Merge)> = merged.into_iter().collect();
        ordered.sort_by(|a, b| b.1.score.total_cmp(&a.1.score));
        ordered.truncate(limit);

        let mut hits = Vec::with_capacity(ordered.len());
        for (id, m) in ordered {
            if let Some(memory) = self.get(&id).await? {
                hits.push(Hit {
                    memory,
                    score: m.score,
                    distance: m.distance,
                    matched_text: m.matched_text,
                });
            }
        }
        Ok(hits)
    }

    pub async fn get(&self, id: &str) -> Result<Option<Memory>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, content, tags, embedding_model, created_at
                 FROM memories WHERE id = ?1 AND deleted_at IS NULL",
                params![id],
            )
            .await?;
        match rows.next().await? {
            Some(row) => Ok(Some(Memory {
                id: row.get(0)?,
                content: row.get(1)?,
                tags: row.get(2)?,
                embedding_model: row.get(3)?,
                created_at: row.get(4)?,
            })),
            None => Ok(None),
        }
    }

    pub async fn forget(&self, id: &str) -> Result<bool> {
        let now = chrono::Utc::now().to_rfc3339();
        let changed = self
            .conn
            .execute(
                "UPDATE memories SET deleted_at = ?2 WHERE id = ?1 AND deleted_at IS NULL",
                params![id, now.as_str()],
            )
            .await?;
        if changed > 0 {
            self.conn
                .execute("DELETE FROM memories_fts WHERE id = ?1", params![id])
                .await?;
        }
        Ok(changed > 0)
    }

    pub async fn recent(&self, limit: usize) -> Result<Vec<Memory>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, content, tags, embedding_model, created_at FROM memories
                 WHERE deleted_at IS NULL ORDER BY created_at DESC LIMIT ?1",
                params![limit as i64],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(Memory {
                id: row.get(0)?,
                content: row.get(1)?,
                tags: row.get(2)?,
                embedding_model: row.get(3)?,
                created_at: row.get(4)?,
            });
        }
        Ok(out)
    }

    pub async fn all_for_export(&self) -> Result<Vec<(Memory, Vec<f32>)>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, content, tags, embedding_model, created_at, embedding
                 FROM memories WHERE deleted_at IS NULL ORDER BY created_at ASC",
                (),
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let memory = Memory {
                id: row.get(0)?,
                content: row.get(1)?,
                tags: row.get(2)?,
                embedding_model: row.get(3)?,
                created_at: row.get(4)?,
            };
            let blob: Vec<u8> = row.get(5)?;
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
        let db_bytes = std::fs::metadata(db_path).map(|m| m.len()).unwrap_or(0);
        Ok(Stats {
            memories,
            model: MODEL_ID.to_string(),
            schema_version: SCHEMA_VERSION,
            db_path: db_path.display().to_string(),
            db_bytes,
        })
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
