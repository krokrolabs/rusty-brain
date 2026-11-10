//! Regression tests for the schema-3 audit. Synthetic vectors only.

use std::path::Path;

use anyhow::Result;
use libsql::{Connection, params};
use rusty_brain::app::{Brain, Prune};
use rusty_brain::config::{Config, model_by_id};
use rusty_brain::store::{AgentFilter, Store};

fn unit(dim: usize, hot: usize) -> Vec<f32> {
    (0..dim).map(|i| if i == hot { 1.0 } else { 0.0 }).collect()
}

fn config(root: &Path) -> Config {
    Config::new(
        Some(root.join("home")),
        Some("audit".into()),
        Some("audit-agent".into()),
        Some("audit-session".into()),
    )
    .unwrap()
}

async fn connection(path: &Path) -> (libsql::Database, Connection) {
    let db = libsql::Builder::new_local(path).build().await.unwrap();
    let conn = db.connect().unwrap();
    (db, conn)
}

async fn count(conn: &Connection, sql: &str) -> i64 {
    let mut rows = conn.query(sql, ()).await.unwrap();
    rows.next().await.unwrap().unwrap().get(0).unwrap()
}

/// A 384-wide fresh brain on the explicit fastembed arm (the registry default
/// is the MLX arm, which these legacy store tests do not exercise).
async fn open_bge_store(path: &Path) -> Store {
    let bge = rusty_brain::config::model_by_id("BGESmallENV15Q").unwrap();
    Store::open_with_model(path, Some(bge)).await.unwrap()
}

async fn one_memory(path: &Path) -> Store {
    let store = open_bge_store(path).await;
    store
        .remember("old model remains usable", "", "a", "s", &unit(384, 0))
        .await
        .unwrap();
    store
}

#[tokio::test]
async fn live_brain_reloads_stats_and_export_after_external_same_width_switch() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let cfg = config(temp.path());
    std::fs::create_dir_all(cfg.project_dir())?;
    let path = cfg.db_path();
    one_memory(&path).await;

    let brain = Brain::open(cfg.clone()).await?;
    assert_eq!(brain.stats().await?.model, "BGESmallENV15Q");

    let external = Brain::open(cfg).await?;
    external
        .reembed_with(model_by_id("AllMiniLML6V2Q").unwrap(), Prune::All, |_| {
            Ok(unit(384, 9))
        })
        .await?;

    assert_eq!(brain.stats().await?.model, "AllMiniLML6V2Q");
    let (_export, exported) = brain.export_parquet().await?;
    assert_eq!(exported, 1);
    Ok(())
}

#[tokio::test]
async fn live_remember_recall_stats_follow_same_width_switch_and_pointer_switch_back() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    let cfg = config(temp.path());
    let brain = Brain::open_synthetic(cfg.clone()).await?;
    brain.remember("before switch", "").await?;
    assert_eq!(brain.stats().await?.model, "mlx-modernbert-4bit");

    let external = Brain::open_synthetic(cfg.clone()).await?;
    external
        .reembed_with(model_by_id("AllMiniLML6V2Q").unwrap(), Prune::Keep, |_| {
            Ok(unit(384, 9))
        })
        .await?;

    brain.use_model("mlx-modernbert-4bit").await?;
    assert_eq!(brain.stats().await?.model, "mlx-modernbert-4bit");
    brain.use_model("AllMiniLML6V2Q").await?;
    assert_eq!(brain.stats().await?.model, "AllMiniLML6V2Q");

    brain.remember("after same-width switch", "").await?;
    assert_eq!(brain.stats().await?.model, "AllMiniLML6V2Q");
    let hits = brain.recall("switch", 10, AgentFilter::All).await?;
    assert_eq!(hits.len(), 2);
    assert!(
        hits.iter()
            .all(|hit| hit.memory.embedding_model == "AllMiniLML6V2Q")
    );
    let store = Store::open(&cfg.db_path()).await?;
    let rows = store.all_for_export().await?;
    let after = rows
        .iter()
        .find(|(memory, _)| memory.content == "after same-width switch")
        .unwrap();
    assert_eq!(after.1.len(), 384);
    assert_eq!(
        after.1[0], 1.0,
        "embedder cache must key on model id, not width"
    );
    assert_eq!(after.1[1], 0.0);

    assert_eq!(brain.recall("switch", 10, AgentFilter::All).await?.len(), 2);
    Ok(())
}

#[tokio::test]
async fn reembed_inference_leaves_the_async_runtime_responsive() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let cfg = config(temp.path());
    let brain = Brain::open_synthetic(cfg.clone()).await?;
    brain.remember("blocking inference", "").await?;
    let external = Brain::open_synthetic(cfg).await?;
    let started_at = std::time::Instant::now();
    let heartbeat = tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        started_at.elapsed()
    });
    external
        .reembed_with(model_by_id("AllMiniLML6V2Q").unwrap(), Prune::Keep, |_| {
            std::thread::sleep(std::time::Duration::from_millis(200));
            Ok(unit(384, 8))
        })
        .await?;
    let heartbeat_elapsed = heartbeat.await?;
    assert!(
        heartbeat_elapsed < std::time::Duration::from_millis(150),
        "async heartbeat was blocked by inference for {heartbeat_elapsed:?}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remember_during_migration_waits_then_uses_the_new_model() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let cfg = config(temp.path());
    let brain = std::sync::Arc::new(Brain::open_synthetic(cfg.clone()).await?);
    brain.remember("before migration", "").await?;
    let external = std::sync::Arc::new(Brain::open_synthetic(cfg).await?);
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let release_rx = std::sync::Mutex::new(release_rx);
    let migration = {
        let external = external.clone();
        tokio::spawn(async move {
            external
                .reembed_with(
                    model_by_id("AllMiniLML6V2Q").unwrap(),
                    Prune::All,
                    move |_| {
                        started_tx.send(()).unwrap();
                        release_rx.lock().unwrap().recv().unwrap();
                        Ok(unit(384, 8))
                    },
                )
                .await
        })
    };
    tokio::task::spawn_blocking(move || started_rx.recv()).await??;
    let writer = {
        let brain = brain.clone();
        tokio::spawn(async move { brain.remember("written during migration", "").await })
    };
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(
        !writer.is_finished(),
        "writer escaped the migration write lock"
    );
    release_tx.send(()).unwrap();
    migration.await??;
    writer.await??;

    assert_eq!(brain.stats().await?.model, "AllMiniLML6V2Q");
    assert_eq!(brain.stats().await?.memories, 2);
    let store = Store::open(&brain.cfg.db_path()).await?;
    assert_eq!(store.count_embeddings("AllMiniLML6V2Q").await?, 2);
    assert_eq!(store.count_embeddings("BGESmallENV15Q").await?, 0);
    Ok(())
}

#[tokio::test]
async fn prune_fault_rolls_back_target_vectors_and_preserves_old_model() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let cfg = config(temp.path());
    std::fs::create_dir_all(cfg.project_dir())?;
    let path = cfg.db_path();
    one_memory(&path).await;
    let (_db, conn) = connection(&path).await;
    conn.execute_batch(
        "CREATE TRIGGER reject_old_prune BEFORE DELETE ON memory_embeddings
         WHEN OLD.model = 'BGESmallENV15Q'
         BEGIN SELECT RAISE(FAIL, 'injected prune fault'); END;",
    )
    .await?;

    let brain = Brain::open(cfg).await?;
    let error = brain
        .reembed_with(
            model_by_id("SnowflakeArcticEmbedMQ").unwrap(),
            Prune::All,
            |_| Ok(unit(768, 2)),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("prun"), "{error}");
    assert_eq!(
        count(
            &conn,
            "SELECT count(*) FROM memory_embeddings WHERE model = 'SnowflakeArcticEmbedMQ'"
        )
        .await,
        0,
        "target writes must roll back with the failed prune"
    );
    assert_eq!(
        count(
            &conn,
            "SELECT count(*) FROM memory_embeddings WHERE model = 'BGESmallENV15Q'"
        )
        .await,
        1
    );
    let old = Store::open(&path).await?;
    assert_eq!(
        old.recall("old model", 5, &AgentFilter::All, &unit(384, 0))
            .await?
            .len(),
        1
    );
    Ok(())
}

#[tokio::test]
async fn pointer_fault_rolls_back_prune_and_preserves_old_model() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let cfg = config(temp.path());
    std::fs::create_dir_all(cfg.project_dir())?;
    let path = cfg.db_path();
    one_memory(&path).await;
    let (_db, conn) = connection(&path).await;
    conn.execute_batch(
        "CREATE TRIGGER reject_pointer_flip BEFORE INSERT ON meta
         WHEN NEW.key = 'current_model' AND NEW.value = 'SnowflakeArcticEmbedMQ'
         BEGIN SELECT RAISE(FAIL, 'injected pointer fault'); END;",
    )
    .await?;

    let brain = Brain::open(cfg).await?;
    brain
        .reembed_with(
            model_by_id("SnowflakeArcticEmbedMQ").unwrap(),
            Prune::Old,
            |_| Ok(unit(768, 2)),
        )
        .await
        .unwrap_err();
    assert_eq!(
        count(
            &conn,
            "SELECT count(*) FROM memory_embeddings WHERE model = 'SnowflakeArcticEmbedMQ'"
        )
        .await,
        0
    );
    assert_eq!(
        count(
            &conn,
            "SELECT count(*) FROM memory_embeddings WHERE model = 'BGESmallENV15Q'"
        )
        .await,
        1,
        "the pointer must never name vectors deleted by a failed migration"
    );
    let old = Store::open(&path).await?;
    assert_eq!(old.all_for_export().await?.len(), 1);
    Ok(())
}

#[tokio::test]
async fn reembed_rejects_wrong_width_and_nonfinite_vectors_before_any_write() -> Result<()> {
    for (label, invalid) in [
        ("wrong-width", unit(383, 0)),
        ("non-finite", {
            let mut vector = unit(768, 0);
            vector[1] = f32::NAN;
            vector
        }),
    ] {
        let temp = tempfile::tempdir()?;
        let cfg = config(temp.path());
        std::fs::create_dir_all(cfg.project_dir())?;
        let path = cfg.db_path();
        let store = open_bge_store(&path).await;
        store.remember("first", "", "a", "s", &unit(384, 0)).await?;
        store
            .remember("second", "", "a", "s", &unit(384, 1))
            .await?;
        drop(store);

        let brain = Brain::open(cfg).await?;
        let invalid = invalid.clone();
        let error = brain
            .reembed_with(
                model_by_id("SnowflakeArcticEmbedMQ").unwrap(),
                Prune::All,
                move |text| {
                    if text == "second" {
                        Ok(invalid.clone())
                    } else {
                        Ok(unit(768, 2))
                    }
                },
            )
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("finite") || error.to_string().contains("768"),
            "{label}: {error}"
        );
        let reopened = Store::open(&path).await?;
        assert_eq!(
            reopened.stats(&path).await?.model,
            "BGESmallENV15Q",
            "{label}"
        );
        assert_eq!(
            reopened.count_embeddings("SnowflakeArcticEmbedMQ").await?,
            0,
            "{label}: validation must happen before the first target write"
        );
        assert_eq!(reopened.all_for_export().await?.len(), 2, "{label}");
    }
    Ok(())
}

#[tokio::test]
async fn empty_interrupted_initialization_is_resumable_but_nonempty_is_refused() -> Result<()> {
    let temp = tempfile::tempdir()?;

    let zero = temp.path().join("zero.db");
    std::fs::write(&zero, [])?;
    let store = open_bge_store(&zero).await;
    assert_eq!(store.stats(&zero).await?.memories, 0);

    let schema_only = temp.path().join("schema-only.db");
    {
        let (_db, conn) = connection(&schema_only).await;
        conn.execute_batch(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
             CREATE TABLE memories (
               id TEXT PRIMARY KEY, content TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '',
               created_at TEXT NOT NULL, deleted_at TEXT, agent TEXT NOT NULL DEFAULT '',
               session TEXT NOT NULL DEFAULT '');
             CREATE TABLE memory_embeddings (
               memory_id TEXT NOT NULL, model TEXT NOT NULL, dim INTEGER NOT NULL,
               embedding BLOB NOT NULL, PRIMARY KEY (memory_id, model));
             CREATE VIRTUAL TABLE memories_fts USING fts5(id UNINDEXED, content);",
        )
        .await?;
    }
    let store = open_bge_store(&schema_only).await;
    store
        .remember("resumed", "", "a", "s", &unit(384, 0))
        .await?;

    let nonempty = temp.path().join("nonempty.db");
    {
        let (_db, conn) = connection(&nonempty).await;
        conn.execute_batch(
            "CREATE TABLE memories (
               id TEXT PRIMARY KEY, content TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '',
               created_at TEXT NOT NULL, deleted_at TEXT);
             INSERT INTO memories VALUES
               ('00000000-0000-4000-8000-000000000001', 'unsafe to guess', '',
                '2026-01-01T00:00:00Z', NULL);",
        )
        .await?;
    }
    let error = match Store::open(&nonempty).await {
        Err(error) => error,
        Ok(_) => panic!("nonempty brain without model metadata must be refused"),
    };
    assert!(
        error.to_string().contains("no embedding model metadata"),
        "{error}"
    );
    Ok(())
}

#[tokio::test]
async fn interrupted_provenance_migration_adds_agent_and_session_independently() -> Result<()> {
    for existing in ["agent", "session"] {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("brain.db");
        let other = if existing == "agent" {
            "session"
        } else {
            "agent"
        };
        let (_db, conn) = connection(&path).await;
        conn.execute_batch(&format!(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
             INSERT INTO meta VALUES ('current_model', 'BGESmallENV15Q');
             CREATE TABLE memories (
               id TEXT PRIMARY KEY, content TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '',
               created_at TEXT NOT NULL, deleted_at TEXT,
               {existing} TEXT NOT NULL DEFAULT '');
             CREATE TABLE memory_embeddings (
               memory_id TEXT NOT NULL, model TEXT NOT NULL, dim INTEGER NOT NULL,
               embedding BLOB NOT NULL, PRIMARY KEY (memory_id, model));
             CREATE VIRTUAL TABLE memories_fts USING fts5(id UNINDEXED, content);"
        ))
        .await?;
        drop(conn);
        let store = Store::open(&path).await?;
        store
            .remember("resumed migration", "", "a", "s", &unit(384, 0))
            .await?;
        let (_db, conn) = connection(&path).await;
        let mut rows = conn.query("PRAGMA table_info(memories)", ()).await?;
        let mut names = Vec::new();
        while let Some(row) = rows.next().await? {
            names.push(row.get::<String>(1)?);
        }
        assert!(names.contains(&other.to_string()), "missing {other}");
    }
    Ok(())
}

#[tokio::test]
async fn tied_semantic_distances_use_stable_id_ranks() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("brain.db");
    let store = open_bge_store(&path).await;
    let mut generated = Vec::new();
    for content in ["first inserted", "second inserted", "third inserted"] {
        generated.push(store.remember(content, "", "a", "s", &unit(384, 4)).await?);
    }
    let fixed = [
        "00000000-0000-4000-8000-000000000003",
        "00000000-0000-4000-8000-000000000002",
        "00000000-0000-4000-8000-000000000001",
    ];
    let (_db, conn) = connection(&path).await;
    for (old, new) in generated.iter().zip(fixed) {
        conn.execute(
            "UPDATE memories SET id = ?2 WHERE id = ?1",
            params![old.as_str(), new],
        )
        .await?;
        conn.execute(
            "UPDATE memory_embeddings SET memory_id = ?2 WHERE memory_id = ?1",
            params![old.as_str(), new],
        )
        .await?;
    }
    let hits = store
        .recall("no lexical match", 3, &AgentFilter::All, &unit(384, 4))
        .await?;
    let ids: Vec<&str> = hits.iter().map(|hit| hit.memory.id.as_str()).collect();
    assert_eq!(ids, vec![fixed[2], fixed[1], fixed[0]]);
    Ok(())
}

#[tokio::test]
async fn query_tags_and_provenance_have_byte_caps() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("brain.db");
    let store = open_bge_store(&path).await;
    let embedding = unit(384, 0);
    assert!(
        store
            .recall(&"q".repeat(4_097), 1, &AgentFilter::All, &embedding)
            .await
            .unwrap_err()
            .to_string()
            .contains("4096")
    );
    for (field, value) in [
        ("tags", "t".repeat(1_025)),
        ("agent", "a".repeat(129)),
        ("session", "s".repeat(129)),
    ] {
        let mut tags = String::new();
        let mut agent = "a".to_string();
        let mut session = "s".to_string();
        match field {
            "tags" => tags = value,
            "agent" => agent = value,
            _ => session = value,
        }
        let error = store
            .remember("bounded", &tags, &agent, &session, &embedding)
            .await
            .unwrap_err();
        assert!(error.to_string().contains(field), "{field}: {error}");
    }
    assert_eq!(store.stats(&path).await?.memories, 0);
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
async fn brain_and_export_paths_use_private_unix_permissions() -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir()?;
    let cfg = config(temp.path());
    cfg.ensure_dirs()?;
    let store = open_bge_store(&cfg.db_path()).await;
    store
        .remember("private", "", "a", "s", &unit(384, 0))
        .await?;
    let rows = store.all_for_export().await?;
    let (export, _) = rusty_brain::export::export(&cfg.exports_dir(), &rows)?;
    for path in [cfg.home.clone(), cfg.project_dir(), cfg.exports_dir()] {
        assert_eq!(std::fs::metadata(path)?.permissions().mode() & 0o777, 0o700);
    }
    for path in [cfg.db_path(), export] {
        assert_eq!(std::fs::metadata(path)?.permissions().mode() & 0o777, 0o600);
    }
    Ok(())
}
