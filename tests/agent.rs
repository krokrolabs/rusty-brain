//! Provenance, filtering and re-embedding — synthetic embeddings, no downloads.

use rusty_brain::app::Brain;
use rusty_brain::config::{Config, MODELS, model_by_id};
use rusty_brain::store::{AgentFilter, Store};

fn unit(dim: usize, hot: usize) -> Vec<f32> {
    (0..dim).map(|i| if i == hot { 1.0 } else { 0.0 }).collect()
}

async fn open_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().unwrap();
    // 384-wide fixtures target the CPU fastembed arm explicitly; the registry
    // default is the MLX arm (covered synthetically in tests/mlx.rs).
    let bge = rusty_brain::config::model_by_id("BGESmallENV15Q").unwrap();
    let store = Store::open_with_model(&dir.path().join("brain.db"), Some(bge))
        .await
        .unwrap();
    (dir, store)
}

#[tokio::test]
async fn provenance_is_recorded_and_recall_filters_by_agent() {
    let (_dir, store) = open_store().await;
    let claude = store
        .remember(
            "from claude",
            "",
            "claude-code",
            "session-abc",
            &unit(384, 0),
        )
        .await
        .unwrap();
    let dsh = store
        .remember("from dsh", "", "dsh", "session-xyz", &unit(384, 1))
        .await
        .unwrap();

    let all = store
        .recall("from", 10, &AgentFilter::All, &unit(384, 0))
        .await
        .unwrap();
    assert_eq!(all.len(), 2);
    let hit = all.iter().find(|h| h.memory.id == claude).unwrap();
    assert_eq!(hit.memory.agent, "claude-code");
    assert_eq!(hit.memory.session, "session-abc");

    let own = store
        .recall(
            "from",
            10,
            &AgentFilter::Own("claude-code".into()),
            &unit(384, 0),
        )
        .await
        .unwrap();
    assert_eq!(own.len(), 1);
    assert_eq!(own[0].memory.id, claude);

    let others = store
        .recall(
            "from",
            10,
            &AgentFilter::Others("claude-code".into()),
            &unit(384, 0),
        )
        .await
        .unwrap();
    assert_eq!(others.len(), 1);
    assert_eq!(others[0].memory.id, dsh);

    let named = store
        .recall("from", 10, &AgentFilter::Named("dsh".into()), &unit(384, 0))
        .await
        .unwrap();
    assert_eq!(named[0].memory.id, dsh);

    let stats = store.stats(&_dir.path().join("brain.db")).await.unwrap();
    assert_eq!(
        stats.by_agent,
        vec![("claude-code".to_string(), 1), ("dsh".to_string(), 1)]
    );
}

#[tokio::test]
async fn schema1_brain_migrates_additively() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brain.db");
    {
        // A schema-1 brain: no agent/session columns at all.
        let db = libsql::Builder::new_local(&path).build().await.unwrap();
        let conn = db.connect().unwrap();
        conn.execute_batch(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
             INSERT INTO meta VALUES ('embedding_model', 'AllMiniLML6V2Q');
             CREATE TABLE memories (
               id TEXT PRIMARY KEY, content TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '',
               embedding F32_BLOB(384) NOT NULL, embedding_model TEXT NOT NULL,
               created_at TEXT NOT NULL, deleted_at TEXT);
             CREATE VIRTUAL TABLE memories_fts USING fts5(id UNINDEXED, content);
             INSERT INTO memories VALUES ('11111111-1111-1111-1111-111111111111', 'legacy row', '',
               vector32('[1,0,0]'), 'AllMiniLML6V2Q', '2026-01-01T00:00:00Z', NULL);",
        )
        .await
        .unwrap();
    }
    let store = Store::open(&path).await.unwrap();
    let legacy = store
        .get("11111111-1111-1111-1111-111111111111")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(legacy.agent, "");
    assert_eq!(legacy.session, "");
    store
        .remember("new row", "", "claude-code", "s1", &unit(384, 0))
        .await
        .unwrap();
    let stats = store.stats(&path).await.unwrap();
    assert_eq!(stats.memories, 2);
    assert_eq!(stats.schema_version, 3);
}

#[tokio::test]
async fn reembed_regenerates_vectors_and_preserves_everything() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = Config::new(
        Some(dir.path().join("home")),
        Some("proj".into()),
        Some("tester".into()),
        None,
    )
    .unwrap();
    std::fs::create_dir_all(cfg.project_dir()).unwrap();
    let db_path = cfg.db_path();
    let bge = model_by_id("BGESmallENV15Q").unwrap();
    let store = Store::open_with_model(&db_path, Some(bge)).await.unwrap();
    let keep = store
        .remember("alpha content", "t1", "claude-code", "s1", &unit(384, 0))
        .await
        .unwrap();
    let deleted = store
        .remember("beta content", "t2", "dsh", "s2", &unit(384, 1))
        .await
        .unwrap();
    store.forget(&deleted).await.unwrap();
    drop(store);

    let brain = Brain::open(cfg.clone()).await.unwrap();
    let arctic = model_by_id("SnowflakeArcticEmbedMQ").unwrap();
    let report = brain
        .reembed_with(arctic, rusty_brain::app::Prune::Keep, |text| {
            let hot = if text.starts_with("alpha") { 3 } else { 4 };
            Ok(unit(768, hot))
        })
        .await
        .unwrap();
    assert_eq!(report.memories, 2); // both rows, soft-deleted included
    assert_eq!(report.from_model, "BGESmallENV15Q");
    assert_eq!(report.to_model, "SnowflakeArcticEmbedMQ");
    assert!(report.pruned_models.is_empty());

    let migrated = Store::open(&db_path).await.unwrap();
    let kept = migrated.get(&keep).await.unwrap().unwrap();
    assert_eq!(kept.content, "alpha content");
    assert_eq!(kept.tags, "t1");
    assert_eq!(kept.agent, "claude-code");
    assert_eq!(kept.session, "s1");
    assert_eq!(kept.embedding_model, "SnowflakeArcticEmbedMQ");
    assert!(
        migrated.get(&deleted).await.unwrap().is_none(),
        "soft-deleted stays deleted"
    );

    // Both models' vectors exist; exports use only the current model (768).
    assert_eq!(
        migrated
            .count_embeddings("SnowflakeArcticEmbedMQ")
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        migrated.count_embeddings("BGESmallENV15Q").await.unwrap(),
        2
    );
    let exported = migrated.all_for_export().await.unwrap();
    assert_eq!(exported.len(), 1);
    assert_eq!(exported[0].1.len(), 768);
    assert_eq!(exported[0].1[3], 1.0);

    // Pruning the old model drops only its vectors.
    let bge = model_by_id("BGESmallENV15Q").unwrap();
    let brain2 = Brain::open(cfg.clone()).await.unwrap();
    brain2
        .reembed_with(bge, rusty_brain::app::Prune::Old, |_| Ok(unit(384, 5)))
        .await
        .unwrap();
    let after = Store::open(&db_path).await.unwrap();
    assert_eq!(
        after
            .count_embeddings("SnowflakeArcticEmbedMQ")
            .await
            .unwrap(),
        0
    );
    assert_eq!(after.count_embeddings("BGESmallENV15Q").await.unwrap(), 2);
    let exported = after.all_for_export().await.unwrap();
    assert_eq!(exported[0].1.len(), 384);
    assert_eq!(exported[0].1[5], 1.0);

    // Prune-all keeps only the current model.
    let arctic2 = model_by_id("SnowflakeArcticEmbedMQ").unwrap();
    let brain3 = Brain::open(cfg.clone()).await.unwrap();
    brain3
        .reembed_with(arctic2, rusty_brain::app::Prune::All, |_| Ok(unit(768, 6)))
        .await
        .unwrap();
    let final_store = Store::open(&db_path).await.unwrap();
    assert_eq!(
        final_store
            .count_embeddings("BGESmallENV15Q")
            .await
            .unwrap(),
        0
    );
    assert_eq!(
        final_store
            .count_embeddings("SnowflakeArcticEmbedMQ")
            .await
            .unwrap(),
        2
    );

    // Re-embedding into the current model is refused.
    let err = brain3
        .reembed_with(arctic2, rusty_brain::app::Prune::Keep, |_| Ok(unit(768, 0)))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("already uses"));
    assert!(MODELS.len() >= 3);
}
