//! Store tests with synthetic embeddings — no model download, no network.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use libsql::{Connection, Database, params};
use rusty_brain::store::Store;

fn unit(dim: usize, hot: usize) -> Vec<f32> {
    (0..dim).map(|i| if i == hot { 1.0 } else { 0.0 }).collect()
}

async fn open_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(&dir.path().join("brain.db")).await.unwrap();
    (dir, store)
}

async fn raw_connection(path: &Path) -> (Database, Connection) {
    let db = libsql::Builder::new_local(path).build().await.unwrap();
    let conn = db.connect().unwrap();
    (db, conn)
}

async fn scalar_i64(conn: &Connection, sql: &str) -> i64 {
    let mut rows = conn.query(sql, ()).await.unwrap();
    rows.next().await.unwrap().unwrap().get(0).unwrap()
}

async fn set_identity_and_time(conn: &Connection, old: &str, new: &str, created: &str) {
    conn.execute(
        "UPDATE memories SET id = ?2, created_at = ?3 WHERE id = ?1",
        params![old, new, created],
    )
    .await
    .unwrap();
    conn.execute(
        "UPDATE memories_fts SET id = ?2 WHERE id = ?1",
        params![old, new],
    )
    .await
    .unwrap();
    conn.execute(
        "UPDATE memory_embeddings SET memory_id = ?2 WHERE memory_id = ?1",
        params![old, new],
    )
    .await
    .unwrap();
}

async fn copy_fixture(name: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(name);
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/base-f586d4e")
        .join(name);
    std::fs::copy(source, &path).unwrap();
    (dir, path)
}

#[tokio::test]
async fn remember_recall_forget_roundtrip() {
    let (_dir, store) = open_store().await;
    let a = store
        .remember(
            "duckdb warm instances",
            "decision",
            "test-agent",
            "test-session",
            &unit(384, 0),
        )
        .await
        .unwrap();
    let b = store
        .remember(
            "parquet exports are derived",
            "principle",
            "test-agent",
            "test-session",
            &unit(384, 1),
        )
        .await
        .unwrap();

    let hits = store
        .recall(
            "warm instances",
            5,
            &rusty_brain::store::AgentFilter::All,
            &unit(384, 0),
        )
        .await
        .unwrap();
    assert_eq!(hits[0].memory.id, a, "nearest vector must rank first");
    assert!(hits[0].matched_text, "keyword leg must match too");
    assert!(hits.iter().any(|h| h.memory.id == b));

    assert!(store.forget(&a).await.unwrap());
    let hits = store
        .recall(
            "warm instances",
            5,
            &rusty_brain::store::AgentFilter::All,
            &unit(384, 0),
        )
        .await
        .unwrap();
    assert!(
        hits.iter().all(|h| h.memory.id != a),
        "forgotten ids never return"
    );
    assert!(!store.forget(&a).await.unwrap(), "second forget is a no-op");
    let _ = b;
}

#[tokio::test]
async fn stats_counts_live_memories() {
    let (dir, store) = open_store().await;
    store
        .remember("one", "", "test-agent", "test-session", &unit(384, 0))
        .await
        .unwrap();
    let id = store
        .remember("two", "", "test-agent", "test-session", &unit(384, 1))
        .await
        .unwrap();
    store
        .remember("three", "", "test-agent", "test-session", &unit(384, 2))
        .await
        .unwrap();
    store.forget(&id).await.unwrap();
    let stats = store.stats(&dir.path().join("brain.db")).await.unwrap();
    assert_eq!(stats.memories, 2);
    assert_eq!(stats.model, "BGESmallENV15Q");
}

#[tokio::test]
async fn export_roundtrip_reads_back_floats() {
    let (_dir, store) = open_store().await;
    let original = unit(384, 7);
    store
        .remember(
            "vector payload",
            "",
            "test-agent",
            "test-session",
            &original,
        )
        .await
        .unwrap();
    let rows = store.all_for_export().await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].1, original, "embeddings must survive byte-exact");
}

#[tokio::test]
async fn recall_ties_use_score_created_at_then_id_across_reopen() -> Result<()> {
    let (dir, store) = open_store().await;
    let path = dir.path().join("brain.db");
    let embedding = unit(384, 3);
    // Equal vectors give semantic ranks A/B/C. Decreasing content lengths give
    // FTS ranks C/B/A, so every reciprocal-rank-fusion score is exactly tied.
    let a = store
        .remember(
            "tie extra extra extra extra extra",
            "",
            "test-agent",
            "test-session",
            &embedding,
        )
        .await?;
    let b = store
        .remember(
            "tie extra extra",
            "",
            "test-agent",
            "test-session",
            &embedding,
        )
        .await?;
    let c = store
        .remember("tie", "", "test-agent", "test-session", &embedding)
        .await?;
    let (_db, conn) = raw_connection(&path).await;
    let a_fixed = "00000000-0000-4000-8000-000000000003";
    let b_fixed = "00000000-0000-4000-8000-000000000002";
    let c_fixed = "00000000-0000-4000-8000-000000000001";
    set_identity_and_time(&conn, &a, a_fixed, "2026-01-01T00:00:00Z").await;
    set_identity_and_time(&conn, &b, b_fixed, "2026-01-02T00:00:00Z").await;
    set_identity_and_time(&conn, &c, c_fixed, "2026-01-02T00:00:00Z").await;

    let expected_created_order = vec![a_fixed, c_fixed, b_fixed];
    for _ in 0..8 {
        let hits = store
            .recall("tie", 3, &rusty_brain::store::AgentFilter::All, &embedding)
            .await?;
        assert_eq!(hits.len(), 3);
        assert_eq!(
            hits[0].score, hits[1].score,
            "first two scores must tie: {hits:#?}"
        );
        assert!(
            hits[1].score > hits[2].score,
            "score DESC must remain significant"
        );
        let actual: Vec<&str> = hits.iter().map(|hit| hit.memory.id.as_str()).collect();
        assert_eq!(
            actual, expected_created_order,
            "score DESC, created_at ASC, id ASC"
        );
    }

    set_identity_and_time(&conn, c_fixed, c_fixed, "2026-01-01T00:00:00Z").await;
    let expected_id_order = vec![c_fixed, a_fixed, b_fixed];
    for _ in 0..8 {
        let hits = store
            .recall("tie", 3, &rusty_brain::store::AgentFilter::All, &embedding)
            .await?;
        let actual: Vec<&str> = hits.iter().map(|hit| hit.memory.id.as_str()).collect();
        assert_eq!(actual, expected_id_order, "equal score/time uses id ASC");
    }

    drop(conn);
    drop(store);
    let copied = dir.path().join("copied.db");
    std::fs::copy(&path, &copied)?;
    let reopened = Store::open(&copied).await?;
    let hits = reopened
        .recall("tie", 3, &rusty_brain::store::AgentFilter::All, &embedding)
        .await?;
    let actual: Vec<&str> = hits.iter().map(|hit| hit.memory.id.as_str()).collect();
    assert_eq!(
        actual, expected_id_order,
        "ordering must survive a copied-store reopen"
    );
    Ok(())
}

#[tokio::test]
async fn limits_and_utf8_content_boundaries_are_exact() -> Result<()> {
    let (dir, store) = open_store().await;
    let path = dir.path().join("brain.db");
    let embedding = unit(384, 0);
    assert!(
        store
            .recall(
                "anything",
                0,
                &rusty_brain::store::AgentFilter::All,
                &embedding
            )
            .await?
            .is_empty()
    );
    assert!(store.recent(0).await?.is_empty());
    let recall_error = store
        .recall(
            "anything",
            51,
            &rusty_brain::store::AgentFilter::All,
            &embedding,
        )
        .await
        .unwrap_err();
    assert!(recall_error.to_string().contains("between 0 and 50"));
    let recent_error = store.recent(51).await.unwrap_err();
    assert!(recent_error.to_string().contains("between 0 and 50"));

    let accepted = "é".repeat(50_000);
    assert_eq!(accepted.len(), 100_000);
    store
        .remember(&accepted, "", "test-agent", "test-session", &embedding)
        .await?;
    let rejected = format!("{accepted}é");
    assert_eq!(rejected.len(), 100_002);
    let error = store
        .remember(&rejected, "", "test-agent", "test-session", &embedding)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("100000"));
    assert_eq!(store.stats(&path).await?.memories, 1);

    let one_too_many_ascii = "x".repeat(100_001);
    assert!(
        store
            .remember(
                &one_too_many_ascii,
                "",
                "test-agent",
                "test-session",
                &embedding
            )
            .await
            .is_err()
    );
    assert!(
        store
            .remember(" \n\t ", "", "test-agent", "test-session", &embedding)
            .await
            .is_err()
    );
    assert_eq!(store.stats(&path).await?.memories, 1);
    Ok(())
}

#[tokio::test]
async fn operator_text_is_escaped_and_uuid_rules_are_canonical() -> Result<()> {
    let (_dir, store) = open_store().await;
    let embedding = unit(384, 0);
    let id = store
        .remember(
            "ordinary lexical content",
            "",
            "test-agent",
            "test-session",
            &embedding,
        )
        .await?;
    let hits = store
        .recall(
            "\" OR * : ( )",
            5,
            &rusty_brain::store::AgentFilter::All,
            &embedding,
        )
        .await?;
    assert!(
        hits.iter().all(|hit| !hit.matched_text),
        "operators must not inject a lexical hit"
    );

    for malformed in [
        "not-a-uuid",
        "00000000000040008000000000000000",
        &id.to_uppercase(),
    ] {
        let error = store.forget(malformed).await.unwrap_err();
        assert!(error.to_string().contains("canonical UUID"), "{malformed}");
    }
    assert!(!store.forget("00000000-0000-4000-8000-000000000000").await?);
    assert!(store.forget(&id).await?);
    assert!(!store.forget(&id).await?);
    Ok(())
}

#[tokio::test]
async fn mismatched_model_is_refused_before_any_store_write() -> Result<()> {
    let (dir, store) = open_store().await;
    let path = dir.path().join("brain.db");
    store
        .remember(
            "must remain unchanged",
            "",
            "test-agent",
            "test-session",
            &unit(384, 0),
        )
        .await?;
    let (_db, conn) = raw_connection(&path).await;
    conn.execute("DROP TABLE memories_fts", ()).await?;
    conn.execute(
        "UPDATE meta SET value = 'DifferentModel' WHERE key IN ('embedding_model', 'current_model')",
        (),
    )
    .await?;
    drop(conn);
    drop(store);

    let before = std::fs::read(&path)?;
    let error = match Store::open(&path).await {
        Err(error) => error,
        Ok(_) => panic!("incompatible model must be refused"),
    };
    assert!(error.to_string().contains("DifferentModel"));
    let after = std::fs::read(&path)?;
    assert!(after == before, "refusal must be byte-for-byte read-only");

    let (_db, conn) = raw_connection(&path).await;
    assert_eq!(scalar_i64(&conn, "SELECT count(*) FROM memories").await, 1);
    assert_eq!(
        scalar_i64(
            &conn,
            "SELECT count(*) FROM sqlite_master WHERE name = 'memories_fts'",
        )
        .await,
        0,
        "schema must not be created before model validation"
    );
    let mut rows = conn
        .query(
            "SELECT value FROM meta WHERE key IN ('embedding_model', 'current_model')",
            (),
        )
        .await?;
    let value: String = rows.next().await?.context("model metadata")?.get(0)?;
    assert_eq!(value, "DifferentModel");
    Ok(())
}

#[tokio::test]
async fn base_revision_brain_reopens_without_migration() -> Result<()> {
    let (_dir, path) = copy_fixture("brain.db").await;
    let store = Store::open(&path).await?;
    let rows = store.all_for_export().await?;
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0.content, "Base fixture: DuckDB remains local");
    assert_eq!(rows[1].0.content, "Base fixture: Parquet preserves vectors");
    assert!(rows.iter().all(|(_, vector)| vector.len() == 384));
    assert_eq!(store.stats(&path).await?.memories, 2);
    assert_eq!(store.recent(50).await?.len(), 2);
    Ok(())
}

#[tokio::test]
async fn remember_second_statement_fault_rolls_back_memory_insert() -> Result<()> {
    let (dir, store) = open_store().await;
    let path = dir.path().join("brain.db");
    let (_db, conn) = raw_connection(&path).await;
    conn.execute("DROP TABLE memories_fts", ()).await?;
    let error = store
        .remember(
            "transaction must roll back",
            "fault",
            "test-agent",
            "test-session",
            &unit(384, 0),
        )
        .await
        .unwrap_err();
    assert_eq!(scalar_i64(&conn, "SELECT count(*) FROM memories").await, 0);
    assert!(error.to_string().contains("search index"));
    Ok(())
}

async fn assert_forgotten_absence(
    store: &Store,
    path: &Path,
    id: &str,
    active_count: i64,
) -> Result<()> {
    let hits = store
        .recall(
            "absence control",
            10,
            &rusty_brain::store::AgentFilter::All,
            &unit(384, 0),
        )
        .await?;
    ensure!(
        hits.iter().all(|hit| hit.memory.id != id),
        "recall still contains row"
    );
    ensure!(
        store.recent(10).await?.iter().all(|memory| memory.id != id),
        "recent still contains row"
    );
    ensure!(
        store.stats(path).await?.memories == active_count,
        "stats count is not expected"
    );
    ensure!(
        store
            .all_for_export()
            .await?
            .iter()
            .all(|(memory, _)| memory.id != id),
        "export selection still contains row"
    );
    Ok(())
}

#[tokio::test]
async fn forget_second_statement_fault_rolls_back_and_absence_controls_fail_while_active()
-> Result<()> {
    let (dir, store) = open_store().await;
    let path = dir.path().join("brain.db");
    let id = store
        .remember(
            "absence control",
            "",
            "test-agent",
            "test-session",
            &unit(384, 0),
        )
        .await?;

    assert!(
        assert_forgotten_absence(&store, &path, &id, 0)
            .await
            .is_err(),
        "active-row positive control must make the absence predicates fail"
    );
    let (_db, conn) = raw_connection(&path).await;
    conn.execute("DROP TABLE memories_fts", ()).await?;
    let error = store.forget(&id).await.unwrap_err();
    let mut rows = conn
        .query(
            "SELECT deleted_at FROM memories WHERE id = ?1",
            params![id.as_str()],
        )
        .await?;
    let deleted_at: Option<String> = rows.next().await?.context("memory row")?.get(0)?;
    drop(rows);
    assert_eq!(deleted_at, None, "soft deletion must roll back");
    assert!(error.to_string().contains("search index"));

    // Recreate only the index so the same active row can prove the post-forget
    // absence predicate after the fault test has completed.
    conn.execute(
        "CREATE VIRTUAL TABLE memories_fts USING fts5(id UNINDEXED, content)",
        (),
    )
    .await?;
    conn.execute(
        "INSERT INTO memories_fts (id, content) SELECT id, content FROM memories WHERE deleted_at IS NULL",
        (),
    )
    .await?;
    drop(conn);
    assert!(store.forget(&id).await?);
    assert_forgotten_absence(&store, &path, &id, 0).await?;
    Ok(())
}
