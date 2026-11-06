//! Store tests with synthetic embeddings — no model download, no network.

use rusty_brain::store::Store;

fn unit(dim: usize, hot: usize) -> Vec<f32> {
    (0..dim).map(|i| if i == hot { 1.0 } else { 0.0 }).collect()
}

async fn open_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(&dir.path().join("brain.db")).await.unwrap();
    (dir, store)
}

#[tokio::test]
async fn remember_recall_forget_roundtrip() {
    let (_dir, store) = open_store().await;
    let a = store
        .remember("duckdb warm instances", "decision", &unit(384, 0))
        .await
        .unwrap();
    let b = store
        .remember("parquet exports are derived", "principle", &unit(384, 1))
        .await
        .unwrap();

    let hits = store
        .recall("warm instances", 5, &unit(384, 0))
        .await
        .unwrap();
    assert_eq!(hits[0].memory.id, a, "nearest vector must rank first");
    assert!(hits[0].matched_text, "keyword leg must match too");
    assert!(hits.iter().any(|h| h.memory.id == b));

    assert!(store.forget(&a).await.unwrap());
    let hits = store
        .recall("warm instances", 5, &unit(384, 0))
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
    store.remember("one", "", &unit(384, 0)).await.unwrap();
    let id = store.remember("two", "", &unit(384, 1)).await.unwrap();
    store.remember("three", "", &unit(384, 2)).await.unwrap();
    store.forget(&id).await.unwrap();
    let stats = store.stats(&dir.path().join("brain.db")).await.unwrap();
    assert_eq!(stats.memories, 2);
    assert_eq!(stats.model, "AllMiniLML6V2Q");
}

#[tokio::test]
async fn export_roundtrip_reads_back_floats() {
    let (_dir, store) = open_store().await;
    let original = unit(384, 7);
    store
        .remember("vector payload", "", &original)
        .await
        .unwrap();
    let rows = store.all_for_export().await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].1, original, "embeddings must survive byte-exact");
}
