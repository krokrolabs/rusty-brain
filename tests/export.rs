//! Synthetic Parquet export regressions; no model or network is used.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rusty_brain::export::{ExportFault, export, export_with_fault};
use rusty_brain::store::{Memory, Store};

fn memory(id: &str, content: &str, tags: &str, created_at: &str) -> Memory {
    Memory {
        id: id.into(),
        content: content.into(),
        tags: tags.into(),
        embedding_model: "AllMiniLML6V2Q".into(),
        created_at: created_at.into(),
        agent: "test-agent".into(),
        session: "test-session".into(),
    }
}

fn fractional_vector(offset: f32) -> Vec<f32> {
    (0..384).map(|index| index as f32 / 17.0 + offset).collect()
}

fn populated_rows() -> Vec<(Memory, Vec<f32>)> {
    vec![
        (
            memory(
                "00000000-0000-4000-8000-000000000001",
                "Fractional vectors and 非 ASCII text round-trip",
                "",
                "2026-01-01T00:00:00Z",
            ),
            fractional_vector(0.25),
        ),
        (
            memory(
                "00000000-0000-4000-8000-000000000002",
                "Second active row",
                "one,two",
                "2026-01-02T00:00:00Z",
            ),
            fractional_vector(-3.5),
        ),
    ]
}

fn assert_parquet_matches(path: &Path, expected: &[(Memory, Vec<f32>)]) {
    let file = std::fs::File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let schema = builder.schema();
    let fields = schema.fields();
    assert_eq!(fields.len(), 8);
    assert_eq!(
        fields
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        vec![
            "id",
            "content",
            "tags",
            "embedding_model",
            "created_at",
            "agent",
            "session",
            "embedding"
        ]
    );
    assert_eq!(
        fields[7].data_type(),
        &DataType::FixedSizeList(
            Arc::new(arrow::datatypes::Field::new(
                "item",
                DataType::Float32,
                true
            )),
            384,
        )
    );

    let batches = builder
        .build()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let actual_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(actual_count, expected.len());
    let mut expected_row = 0;
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let contents = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let tags = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let models = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let created = batch
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let embeddings = batch
            .column(7)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let (memory, vector) = &expected[expected_row];
            assert_eq!(ids.value(row), memory.id);
            assert_eq!(contents.value(row), memory.content);
            assert_eq!(tags.value(row), memory.tags);
            assert_eq!(models.value(row), memory.embedding_model);
            assert_eq!(created.value(row), memory.created_at);
            let values = embeddings.value(row);
            let floats = values.as_any().downcast_ref::<Float32Array>().unwrap();
            assert_eq!(floats.values().as_ref(), vector.as_slice());
            expected_row += 1;
        }
    }
}

fn file_names(dir: &Path) -> Vec<String> {
    let mut names = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    names
}

#[test]
fn populated_zero_and_rapid_exports_keep_schema_and_prior_file() {
    let temp = tempfile::tempdir().unwrap();
    let rows = populated_rows();
    let (first, count) = export(temp.path(), &rows).unwrap();
    assert_eq!(count, rows.len());
    assert_parquet_matches(&first, &rows);
    let first_hash = std::fs::read(&first).unwrap();

    let (second, second_count) = export(temp.path(), &rows).unwrap();
    assert_eq!(second_count, rows.len());
    assert_ne!(first, second, "rapid exports must never collide or sleep");
    assert_parquet_matches(&second, &rows);
    assert!(
        std::fs::read(&first).unwrap() == first_hash,
        "prior snapshot changed"
    );

    let (empty, empty_count) = export(temp.path(), &[]).unwrap();
    assert_eq!(empty_count, 0);
    assert_parquet_matches(&empty, &[]);
}

#[tokio::test]
async fn reported_count_comes_from_selection_not_later_stats() {
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("brain.db");
    let export_dir = temp.path().join("exports");
    let store = Store::open(&db_path).await.unwrap();
    let embedding = vec![0.125; 384];
    store
        .remember("selected one", "", "test-agent", "test-session", &embedding)
        .await
        .unwrap();
    store
        .remember("selected two", "", "test-agent", "test-session", &embedding)
        .await
        .unwrap();
    let selected = store.all_for_export().await.unwrap();
    assert_eq!(selected.len(), 2);

    // Deterministic interleaving: the source changes after selection. The
    // exported count must still describe this exact two-row snapshot.
    store
        .remember(
            "later concurrent row",
            "",
            "test-agent",
            "test-session",
            &embedding,
        )
        .await
        .unwrap();
    let source_before = std::fs::read(&db_path).unwrap();
    let (path, count) = export(&export_dir, &selected).unwrap();
    assert_eq!(store.stats(&db_path).await.unwrap().memories, 3);
    assert_eq!(count, selected.len());
    assert_parquet_matches(&path, &selected);
    let source_after = std::fs::read(&db_path).unwrap();
    assert!(
        source_after == source_before,
        "export mutated the source brain"
    );
}

#[test]
fn write_close_hard_link_and_post_link_cleanup_faults_leave_no_artifact() {
    let temp = tempfile::tempdir().unwrap();
    let export_dir = temp.path().join("exports");
    let rows = populated_rows();
    let (prior, _) = export(&export_dir, &rows).unwrap();
    let prior_bytes = std::fs::read(&prior).unwrap();
    let before = file_names(&export_dir);

    // The unchanged predicate must detect a mutated copy before it can be
    // trusted for the real prior snapshot.
    let mut mutated = prior_bytes.clone();
    mutated.push(0);
    assert!(
        mutated != prior_bytes,
        "mutated-copy hash control was not detected"
    );

    for (fault, stage) in [
        (ExportFault::Write, "write"),
        (ExportFault::Close, "close"),
        (ExportFault::HardLink, "hard link"),
        (ExportFault::RemoveTemp, "cleanup"),
    ] {
        let error = export_with_fault(&export_dir, &rows, Some(fault)).unwrap_err();
        assert!(
            error.to_string().contains(stage),
            "{fault:?} named wrong stage: {error}"
        );
        assert_eq!(
            file_names(&export_dir),
            before,
            "{fault:?} left a final/temp artifact"
        );
        assert!(
            std::fs::read(&prior).unwrap() == prior_bytes,
            "{fault:?} changed prior export"
        );
    }

    // A planted final file must make the no-new-artifact predicate fail.
    std::fs::write(
        export_dir.join("memories-planted-final.parquet"),
        b"control",
    )
    .unwrap();
    assert_ne!(
        file_names(&export_dir),
        before,
        "planted-final positive control passed"
    );
    std::fs::remove_file(export_dir.join("memories-planted-final.parquet")).unwrap();
    assert_eq!(file_names(&export_dir), before);
}
