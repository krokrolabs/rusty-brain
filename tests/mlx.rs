//! MLX backend: registry geometry, setup gating, and sidecar protocol framing.
//! The protocol tests run a deterministic fake sidecar (`tests/fixtures/
//! fake_mlx_serve.py`) with the system python, so no model download and no MLX
//! install is needed; the real sidecar and model download are covered by the
//! gated live e2e (`RUSTY_BRAIN_MLX_E2E=1 scripts/verify-mlx.sh`).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;

use anyhow::Result;
use rusty_brain::app::{Brain, Prune};
use rusty_brain::config::{Backend, Config, MODEL_ID, MODELS, default_model, model_by_id};
use rusty_brain::embed::Embedder;
use rusty_brain::mlx::MlxEmbedder;
use sha2::{Digest, Sha256};

const MLX_ID: &str = "mlx-modernbert-4bit";
const DOC_PREFIX: &str = "search_document: ";
const QUERY_PREFIX: &str = "search_query: ";

fn fake_sidecar_source() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_mlx_serve.py")
}

fn find_test_python() -> Option<String> {
    let mut candidates = vec!["python3".to_string(), "/usr/bin/python3".to_string()];
    if let Some(custom) = std::env::var("RUSTY_BRAIN_TEST_PYTHON")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        candidates.insert(0, custom);
    }
    candidates.into_iter().find(|candidate| {
        Command::new(candidate)
            .arg("-c")
            .arg("print(1)")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    })
}

/// The mirror of the fake sidecar's deterministic embedding, computed in f64
/// exactly like the Python side (f32-only accumulation would drift at 1e-8).
fn fake_embedding(text: &str) -> Vec<f32> {
    let digest = Sha256::digest(text.as_bytes());
    let raw: Vec<f64> = (0..768)
        .map(|index| (digest[index % 32] as f64 - 128.0) / 128.0)
        .collect();
    let norm = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
    raw.iter().map(|v| (v / norm) as f32).collect()
}

/// A home that pretends setup-mlx ran: embeds the fake sidecar in place of the
/// real one and touches the model file the embedder requires.
fn fake_setup_home() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let models = home.join("models");
    std::fs::create_dir_all(models.join(MLX_ID)).unwrap();
    std::fs::create_dir_all(home.join("sidecar")).unwrap();
    std::fs::copy(
        fake_sidecar_source(),
        home.join("sidecar").join("mlx_serve.py"),
    )
    .unwrap();
    std::fs::write(models.join(MLX_ID).join("model.safetensors"), b"fake").unwrap();
    (dir, models)
}

/// Test processes run concurrently; the interpreter override is set once and
/// never changed (the missing-setup paths never consult it meaningfully).
fn set_mlx_python_once(python: &str) {
    static SET: Once = Once::new();
    SET.call_once(|| {
        // Test-only global env write: every fake-sidecar test resolves the
        // same interpreter, so one write cannot race a different value.
        unsafe { std::env::set_var("RUSTY_BRAIN_MLX_PYTHON", python) };
    });
}

// (a) Registry geometry: the MLX arm exists with the exact ModernBERT contract.
#[test]
fn mlx_arm_is_registered_with_exact_geometry() {
    let mlx = model_by_id(MLX_ID).expect("MLX arm must be registered");
    assert_eq!(mlx.dim, 768);
    assert_eq!(mlx.backend, Backend::Mlx);
    assert_eq!(mlx.doc_prefix, DOC_PREFIX);
    assert_eq!(mlx.query_prefix, QUERY_PREFIX);
    assert!(mlx.fastembed_variant.is_none());
    assert_eq!(
        mlx.source,
        "mlx-community/nomicai-modernbert-embed-base-4bit"
    );
    assert!(
        MODELS.iter().any(|m| m.id == mlx.id),
        "the MLX arm must live in the same registry"
    );
    for fastembed_arm in MODELS.iter().filter(|m| m.backend == Backend::Fastembed) {
        assert_eq!(fastembed_arm.doc_prefix, "");
        assert!(fastembed_arm.fastembed_variant.is_some());
        assert_eq!(fastembed_arm.backend, Backend::Fastembed);
    }
    assert_eq!(MODELS.len(), 4, "three fastembed arms plus the MLX arm");
}

// (b) The owner decision: new brains default to the MLX arm.
#[test]
fn default_model_is_the_mlx_arm() {
    assert_eq!(default_model().id, MLX_ID);
    assert_eq!(MODEL_ID, MLX_ID);
    assert_eq!(default_model().dim, 768);
}

// (c) Without setup, a new MLX-default brain fails with the actionable error
// and never creates the brain file.
#[tokio::test]
async fn mlx_default_brain_without_setup_reports_actionable_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let cfg = Config::new(
        Some(temp.path().join("home")),
        Some("mlx-gate".into()),
        Some("mlx-agent".into()),
        Some("mlx-session".into()),
    )?;
    let brain = Brain::open(cfg).await?;
    let error = brain
        .remember("must fail before setup", "")
        .await
        .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("run `rusty-brain setup-mlx` to install the MLX sidecar venv and model"),
        "actionable setup guidance required: {message}"
    );
    let brain_db = temp
        .path()
        .join("home")
        .join("projects")
        .join("mlx-gate")
        .join("brain.db");
    assert!(
        !brain_db.exists(),
        "a failed first embed must not create a brain"
    );
    Ok(())
}

// (d) Protocol framing against the fake sidecar, via the system python.
#[test]
fn mlx_embedder_speaks_the_sidecar_protocol() {
    let Some(python) = find_test_python() else {
        eprintln!("skipping: no python interpreter found for the fake sidecar");
        return;
    };
    set_mlx_python_once(&python);
    let (_dir, models) = fake_setup_home();
    let spec = model_by_id(MLX_ID).unwrap();
    let mut embedder = MlxEmbedder::new(&models, spec).expect("fake sidecar handshake");

    let document = embedder.embed("hello world").expect("embed reply");
    assert_eq!(
        document,
        fake_embedding(&format!("{DOC_PREFIX}hello world"))
    );
    let query = embedder.embed_query("hello world").expect("embed reply");
    assert_eq!(query, fake_embedding(&format!("{QUERY_PREFIX}hello world")));
    assert_ne!(document, query, "prefixes must change the embedding");

    // The same framing via the shared Embedder surface used by Brain.
    let mut shared = Embedder::new(&models, spec).expect("fake sidecar handshake");
    let via_shared = shared.embed("framing").expect("embed reply");
    assert_eq!(via_shared, fake_embedding(&format!("{DOC_PREFIX}framing")));
    let query_shared = shared.embed_query("framing").expect("embed reply");
    assert_eq!(
        query_shared,
        fake_embedding(&format!("{QUERY_PREFIX}framing"))
    );
}

#[test]
fn mlx_sidecar_load_errors_surface_as_actionable_err() {
    let Some(python) = find_test_python() else {
        eprintln!("skipping: no python interpreter found for the fake sidecar");
        return;
    };
    set_mlx_python_once(&python);
    let (dir, models) = fake_setup_home();
    std::fs::write(
        dir.path().join("home").join("models").join("LOAD_ERROR"),
        b"1",
    )
    .unwrap();
    let spec = model_by_id(MLX_ID).unwrap();
    let error = MlxEmbedder::new(&models, spec).expect_err("load failure must propagate");
    let message = error.to_string();
    assert!(
        message.contains("fake sidecar load failure"),
        "the sidecar's error must surface: {message}"
    );
    assert!(
        message.to_lowercase().contains("setup-mlx"),
        "load failure guidance must stay actionable: {message}"
    );
}

#[test]
fn mlx_sidecar_embed_errors_surface_as_err() {
    let Some(python) = find_test_python() else {
        eprintln!("skipping: no python interpreter found for the fake sidecar");
        return;
    };
    set_mlx_python_once(&python);
    let (_dir, models) = fake_setup_home();
    let spec = model_by_id(MLX_ID).unwrap();
    let mut embedder = MlxEmbedder::new(&models, spec).expect("fake sidecar handshake");
    let error = embedder
        .embed("poison __fail__")
        .expect_err("embed failure");
    assert!(
        error.to_string().contains("fake embed failure"),
        "the sidecar's embed error must surface: {error}"
    );
}

#[test]
fn mlx_missing_sidecar_or_model_reports_the_setup_hint() {
    let dir = tempfile::tempdir().unwrap();
    let models = dir.path().join("home").join("models");
    std::fs::create_dir_all(models.join(MLX_ID)).unwrap();
    let spec = model_by_id(MLX_ID).unwrap();
    let error = MlxEmbedder::new(&models, spec).unwrap_err();
    assert!(error.to_string().contains("setup-mlx"), "{error}");
}

// (e) Docs checker anchors: the new commands/model must be documented.
#[test]
fn mlx_docs_are_present() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let readme = std::fs::read_to_string(manifest.join("README.md")).unwrap();
    for required in [
        "rusty-brain setup-mlx",
        "mlx-modernbert-4bit",
        "Metal acceleration (optional)",
    ] {
        assert!(readme.contains(required), "README.md missing: {required}");
    }
    let models_doc = std::fs::read_to_string(manifest.join("docs/models.md")).unwrap();
    for required in [
        "mlx-community/nomicai-modernbert-embed-base-4bit",
        "search_document: ",
        "search_query: ",
        "Apache-2.0",
    ] {
        assert!(
            models_doc.contains(required),
            "docs/models.md missing: {required}"
        );
    }
    let verification = std::fs::read_to_string(manifest.join("docs/verification.md")).unwrap();
    for required in ["RUSTY_BRAIN_MLX_E2E=1", "scripts/verify-mlx.sh"] {
        assert!(
            verification.contains(required),
            "docs/verification.md missing: {required}"
        );
    }
}

// Synthetic embeddings must work for ANY registered arm, including the MLX
// one: Brain::open_synthetic never touches the sidecar.
#[tokio::test]
async fn synthetic_embeddings_work_for_the_mlx_arm() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let cfg = Config::new(
        Some(temp.path().join("home")),
        Some("synthetic-mlx".into()),
        Some("tester".into()),
        Some("session".into()),
    )?;
    let brain = Brain::open_synthetic(cfg.clone()).await?;
    brain.remember("mlx arm memory", "").await?;
    let stats = brain.stats().await?;
    assert_eq!(stats.model, MLX_ID);

    let spec = model_by_id(MLX_ID).unwrap();
    let vector = Embedder::synthetic(spec).embed("any text")?;
    assert_eq!(vector.len(), 768);
    assert!(vector.iter().all(|v| v.is_finite()));

    brain.reembed("BGESmallENV15Q", Prune::Keep).await?;
    brain.use_model(MLX_ID).await?;
    assert_eq!(brain.stats().await?.model, MLX_ID);
    Ok(())
}
