//! Real-binary CLI contracts. No test needs a model download or cloud access.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use libsql::params;
use uuid::Uuid;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_rusty-brain")
}

fn command(root: &Path, project: &str) -> Command {
    let mut command = Command::new(binary());
    command
        .arg("--data-dir")
        .arg(root)
        .arg("--project")
        .arg(project)
        .env_remove("RUSTY_BRAIN_HOME")
        .env("HF_HOME", root.join("empty-hf-cache"))
        // Before lazy validation is fixed, an accidental model initialization can
        // only reach this closed loopback port, never an external service.
        .env("HF_ENDPOINT", "http://127.0.0.1:9")
        .env_remove("HF_TOKEN")
        .env_remove("RB_TEST_SECRET");
    command
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn safety_violations(output: &Output, marker: &str) -> Vec<&'static str> {
    let combined = [output.stdout.as_slice(), output.stderr.as_slice()].concat();
    let text = String::from_utf8_lossy(&combined);
    let mut violations = Vec::new();
    if text.contains(marker) {
        violations.push("marker credential");
    }
    if text.contains("Caused by:") {
        violations.push("diagnostic chain");
    }
    if text.contains("Stack backtrace") || text.to_lowercase().contains("stack backtrace:") {
        violations.push("stack trace");
    }
    violations
}

fn assert_safe(output: &Output, marker: &str) {
    let violations = safety_violations(output, marker);
    assert!(
        violations.is_empty(),
        "unsafe CLI diagnostics: {violations:?}"
    );
}

fn generated_marker() -> String {
    format!("rb006-unrecognized-cli-marker-{}", Uuid::new_v4())
}

fn base_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/base-f586d4e")
        .join(name)
}

#[test]
fn global_overrides_and_path_do_not_initialize_anything() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("isolated-root");
    let output = command(&root, "cli-contract").arg("path").output().unwrap();
    assert!(output.status.success(), "path failed: {}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains(&format!("home:     {}", root.display())));
    assert!(text.contains("project:  cli-contract"));
    assert!(text.contains("brain:"));
    assert!(text.contains("models:"));
    assert!(text.contains("exports:"));
    assert!(
        !root.exists(),
        "path must not create a database, model cache, or root"
    );

    let missing = command(&root, "cli-contract").output().unwrap();
    assert!(!missing.status.success());
    assert!(missing.stdout.is_empty());
    assert!(stderr(&missing).contains("Usage"));

    let unknown = command(&root, "cli-contract")
        .arg("not-a-command")
        .output()
        .unwrap();
    assert!(!unknown.status.success());
    assert!(unknown.stdout.is_empty());
    assert!(stderr(&unknown).contains("unrecognized subcommand"));
}

#[test]
fn zero_limits_succeed_and_51_fails_without_success_stdout() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("isolated-root");
    for subcommand in ["recall", "recent"] {
        let mut zero = command(&root, "cli-contract");
        if subcommand == "recall" {
            zero.arg("recall").arg("unused for zero limit");
        } else {
            zero.arg("recent");
        }
        let output = zero.args(["--limit", "0"]).output().unwrap();
        assert!(
            output.status.success(),
            "{subcommand} zero failed: {}",
            stderr(&output)
        );
        assert_eq!(stdout(&output).trim(), "[]");

        let mut over = command(&root, "cli-contract");
        if subcommand == "recall" {
            over.arg("recall").arg("unused for invalid limit");
        } else {
            over.arg("recent");
        }
        let output = over.args(["--limit", "51"]).output().unwrap();
        assert!(!output.status.success(), "{subcommand} accepted 51");
        assert!(
            output.stdout.is_empty(),
            "failure must not print a success array"
        );
        assert!(stderr(&output).contains("between 0 and 50"));
    }
}

#[test]
fn prune_flags_conflict_before_any_model_or_database_work() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("isolated-root");
    let output = command(&root, "cli-contract")
        .args(["reembed", "BGESmallENV15Q", "--prune", "--prune-all"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(stdout(&output).is_empty());
    let diagnostic = stderr(&output);
    assert!(diagnostic.contains("cannot be used with"), "{diagnostic}");
    assert!(!root.exists());
}

#[tokio::test]
async fn use_model_flips_only_a_fully_covered_pointer_without_model_loading() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("isolated-root");
    let project = "cli-contract";
    let project_dir = root.join("projects").join(project);
    std::fs::create_dir_all(&project_dir).unwrap();
    let path = project_dir.join("brain.db");
    let store = rusty_brain::store::Store::open(&path).await.unwrap();
    let id = store
        .remember("retained vectors", "", "a", "s", &vec![0.0; 384])
        .await
        .unwrap();
    store
        .insert_embedding(&id, "AllMiniLML6V2Q", 384, &vec![0.25; 384])
        .await
        .unwrap();
    drop(store);

    let output = command(&root, project)
        .args(["use-model", "AllMiniLML6V2Q"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(stdout(&output).contains("BGESmallENV15Q -> AllMiniLML6V2Q"));

    let output = command(&root, project)
        .args(["use-model", "SnowflakeArcticEmbedMQ"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(stderr(&output).contains("every active memory"));
    let stats = command(&root, project).arg("stats").output().unwrap();
    assert!(stdout(&stats).contains("AllMiniLML6V2Q"));

    let output = command(&root, project)
        .args(["use-model", "BGESmallENV15Q"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(stdout(&output).contains("AllMiniLML6V2Q -> BGESmallENV15Q"));
}

#[test]
fn invalid_inputs_are_actionable_and_safe_with_leaking_control() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("isolated-root");
    let marker = generated_marker();

    let leaking = Command::new("sh")
        .arg(base_fixture("../leaking_cli.sh"))
        .env("HF_TOKEN", &marker)
        .env("RUST_BACKTRACE", "1")
        .output()
        .unwrap();
    assert!(!leaking.status.success());
    let control_violations = safety_violations(&leaking, &marker);
    assert_eq!(
        control_violations,
        vec!["marker credential", "diagnostic chain", "stack trace"],
        "positive control must make every absence predicate fail"
    );

    let malformed = command(&root, "cli-contract")
        .arg("forget")
        .arg("not-a-uuid")
        .env("HF_TOKEN", &marker)
        .env("RB_TEST_SECRET", &marker)
        .env("RUST_BACKTRACE", "1")
        .output()
        .unwrap();
    assert!(!malformed.status.success());
    assert!(malformed.stdout.is_empty());
    assert!(stderr(&malformed).contains("canonical UUID"));
    assert_safe(&malformed, &marker);

    let invalid_query = command(&root, "cli-contract")
        .arg("recall")
        .arg(" \n\t ")
        .env("HF_TOKEN", &marker)
        .env("RB_TEST_SECRET", &marker)
        .env("RUST_BACKTRACE", "1")
        .output()
        .unwrap();
    assert!(!invalid_query.status.success());
    assert!(invalid_query.stdout.is_empty());
    assert!(stderr(&invalid_query).contains("recall query must not be empty"));
    assert_safe(&invalid_query, &marker);
}

#[tokio::test]
async fn incompatible_model_is_actionable_safe_and_non_mutating() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("isolated-root");
    let project_dir = root.join("projects/cli-contract");
    std::fs::create_dir_all(&project_dir).unwrap();
    let db_path = project_dir.join("brain.db");
    std::fs::copy(base_fixture("brain.db"), &db_path).unwrap();
    let db = libsql::Builder::new_local(&db_path).build().await.unwrap();
    let conn = db.connect().unwrap();
    conn.execute(
        "UPDATE meta SET value = 'DifferentModel' WHERE key = 'embedding_model'",
        params![],
    )
    .await
    .unwrap();
    drop(conn);
    drop(db);
    let before = std::fs::read(&db_path).unwrap();
    let marker = generated_marker();

    let output = command(&root, "cli-contract")
        .arg("stats")
        .env("HF_TOKEN", &marker)
        .env("RB_TEST_SECRET", &marker)
        .env("RUST_BACKTRACE", "1")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(stderr(&output).contains("DifferentModel"));
    assert!(stderr(&output).contains("AllMiniLML6V2Q"));
    assert_safe(&output, &marker);
    let after = std::fs::read(&db_path).unwrap();
    assert!(
        after == before,
        "incompatible-model CLI failure mutated the brain"
    );
}
