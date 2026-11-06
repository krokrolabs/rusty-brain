//! Real rmcp child-process MCP contracts. Model-backed checks are explicitly gated.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::RunningService;
use rmcp::transport::TokioChildProcess;
use rmcp::{RoleClient, ServiceExt};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdout, Command};
use tokio::task::JoinHandle;
use uuid::Uuid;

const NAMES: [&str; 5] = ["data_path", "forget", "recall", "remember", "stats"];

type Client = RunningService<RoleClient, ()>;

fn enabled() -> bool {
    if std::env::var("RUSTY_BRAIN_E2E").as_deref() != Ok("1") {
        eprintln!(
            "skipping real MCP stdio check; set RUSTY_BRAIN_E2E=1 with a primed isolated home"
        );
        return false;
    }
    true
}

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_rusty-brain")
}

fn isolated_environment() -> (PathBuf, String) {
    let home = PathBuf::from(std::env::var_os("RUSTY_BRAIN_HOME").expect("RUSTY_BRAIN_HOME"));
    let project = std::env::var("RUSTY_BRAIN_PROJECT").expect("RUSTY_BRAIN_PROJECT");
    assert!(home.is_absolute());
    assert!(
        home.to_str()
            .is_some_and(|path| path.starts_with("/tmp/rusty-brain-verify.")),
        "real stdio tests require an isolated quickstart home, got {}",
        home.display()
    );
    assert!(!project.is_empty());
    (home, project)
}

fn assert_cache_primed(home: &Path) {
    let ready = walkdir_files(&home.join("models"))
        .into_iter()
        .any(|path| std::fs::metadata(path).is_ok_and(|metadata| metadata.len() > 0));
    assert!(
        ready,
        "run scripts/verify-local.sh first to prime {}",
        home.join("models").display()
    );
}

fn walkdir_files(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut files = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files.extend(walkdir_files(&path));
        } else {
            files.push(path);
        }
    }
    files
}

fn configure(command: &mut Command, home: &Path, project: &str, marker: &str) {
    let os_home = home.join("os-home");
    std::fs::create_dir_all(&os_home).unwrap();
    command
        .env_clear()
        .env("HOME", os_home)
        .env(
            "PATH",
            std::env::var("PATH").unwrap_or_else(|_| "/usr/local/bin:/usr/bin:/bin".into()),
        )
        .env(
            "TMPDIR",
            std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into()),
        )
        .env("LANG", "en_US.UTF-8")
        .env("RUSTY_BRAIN_HOME", home)
        .env("RUSTY_BRAIN_PROJECT", project)
        .env("RB_TEST_SECRET", marker)
        .env("DEEPSEEK_API_KEY", marker)
        .env("HF_TOKEN", marker)
        .env("AWS_SECRET_ACCESS_KEY", marker)
        .env("RUST_BACKTRACE", "1");
}

async fn spawn_client(home: &Path, project: &str, marker: &str) -> (Client, JoinHandle<String>) {
    let mut command = Command::new(binary());
    command.args(["--project", project, "serve"]);
    configure(&mut command, home, project, marker);
    let (transport, stderr) = TokioChildProcess::builder(command)
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stderr = stderr.unwrap();
    let stderr_task = tokio::spawn(async move {
        let mut bytes = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut stderr, &mut bytes)
            .await
            .unwrap();
        String::from_utf8_lossy(&bytes).into_owned()
    });
    let client = ().serve(transport).await.expect("initialize real MCP server");
    (client, stderr_task)
}

async fn call(
    client: &Client,
    name: &str,
    arguments: Value,
) -> Result<CallToolResult, rmcp::ServiceError> {
    let params = CallToolRequestParams::new(name.to_string())
        .with_arguments(arguments.as_object().unwrap().clone());
    client.call_tool(params).await
}

fn text(result: &CallToolResult) -> &str {
    result.content[0]
        .as_text()
        .expect("text fallback")
        .text
        .as_str()
}

fn structured(result: &CallToolResult) -> &Value {
    result
        .structured_content
        .as_ref()
        .expect("structuredContent")
}

fn safety_violations(text: &str, marker: &str) -> Vec<&'static str> {
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

fn assert_safe_result(result: &CallToolResult, marker: &str) {
    let rendered = format!("{result:?}");
    assert!(
        safety_violations(&rendered, marker).is_empty(),
        "unsafe MCP result: {rendered}"
    );
    assert!(
        text(result).len() <= 4_096,
        "error text exceeds 4096 UTF-8 bytes"
    );
}

#[test]
fn marker_chain_and_stack_absence_checks_reject_deliberate_leak() {
    let marker = format!("rb006-unrecognized-mcp-control-{}", Uuid::new_v4());
    let output = std::process::Command::new("sh")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/leaking_cli.sh"))
        .env("HF_TOKEN", &marker)
        .env("RUST_BACKTRACE", "1")
        .output()
        .unwrap();
    let combined_bytes = [output.stdout, output.stderr].concat();
    let combined = String::from_utf8_lossy(&combined_bytes);
    assert_eq!(
        safety_violations(&combined, &marker),
        ["marker credential", "diagnostic chain", "stack trace"]
    );
}

#[tokio::test]
async fn real_stdio_catalog_results_errors_lifecycle_and_cap() {
    if !enabled() {
        return;
    }
    let (home, project) = isolated_environment();
    assert_cache_primed(&home);
    let marker = format!("rb006-unrecognized-mcp-live-{}", Uuid::new_v4());
    let (client, stderr_task) = spawn_client(&home, &project, &marker).await;

    let listed = client.list_tools(None).await.unwrap();
    let names: Vec<&str> = listed.tools.iter().map(|tool| tool.name.as_ref()).collect();
    assert_eq!(names, NAMES);
    for tool in &listed.tools {
        assert_eq!(
            tool.input_schema.get("additionalProperties"),
            Some(&Value::Bool(false))
        );
        assert!(
            tool.output_schema.is_some(),
            "{} has no output schema",
            tool.name
        );
        let annotations = tool.annotations.as_ref().unwrap();
        assert!(annotations.read_only_hint.is_some());
        assert!(annotations.destructive_hint.is_some());
        assert!(annotations.idempotent_hint.is_some());
        assert_eq!(annotations.open_world_hint, Some(false));
    }
    let catalog: Value = serde_json::from_str(include_str!("../contracts/mcp-tools.json")).unwrap();
    rusty_brain::mcp::validate_catalog(&catalog, &listed.tools).unwrap();

    let unknown = call(
        &client,
        "remember",
        json!({"content": "not stored", "unknown": true}),
    )
    .await;
    assert!(
        unknown.is_err(),
        "unknown input fields must be invalid params: {unknown:?}"
    );

    let zero = call(&client, "recall", json!({"query": "unused", "limit": 0}))
        .await
        .unwrap();
    assert_eq!(zero.is_error, Some(false));
    assert_eq!(structured(&zero), &json!({"hits": [], "truncated": false}));
    assert_eq!(text(&zero), "[]");

    for (name, arguments, expected) in [
        (
            "recall",
            json!({"query": " \n\t ", "limit": 5}),
            "query must not be empty",
        ),
        (
            "recall",
            json!({"query": "unused", "limit": 51}),
            "between 0 and 50",
        ),
        ("recall", json!({"query": "q".repeat(4_097)}), "4096"),
        (
            "remember",
            json!({"content": "not stored", "tags": "t".repeat(1_025)}),
            "1024",
        ),
        (
            "recall",
            json!({"query": "bounded", "agent": "a".repeat(129)}),
            "128",
        ),
        ("forget", json!({"id": "not-a-uuid"}), "canonical UUID"),
    ] {
        let result = call(&client, name, arguments).await.unwrap();
        assert_eq!(result.is_error, Some(true), "{name} must be a tool error");
        assert!(
            text(&result).contains(expected),
            "{} lacked {expected:?}",
            text(&result)
        );
        assert_safe_result(&result, &marker);
    }

    let unknown_id = "00000000-0000-4000-8000-000000000000";
    let noop = call(&client, "forget", json!({"id": unknown_id}))
        .await
        .unwrap();
    assert_eq!(
        structured(&noop),
        &json!({"id": unknown_id, "forgotten": false})
    );

    let path = call(&client, "data_path", json!({})).await.unwrap();
    assert_eq!(structured(&path)["home"], home.display().to_string());
    assert_eq!(structured(&path)["project"], project);
    assert_eq!(
        structured(&path)["brain_file"],
        home.join("projects")
            .join(&project)
            .join("brain.db")
            .display()
            .to_string()
    );
    assert_eq!(structured(&path)["model"], "BGESmallENV15Q");
    assert_eq!(
        structured(&path)["models_dir"],
        home.join("models").display().to_string()
    );
    assert_eq!(
        structured(&path)["project_dir"],
        home.join("projects").join(&project).display().to_string()
    );
    assert_eq!(
        text(&path).parse::<Value>().unwrap(),
        structured(&path).clone()
    );

    let before = call(&client, "stats", json!({})).await.unwrap();
    let initial_count = structured(&before)["memories"].as_i64().unwrap();
    assert_eq!(structured(&before)["model"], "BGESmallENV15Q");
    assert_eq!(structured(&before)["schema_version"], 3);

    let token = format!("RB-MCP-{}", Uuid::new_v4());
    let remembered = call(
        &client,
        "remember",
        json!({"content": format!("Verification token: {token}"), "tags": "verification"}),
    )
    .await
    .unwrap();
    assert_eq!(remembered.is_error, Some(false), "{}", text(&remembered));
    let id = structured(&remembered)["id"].as_str().unwrap().to_string();
    assert_eq!(
        text(&remembered).parse::<Value>().unwrap(),
        structured(&remembered).clone()
    );

    let recalled = call(&client, "recall", json!({"query": token, "limit": 5}))
        .await
        .unwrap();
    let hits = structured(&recalled)["hits"].as_array().unwrap();
    assert!(
        hits.iter().any(|hit| hit["memory"]["id"] == id),
        "remembered id absent: {hits:?}"
    );
    let text_hits: Value = text(&recalled).parse().unwrap();
    assert_eq!(text_hits, Value::Array(hits.clone()));

    let during = call(&client, "stats", json!({})).await.unwrap();
    assert_eq!(structured(&during)["memories"], initial_count + 1);
    let forgotten = call(&client, "forget", json!({"id": id})).await.unwrap();
    assert_eq!(
        structured(&forgotten),
        &json!({"id": id, "forgotten": true})
    );
    let absent = call(&client, "recall", json!({"query": token, "limit": 5}))
        .await
        .unwrap();
    assert!(
        structured(&absent)["hits"]
            .as_array()
            .unwrap()
            .iter()
            .all(|hit| hit["memory"]["id"] != id)
    );

    let cap_token = format!("RB-CAP-{}", Uuid::new_v4());
    let mut cap_ids = Vec::new();
    for _ in 0..6 {
        let content = format!("{cap_token} {}", "x".repeat(100_000 - cap_token.len() - 1));
        assert_eq!(content.len(), 100_000);
        let result = call(
            &client,
            "remember",
            json!({"content": content, "tags": "cap"}),
        )
        .await
        .unwrap();
        assert_eq!(result.is_error, Some(false), "{}", text(&result));
        cap_ids.push(structured(&result)["id"].as_str().unwrap().to_string());
    }
    let one = call(&client, "recall", json!({"query": cap_token, "limit": 1}))
        .await
        .unwrap();
    assert_eq!(
        one.is_error,
        Some(false),
        "one large hit must remain below cap: {}",
        text(&one)
    );
    assert!(serde_json::to_vec(&one).unwrap().len() <= 1_000_000);
    let over = call(&client, "recall", json!({"query": cap_token, "limit": 50}))
        .await
        .unwrap();
    assert_eq!(
        over.is_error,
        Some(true),
        "real over-cap recall was not rejected"
    );
    assert!(text(&over).contains("lower the limit") && text(&over).contains("narrow the query"));
    assert_safe_result(&over, &marker);
    for id in cap_ids {
        let result = call(&client, "forget", json!({"id": id})).await.unwrap();
        assert_eq!(structured(&result)["forgotten"], true);
    }
    let after = call(&client, "stats", json!({})).await.unwrap();
    assert_eq!(structured(&after)["memories"], initial_count);

    client.cancel().await.unwrap();
    let stderr = stderr_task.await.unwrap();
    assert!(
        safety_violations(&stderr, &marker).is_empty(),
        "unsafe stderr: {stderr}"
    );
}

#[tokio::test]
async fn missing_cache_is_bounded_actionable_and_non_mutating() {
    if !enabled() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("empty-home");
    let project = "missing-cache";
    let marker = format!("rb006-unrecognized-mcp-cache-{}", Uuid::new_v4());
    let mut command = Command::new(binary());
    command.args(["--project", project, "serve"]);
    configure(&mut command, &home, project, &marker);
    command.env("HF_HOME", temp.path().join("empty-hf"));
    command.env("HF_ENDPOINT", "http://127.0.0.1:9");
    let (transport, stderr) = TokioChildProcess::builder(command)
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stderr = stderr.unwrap();
    let stderr_task = tokio::spawn(async move {
        let mut bytes = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut stderr, &mut bytes)
            .await
            .unwrap();
        String::from_utf8_lossy(&bytes).into_owned()
    });
    let client = ().serve(transport).await.unwrap();
    let result = call(&client, "remember", json!({"content": "must not persist"}))
        .await
        .unwrap();
    assert_eq!(result.is_error, Some(true));
    let message = text(&result);
    assert!(
        message.contains("first") && message.contains("download"),
        "missing first-download guidance: {message}"
    );
    assert_safe_result(&result, &marker);
    client.cancel().await.unwrap();
    let stderr = stderr_task.await.unwrap();
    assert!(
        safety_violations(&stderr, &marker).is_empty(),
        "unsafe stderr: {stderr}"
    );
    assert!(
        !home
            .join("projects")
            .join(project)
            .join("brain.db")
            .exists()
    );
}

async fn read_response(reader: &mut BufReader<ChildStdout>, id: u64, lines: &mut Vec<String>) {
    loop {
        let mut line = String::new();
        let read = tokio::time::timeout(Duration::from_secs(30), reader.read_line(&mut line))
            .await
            .expect("timed out waiting for MCP response")
            .unwrap();
        assert_ne!(
            read, 0,
            "MCP child exited before response id={id}; lines={lines:?}"
        );
        let trimmed = line.trim_end().to_string();
        if serde_json::from_str::<Value>(&trimmed).is_ok_and(|value| value["id"] == id) {
            lines.push(trimmed);
            return;
        }
        lines.push(trimmed);
    }
}

fn purity_violations(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|line| {
            !matches!(serde_json::from_str::<Value>(line), Ok(value) if value["jsonrpc"] == "2.0")
        })
        .cloned()
        .collect()
}

async fn raw_conversation(
    executable: &Path,
    mode: Option<&str>,
    home: &Path,
    project: &str,
    calls: &[(&str, Value)],
) -> Vec<String> {
    let mut command = Command::new(executable);
    if let Some(mode) = mode {
        command.arg(mode);
    }
    // The example ignores extra arguments after its contamination mode.
    command.args(["--project", project, "serve"]);
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    if mode.is_none() {
        configure(
            &mut command,
            home,
            project,
            "rb006-raw-purity-marker-not-used",
        );
    }
    let mut child = command.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut reader = BufReader::new(child.stdout.take().unwrap());
    let mut lines = Vec::new();
    stdin.write_all(json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"purity-test","version":"1"}}}).to_string().as_bytes()).await.unwrap();
    stdin.write_all(b"\n").await.unwrap();
    stdin.flush().await.unwrap();
    read_response(&mut reader, 0, &mut lines).await;
    stdin
        .write_all(
            json!({"jsonrpc":"2.0","method":"notifications/initialized"})
                .to_string()
                .as_bytes(),
        )
        .await
        .unwrap();
    stdin.write_all(b"\n").await.unwrap();
    for (index, (name, arguments)) in calls.iter().enumerate() {
        stdin.write_all(json!({"jsonrpc":"2.0","id":index+1,"method":"tools/call","params":{"name":name,"arguments":arguments}}).to_string().as_bytes()).await.unwrap();
        stdin.write_all(b"\n").await.unwrap();
        stdin.flush().await.unwrap();
        read_response(&mut reader, (index + 1) as u64, &mut lines).await;
    }
    drop(stdin);
    tokio::time::timeout(Duration::from_secs(10), child.wait())
        .await
        .unwrap()
        .unwrap();
    lines
}

#[tokio::test]
async fn stdout_purity_rejects_all_controls_before_accepting_real_server() {
    if !enabled() {
        return;
    }
    let (home, project) = isolated_environment();
    let cargo = std::process::Command::new(env!("CARGO"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "build",
            "--offline",
            "--example",
            "stdout_contaminating_server",
        ])
        .output()
        .unwrap();
    assert!(
        cargo.status.success(),
        "{}",
        String::from_utf8_lossy(&cargo.stderr)
    );
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
    let example = target.join("debug/examples/stdout_contaminating_server");
    for (mode, calls) in [
        ("startup", &[][..]),
        ("successful-call", &[("probe", json!({}))][..]),
        ("handled-error", &[("probe", json!({}))][..]),
    ] {
        let lines = raw_conversation(&example, Some(mode), &home, &project, calls).await;
        assert!(
            lines
                .iter()
                .any(|line| line.contains(&format!("stdout-contamination-{mode}"))),
            "control did not contaminate: {lines:?}"
        );
        assert!(
            !purity_violations(&lines).is_empty(),
            "{mode} contamination was accepted: {lines:?}"
        );
    }
    let actual = raw_conversation(
        Path::new(binary()),
        None,
        &home,
        &project,
        &[
            ("data_path", json!({})),
            ("forget", json!({"id":"not-a-uuid"})),
            ("recall", json!({"query":" \n\t "})),
        ],
    )
    .await;
    assert!(
        purity_violations(&actual).is_empty(),
        "real stdout was contaminated: {actual:?}"
    );
}
