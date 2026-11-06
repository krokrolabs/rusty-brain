//! MCP server over stdio: mount rusty-brain from any MCP-capable agent
//! harness (e.g. deepseek-harness's `dsh-mcp-client`).

use std::sync::Arc;

use rmcp::handler::server::tool::ToolCallContext;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
    JsonObject, ListToolsResult, PaginatedRequestParams, ResultType, ServerCapabilities,
    ServerConfig, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{
    ErrorData, RoleServer, ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::app::Brain;

pub const MCP_SUCCESS_CAP_BYTES: usize = 1_000_000;
pub const MCP_ERROR_TEXT_CAP_BYTES: usize = 4_096;
const CATALOG_VERSION: i64 = 1;
const CONTRACT_VERSION: &str = "initial-verified-v1";

/// Recursively key-sorted compact JSON bytes, with null-valued keys rejected.
pub fn canonical_json_bytes(value: &Value) -> anyhow::Result<Vec<u8>> {
    fn normalize(value: &Value) -> anyhow::Result<Value> {
        match value {
            Value::Null => anyhow::bail!("null-valued catalog keys are not allowed"),
            Value::Object(object) => {
                let mut entries = object.iter().collect::<Vec<_>>();
                entries.sort_by(|(left, _), (right, _)| left.as_bytes().cmp(right.as_bytes()));
                let mut normalized = Map::new();
                for (key, value) in entries {
                    normalized.insert(key.clone(), normalize(value)?);
                }
                Ok(Value::Object(normalized))
            }
            Value::Array(values) => values
                .iter()
                .map(normalize)
                .collect::<anyhow::Result<_>>()
                .map(Value::Array),
            _ => Ok(value.clone()),
        }
    }
    serde_json::to_vec(&normalize(value)?).map_err(Into::into)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Apply the cap after every CallToolResult field has been populated.
pub fn bounded_success(result: CallToolResult) -> CallToolResult {
    match serde_json::to_vec(&result) {
        Ok(bytes) if bytes.len() <= MCP_SUCCESS_CAP_BYTES => result,
        Ok(_) => CallToolResult::error(vec![ContentBlock::text(
            "MCP result exceeds 1000000 compact UTF-8 bytes; lower the limit or narrow the query.",
        )]),
        Err(_) => CallToolResult::error(vec![ContentBlock::text(
            "Could not serialize the MCP result; lower the limit or narrow the query.",
        )]),
    }
}

fn truncate_utf8(mut text: String, cap: usize) -> String {
    if text.len() > cap {
        let mut end = cap;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    text
}

/// Convert an application failure into a bounded tool result, never a protocol error.
pub fn application_error(error: &anyhow::Error) -> CallToolResult {
    // Only the top-level message is exposed; anyhow chains and backtraces stay out.
    let mut message = error.to_string();
    for name in [
        "HF_TOKEN",
        "RB_TEST_SECRET",
        "DEEPSEEK_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
    ] {
        if let Ok(value) = std::env::var(name)
            && !value.is_empty()
        {
            message = message.replace(&value, "[redacted]");
        }
    }
    if let Some(index) = message.find("\nCaused by:") {
        message.truncate(index);
    }
    if let Some(index) = message.find("\nStack backtrace") {
        message.truncate(index);
    }
    if message.contains("Caused by:") || message.to_lowercase().contains("stack backtrace:") {
        message =
            "The operation failed with an internal diagnostic; retry with corrected input.".into();
    }
    let message = truncate_utf8(message, MCP_ERROR_TEXT_CAP_BYTES);
    CallToolResult::error(vec![ContentBlock::text(message)])
}

fn invalid_params(error: serde_json::Error) -> ErrorData {
    ErrorData::invalid_params(
        truncate_utf8(
            format!("failed to deserialize parameters: {error}"),
            MCP_ERROR_TEXT_CAP_BYTES,
        ),
        None,
    )
}

fn structured_success<T: Serialize>(value: T, text: Option<String>) -> CallToolResult {
    match serde_json::to_value(value) {
        Ok(value) => {
            let mut result = CallToolResult::structured(value);
            if let Some(text) = text {
                result.content = vec![ContentBlock::text(text)];
            }
            bounded_success(result)
        }
        Err(error) => application_error(&anyhow::Error::new(error)),
    }
}

fn schema(value: Value) -> Arc<JsonObject> {
    match value {
        Value::Object(object) => Arc::new(object),
        _ => unreachable!("tool schemas are objects"),
    }
}

fn remember_output_schema() -> Arc<JsonObject> {
    schema(
        json!({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string","format":"uuid"}}}),
    )
}

fn forget_output_schema() -> Arc<JsonObject> {
    schema(
        json!({"type":"object","additionalProperties":false,"required":["id","forgotten"],"properties":{"id":{"type":"string","format":"uuid"},"forgotten":{"type":"boolean"}}}),
    )
}

fn recall_output_schema() -> Arc<JsonObject> {
    schema(json!({
        "type":"object","additionalProperties":false,"required":["hits","truncated"],
        "properties":{
            "hits":{"type":"array","items":{
                "type":"object","additionalProperties":false,"required":["memory","score","distance","matched_text"],
                "properties":{
                    "memory":{"type":"object","additionalProperties":false,"required":["id","content","tags","embedding_model","created_at","agent","session"],"properties":{
                        "id":{"type":"string","format":"uuid"},"content":{"type":"string"},"tags":{"type":"string"},
                        "embedding_model":{"type":"string"},"created_at":{"type":"string","format":"date-time"},
                        "agent":{"type":"string"},"session":{"type":"string"}
                    }},
                    "score":{"type":"number"},"distance":{"type":["number","null"]},"matched_text":{"type":"boolean"}
                }
            }},
            "truncated":{"type":"boolean","const":false}
        }
    }))
}

fn stats_output_schema() -> Arc<JsonObject> {
    schema(
        json!({"type":"object","additionalProperties":false,"required":["memories","model","schema_version","db_path","db_bytes"],"properties":{
            "memories":{"type":"integer","minimum":0},"model":{"type":"string"},
            "schema_version":{"type":"integer","const":3},"db_path":{"type":"string"},"db_bytes":{"type":"integer","minimum":0},
            "by_agent":{"type":"array","items":{"type":"array","prefixItems":[{"type":"string"},{"type":"integer"}]}},
            "models":{"type":"array","items":{"type":"string"}}
        }}),
    )
}

fn path_output_schema() -> Arc<JsonObject> {
    schema(
        json!({"type":"object","additionalProperties":false,"required":["home","brain_file","project","model","models_dir","project_dir"],"properties":{
            "home":{"type":"string"},"brain_file":{"type":"string"},"project":{"type":"string"},"model":{"type":"string"},
            "models_dir":{"type":"string"},"project_dir":{"type":"string"}
        }}),
    )
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RememberArgs {
    /// The fact or note to store for this project.
    #[schemars(length(max = 100_000))]
    pub content: String,
    /// Optional comma-separated tags (e.g. "decision,duckdb").
    #[schemars(length(max = 1_024))]
    pub tags: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecallArgs {
    /// Natural-language or keyword query.
    #[schemars(length(max = 4_096))]
    pub query: String,
    /// Maximum number of memories to return (default 5).
    pub limit: Option<usize>,
    /// Recall scope: `all` (default), `self`, `others`, or an agent name.
    #[schemars(length(max = 128))]
    pub agent: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ForgetArgs {
    /// The memory id to remove (soft delete).
    pub id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EmptyArgs {}

#[derive(Debug, Serialize)]
struct PathInfo {
    home: String,
    brain_file: String,
    project: String,
    model: String,
    models_dir: String,
    project_dir: String,
}

#[derive(Clone)]
pub struct BrainServer {
    brain: Arc<Brain>,
}

#[tool_router]
impl BrainServer {
    pub fn new(brain: Arc<Brain>) -> Self {
        Self { brain }
    }

    #[tool(
        description = "Store a fact about this project in the local brain (embedded locally, no network).",
        output_schema = remember_output_schema(),
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false)
    )]
    async fn remember(
        &self,
        Parameters(args): Parameters<RememberArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let tags = args.tags.unwrap_or_default();
        match self.brain.remember(&args.content, &tags).await {
            Ok(id) => Ok(structured_success(json!({"id": id}), None)),
            Err(error) => Ok(application_error(&error)),
        }
    }

    #[tool(
        description = "Recall relevant memories for this project (hybrid semantic + keyword search).",
        output_schema = recall_output_schema(),
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn recall(
        &self,
        Parameters(args): Parameters<RecallArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        match self
            .brain
            .recall(
                &args.query,
                args.limit.unwrap_or(5),
                self.brain.agent_filter(args.agent.as_deref()),
            )
            .await
        {
            Ok(hits) => match serde_json::to_string(&hits) {
                Ok(text) => Ok(structured_success(
                    json!({"hits": hits, "truncated": false}),
                    Some(text),
                )),
                Err(error) => Ok(application_error(&anyhow::Error::new(error))),
            },
            Err(error) => Ok(application_error(&error)),
        }
    }

    #[tool(
        description = "Remove a memory by id (soft delete; it disappears from recall and exports).",
        output_schema = forget_output_schema(),
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = true, open_world_hint = false)
    )]
    async fn forget(
        &self,
        Parameters(args): Parameters<ForgetArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        match self.brain.forget(&args.id).await {
            Ok(forgotten) => Ok(structured_success(
                json!({"id": args.id, "forgotten": forgotten}),
                None,
            )),
            Err(error) => Ok(application_error(&error)),
        }
    }

    #[tool(
        description = "Show the brain's size, model and file locations — where your data lives.",
        output_schema = stats_output_schema(),
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn stats(
        &self,
        Parameters(_args): Parameters<EmptyArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        match self.brain.stats().await {
            Ok(stats) => Ok(structured_success(stats, None)),
            Err(error) => Ok(application_error(&error)),
        }
    }

    #[tool(
        description = "Return the exact paths of the brain file, model cache and project directory.",
        output_schema = path_output_schema(),
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false)
    )]
    async fn data_path(
        &self,
        Parameters(_args): Parameters<EmptyArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        match self.brain.current_model().await {
            Ok(model) => {
                let info = PathInfo {
                    home: self.brain.cfg.home.display().to_string(),
                    brain_file: self.brain.cfg.db_path().display().to_string(),
                    project: self.brain.cfg.project.clone(),
                    model,
                    models_dir: self.brain.cfg.models_dir().display().to_string(),
                    project_dir: self.brain.cfg.project_dir().display().to_string(),
                };
                Ok(structured_success(info, None))
            }
            Err(error) => Ok(application_error(&error)),
        }
    }
}

pub fn tool_definitions() -> Vec<Tool> {
    let mut tools = vec![
        BrainServer::remember_tool_attr(),
        BrainServer::recall_tool_attr(),
        BrainServer::forget_tool_attr(),
        BrainServer::stats_tool_attr(),
        BrainServer::data_path_tool_attr(),
    ];
    tools.sort_by(|left, right| left.name.cmp(&right.name));
    tools
}

pub fn validate_catalog(catalog: &Value, tools: &[Tool]) -> anyhow::Result<()> {
    canonical_json_bytes(catalog)?;
    anyhow::ensure!(
        catalog["catalogVersion"] == CATALOG_VERSION,
        "catalogVersion drift"
    );
    anyhow::ensure!(
        catalog["contractVersion"] == CONTRACT_VERSION,
        "contractVersion drift"
    );
    anyhow::ensure!(catalog["transport"] == "stdio", "transport drift");
    anyhow::ensure!(catalog["serverName"] == "rusty-brain", "serverName drift");
    anyhow::ensure!(
        catalog["localIdentity"] == json!(["serverName", "project", "cwd"]),
        "local identity drift"
    );
    anyhow::ensure!(
        catalog["outputCapBytes"] == MCP_SUCCESS_CAP_BYTES,
        "output cap drift"
    );
    anyhow::ensure!(
        catalog["errorTextCapBytes"] == MCP_ERROR_TEXT_CAP_BYTES,
        "error cap drift"
    );
    anyhow::ensure!(
        catalog["protocolRevisions"]
            == json!([
                "2024-11-05",
                "2025-03-26",
                "2025-06-18",
                "2025-11-25",
                "2026-07-28"
            ]),
        "protocol revisions drift"
    );
    let entries = catalog["tools"].as_array().cloned().unwrap_or_default();
    anyhow::ensure!(
        entries.len() == tools.len() && tools.len() == 5,
        "tool count drift"
    );
    for (tool, entry) in tools.iter().zip(entries.iter()) {
        anyhow::ensure!(entry["name"] == tool.name.as_ref(), "tool name/order drift");
        let description = tool
            .description
            .as_ref()
            .map(|value| value.as_ref())
            .unwrap_or_default();
        anyhow::ensure!(
            entry["description"] == description,
            "{} description drift",
            tool.name
        );
        let input = Value::Object(tool.input_schema.as_ref().clone());
        let output = Value::Object(
            tool.output_schema
                .as_ref()
                .expect("output schema")
                .as_ref()
                .clone(),
        );
        let annotations = serde_json::to_value(tool.annotations.as_ref().expect("annotations"))?;
        anyhow::ensure!(
            entry["inputSchema"] == input,
            "{} input schema drift",
            tool.name
        );
        anyhow::ensure!(
            entry["outputSchema"] == output,
            "{} output schema drift",
            tool.name
        );
        anyhow::ensure!(
            entry["annotations"] == annotations,
            "{} annotation drift",
            tool.name
        );
        anyhow::ensure!(entry["scope"] == "local", "{} scope drift", tool.name);
        anyhow::ensure!(
            entry["outputCapBytes"] == MCP_SUCCESS_CAP_BYTES,
            "{} cap drift",
            tool.name
        );
        for field in [
            "effect",
            "risk",
            "confirmation",
            "idempotency",
            "sharedOperation",
        ] {
            anyhow::ensure!(
                entry[field].is_string() && !entry[field].as_str().unwrap().is_empty(),
                "{} missing {field}",
                tool.name
            );
        }
        anyhow::ensure!(
            entry["requiredScopes"] == json!([]),
            "{} scopes must be empty for local trust",
            tool.name
        );
        anyhow::ensure!(entry["limits"].is_object(), "{} limits missing", tool.name);
        let hashes = &entry["hashes"];
        anyhow::ensure!(
            hashes["description"] == sha256_hex(description.as_bytes()),
            "{} description hash drift",
            tool.name
        );
        anyhow::ensure!(
            hashes["inputSchema"] == sha256_hex(&canonical_json_bytes(&input)?),
            "{} input hash drift",
            tool.name
        );
        anyhow::ensure!(
            hashes["outputSchema"] == sha256_hex(&canonical_json_bytes(&output)?),
            "{} output hash drift",
            tool.name
        );
        anyhow::ensure!(
            hashes["annotations"] == sha256_hex(&canonical_json_bytes(&annotations)?),
            "{} annotation hash drift",
            tool.name
        );
    }
    Ok(())
}

#[tool_handler]
impl ServerHandler for BrainServer {
    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        // rmcp 3.9 normally converts argument deserialization invalid-params into
        // isError. Preserve the wire contract by checking closed input shapes
        // before its router; semantic validation remains an application error.
        let arguments = Value::Object(request.arguments.clone().unwrap_or_default());
        let parsed = match request.name.as_ref() {
            "remember" => serde_json::from_value::<RememberArgs>(arguments).map(|_| ()),
            "recall" => serde_json::from_value::<RecallArgs>(arguments).map(|_| ()),
            "forget" => serde_json::from_value::<ForgetArgs>(arguments).map(|_| ()),
            "stats" | "data_path" => serde_json::from_value::<EmptyArgs>(arguments).map(|_| ()),
            _ => Ok(()),
        };
        if let Err(error) = parsed {
            return Err(invalid_params(error));
        }
        Self::tool_router()
            .call(ToolCallContext::new(self, request, context))
            .await
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult {
            result_type: Some(ResultType::COMPLETE),
            tools: tool_definitions(),
            meta: None,
            next_cursor: None,
            ttl_ms: None,
            cache_scope: None,
        })
    }

    fn get_info(&self) -> ServerConfig {
        let mut info = ServerConfig::new(ServerCapabilities::builder().enable_tools().build());
        info.server_info = Implementation::new("rusty-brain", env!("CARGO_PKG_VERSION"));
        info.instructions = Some(
            "Local project memory. Use `remember` to store durable facts, `recall` before \
             answering questions about past work, `data_path` to show where data lives. \
             All data stays on this machine."
                .into(),
        );
        info
    }
}

pub async fn serve(brain: Arc<Brain>) -> anyhow::Result<()> {
    let service = BrainServer::new(brain)
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::ToolAnnotations;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::process::Command;

    fn result_with_exact_bytes(target: usize) -> CallToolResult {
        let mut low = 0;
        let mut high = target;
        while low < high {
            let middle = (low + high) / 2;
            let candidate = CallToolResult::success(vec![ContentBlock::text("x".repeat(middle))]);
            if serde_json::to_vec(&candidate).unwrap().len() < target {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        let result = CallToolResult::success(vec![ContentBlock::text("x".repeat(low))]);
        assert_eq!(serde_json::to_vec(&result).unwrap().len(), target);
        result
    }

    fn catalog() -> Value {
        serde_json::from_str(include_str!("../contracts/mcp-tools.json")).unwrap()
    }

    #[test]
    fn canonical_hash_known_vectors_match_rust_and_independent_python() {
        let schema = json!({
            "type": "object",
            "required": ["content"],
            "properties": {"content": {"type": "string"}},
            "additionalProperties": false
        });
        let annotations = json!({
            "openWorldHint": false,
            "readOnlyHint": false,
            "destructiveHint": false,
            "idempotentHint": false
        });
        let schema_bytes = canonical_json_bytes(&schema).unwrap();
        let annotation_bytes = canonical_json_bytes(&annotations).unwrap();
        assert_eq!(
            schema_bytes,
            br#"{"additionalProperties":false,"properties":{"content":{"type":"string"}},"required":["content"],"type":"object"}"#
        );
        assert_eq!(
            annotation_bytes,
            br#"{"destructiveHint":false,"idempotentHint":false,"openWorldHint":false,"readOnlyHint":false}"#
        );
        let vectors = [
            (
                "Store a fact.".as_bytes(),
                "339ae9685464f75e8c72a4f1a56c3e9bb29cac225138bb5b15d7e71ea8b24093",
            ),
            (
                schema_bytes.as_slice(),
                "c9fd9a503f520ca7c958067d2db0bc2b1eef6fe5bbae6c7b9356cb921593c49a",
            ),
            (
                annotation_bytes.as_slice(),
                "d72d3cb182e3d7fcf0b1c6782251eee772cdf2a99470e1a4bd6a006b5c8f4e25",
            ),
        ];
        for (bytes, expected) in vectors {
            assert_eq!(format!("{:x}", Sha256::digest(bytes)), expected);
        }

        let python = r#"
import hashlib, json, sys
schema = {"type":"object","required":["content"],"properties":{"content":{"type":"string"}},"additionalProperties":False}
annotations = {"openWorldHint":False,"readOnlyHint":False,"destructiveHint":False,"idempotentHint":False}
values = [sys.argv[1].encode(), json.dumps(schema, sort_keys=True, separators=(",", ":")).encode(), json.dumps(annotations, sort_keys=True, separators=(",", ":")).encode()]
print(json.dumps([hashlib.sha256(value).hexdigest() for value in values]))
"#;
        let output = Command::new("uv")
            .args([
                "run",
                "--no-project",
                "python",
                "-I",
                "-c",
                python,
                "Store a fact.",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let actual: Vec<String> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(actual, vectors.map(|(_, expected)| expected.to_string()));
    }

    #[test]
    fn exact_call_tool_result_cap_boundaries_and_utf8_error_bound() {
        for size in [999_999, 1_000_000] {
            let result = result_with_exact_bytes(size);
            let bounded = bounded_success(result.clone());
            assert_eq!(bounded, result, "{size} bytes must succeed unchanged");
        }
        let over = bounded_success(result_with_exact_bytes(1_000_001));
        assert_eq!(over.is_error, Some(true));
        let text = over.content[0].as_text().unwrap().text.as_str();
        assert!(text.contains("lower the limit") && text.contains("narrow the query"));
        assert!(text.len() <= 4_096);
        assert!(serde_json::to_vec(&over).unwrap().len() < 1_000_000);
    }

    #[test]
    fn application_errors_are_bounded_and_marker_chain_stack_safe() {
        let marker = format!("rb006-unrecognized-mcp-marker-{}", uuid::Uuid::new_v4());
        temp_env_marker(&marker, || {
            let error = anyhow::anyhow!(
                "top-level failure\nCaused by: {marker}\nStack backtrace:\nsecret details"
            );
            let result = application_error(&error);
            assert_eq!(result.is_error, Some(true));
            let text = &result.content[0].as_text().unwrap().text;
            assert!(!text.contains(&marker));
            assert!(!text.contains("Caused by:"));
            assert!(!text.contains("Stack backtrace"));
            assert!(text.len() <= 4_096);

            let long = application_error(&anyhow::anyhow!("é".repeat(5_000)));
            assert_eq!(long.content[0].as_text().unwrap().text.len(), 4_096);
        });
    }

    fn temp_env_marker(marker: &str, run: impl FnOnce()) {
        // This test is synchronous and restores its process-local test marker.
        let old = std::env::var_os("RB_TEST_SECRET");
        unsafe { std::env::set_var("RB_TEST_SECRET", marker) };
        run();
        match old {
            Some(value) => unsafe { std::env::set_var("RB_TEST_SECRET", value) },
            None => unsafe { std::env::remove_var("RB_TEST_SECRET") },
        }
    }

    #[tokio::test]
    async fn data_path_reports_the_active_nondefault_model_and_all_paths() {
        let temp = tempfile::tempdir().unwrap();
        let cfg = crate::config::Config::new(
            Some(temp.path().join("home")),
            Some("paths".into()),
            Some("test".into()),
            None,
        )
        .unwrap();
        let brain = Arc::new(
            crate::app::Brain::open_synthetic(cfg.clone())
                .await
                .unwrap(),
        );
        brain.remember("model identity", "").await.unwrap();
        brain
            .reembed("AllMiniLML6V2Q", crate::app::Prune::Keep)
            .await
            .unwrap();
        let server = BrainServer::new(brain);
        let result = server.data_path(Parameters(EmptyArgs {})).await.unwrap();
        let value = result.structured_content.unwrap();
        assert_eq!(value["model"], "AllMiniLML6V2Q");
        assert_eq!(value["models_dir"], cfg.models_dir().display().to_string());
        assert_eq!(
            value["project_dir"],
            cfg.project_dir().display().to_string()
        );
    }

    #[test]
    fn input_schema_limits_and_protocol_error_text_are_bounded() {
        let tools = tool_definitions();
        let remember = tools.iter().find(|tool| tool.name == "remember").unwrap();
        assert_eq!(
            remember.input_schema["properties"]["content"]["maxLength"],
            100_000
        );
        assert_eq!(
            remember.input_schema["properties"]["tags"]["maxLength"],
            1_024
        );
        let recall = tools.iter().find(|tool| tool.name == "recall").unwrap();
        assert_eq!(
            recall.input_schema["properties"]["query"]["maxLength"],
            4_096
        );
        assert_eq!(recall.input_schema["properties"]["agent"]["maxLength"], 128);

        let unknown = "é".repeat(5_000);
        let error = serde_json::from_value::<RememberArgs>(json!({
            "content": "not stored", unknown: true
        }))
        .unwrap_err();
        let error = invalid_params(error);
        assert!(error.message.len() <= MCP_ERROR_TEXT_CAP_BYTES);
        assert!(error.message.is_char_boundary(error.message.len()));
        assert!(
            error
                .message
                .starts_with("failed to deserialize parameters:")
        );
    }

    #[test]
    fn catalog_matches_sorted_closed_annotated_tools_and_controls_fail() {
        let tools = tool_definitions();
        let names: Vec<&str> = tools.iter().map(|tool| tool.name.as_ref()).collect();
        assert_eq!(
            names,
            ["data_path", "forget", "recall", "remember", "stats"]
        );
        for tool in &tools {
            assert_eq!(
                tool.input_schema.get("additionalProperties"),
                Some(&Value::Bool(false))
            );
            assert!(
                tool.output_schema.is_some(),
                "{} lacks outputSchema",
                tool.name
            );
            let annotations = tool.annotations.as_ref().unwrap();
            assert_eq!(annotations.open_world_hint, Some(false));
            assert!(annotations.read_only_hint.is_some());
            assert!(annotations.destructive_hint.is_some());
            assert!(annotations.idempotent_hint.is_some());
        }
        let remember = tools.iter().find(|tool| tool.name == "remember").unwrap();
        assert_eq!(
            remember.annotations,
            Some(ToolAnnotations::from_raw(
                None,
                Some(false),
                Some(false),
                Some(false),
                Some(false)
            ))
        );
        let forget = tools.iter().find(|tool| tool.name == "forget").unwrap();
        assert_eq!(
            forget.annotations,
            Some(ToolAnnotations::from_raw(
                None,
                Some(false),
                Some(true),
                Some(true),
                Some(false)
            ))
        );
        for name in ["data_path", "recall", "stats"] {
            let tool = tools.iter().find(|tool| tool.name == name).unwrap();
            assert_eq!(
                tool.annotations,
                Some(ToolAnnotations::from_raw(
                    None,
                    Some(true),
                    Some(false),
                    Some(true),
                    Some(false)
                ))
            );
        }

        let original = catalog();
        validate_catalog(&original, &tools).unwrap();

        let mut null_control = original.clone();
        null_control["tools"][0]["effect"] = Value::Null;
        assert!(validate_catalog(&null_control, &tools).is_err());

        let mut description_control = original.clone();
        description_control["tools"][0]["description"] = Value::String("drift".into());
        assert!(validate_catalog(&description_control, &tools).is_err());

        let mut schema_control = original.clone();
        schema_control["tools"][0]["inputSchema"]["additionalProperties"] = Value::Bool(true);
        assert!(validate_catalog(&schema_control, &tools).is_err());

        let mut annotation_control = original.clone();
        annotation_control["tools"][0]["annotations"]["openWorldHint"] = Value::Bool(true);
        assert!(validate_catalog(&annotation_control, &tools).is_err());

        let mut value_control = original.clone();
        value_control["outputCapBytes"] = Value::from(999_999);
        assert!(validate_catalog(&value_control, &tools).is_err());
    }
}
