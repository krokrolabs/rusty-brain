//! MCP server over stdio: mount rusty-brain from any MCP-capable agent
//! harness (e.g. deepseek-harness's `dsh-mcp-client`).

use std::sync::Arc;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig};
use rmcp::{ErrorData, ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router};
use serde::{Deserialize, Serialize};

use crate::app::Brain;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RememberArgs {
    /// The fact or note to store for this project.
    pub content: String,
    /// Optional comma-separated tags (e.g. "decision,duckdb").
    pub tags: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecallArgs {
    /// Natural-language or keyword query.
    pub query: String,
    /// Maximum number of memories to return (default 5).
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ForgetArgs {
    /// The memory id to remove (soft delete).
    pub id: String,
}

#[derive(Debug, Serialize)]
struct PathInfo {
    home: String,
    brain_file: String,
    project: String,
    model: String,
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
        description = "Store a fact about this project in the local brain (embedded locally, no network)."
    )]
    async fn remember(
        &self,
        Parameters(args): Parameters<RememberArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let tags = args.tags.unwrap_or_default();
        match self.brain.remember(&args.content, &tags).await {
            Ok(id) => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "remembered as {id}"
            ))])),
            Err(e) => Err(ErrorData::internal_error(e.to_string(), None)),
        }
    }

    #[tool(
        description = "Recall relevant memories for this project (hybrid semantic + keyword search)."
    )]
    async fn recall(
        &self,
        Parameters(args): Parameters<RecallArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        match self
            .brain
            .recall(&args.query, args.limit.unwrap_or(5))
            .await
        {
            Ok(hits) => Ok(CallToolResult::success(vec![ContentBlock::text(
                serde_json::to_string_pretty(&hits).unwrap_or_else(|_| "[]".into()),
            )])),
            Err(e) => Err(ErrorData::internal_error(e.to_string(), None)),
        }
    }

    #[tool(
        description = "Remove a memory by id (soft delete; it disappears from recall and exports)."
    )]
    async fn forget(
        &self,
        Parameters(args): Parameters<ForgetArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        match self.brain.forget(&args.id).await {
            Ok(true) => Ok(CallToolResult::success(vec![ContentBlock::text(
                "forgotten",
            )])),
            Ok(false) => Ok(CallToolResult::success(vec![ContentBlock::text(
                "no such memory (or already forgotten)",
            )])),
            Err(e) => Err(ErrorData::internal_error(e.to_string(), None)),
        }
    }

    #[tool(
        description = "Show the brain's size, model and file locations — where your data lives."
    )]
    async fn stats(&self) -> Result<CallToolResult, ErrorData> {
        match self.brain.stats().await {
            Ok(stats) => Ok(CallToolResult::success(vec![ContentBlock::text(
                serde_json::to_string_pretty(&stats).unwrap_or_default(),
            )])),
            Err(e) => Err(ErrorData::internal_error(e.to_string(), None)),
        }
    }

    #[tool(
        description = "Return the exact paths of the brain file, model cache and project directory."
    )]
    async fn data_path(&self) -> Result<CallToolResult, ErrorData> {
        let info = PathInfo {
            home: self.brain.cfg.home.display().to_string(),
            brain_file: self.brain.cfg.db_path().display().to_string(),
            project: self.brain.cfg.project.clone(),
            model: crate::config::MODEL_ID.to_string(),
        };
        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string_pretty(&info).unwrap_or_default(),
        )]))
    }
}

#[tool_handler]
impl ServerHandler for BrainServer {
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
