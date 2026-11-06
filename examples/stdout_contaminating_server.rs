//! Verification-only MCP server that deliberately breaks stdout purity.

use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig};
use rmcp::{ErrorData, ServerHandler, ServiceExt, tool, tool_handler, tool_router};

#[derive(Clone)]
struct ContaminatingServer {
    mode: String,
}

#[tool_router]
impl ContaminatingServer {
    #[tool(description = "Verification-only probe.")]
    async fn probe(&self) -> Result<CallToolResult, ErrorData> {
        match self.mode.as_str() {
            "successful-call" => {
                println!("stdout-contamination-successful-call");
                Ok(CallToolResult::success(vec![ContentBlock::text("ok")]))
            }
            "handled-error" => {
                println!("stdout-contamination-handled-error");
                Ok(CallToolResult::error(vec![ContentBlock::text("handled")]))
            }
            _ => Ok(CallToolResult::success(vec![ContentBlock::text("ok")])),
        }
    }
}

#[tool_handler]
impl ServerHandler for ContaminatingServer {
    fn get_info(&self) -> ServerConfig {
        let mut info = ServerConfig::new(ServerCapabilities::builder().enable_tools().build());
        info.server_info = Implementation::new("stdout-contaminating-server", "0.0.0");
        info
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_default();
    if mode == "startup" {
        println!("stdout-contamination-startup");
    }
    let service = ContaminatingServer { mode }
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await?;
    service.waiting().await?;
    Ok(())
}
