// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The stdio MCP server (rmcp 0.16): `tools/list` serves
//! `metrale_kg_derive::tools::tools()` verbatim (the schemas the token gate measured) and
//! `tools/call` dispatches to [`crate::handlers::call`]. A tool error is a tool result with
//! `isError` and the registry code, so the client sees the code, not a transport failure.
//!
//! Owner: metrale-kg-mcp.
//! Invariants: no tool is served that `tools()` does not list.

use crate::app::App;
use crate::handlers;
use metrale_kg_derive::tools::{INSTRUCTIONS, tools};
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, Implementation, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

/// 2026-10-23: The server handler around the shared app.
#[derive(Clone)]
pub struct KgServer {
    pub app: Arc<Mutex<App>>,
}

/// 2026-10-23: The rmcp `Tool` list from the one schema definition.
pub fn tool_list() -> Vec<Tool> {
    tools()
        .into_iter()
        .filter_map(|t| serde_json::from_value::<Tool>(t).ok())
        .collect()
}

impl ServerHandler for KgServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "metrale-kg".into(),
                title: None,
                version: env!("CARGO_PKG_VERSION").into(),
                description: None,
                icons: None,
                website_url: None,
            },
            instructions: Some(INSTRUCTIONS.into()),
            ..ServerInfo::default()
        }
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(tool_list()))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let args = Value::Object(request.arguments.unwrap_or_default());
        let mut app = self.app.lock().await;
        Ok(match handlers::call(&mut app, &request.name, &args).await {
            Ok(v) => CallToolResult::success(vec![Content::text(v.to_string())]),
            Err(e) => {
                let body = serde_json::json!({"error": e.code().name(), "message": e.to_string()});
                CallToolResult::error(vec![Content::text(body.to_string())])
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_schema_parses_as_an_rmcp_tool() {
        let names: Vec<String> = tool_list().iter().map(|t| t.name.to_string()).collect();
        assert_eq!(
            names,
            [
                "kg_query",
                "kg_search",
                "kg_impact",
                "kg_record",
                "kg_status"
            ]
        );
    }
}
