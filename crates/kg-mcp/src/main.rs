// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `metrale-kg`: the kernel knowledge graph's CLI and stdio MCP server
//! (DESIGN-kernel-knowledge-graph.md). `derive|check|status|impact|serve` plus the ledger.py-format
//! views (`tainted`, `deps`), the real-host parity gate (`parity`), `query` and `tools-list`.
//!
//! Owner: metrale-kg-mcp.
//! Invariants:
//! - The only place the knowledge graph's hosts are chosen (FsSources, LedgerPy, SystemClock).
//! - `serve` writes nothing to stdout but the MCP protocol.

mod app;
mod cypher;
mod gates;
mod handlers;
mod mcp;
mod store;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// 2026-10-23: The command line.
#[derive(Parser)]
#[command(
    name = "metrale-kg",
    about = "Kernel knowledge graph: derive, check, query and serve (MCP stdio)"
)]
struct Cli {
    /// The kg-config/v1 file (no default).
    #[arg(long)]
    config: PathBuf,
    /// Use this store directory instead of the config's `store.data_dir`.
    #[arg(long)]
    data_dir: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the kg-bundle/v1 JSONL (or write it to --out).
    Derive {
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Derive twice, compare digests, list fatal findings; --store-roundtrip also rebuilds two
    /// fresh stores and compares their exports.
    Check {
        #[arg(long)]
        store_roundtrip: bool,
    },
    /// kg_status sections as JSON.
    Status {
        #[arg(long)]
        campaign: String,
        #[arg(long = "section")]
        sections: Vec<String>,
        #[arg(long)]
        instance: Option<String>,
    },
    /// kg_impact as JSON.
    Impact {
        id: String,
        #[arg(long)]
        direction: String,
        #[arg(long = "relation")]
        relations: Vec<String>,
        #[arg(long)]
        max_depth: Option<u32>,
    },
    /// `ledger.py tainted`, computed over the bundle.
    Tainted {
        #[arg(long)]
        campaign: String,
    },
    /// `ledger.py deps <id>`, computed over the bundle.
    Deps { id: String },
    /// The P0 parity gate against the campaign's real ledger.py (or --ledger-file, a copy).
    Parity {
        #[arg(long)]
        campaign: String,
        #[arg(long)]
        ledger_file: Option<PathBuf>,
    },
    /// Read-only Cypher.
    Query {
        cypher: String,
        #[arg(long)]
        limit: Option<u32>,
    },
    /// The tools/list body the server sends (for the token gate).
    ToolsList,
    /// Serve the 5 tools over MCP stdio.
    Serve,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            // 2026-10-23: stdout stays JSON on every path (a caller parses it); the human line
            // goes to stderr.
            let code = e
                .downcast_ref::<metrale_error::Error>()
                .map_or("Failed", |k| k.code().name());
            println!(
                "{}",
                serde_json::json!({"error": code, "message": format!("{e:#}")})
            );
            eprintln!("metrale-kg: {code}: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<()> {
    let data = cli.data_dir.as_deref();
    match cli.cmd {
        Cmd::Derive { out } => gates::derive_cmd(&cli.config, out.as_deref()),
        Cmd::Check { store_roundtrip } => gates::check_cmd(&cli.config, store_roundtrip).await,
        Cmd::Parity {
            campaign,
            ledger_file,
        } => gates::parity_cmd(&cli.config, &campaign, ledger_file.as_deref()),
        Cmd::Tainted { campaign } => gates::tainted_cmd(&cli.config, &campaign),
        Cmd::Deps { id } => gates::deps_cmd(&cli.config, &id),
        Cmd::ToolsList => {
            println!("{}", metrale_kg_derive::tools::tools_list_json());
            Ok(())
        }
        Cmd::Serve => serve(&cli.config, data).await,
        Cmd::Status {
            campaign,
            sections,
            instance,
        } => {
            let args = serde_json::json!({"campaign": campaign, "sections": sections, "instance": instance});
            tool(&cli.config, data, "kg_status", args).await
        }
        Cmd::Impact {
            id,
            direction,
            relations,
            max_depth,
        } => {
            let args = serde_json::json!({"id": id, "direction": direction, "relations": relations, "max_depth": max_depth});
            tool(&cli.config, data, "kg_impact", args).await
        }
        Cmd::Query { cypher, limit } => {
            let args = serde_json::json!({"cypher": cypher, "limit": limit});
            tool(&cli.config, data, "kg_query", args).await
        }
    }
}

async fn tool(
    cfg: &std::path::Path,
    data: Option<&std::path::Path>,
    name: &str,
    args: serde_json::Value,
) -> Result<()> {
    let mut app = app::App::open(cfg, data).await?;
    match handlers::call(&mut app, name, &args).await {
        Ok(v) => {
            println!("{}", serde_json::to_string_pretty(&v)?);
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}

async fn serve(cfg: &std::path::Path, data: Option<&std::path::Path>) -> Result<()> {
    use rmcp::ServiceExt;
    let app = app::App::open(cfg, data).await?;
    let server = mcp::KgServer {
        app: std::sync::Arc::new(tokio::sync::Mutex::new(app)),
    };
    let running = server.serve(rmcp::transport::stdio()).await?;
    running.waiting().await?;
    Ok(())
}
