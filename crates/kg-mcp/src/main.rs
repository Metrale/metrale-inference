// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `metrale-kg`: the metrale pack on Latlas (DESIGN-latlas.md §5.3, K1): latlas-cli's
//! `derive|check|status|impact|query|search|probes|tools-list|serve` over `Metrale` and
//! `MetraleHost`, plus the pack's own `tainted`, `deps`, `parity` and `refresh`.
//!
//! Owner: metrale-kg-mcp.
//! Invariants:
//! - The only place the metrale pack's hosts are chosen (latlas-host-native, MetraleHost).
//! - `serve` writes nothing to stdout but the MCP protocol; every other path keeps stdout one
//!   JSON document.
//! - CLER: a Latlas code leaves this binary as its Metrale `KnowledgeGraph 0x0800` code
//!   ([`metrale_code`]).

mod gates;
mod host;

use anyhow::Result;
use clap::{Parser, Subcommand};
use host::MetraleHost;
use latlas_cli::{CoreCmd, Identity, run_core};
use metrale_latlas_pack::Metrale;
use std::path::PathBuf;

/// 2026-10-23: The command line.
#[derive(Parser)]
#[command(
    name = "metrale-kg",
    about = "Kernel knowledge graph (the metrale pack on Latlas): derive, check, query and serve (MCP stdio)"
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
    #[command(flatten)]
    Core(CoreCmd),
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
    /// The integrity score vector and ratchet (latlas-check): the head is --config, the base (when
    /// given) a second config naming the base's sources; prints the report JSON, exits 3 on a
    /// regression and 4 on a fatal finding.
    Score {
        #[arg(long)]
        check: PathBuf,
        #[arg(long)]
        base_config: Option<PathBuf>,
        /// Also write the PR comment (markdown) here.
        #[arg(long)]
        comment: Option<PathBuf>,
    },
    /// Move the read-only LKB checkout to `[sources.lkb]` remote/branch (fetch + detached
    /// checkout); the only command that changes a source. Prints the old and new commit.
    Refresh,
}

/// 2026-10-23: A Latlas code as the Metrale code it was cut from (`0x01nn` ↔ `0x08nn`); a
/// general code keeps its name.
fn metrale_code(c: latlas_error::ErrorCode) -> &'static str {
    let n = c.as_u16();
    if (0x0100..=0x01FF).contains(&n)
        && let Some(m) = metrale_error::ErrorCode::from_u16(0x0800 | (n & 0x00FF))
    {
        return m.name();
    }
    c.name()
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) if e.downcast_ref::<latlas_cli::score::CheckExit>().is_some() => {
            eprintln!("metrale-kg: {e}");
            let n = e.downcast_ref::<latlas_cli::score::CheckExit>().map_or(1, |x| x.0);
            std::process::ExitCode::from(n)
        }
        Err(e) if e.downcast_ref::<latlas_cli::gates::GateFailed>().is_some() => {
            eprintln!("metrale-kg: {e}");
            std::process::ExitCode::FAILURE
        }
        Err(e) => {
            // 2026-10-23: stdout stays JSON on every path (a caller parses it); the human line
            // goes to stderr.
            let code = e
                .downcast_ref::<latlas_error::Error>()
                .map_or("Failed", |k| metrale_code(k.code()));
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
        Cmd::Core(c) => {
            let id = Identity {
                name: "metrale-kg",
                version: env!("CARGO_PKG_VERSION"),
            };
            run_core::<Metrale, MetraleHost>(&cli.config, data, c, id).await
        }
        Cmd::Parity {
            campaign,
            ledger_file,
        } => gates::parity_cmd(&cli.config, &campaign, ledger_file.as_deref()),
        Cmd::Tainted { campaign } => gates::tainted_cmd(&cli.config, &campaign),
        Cmd::Deps { id } => gates::deps_cmd(&cli.config, &id),
        Cmd::Score {
            check,
            base_config,
            comment,
        } => latlas_cli::score::score_cmd::<Metrale, MetraleHost>(
            &cli.config,
            base_config.as_deref(),
            &check,
            comment.as_deref(),
        ),
        Cmd::Refresh => refresh(&cli.config),
    }
}

fn refresh(cfg_path: &std::path::Path) -> Result<()> {
    let cfg = latlas_host_native::fs::read_config::<Metrale>(cfg_path)?;
    let t = cfg.sources.lkb.as_ref().ok_or_else(|| {
        latlas_error::error!(
            latlas_error::ErrorCode::LatlasConfig,
            "sources.lkb",
            "is not configured: nothing to refresh"
        )
    })?;
    let lkb = metrale_latlas_pack::lkb_derive::LkbSources::parse(t)?;
    let (old, new) =
        metrale_io_host_kg::lkb::refresh(std::path::Path::new(&cfg.sources.root), &lkb)?;
    let track = format!("{}/{}", lkb.remote, lkb.branch);
    println!(
        "{}",
        serde_json::json!({"repo": lkb.repo, "track": track, "from": old, "to": new, "moved": old != new})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::metrale_code;
    use latlas_error::ErrorCode;

    #[test]
    fn a_latlas_code_leaves_as_the_metrale_code_it_was_cut_from() {
        for c in ErrorCode::ALL {
            let n = c.as_u16();
            if (0x0100..=0x01FF).contains(&n) {
                let want = c.name().replacen("Latlas", "Kg", 1);
                assert_eq!(metrale_code(*c), want, "{c:?}");
            }
        }
        assert_eq!(metrale_code(ErrorCode::Io), "Io");
    }

    #[test]
    fn every_schema_parses_as_an_rmcp_tool() {
        let names: Vec<String> = latlas_cli::mcp::tool_list::<metrale_latlas_pack::Metrale>()
            .iter()
            .map(|t| t.name.to_string())
            .collect();
        assert_eq!(
            names,
            [
                "la_query",
                "la_search",
                "la_impact",
                "la_record",
                "la_status"
            ]
        );
    }
}
