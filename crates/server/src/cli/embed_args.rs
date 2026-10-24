// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: `met embed`: an alias of `met serve` for one embedding client on stdin/stdout.
//! It boots the serve the blueprint pins, exactly as `met serve` would, and swaps only the
//! transport: length-prefixed frames on stdio instead of a TCP listener, answered by the same
//! router and `/v1/embeddings` handler (`main_modules::embed_stdio`).
//!
//! Owner: server CLI.
//! Invariants:
//! - Every argument is required (PCND); none has a default.
//! - The blueprint is read and hashed once, here (ROTI); the hash is what every request must
//!   name and what every response echoes.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, bail};

/// `met embed`: boot the serve a blueprint pins and answer embedding requests on stdio.
#[derive(clap::Args, Debug, Clone)]
pub struct EmbedArgs {
    /// The pinned serve: a `runtime: metrale` recipe file (`<family>/<stem>.yaml`). Its
    /// SHA-256 is echoed in every response as `blueprint_sha256`. A signed `.kbp` is refused
    /// until serve boots from one.
    #[arg(long)]
    pub blueprint: PathBuf,
    /// The `input_type` every request to this process carries (`query` or `passage` for
    /// nemotron-embed); a request naming another is refused.
    #[arg(long)]
    pub input_type: String,
    /// Exit after this many milliseconds with no request; 0 exits after the first response.
    /// End of input always exits.
    #[arg(long)]
    pub idle_exit_ms: u64,
}

/// 2026-10-24: What the stdio transport checks and echoes, resolved once from [`EmbedArgs`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedStdio {
    pub blueprint_sha256: String,
    pub input_type: String,
    /// 2026-10-24: `None`: exit after the first response (`--idle-exit-ms 0`).
    pub idle_exit: Option<Duration>,
}

impl EmbedStdio {
    /// 2026-10-24: The transport settings for `args` and the blueprint's hash.
    pub fn new(args: &EmbedArgs, blueprint_sha256: String) -> Self {
        Self {
            blueprint_sha256,
            input_type: args.input_type.clone(),
            idle_exit: (args.idle_exit_ms > 0).then(|| Duration::from_millis(args.idle_exit_ms)),
        }
    }
}

/// 2026-10-24: The serve the blueprint pins (its recipe's argv, without the dashboard) and the
/// transport settings.
pub fn into_serve(args: &EmbedArgs) -> Result<(super::ServeArgs, EmbedStdio)> {
    if args.blueprint.extension().is_some_and(|e| e == "kbp") {
        bail!(
            "{}: serve does not boot from a signed blueprint yet; pass the recipe it was built from",
            args.blueprint.display()
        );
    }
    let recipe = super::blueprint_build::recipe_at(&args.blueprint)?;
    let sha = super::blueprint_build::sha_file(&args.blueprint)?.hex();
    let overrides = BTreeMap::from([("no_tui".to_string(), "true".to_string())]);
    let serve = recipe.serve_args(&overrides)?;
    Ok((serve, EmbedStdio::new(args, sha)))
}

/// 2026-10-24: `met embed` as the serve it aliases: the command becomes that `serve`, and the
/// transport settings and the claimed stdout come beside it. Any other command is unchanged.
pub fn rewrite(cli: super::Cli) -> Result<(super::Cli, Option<(EmbedStdio, std::fs::File)>)> {
    let args = match cli.command {
        super::Command::Embed(args) => args,
        command => return Ok((super::Cli { command }, None)),
    };
    let out = crate::main_modules::embed_stdio::claim_stdout()?;
    let (serve, stdio) = into_serve(&args)?;
    Ok((
        super::Cli {
            command: super::Command::Serve(serve),
        },
        Some((stdio, out)),
    ))
}

#[cfg(test)]
#[path = "embed_args_tests.rs"]
mod tests;
