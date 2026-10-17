// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The serve's lever settings, resolved once per load: a command-line flag that
//! names a lever, then the environment, then the lever's declared default.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - A load reads its levers only from the value [`resolve`] returns.
//! - A malformed value stops the load before any model work.
//! - A recipe reaches a serve as argv (its `defaults:` expand into flags), so recipe values
//!   arrive through the CLI layer and the recipe layer is empty.

use std::sync::Arc;

use anyhow::Result;
use metrale_settings::sources::{CliLayer, RecipeLayer};
use metrale_settings::{Env, InferenceServerSettings};

use crate::cli;
use crate::cli::flag_values::Tristate;

/// 2026-10-17: Resolve every Runtime lever for one load of `args`.
///
/// # Errors
/// A flag whose value the lever grammar rejects, or a `METRALE_*` variable that does not parse
/// as its lever's type.
pub(crate) fn resolve(
    env: &impl Env,
    args: &cli::ServeArgs,
) -> Result<Arc<InferenceServerSettings>> {
    let cli = cli_layer(args)?;
    let settings = metrale_settings::resolve(env, &cli, &RecipeLayer::empty())
        .map_err(|e| anyhow::anyhow!("lever settings: {e}"))?;
    Ok(Arc::new(settings))
}

/// 2026-10-17: The flags that set a lever, keyed by the lever's variable. An absent flag (or
/// `auto`, or a `0` that means "unset") adds nothing, so the variable stays reachable.
fn cli_layer(args: &cli::ServeArgs) -> Result<CliLayer> {
    let mut pairs: Vec<(&str, String)> = Vec::new();
    if let Some(force) = args.mtp_gate_force() {
        pairs.push(("METRALE_MTP_GATE_FORCE", switch(force)));
    }
    if let Some(ratio) = args.mtp_shape.mtp_dcut_ratio {
        // 2026-10-17: Widened first: the f64 text parses back to the exact f32.
        pairs.push(("METRALE_MTP_DCUT_RATIO", f64::from(ratio).to_string()));
    }
    let watchdog: Tristate = args
        .content_loop_watchdog
        .parse()
        .map_err(|e: String| anyhow::anyhow!("--content-loop-watchdog: {e}"))?;
    if let Some(on) = watchdog.pinned() {
        pairs.push(("METRALE_CONTENT_LOOP_WATCHDOG", switch(on)));
    }
    if let Some(ladder) = args.mtp_shape.mtp_k_ladder.as_deref() {
        pairs.push(("METRALE_MTP_K_LADDER", ladder.to_string()));
    }
    if let Some(n) = args.content_loop_min_repeats {
        pairs.push(("METRALE_CONTENT_LOOP_MIN_REPEATS", n.to_string()));
    }
    if let Some(n) = args.max_inter_tool_prose {
        pairs.push(("METRALE_MAX_INTER_TOOL_PROSE", n.to_string()));
    }
    // 2026-10-17: Presence-style flags: given, they pin the variable's on (`--hermetic`) or
    // off (`--no-ssm-tail-midchunk`) spelling; absent, the variable stays reachable.
    if args.hermetic {
        pairs.push(("METRALE_HERMETIC", String::from("1")));
    }
    if args.no_ssm_tail_midchunk {
        pairs.push(("METRALE_SSM_TAIL_MIDCHUNK", String::from("0")));
    }
    if args.vision_max_pixels > 0 {
        pairs.push((
            "METRALE_VISION_MAX_PIXELS",
            args.vision_max_pixels.to_string(),
        ));
    }
    CliLayer::from_pairs(pairs).map_err(|e| anyhow::anyhow!("lever flags: {e}"))
}

fn switch(on: bool) -> String {
    String::from(if on { "on" } else { "off" })
}

#[cfg(test)]
#[path = "serve_settings_tests.rs"]
mod tests;
