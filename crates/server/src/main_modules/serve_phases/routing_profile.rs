// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `met serve --record-routing <path>`: arm the MoE expert-load recorder
//! (`metrale_model_layers::layers::moe::routing_record`) and rewrite the profile file every two
//! seconds, in the format `metrale_ml_utils::RoutingProfile` reads.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - The file is replaced atomically (write a sibling, then rename), so a reader never sees a
//!   partial profile.
//! - A write failure is logged and retried on the next tick; it never stops the serve.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use metrale_model_layers::layers::moe::routing_record;

/// 2026-10-03: Seconds between rewrites.
const PERIOD_SECS: u64 = 2;

/// 2026-10-03: The profile JSON for `rows` (one per MoE layer, first-touch order).
pub(crate) fn profile_json(
    source: &str,
    experts: usize,
    top_k: usize,
    rows: &[Vec<u64>],
) -> String {
    serde_json::json!({
        "schema": 1,
        "source": source,
        "experts": experts,
        "top_k": top_k,
        "prompt_set": "prefill rows served while recording",
        "layers": rows,
    })
    .to_string()
}

fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

/// 2026-10-03: Arm the recorder and start the writer when `path` is given.
pub(crate) fn start(
    path: Option<&Path>,
    source: &str,
    config: &metrale_config::ModelConfig,
) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    if config.num_experts == 0 {
        bail!(
            "--record-routing needs a MoE model; {} has no routed experts",
            config.model_type
        );
    }
    routing_record::arm();
    let (path, source) = (path.to_path_buf(), source.to_string());
    let (experts, top_k) = (config.num_experts, config.num_experts_per_tok);
    tracing::warn!(
        "--record-routing: expert loads are recorded to {} (one device sync per MoE layer in \
         prefill; not a measurement run)",
        path.display()
    );
    std::thread::Builder::new()
        .name("routing-profile".into())
        .spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(PERIOD_SECS));
                let Some(rows) = routing_record::snapshot() else {
                    continue;
                };
                if rows.is_empty() {
                    continue;
                }
                if let Err(e) = write_atomic(&path, &profile_json(&source, experts, top_k, &rows)) {
                    tracing::warn!("--record-routing: writing {}: {e}", path.display());
                }
            }
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-03: What the recorder writes is a profile the mock planner reads back.
    #[test]
    fn the_written_profile_parses_as_a_routing_profile() {
        let text = profile_json("org/model", 4, 2, &[vec![1, 2, 3, 4], vec![4, 3, 2, 1]]);
        let p = metrale_ml_utils::RoutingProfile::parse(&text).unwrap();
        assert_eq!((p.experts, p.top_k, p.layers.len()), (4, 2, 2));
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("p.json");
        write_atomic(&f, &text).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), text);
    }
}
