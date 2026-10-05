// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: I/O adapter for the vLLM-image-pinned check: `docker inspect` the local
//! image, by shelling out, exactly once.
//!
//! Owner: server CLI (`met bench preflight`).
//! Invariants:
//! - No decision lives here: `core::evaluate` decides what a pinned reference looks like
//!   and whether the digests match. This module only answers "what RepoDigests does the
//!   local image carry, or why could it not say".

use std::process::Command;

/// 2026-10-05: The local image's `RepoDigests`, via `docker inspect --format
/// '{{json .RepoDigests}}' <image_ref>`. `Err` when docker is not available, the image is
/// not present locally, or the output does not parse — any of which means "cannot verify",
/// which `core::evaluate` folds into FAIL, never into a silent pass.
pub fn repo_digests(image_ref: &str) -> Result<Vec<String>, String> {
    let out = Command::new("docker")
        .args(["inspect", "--format", "{{json .RepoDigests}}", image_ref])
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("failed to run docker: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "docker inspect {image_ref:?} failed: {}",
            stderr.trim()
        ));
    }
    parse_repo_digests(&String::from_utf8_lossy(&out.stdout))
}

/// 2026-10-05: Parsing split out of [`repo_digests`] so it is tested without a docker
/// daemon: `docker inspect`'s stdout, verbatim, in -> the digest list or an error that
/// names what was wrong with it.
fn parse_repo_digests(stdout: &str) -> Result<Vec<String>, String> {
    serde_json::from_str::<Vec<String>>(stdout.trim())
        .map_err(|e| format!("could not parse `docker inspect` output {stdout:?}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-05: Path A — a well-formed `docker inspect` RepoDigests array parses.
    #[test]
    fn a_well_formed_digest_array_parses() {
        let out = parse_repo_digests(
            "[\"vllm/vllm-openai@sha256:deadbeef00000000000000000000000000000000000000000000000000000000\"]\n",
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].starts_with("vllm/vllm-openai@sha256:"));
    }

    /// 2026-10-05: Path B — an image with no digests at all (built locally, never pushed)
    /// parses to an empty list, not an error: `core::evaluate` then fails it for the right
    /// reason ("RepoDigests do not contain the pinned digest"), not a parse error.
    #[test]
    fn an_empty_digest_array_parses_to_empty_not_an_error() {
        assert_eq!(parse_repo_digests("[]\n").unwrap(), Vec::<String>::new());
    }

    /// 2026-10-05: Path C — malformed JSON (a corrupt or unexpected output shape, e.g.
    /// docker printing `<no value>` for a missing field) is reported as an error, not
    /// silently treated as "no digests" — which would make `core::evaluate` fail it for
    /// the wrong reason ("not pinned" instead of "could not inspect").
    #[test]
    fn malformed_inspect_output_is_an_error_not_an_empty_list() {
        let err = parse_repo_digests("<no value>").unwrap_err();
        assert!(err.contains("could not parse"), "{err}");
    }
}
