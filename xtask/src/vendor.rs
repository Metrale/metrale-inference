// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `cargo xtask vendor-ban` (VBOP): GPU vendor symbols (CUDA, cudarc, cuBLAS, NCCL,
//! NVML, Metal, HIP) appear only in `device` crates, behind a vendor-neutral trait. Every `.rs`
//! file of every other crate except `tool` crates is scanned as code only (comments and literal
//! contents blanked, so prose and log text never count); each identifier naming a vendor counts.
//! Offenders are keyed by path with their count in `xtask/vendor-ban-allow.txt`.
//!
//! Files no workspace member owns are not scanned. That is deliberate for `vendor/cudarc` (the
//! `metrale-cudarc` fork, excluded from the workspace): it IS the vendor binding, so naming the
//! vendor is its job, and depgraph's VBOP rule holds every crate that links it to `device`.

use crate::allow::{Finding, Findings};
use crate::rust_text;

/// 2026-10-17: Lower-cased fragments that name a vendor wherever they sit in an identifier.
const FRAGMENTS: [&str; 5] = ["cuda", "cublas", "nccl", "nvml", "metal"];

/// 2026-10-17: Whether the identifier `ident` names a GPU vendor API.
pub fn is_vendor_ident(ident: &str) -> bool {
    let lower = ident.to_ascii_lowercase();
    if FRAGMENTS.iter().any(|f| lower.contains(f)) {
        return true;
    }
    let upper_after = |prefix: &str| {
        ident
            .strip_prefix(prefix)
            .and_then(|rest| rest.chars().next())
            .is_some_and(|c| c.is_ascii_uppercase())
    };
    upper_after("MTL")
        || upper_after("hip")
        || lower == "hip"
        || lower.starts_with("hip_")
        || ["hipblas", "hiprtc", "hipcc"]
            .iter()
            .any(|p| lower.starts_with(p))
}

/// 2026-10-17: Each vendor identifier in `src` with its 1-based line.
pub fn hits(src: &str) -> Vec<(usize, String)> {
    let code = rust_text::code_only(src);
    let mut out = Vec::new();
    for (index, line) in code.lines().enumerate() {
        let idents = line
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .filter(|w| w.chars().next().is_some_and(|c| !c.is_ascii_digit()));
        for ident in idents {
            if is_vendor_ident(ident) {
                out.push((index.saturating_add(1), ident.to_owned()));
            }
        }
    }
    out
}

/// 2026-10-17: The offending files among `files` (path, source, owning crate, its role name).
pub fn findings<'a>(
    files: impl IntoIterator<Item = (&'a str, &'a str, &'a str, &'a str)>,
) -> Findings {
    let mut out = Findings::new();
    for (path, src, krate, role) in files {
        let found = hits(src);
        let Some((line, ident)) = found.first() else {
            continue;
        };
        let count = u64::try_from(found.len()).unwrap_or(u64::MAX);
        out.insert(
            path.to_owned(),
            Finding {
                count: Some(count),
                why: format!(
                    "{count} GPU vendor identifier(s) in the `{role}` crate `{krate}` (first: line {line} `{ident}`); only `device` crates name a vendor"
                ),
            },
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{hits, is_vendor_ident};

    #[test]
    fn vendor_identifiers_are_named_and_lookalikes_are_not() {
        for ident in [
            "cudarc",
            "CudaSlice",
            "cuBLASLt",
            "ncclComm",
            "nvmlDevice",
            "MetalBackend",
            "metal",
            "MTLDevice",
            "hipMalloc",
            "hip_backend",
            "HIP_PATH",
            "hipblasLt",
            "hip",
        ] {
            assert!(is_vendor_ident(ident), "{ident} should be named");
        }
        for ident in [
            "chip", "hippo", "MTLs", "metadata", "hint", "Mtl", "ship_it",
        ] {
            assert!(!is_vendor_ident(ident), "{ident} should not be named");
        }
    }

    #[test]
    fn only_code_counts_never_comments_or_literal_text() {
        let src = "// cudarc here\nlet d = cudarc::driver::CudaContext::new(0); // cuda\nlet s = \"CUDA error\";\n/* MTLDevice */ let m = MTLDevice;\n";
        assert_eq!(
            hits(src),
            vec![
                (2, "cudarc".to_owned()),
                (2, "CudaContext".to_owned()),
                (4, "MTLDevice".to_owned()),
            ]
        );
    }
}
