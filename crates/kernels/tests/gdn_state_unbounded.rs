// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: No gated delta rule (GDN) recurrence source, in any hardware tree or model copy,
//! and no per-chunk SSM state pass rescales the recurrent state by its norm. The reference recurrence bounds nothing; a kernel
//! that clamps the state's Frobenius norm gives a sequence whose norm passes the bound different
//! bits from a kernel that does not, so the batched and per-sequence decodes stop agreeing.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! A plain-text scan with comments stripped. It flags the three forms every clamp in the tree
//! took before its removal (`MAX_NORM` bounds, `rsqrt` of a squared norm without an epsilon, and
//! an in-place `*=` on a state array), and cannot prove a kernel's arithmetic. The byte-for-byte
//! agreement of the strided and per-sequence decodes past norm 1000 is checked on a GPU by
//! `examples/gdn_decode_strided_microtest` (model-arch).

use std::path::{Path, PathBuf};

/// 2026-10-10: File-name fragments of the sources that write a GDN state: every
/// `gated_delta_rule*` and `gdn_*` source, its shadows and its Metal port, and `ssm_state_norm`
/// (the pass the engine runs over every SSM state after each prefill chunk).
const RECURRENCE_FILES: &[&str] = &["gated_delta", "gdn", "ssm_state_norm"];

/// 2026-10-10: Extensions of kernel sources across the hardware trees.
const SOURCE_EXT: &[&str] = &["cu", "cuh", "metal", "hip", "h"];

/// 2026-10-10: Names the kernels give a head's recurrent state.
const STATE_ARRAYS: &[&str] = &["H", "H_reg", "H_smem", "H_global", "hreg", "smem_h"];

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .join("kernels")
}

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let p = e.expect("directory entry").path();
        if p.is_dir() {
            sources(&p, out);
            continue;
        }
        let name = p.file_name().unwrap_or_default().to_string_lossy();
        let ext = p.extension().unwrap_or_default().to_string_lossy();
        if SOURCE_EXT.contains(&ext.as_ref()) && RECURRENCE_FILES.iter().any(|f| name.contains(f)) {
            out.push(p);
        }
    }
}

/// 2026-10-10: `text` without `//` line comments and `/* */` block comments (string literals
/// are not special-cased: kernel sources carry none that hold comment markers).
fn strip_comments(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_block = false;
    for line in text.lines() {
        let mut kept = String::new();
        let mut rest = line;
        loop {
            if in_block {
                match rest.find("*/") {
                    Some(i) => {
                        rest = &rest[i + 2..];
                        in_block = false;
                    }
                    None => break,
                }
            } else {
                let lc = rest.find("//");
                let bc = rest.find("/*");
                match (lc, bc) {
                    (Some(l), Some(b)) if b < l => {
                        kept.push_str(&rest[..b]);
                        rest = &rest[b + 2..];
                        in_block = true;
                    }
                    (Some(l), _) => {
                        kept.push_str(&rest[..l]);
                        break;
                    }
                    (None, Some(b)) => {
                        kept.push_str(&rest[..b]);
                        rest = &rest[b + 2..];
                        in_block = true;
                    }
                    (None, None) => {
                        kept.push_str(rest);
                        break;
                    }
                }
            }
        }
        out.push(kept);
    }
    out
}

/// 2026-10-10: The array name an assignment's left side indexes (`H` in `H[j * v_dim] *=`).
fn indexed_name(lhs: &str) -> Option<&str> {
    let lhs = lhs.trim_end();
    if !lhs.ends_with(']') {
        return None;
    }
    let open = lhs.find('[')?;
    let head = lhs[..open].trim_end();
    let start = head
        .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .map_or(0, |i| i + 1);
    Some(&head[start..])
}

/// 2026-10-10: The argument text of each `rsqrt`/`rsqrtf` call on `line`.
fn rsqrt_args(line: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(i) = line[at..].find("rsqrt") {
        let call = at + i;
        let Some(open) = line[call..].find('(').map(|o| call + o) else {
            break;
        };
        let mut depth = 0usize;
        let mut end = line.len();
        for (j, c) in line[open..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + j;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push(&line[open + 1..end]);
        at = open + 1;
    }
    out
}

/// 2026-10-10: 1-based lines of `text` that bound or rescale a state by its norm.
fn rescale_sites(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, line) in strip_comments(text).iter().enumerate() {
        let bound = line.contains("MAX_NORM");
        let norm_scale = rsqrt_args(line)
            .iter()
            .any(|a| a.contains("norm") && !a.contains("eps"));
        let in_place = line
            .find("*=")
            .and_then(|p| indexed_name(&line[..p]))
            .is_some_and(|n| STATE_ARRAYS.contains(&n));
        if bound || norm_scale || in_place {
            out.push(i + 1);
        }
    }
    out
}

/// 2026-10-10: Negative control: the per-head clamp as the 27B strided decode compiled it.
#[test]
fn the_scanner_flags_the_removed_decode_clamp() {
    let src = "\
    float head_norm_sq = norm_sums[0];
    if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
        float scale = SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq);
        for (unsigned int j = 0; j < k_dim; j++) {
            H[j * v_dim + tid] *= scale;
        }
    }
";
    assert_eq!(rescale_sites(src), vec![2, 3, 5]);
}

/// 2026-10-10: Negative control: the register form (exact carry) under a bound of another name.
/// The FP16 twins' store-side rescale (line 4) is not matched itself; its scale's `rsqrt` of the
/// squared norm is, as on line 2.
#[test]
fn the_scanner_flags_the_register_rescale_and_its_scale() {
    let src = "\
    if (ns[0] > LIMIT * LIMIT) {
        const float scale = LIMIT * rsqrt(head_norm_sq);
        for (int j = 0; j < CARRY_KD; j++) H_reg[j] *= scale;
        H[j * v_dim + tid] = gdn_f16_store(__half2float(H[j * v_dim + tid]) * scale);
";
    assert_eq!(rescale_sites(src), vec![2, 3]);
}

/// 2026-10-10: Positive control: the update, the decay clamp, the q/k L2 norm, the gated RMS
/// norm and prose about the removed clamp are not flagged.
#[test]
fn the_scanner_passes_the_recurrence_and_its_norms() {
    let src = "\
    // 2026-10-10: No kernel bounds the state norm (no MAX_NORM, no H[j] *= scale).
    const float g = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    h0 = g * h0 + smem_k[j] * v_new_i;
    H[(j + 0) * v_dim + tid] = h0;
    float inv = rsqrtf(total + l2_eps);
    const float rms = rsqrtf(rms_sums[0] / (float)v_dim + eps);
    float inv_norm = rsqrtf(q_norm_sq + eps);
    acc *= inv_sqrt_d; /* H[j] *= scale; */
";
    assert!(rescale_sites(src).is_empty(), "{:?}", rescale_sites(src));
}

/// 2026-10-10: Every recurrence source under kernels/ leaves the state's norm unbounded.
#[test]
fn no_recurrence_source_rescales_the_state() {
    let mut files = Vec::new();
    sources(&kernels_root(), &mut files);
    files.sort();
    assert!(
        files.len() >= 20,
        "only {} recurrence sources found; the file filter is not seeing the tree",
        files.len()
    );
    let mut faults = Vec::new();
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        for line in rescale_sites(&text) {
            faults.push(format!("{}:{line}", f.display()));
        }
    }
    assert!(
        faults.is_empty(),
        "a recurrence source rescales the state by its norm:\n{}",
        faults.join("\n")
    );
}
