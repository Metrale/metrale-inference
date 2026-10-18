// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The audit of every LM-head call's [`HeadRanks`]: which ranks make it. A
//! vocab-parallel head ends in an all-reduce, so a call made by one rank alone that passes
//! `Every` waits forever for partners (rank 0's prompt-logprob collection did, under the FP8
//! head, and hung `/v1/completions` with `echo` and `logprobs`).
//!
//! The test reads `src/` and lists, per file, the ranks argument of each head call in source
//! order. It fails when a call is added, removed, moved to another file or changes its ranks
//! without this table changing with it, so each new call gets the question asked.
//!
//! Audit (2026-10-10), `Every` where the call ends a forward whose layers ran their
//! collectives on every rank in the same command (a rank skipping it would already hang in
//! the layers):
//! - decode (`decode_a3.rs`, `impl_b1_decode.rs` profiled and draft decode) and batched
//!   decode (`decode_a2.rs`): every rank runs the same decode command;
//! - verify K=2/3/4/γ, batched and fused (`verify_*.rs`): workers run the same verify;
//! - prefill (`prefill_a.rs`, `prefill_c.rs`, `prefill_b/finalize_last.rs`): workers run the
//!   same chunks and `is_last` from the same bounds;
//! - mixed decode + prefill (`decode_b2.rs`): runs only without a communicator
//!   (`decode_b.rs` takes the split path whenever `comm` is set), so it never shards.
//!
//! `Local`:
//! - prompt-logprob collection (`prefill_b/prompt_logprobs.rs`): rank 0 only, no layer runs;
//! - the full-prompt cache hit (`prefill_d.rs`): no layer runs, nothing pairs the ranks.
//!
//! `ranks`: the head entries passing their caller's choice on.
//!
//! Not here: the MTP and DFlash drafters project through their own heads (model-arch), which
//! never call these entries.
//!
//! Owner: model-engine (LM head).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;
use std::path::Path;

/// 2026-10-10: The head entries whose call sites are audited (as `self.` calls, so another
/// type's `lm_head`, such as `WeightQuantPolicy::lm_head`, is not one).
const ENTRIES: [&str; 4] = [
    "self.lm_head(",
    "self.lm_head_batched(",
    "self.lm_head_project_batched(",
    "self.lm_head_fp8_run(",
];

/// 2026-10-10: Per file under `src/`, the ranks argument of each head call in source order.
const AUDIT: &[(&str, &[&str])] = &[
    ("model/impl_a3_lm_head.rs", &["ranks", "ranks", "ranks"]),
    ("model/impl_b1_decode.rs", &["Every", "Every"]),
    ("model/trait_impl/decode_a2.rs", &["Every"]),
    ("model/trait_impl/decode_a3.rs", &["Every"]),
    ("model/trait_impl/decode_b2.rs", &["Every"]),
    ("model/trait_impl/lm_head_batched.rs", &["ranks"]),
    ("model/trait_impl/prefill_a.rs", &["Every"]),
    (
        "model/trait_impl/prefill_b/finalize_last.rs",
        &["Every", "Every"],
    ),
    ("model/trait_impl/prefill_b/prompt_logprobs.rs", &["Local"]),
    ("model/trait_impl/prefill_c.rs", &["Every"]),
    ("model/trait_impl/prefill_d.rs", &["Local"]),
    ("model/trait_impl/verify_a.rs", &["Every"]),
    ("model/trait_impl/verify_b.rs", &["Every"]),
    ("model/trait_impl/verify_c.rs", &["Every"]),
    ("model/trait_impl/verify_c2.rs", &["Every"]),
    ("model/trait_impl/verify_d.rs", &["Every"]),
    ("model/trait_impl/verify_e.rs", &["Every"]),
    ("model/trait_impl/verify_fused.rs", &["Every"]),
];

/// 2026-10-10: The argument list of the call whose `(` is at `open`, up to its matching `)`.
fn call_args(src: &str, open: usize) -> &str {
    let mut depth = 0usize;
    for (i, c) in src[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return &src[open + 1..open + i];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced call at byte {open}");
}

/// 2026-10-10: `Every`, `Local` or `ranks`: the single ranks token among a call's arguments,
/// or the arguments themselves when there is not exactly one.
fn ranks_of(args: &str) -> String {
    let found: Vec<&str> = ["HeadRanks::Every", "HeadRanks::Local"]
        .into_iter()
        .filter(|t| args.contains(t))
        .map(|t| t.trim_start_matches("HeadRanks::"))
        .chain(
            args.split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .filter(|w| *w == "ranks"),
        )
        .collect();
    match found.as_slice() {
        [one] => one.to_string(),
        _ => format!(
            "<{}>",
            args.split_whitespace().collect::<Vec<_>>().join(" ")
        ),
    }
}

/// 2026-10-10: The head calls in one source file, comment lines skipped.
fn head_calls(src: &str) -> Vec<String> {
    let code: String = src
        .lines()
        .map(|l| {
            if l.trim_start().starts_with("//") {
                ""
            } else {
                l
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut calls: Vec<(usize, String)> = Vec::new();
    for e in ENTRIES {
        for (at, _) in code.match_indices(e) {
            calls.push((at, ranks_of(call_args(&code, at + e.len() - 1))));
        }
    }
    calls.sort();
    calls.into_iter().map(|(_, r)| r).collect()
}

fn walk(dir: &Path, root: &Path, out: &mut BTreeMap<String, Vec<String>>) {
    for entry in std::fs::read_dir(dir).expect("read src dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            walk(&path, root, out);
        } else if path.extension().is_some_and(|x| x == "rs")
            && !path.ends_with("model/lm_head_vocab_tp_audit_tests.rs")
        {
            // 2026-10-10: This file's own string fixtures are not calls.
            let src = std::fs::read_to_string(&path).expect("read source");
            let calls = head_calls(&src);
            if !calls.is_empty() {
                let rel = path.strip_prefix(root).expect("under src");
                out.insert(rel.to_string_lossy().replace('\\', "/"), calls);
            }
        }
    }
}

#[test]
fn every_lm_head_call_names_the_ranks_that_make_it() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut found = BTreeMap::new();
    walk(&root, &root, &mut found);
    let want: BTreeMap<String, Vec<String>> = AUDIT
        .iter()
        .map(|(f, r)| (f.to_string(), r.iter().map(|s| s.to_string()).collect()))
        .collect();
    assert_eq!(
        found, want,
        "LM-head call sites changed: decide for each new or moved call whether every rank \
         makes it (HeadRanks::Every) or only this one (HeadRanks::Local), then update AUDIT"
    );
}

#[test]
fn the_scanner_reads_ranks_from_multiline_calls() {
    let src = "self.lm_head_batched(\n    normed,\n    f(a, b),\n    HeadRanks::Local,\n    s,\n)?;\n\
               // self.lm_head(n, HeadRanks::Every, s)\n\
               self.lm_head(n, s)?;\n\
               self.lm_head_fp8_run(h, 1, l, ranks, s)?;";
    assert_eq!(
        head_calls(src),
        vec![
            "Local".to_string(),
            "<n, s>".to_string(),
            "ranks".to_string()
        ]
    );
}
