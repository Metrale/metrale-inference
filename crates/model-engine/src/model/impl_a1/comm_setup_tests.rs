// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The rank-agreed DFlash context window against a frozen copy of the environment
//! reader it replaced, through `MapEnv`-resolved settings (never the process environment).
//!
//! Owner: model-engine.
//! Invariants: every probe reads as the old reader did.

use metrale_model_layers::lever_equivalence::{Old, check, settings_from};

use super::dflash_ctx_window;

#[test]
fn dflash_ctx_window_matches_the_old_reader() {
    // 2026-10-17: Frozen `env::var(..).ok().and_then(parse::<usize>).unwrap_or(4096)`.
    let old = |v: Option<&str>| v.and_then(|s| s.parse::<usize>().ok()).unwrap_or(4096);
    let changes = check(
        "METRALE_DFLASH_CTX_WINDOW",
        Old::Verbatim,
        old,
        dflash_ctx_window,
    );
    assert!(changes.is_empty(), "{changes:?}");
    for v in ["", " 8", "8192", "-1", "unset"] {
        let s = settings_from(&[("METRALE_DFLASH_CTX_WINDOW", v)]).unwrap();
        assert_eq!(dflash_ctx_window(&s), old(Some(v)), "{v:?}");
    }
}
