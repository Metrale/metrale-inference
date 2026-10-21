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

/// 2026-10-21: The control channel's lever and port join the rank agreement as set, so a rank
/// with the lever or the port alone differing fails every rank's start; an unset port is a value
/// no set port takes.
#[test]
fn the_ctrl_comm_lever_and_port_are_rank_agreed_as_set() {
    let agree = |env: &[(&str, &str)]| super::ctrl_comm_agreement(&settings_from(env).unwrap());
    let off = agree(&[]);
    assert_eq!(off[0], ("METRALE_GLM_EP_CTRL_COMM", 0));
    assert_eq!(off[1], ("METRALE_GLM_EP_CTRL_PORT", u64::MAX));
    let on = agree(&[
        ("METRALE_GLM_EP_CTRL_COMM", "1"),
        ("METRALE_GLM_EP_CTRL_PORT", "29611"),
    ]);
    assert_eq!(on[0].1, 1);
    assert_eq!(on[1].1, 29611);
    let other_port = agree(&[
        ("METRALE_GLM_EP_CTRL_COMM", "1"),
        ("METRALE_GLM_EP_CTRL_PORT", "29612"),
    ]);
    assert_ne!(on[1], other_port[1], "a port skew is a disagreement");
    assert_ne!(on[0], off[0], "a lever skew is a disagreement");
}

/// 2026-10-21: The A2 levers are rank-agreed: a rank with either one alone differing differs in
/// the compared table, and the early launch is compared as resolved (beside the NCCL host hoist,
/// which refuses it, `1` agrees with off).
#[test]
fn the_a2_levers_are_rank_agreed() {
    let table =
        |env: &[(&str, &str)]| super::rank_lever_table(&settings_from(env).unwrap()).unwrap();
    let off = table(&[]);
    for env in [
        "METRALE_GLM_GRAPH_EARLY_LAUNCH",
        "METRALE_GLM_MTP_ROW_NOSYNC",
    ] {
        let on = table(&[(env, "1")]);
        assert_ne!(on, off, "{env} skew is not compared");
        let diff: Vec<_> = on.iter().zip(&off).filter(|(a, b)| a != b).collect();
        assert_eq!(diff.len(), 1, "{env}: {diff:?}");
        assert_eq!(diff[0].0.0, env);
    }
    let hoisted = table(&[
        ("METRALE_GLM_GRAPH_EARLY_LAUNCH", "1"),
        ("METRALE_NCCL_HOST_HOIST", "1"),
    ]);
    assert_eq!(hoisted, off, "a refused early launch is compared as off");
}
