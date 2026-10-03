// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The GatedDeltaNet prefill switch table: a switch refuses only at a value that
//! moves the route, a switch outside the asked list never refuses, and the other values pass.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use super::switch_refusal;

fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
    move |v| {
        pairs
            .iter()
            .find(|(k, _)| *k == v)
            .map(|(_, x)| x.to_string())
    }
}

#[test]
fn a_switch_refuses_only_at_the_value_that_moves_the_route() {
    let vars = ["METRALE_FP8_LDMAB", "METRALE_CONV1D_TP"];
    assert!(switch_refusal(&vars, env(&[])).is_none());
    assert!(switch_refusal(&vars, env(&[("METRALE_FP8_LDMAB", "1")])).is_none());
    let why = switch_refusal(&vars, env(&[("METRALE_FP8_LDMAB", "0")])).expect("refused");
    assert!(why.contains("METRALE_FP8_LDMAB=0"), "{why}");
    assert!(switch_refusal(&vars, env(&[("METRALE_CONV1D_TP", "0")])).is_some());
}

#[test]
fn an_any_value_switch_refuses_at_every_value_and_only_when_asked() {
    let fla = ["METRALE_GDN_VTILE", "METRALE_NO_GDN_FWD_O_MMA8"];
    for v in ["0", "1", ""] {
        let set: &'static [(&'static str, &'static str)] = match v {
            "0" => &[("METRALE_GDN_VTILE", "0")],
            "1" => &[("METRALE_GDN_VTILE", "1")],
            _ => &[("METRALE_GDN_VTILE", "")],
        };
        assert!(switch_refusal(&fla, env(set)).is_some(), "VTILE={v}");
    }
    assert!(
        switch_refusal(&["METRALE_CONV1D_TP"], env(&[("METRALE_GDN_VTILE", "1")])).is_none(),
        "a switch the launch does not read never refuses it"
    );
}
