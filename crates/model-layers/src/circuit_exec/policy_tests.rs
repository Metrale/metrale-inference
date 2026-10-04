// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `ffn_act_fixed`, the policy setting of the dense FFN's fixed-activation path: the
//! two ladders the rules plan, and every other one refused.
//!
//! Owner: model-layers circuit executor tests.
//! Invariants: none beyond the types.

use metrale_config::{ActivationQuantization, ProjFamily};

use super::{PREFILL_ENV_SWITCHES, ffn_act_fixed, prefill_switch_refusals};

fn of(flag: &str) -> anyhow::Result<&'static str> {
    let a = ActivationQuantization::parse(flag).expect("a valid flag");
    ffn_act_fixed(a.ladder(ProjFamily::Ffn))
}

// 2026-10-03: Mutation: reading `off` for any ladder with an adaptive rung plans today's arms
// for a serve that runs `forward_fixed` at its fixed rows; reading `declared` for a uniform
// fixed format other than declared plans the declared kernels for another format.
#[test]
fn only_an_all_adaptive_or_all_declared_ffn_ladder_has_rules() {
    assert_eq!(of("adaptive").unwrap(), "off");
    assert_eq!(of("declared").unwrap(), "declared");
    // 2026-10-03: Another family's fixed format leaves the ffn ladder adaptive.
    assert_eq!(of("adaptive,gdn:declared").unwrap(), "off");
    assert_eq!(of("adaptive,ffn:declared").unwrap(), "declared");
    for refused in [
        "nvfp4",
        "fp8",
        "bf16",
        "ffn:1-4=adaptive;5-=declared",
        "1=declared;2-=nvfp4",
    ] {
        let flag = if refused.contains(':') {
            format!("adaptive,{refused}")
        } else {
            refused.to_string()
        };
        let e = of(&flag).expect_err(&flag);
        assert!(
            format!("{e:#}").contains("adaptive or declared"),
            "{flag}: {e:#}"
        );
    }
}

fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
    move |v| {
        pairs
            .iter()
            .find(|(k, _)| *k == v)
            .map(|(_, x)| x.to_string())
    }
}

// 2026-10-04 (moved from the GDN prefill emitter's table): a switch refuses only at a value that
// moves the route, an any-value switch at every value, and an unset or other-valued one never.
// Mutation: refusing at the default value refuses every serve; accepting the moving value runs a
// kernel the plan does not name.
#[test]
fn a_prefill_switch_refuses_only_at_the_value_that_moves_the_route() {
    assert!(prefill_switch_refusals(env(&[])).is_empty());
    assert!(prefill_switch_refusals(env(&[("METRALE_FP8_LDMAB", "1")])).is_empty());
    let why = prefill_switch_refusals(env(&[("METRALE_FP8_LDMAB", "0")]));
    assert_eq!(why.len(), 1);
    assert!(why[0].contains("METRALE_FP8_LDMAB=0"), "{why:?}");
    for v in ["0", "1", ""] {
        let set: &'static [(&'static str, &'static str)] = match v {
            "0" => &[("METRALE_GDN_VTILE", "0")],
            "1" => &[("METRALE_GDN_VTILE", "1")],
            _ => &[("METRALE_GDN_VTILE", "")],
        };
        assert_eq!(prefill_switch_refusals(env(set)).len(), 1, "VTILE={v}");
    }
    assert!(prefill_switch_refusals(env(&[("METRALE_FUSED_KV", "0")])).is_empty());
    let both = prefill_switch_refusals(env(&[
        ("METRALE_CONV1D_TP", "0"),
        ("METRALE_NO_ATTN_FA128", "1"),
    ]));
    assert_eq!(both.len(), 2, "every refusal is reported: {both:?}");
}

// 2026-10-04: Mutation: a duplicated or unattributed row hides which reader a refusal is about.
#[test]
fn every_prefill_switch_is_listed_once_with_its_reader() {
    let mut names: Vec<&str> = PREFILL_ENV_SWITCHES.iter().map(|r| r.0).collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), PREFILL_ENV_SWITCHES.len());
    for (var, _, what, reader) in PREFILL_ENV_SWITCHES {
        assert!(var.starts_with("METRALE_") && !what.is_empty() && reader.ends_with(".rs"));
    }
}
