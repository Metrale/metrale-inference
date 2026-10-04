// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `ffn_act_fixed`, the policy setting of the dense FFN's fixed-activation path: the
//! two ladders the rules plan, and every other one refused.
//!
//! Owner: model-layers circuit executor tests.
//! Invariants: none beyond the types.

use metrale_config::{ActivationQuantization, ProjFamily};

use super::ffn_act_fixed;

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
