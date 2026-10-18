// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The loader switches read from the settings (`moe_prefill_copies_fit`, the qwen3
//! BF16 router, the DeepSeek-V4 expert-0 dump) against the environment reads they replaced.
//!
//! Owner: model-arch weight loaders.
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a
//! strict-`1` switch now also arms at `true`, the prefill-copies opt-out now also turns off at
//! `off`, and a value the switch rule cannot read is refused at startup.

use metrale_model_layers::lever_equivalence::{Old, Outcome, check};
use metrale_settings::InferenceServerSettings;

#[test]
fn the_prefill_copies_opt_out_reads_as_before_except_off() {
    const ENV: &str = "METRALE_MOE_PREFILL_COPIES";
    assert_eq!(
        check(
            ENV,
            Old::OptOut,
            |probe| probe != Some("0"),
            |s| s.scheduler.moe_prefill_copies
        ),
        [
            (ENV, Some("off"), Outcome::Changed),
            (ENV, Some("garbage"), Outcome::Rejected)
        ]
    );
}

#[test]
fn the_router_and_dump_opt_ins_read_as_before_except_true() {
    let rows: [(&'static str, fn(&InferenceServerSettings) -> bool); 2] = [
        ("METRALE_BF16_ROUTER", |s| s.kernels.bf16_router),
        ("METRALE_DUMP_EXPERT0", |s| s.telemetry.dump_expert0),
    ];
    for (env, new) in rows {
        assert_eq!(
            check(env, Old::OptIn, |probe| probe == Some("1"), new),
            [
                (env, Some("true"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ],
            "{env}"
        );
    }
}
