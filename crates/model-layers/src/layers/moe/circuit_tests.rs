// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Tests of the MoE binding's per-width admission (`MoeFacts::check_rows`): a plan
//! must be refused exactly where legacy's grouped decode declines the width, by row envelope
//! or by an arena buffer it checks.
//!
//! Owner: model-layers (MoE).
//! Invariants: the shapes are Qwen3.6-35B-A3B's (hidden 2048, 256 experts, top-8, inter 512).

use super::*;

fn facts(tensor_core: bool, w8a8: bool) -> MoeFacts {
    MoeFacts {
        num_experts: 256,
        top_k: 8,
        hidden: 2048,
        inter: 512,
        norm_topk_prob: true,
        tensor_core,
        w8a8,
    }
}

/// 2026-10-03: An arena exactly as large as `m` rows need, so one byte less refuses.
fn arena_for(f: &MoeFacts, m: usize) -> MoeScratch {
    let need = grouped_decode_buffer_need(
        m,
        f.hidden as usize,
        f.inter as usize,
        f.num_experts as usize,
        f.top_k as usize,
    );
    MoeScratch {
        sort: DevicePtr::NULL,
        scratch_bytes: need.scratch,
        gate_logits_bytes: need.gate_logits,
        expert_gate_out_bytes: need.expert_gate_out,
        expert_down_out_bytes: need.expert_down_out,
        logits_bytes: need.shared_act,
        attn_output_bytes: need.row_hidden,
        moe_output_bytes: need.row_hidden,
    }
}

#[test]
fn the_tensor_core_kernels_take_one_to_256_rows_and_the_scalar_ones_two_to_64() {
    let big = arena_for(&facts(true, false), 256);
    let tc = facts(true, false);
    for m in [1, 2, 64, 65, 128, 256] {
        tc.check_rows(m, &big)
            .unwrap_or_else(|e| panic!("tc m={m}: {e:#}"));
    }
    assert!(tc.check_rows(257, &arena_for(&tc, 257)).is_err());
    let scalar = facts(false, false);
    for m in [2, 64] {
        scalar.check_rows(m, &big).unwrap();
    }
    for m in [1, 65] {
        assert!(scalar.check_rows(m, &big).is_err(), "scalar m={m}");
    }
}

#[test]
fn a_width_past_any_checked_arena_buffer_is_refused() {
    let f = facts(true, false);
    let exact = arena_for(&f, 16);
    f.check_rows(16, &exact).unwrap();
    let shrink: [fn(&mut MoeScratch); 7] = [
        |s| s.scratch_bytes -= 1,
        |s| s.gate_logits_bytes -= 1,
        |s| s.expert_gate_out_bytes -= 1,
        |s| s.expert_down_out_bytes -= 1,
        |s| s.logits_bytes -= 1,
        |s| s.attn_output_bytes -= 1,
        |s| s.moe_output_bytes -= 1,
    ];
    for (i, cut) in shrink.iter().enumerate() {
        let mut s = exact;
        cut(&mut s);
        assert!(f.check_rows(16, &s).is_err(), "buffer {i} one byte short");
    }
}
