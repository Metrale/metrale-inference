// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: the MTP bootstrap's pick: one decode row, picked as verify
//! position 0 is (`verify_pick_with_pipeline`), so the bootstrap token takes
//! the logits pipeline: no grammar inside thinking, the forced `</think>`, the
//! think and tool-call masks, the EOS hold, the penalties and the request's
//! `logit_bias`. It replaced `sample_token_with_grammar`, which applied only
//! the grammar (inside thinking too), the penalties and an EOS mask.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types. The grammar matcher is not advanced;
//! `emit_token` advances it for the token it emits.

use super::fast_masked::{is_think_or_tool_id, stages_may_act};
use super::verify_pick_with_pipeline;
use crate::scheduler::ActiveSeq;
use crate::scheduler::fast_greedy::{PenaltyGate, argmax_survives, classify_penalties_except_bias};
use crate::scheduler::logit_processors::{LogitsContext, SamplingLevers};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::Model;

/// 2026-10-21: pick the token for row `row` of the decode logits at `base`.
///
/// The device argmax (`device_argmax`, or one `argmax_on_device`) is returned
/// when nothing in the pipeline can move it (`device_argmax_is_pick`).
/// Otherwise the row is copied to host and `verify_pick_with_pipeline` picks
/// it at position 0. A bootstrap token is a decode step, so it is sampled at
/// temperature above 0 whatever `mtp_verify_sample` says, as before.
pub fn bootstrap_pick(
    model: &dyn Model,
    base: DevicePtr,
    row: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
    device_argmax: Option<u32>,
) -> anyhow::Result<u32> {
    let vocab = model.vocab_size();
    let is_fp32 = model.logits_ptr_is_fp32(base);
    let elem = if is_fp32 { 4 } else { 2 };
    let row_ptr = base.offset(row * vocab * elem);
    if greedy_fast_path_open(a, ctx) {
        let top1 = match device_argmax {
            Some(t) => t,
            None => model.argmax_on_device(row_ptr, 0)?,
        };
        if device_argmax_is_pick(model, base, row, vocab, top1, a, ctx) {
            return Ok(top1);
        }
    }
    let mut buf = vec![0u8; vocab * elem];
    model.copy_logits_to_host(row_ptr, &mut buf)?;
    let decode_ctx = LogitsContext {
        sampling: SamplingLevers {
            mtp_verify_sample: true,
            ..ctx.sampling
        },
        ..ctx.clone()
    };
    Ok(verify_pick_with_pipeline(
        &buf,
        is_fp32,
        vocab,
        a,
        &decode_ctx,
        0,
    ))
}

/// 2026-10-21: greedy decoding with the `fast_greedy_grammar` lever on
/// (`METRALE_DISABLE_FAST_GREEDY=1` turns it off), and the AdaDec diagnostic
/// off (it records the full pipeline).
fn greedy_fast_path_open(a: &ActiveSeq, ctx: &LogitsContext) -> bool {
    ctx.sampling.fast_greedy_grammar
        && !ctx.sampling.adadec_diagnostic
        && (a.temperature == 0.0 || ctx.sampling.force_temp_zero)
}

/// 2026-10-21: whether the pipeline at position 0 picks `top1`, the device
/// argmax: under `force_temp_zero` it is the pick (`process_position_logits`
/// returns the raw argmax). Otherwise it must not be a held EOS or an id the
/// think and tool-call masks touch, no other stage may act
/// (`stages_may_act`), the penalties and bias must keep it
/// (`argmax_survives`), and outside thinking the grammar must allow it.
fn device_argmax_is_pick(
    model: &dyn Model,
    base: DevicePtr,
    row: usize,
    vocab: usize,
    top1: u32,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> bool {
    if ctx.sampling.force_temp_zero {
        return true;
    }
    if crate::scheduler::eos_hold::eos_masked_in_window(a, &[top1])
        || is_think_or_tool_id(top1, a, ctx)
        || stages_may_act(a, ctx, 1)
    {
        return false;
    }
    let penalties = crate::scheduler::sample_step::penalty_params_for(
        a,
        crate::scheduler::sample_step::PositionKind::Verify,
        0.0,
        None,
        a.logit_bias.clone(),
        ctx.watchdog.min_reasoning_floor,
    );
    let gate = classify_penalties_except_bias(&penalties);
    if gate == PenaltyGate::Blocked {
        return false;
    }
    let history =
        crate::scheduler::sample_step::penalty_history_scope(&a.output_tokens, ctx.tool_call_end_token);
    if !argmax_survives(
        model,
        base,
        row,
        vocab,
        top1,
        gate,
        history,
        &penalties.logit_bias,
    ) {
        return false;
    }
    match a.grammar_state.as_mut() {
        Some(gs) if !a.inside_thinking && !gs.is_terminated() => {
            gs.fill_bitmask();
            gs.is_token_allowed(top1)
        }
        _ => true,
    }
}
