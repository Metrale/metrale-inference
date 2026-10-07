// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Opt-in host-equivalent masks for a greedy minimum-token batch.
//! Owner: scheduler. Invariant: every other logit transformation stays on the host.

use crate::scheduler::{ActiveSeq, sched_ctx::SchedCtx};

pub(super) fn plan(active: &[ActiveSeq], sched: &SchedCtx) -> Option<Vec<Vec<u32>>> {
    let dumps = sched.io.tel.dumps();
    if !sched.levers.min_tokens_gpu_greedy
        || !sched.masked_greedy_sampling_allowed
        || sched.levers.force_temp_zero
        || sched.levers.adadec_diagnostic
        || sched.levers.decode_timing
        || dumps.logits.is_some()
        || dumps.adadec.is_some()
        || dumps.raw_logits_dir.is_some()
        || sched.io.dev.model().decode_logits_fp32()
        // 2026-10-07: Preserve the existing device tie policy once the host-only floor expires.
        || !active.iter().any(|a| a.output_tokens.len() < a.min_tokens)
        || active.iter().any(|a| {
            a.temperature != 0.0
                || a.repetition_penalty != 1.0
                || a.presence_penalty != 0.0
                || a.frequency_penalty != 0.0
                || a.lz_penalty != 0.0
                || a.dry_multiplier != 0.0
                || !a.logit_bias.is_empty()
                || a.grammar_state.is_some()
                || a.top_logprobs.is_some()
                || a.inside_thinking
                || a.force_end_thinking
                || a.require_tool_call
                || a.tools_present
                || a.tool_request
                || a.inside_parameter_body
                || a.inside_tool_body
                || a.suppress_tool_call
        })
    {
        return None;
    }
    Some(
        active
            .iter()
            .map(|a| {
                let mut mask = Vec::new();
                if a.output_tokens.len() < a.min_tokens {
                    mask.extend_from_slice(&a.eos_tokens);
                }
                if a.think_ended {
                    mask.extend(sched.masked_greedy_think_end);
                    mask.extend(a.think_start_token);
                }
                mask.sort_unstable();
                mask.dedup();
                mask
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::test_support::{PreemptStubModel, test_seq};
    use std::sync::Arc;

    fn context() -> SchedCtx {
        let mut ctx = SchedCtx::for_test_with(Arc::new(PreemptStubModel::default()));
        ctx.masked_greedy_sampling_allowed = true;
        Arc::get_mut(&mut ctx.levers).unwrap().min_tokens_gpu_greedy = true;
        ctx
    }
    fn row() -> ActiveSeq {
        let (mut a, _) = test_seq(vec![1], 20, None, 3);
        a.lz_penalty = 0.0;
        a.eos_tokens = vec![9, 7, 9];
        a.min_tokens = 2;
        a
    }

    #[test]
    fn floor_boundary_and_mixed_rows_preserve_host_masks() {
        let mut rows = vec![row(), row()];
        rows[1].output_tokens.push(2);
        assert_eq!(plan(&rows, &context()), Some(vec![vec![7, 9], vec![]]));
        rows[0].output_tokens.push(2);
        assert!(plan(&rows, &context()).is_none());
        assert!(plan(&[], &context()).is_none());
    }

    #[test]
    fn post_thinking_uses_global_close_id_like_host_pipeline() {
        let mut ctx = context();
        ctx.masked_greedy_think_end = Some(4);
        let mut a = row();
        a.think_ended = true;
        a.think_start_token = Some(3);
        a.think_end_token = Some(99);
        assert_eq!(plan(&[a], &ctx), Some(vec![vec![3, 4, 7, 9]]));
    }

    #[test]
    fn default_and_adaptive_modes_keep_host_sampling() {
        assert!(plan(&[row()], &SchedCtx::for_test()).is_none());
        let mut ctx = context();
        ctx.masked_greedy_sampling_allowed = false;
        assert!(plan(&[row()], &ctx).is_none());
    }

    #[test]
    fn planned_masks_match_real_host_pipeline_at_each_floor_boundary() {
        use crate::scheduler::logit_processors::run_pipeline;
        for length in 0..=3 {
            for closed in [false, true] {
                let mut ctx = context();
                ctx.masked_greedy_think_end = Some(4);
                let mut rows = vec![row(), row()];
                rows[0].output_tokens.resize(length, 1);
                rows[0].think_ended = closed;
                rows[0].think_start_token = Some(3);
                rows[0].eos_tokens.extend([99, 7]);
                let masks = plan(&rows, &ctx).unwrap();
                let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
                let pipeline =
                    super::super::logits_ctx(&ctx, &scratch, Some(4), Some(3), None, None);
                for (a, mask) in rows.iter_mut().zip(masks) {
                    let mut actual = vec![1.0; 12];
                    assert!(run_pipeline(&mut actual, a, &pipeline).is_none());
                    let mut planned = vec![1.0; 12];
                    for id in mask {
                        if let Some(slot) = planned.get_mut(id as usize) {
                            *slot = f32::NEG_INFINITY;
                        }
                    }
                    assert_eq!(actual, planned);
                }
            }
        }
    }

    #[test]
    fn transformations_and_observers_refuse_the_shortcut() {
        let mutations: &[fn(&mut ActiveSeq)] = &[
            |a| a.temperature = 0.1,
            |a| a.repetition_penalty = 1.1,
            |a| a.presence_penalty = 0.1,
            |a| a.frequency_penalty = 0.1,
            |a| a.lz_penalty = 0.1,
            |a| a.dry_multiplier = 0.1,
            |a| a.logit_bias.push((2, 1.0)),
            |a| a.top_logprobs = Some(1),
            |a| a.inside_thinking = true,
            |a| a.force_end_thinking = true,
            |a| a.require_tool_call = true,
            |a| a.tools_present = true,
            |a| a.tool_request = true,
            |a| a.inside_tool_body = true,
            |a| a.suppress_tool_call = true,
            |a| a.inside_parameter_body = true,
        ];
        for mutate in mutations {
            let mut a = row();
            mutate(&mut a);
            assert!(plan(&[a], &context()).is_none());
        }
        for which in 0..3 {
            let mut ctx = context();
            let levers = Arc::get_mut(&mut ctx.levers).unwrap();
            match which {
                0 => levers.force_temp_zero = true,
                1 => levers.adadec_diagnostic = true,
                _ => levers.decode_timing = true,
            }
            assert!(plan(&[row()], &ctx).is_none());
        }
    }
}
