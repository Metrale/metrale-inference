// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The penalty gate for greedy fast paths that take the GPU
//! argmax instead of running the host pipeline.
//!
//! Callers classify the request's penalties with [`classify_penalties`]:
//! `Neutral` cannot change the argmax, `ReduceOnly` cannot when
//! [`argmax_immune`] holds for that position's argmax, and `Blocked` rules
//! the fast path out.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! ## Why `ReduceOnly` plus `argmax_immune` keeps the argmax
//!
//! Let `g` be the argmax of the raw logits (the GPU argmax). The pipeline's
//! pick is the argmax of the masked-and-penalised logits. If:
//!  * every configured penalty only ever lowers logits of tokens in the
//!    scoped penalty history — `repetition_penalty >= 1.0` (divides positive
//!    logits: shrink; multiplies negative logits: more negative — both
//!    decreases), `presence_penalty >= 0.0` and `frequency_penalty >= 0.0`
//!    (subtract), while `lz_penalty == 0.0` and `dry_multiplier == 0.0`
//!    (those penalize pattern-extending tokens, not history members, so they
//!    fall outside the bound and must be off); and
//!  * `g` is not in the scoped history (its logit is untouched); and
//!  * `g`'s raw logit is > 0 (a conservative guard, one 2- or 4-byte read);
//!
//! then after penalties every token's logit is `<=` its raw value while
//! `g`'s is unchanged, and `g` was already the raw maximum, so `g` remains
//! the argmax. The bound says nothing about the grammar or other masks;
//! callers must check those separately.
//!
//! ## `logit_bias` (2026-10-21)
//!
//! A bias can raise a competitor, so [`classify_penalties`] blocks any bias.
//! A fast path that can still read single logits instead classifies with
//! [`classify_penalties_except_bias`] and checks each position with
//! [`bias_keeps_argmax`]: `g`'s net bias is `>= 0` and every token with a
//! positive net bias stays strictly below `g` after both biases. Unbiased
//! tokens end `<=` their raw value `<= raw[g] <= raw[g] + bias[g]`; a biased
//! one ends `<= raw[t] + bias[t]` (the reduce-only penalties only lower it,
//! the masks only lower it), so `g` remains the argmax.
//!
//! The scoped history used here must be the same one the pipeline hands to
//! `apply_penalties_and_bias` (`sample_step::penalty_history_scope`). A
//! `repetition_penalty_window` narrower than that history only shrinks the
//! penalized set, so membership in the full scoped history stays a
//! conservative superset test.

use metrale_sampling::SamplingParams;

/// 2026-09-25: How the configured penalties interact with the greedy fast
/// path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PenaltyGate {
    /// 2026-09-25: Every penalty is exactly neutral — the argmax is
    /// untouched, no per-token immunity check needed.
    Neutral,
    /// 2026-09-25: Reduce-only penalties (see module docs): fast path may
    /// fire per position iff [`argmax_immune`] holds for that position's
    /// argmax.
    ReduceOnly,
    /// 2026-09-25: Penalties/bias can raise a competitor or penalize
    /// non-history tokens (LZ/DRY) — the fast path must not fire.
    Blocked,
}

/// 2026-09-25: Classify built penalty params for fast-path eligibility.
/// Any `logit_bias` entry gives `Blocked`: bias is additive and can be
/// positive.
pub(super) fn classify_penalties(p: &SamplingParams) -> PenaltyGate {
    if !p.logit_bias.is_empty() {
        return PenaltyGate::Blocked;
    }
    classify_penalties_except_bias(p)
}

/// 2026-10-21: [`classify_penalties`] without the `logit_bias` check, for a
/// fast path that checks the bias per position with [`bias_keeps_argmax`].
pub(super) fn classify_penalties_except_bias(p: &SamplingParams) -> PenaltyGate {
    if p.lz_penalty != 0.0 || p.dry_multiplier != 0.0 {
        return PenaltyGate::Blocked;
    }
    if p.repetition_penalty == 1.0 && p.presence_penalty == 0.0 && p.frequency_penalty == 0.0 {
        return PenaltyGate::Neutral;
    }
    if p.repetition_penalty >= 1.0 && p.presence_penalty >= 0.0 && p.frequency_penalty >= 0.0 {
        return PenaltyGate::ReduceOnly;
    }
    PenaltyGate::Blocked
}

/// 2026-09-25: Per-position immunity check for [`PenaltyGate::ReduceOnly`]:
/// the argmax token is untouched by reduce-only penalties iff it is not in
/// the scoped history; `positive_logit` lazily reads the conservative
/// raw-logit > 0 guard off the device (see [`logit_is_positive`]) — lazy
/// because the single-element D2H carries a stream sync, only paid after
/// the membership test passes.
pub(super) fn argmax_immune(
    tok: u32,
    scoped_history: &[u32],
    positive_logit: impl FnOnce() -> bool,
) -> bool {
    !scoped_history.contains(&tok) && positive_logit()
}

/// 2026-10-21: Whether `logit_bias` cannot move the argmax `tok` (module
/// docs): `tok`'s net bias is `>= 0` and every other token with a positive net
/// bias stays strictly below `tok` after both biases. `read_logit` returns a
/// raw logit, `None` on a failed read (which declines); it is called only
/// when some other token has a positive net bias.
pub(super) fn bias_keeps_argmax(
    tok: u32,
    bias: &[(u32, f32)],
    mut read_logit: impl FnMut(u32) -> Option<f32>,
) -> bool {
    let net = |id: u32| -> f32 {
        bias.iter()
            .filter(|&&(t, _)| t == id)
            .map(|&(_, d)| d)
            .sum()
    };
    let tok_bias = net(tok);
    if tok_bias < 0.0 {
        return false;
    }
    let mut tok_biased: Option<f32> = None;
    for &(id, _) in bias {
        let d = net(id);
        if id == tok || d <= 0.0 {
            continue;
        }
        let top = match tok_biased {
            Some(v) => v,
            None => match read_logit(tok) {
                Some(v) => *tok_biased.insert(v + tok_bias),
                None => return false,
            },
        };
        match read_logit(id) {
            Some(raw) if raw + d < top => {}
            _ => return false,
        }
    }
    true
}

/// 2026-10-21: Whether the device argmax `tok` of row `row` is also the
/// argmax after the penalties `gate` classified (not `Blocked`) and `bias`:
/// [`argmax_immune`] for `ReduceOnly`, then [`bias_keeps_argmax`].
#[allow(clippy::too_many_arguments)]
pub(super) fn argmax_survives(
    model: &dyn metrale_model_engine::traits::Model,
    base: metrale_gpu_runtime::gpu::DevicePtr,
    row: usize,
    vocab: usize,
    tok: u32,
    gate: PenaltyGate,
    scoped_history: &[u32],
    bias: &[(u32, f32)],
) -> bool {
    debug_assert!(gate != PenaltyGate::Blocked, "a Blocked gate has no fast path");
    (gate == PenaltyGate::Neutral
        || argmax_immune(tok, scoped_history, || {
            logit_is_positive(model, base, row, vocab, tok)
        }))
        && bias_keeps_argmax(tok, bias, |id| read_logit(model, base, row, vocab, id))
}

/// 2026-09-25: Read one logit for `tok` from the device logits buffer at
/// `base` (row `row` of a `[*, vocab]` layout) and test strict positivity.
/// NaN and zero fail the `> 0.0` comparison; a failed copy returns `false`.
pub(super) fn logit_is_positive(
    model: &dyn metrale_model_engine::traits::Model,
    base: metrale_gpu_runtime::gpu::DevicePtr,
    row: usize,
    vocab: usize,
    tok: u32,
) -> bool {
    read_logit(model, base, row, vocab, tok).is_some_and(|v| v > 0.0)
}

/// 2026-10-21: Read one raw logit for `tok` from row `row` of the device
/// logits buffer at `base`: BF16 unless the model reports the buffer as FP32.
/// `None` when the copy fails.
pub(super) fn read_logit(
    model: &dyn metrale_model_engine::traits::Model,
    base: metrale_gpu_runtime::gpu::DevicePtr,
    row: usize,
    vocab: usize,
    tok: u32,
) -> Option<f32> {
    let idx = row * vocab + tok as usize;
    if model.logits_ptr_is_fp32(base) {
        let mut b = [0u8; 4];
        model
            .copy_logits_to_host(base.offset(idx * 4), &mut b)
            .ok()?;
        Some(f32::from_le_bytes(b))
    } else {
        let mut b = [0u8; 2];
        model
            .copy_logits_to_host(base.offset(idx * 2), &mut b)
            .ok()?;
        Some(crate::scheduler::helpers::bf16_to_f32(b[0], b[1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(rep: f32, pres: f32, freq: f32, lz: f32, dry: f32) -> SamplingParams {
        SamplingParams {
            // 2026-09-25: fields the gate ignores, set explicitly because
            // `SamplingParams` has no `Default` impl.
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            top_n_sigma: 0.0,
            min_p: 0.0,
            logit_bias: Vec::new(),
            repetition_penalty: rep,
            repetition_penalty_window: 0,
            presence_penalty: pres,
            frequency_penalty: freq,
            lz_penalty: lz,
            dry_multiplier: dry,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_sequence_breakers: Vec::new(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: None,
        }
    }

    #[test]
    fn neutral_params_classify_neutral() {
        assert_eq!(
            classify_penalties(&params(1.0, 0.0, 0.0, 0.0, 0.0)),
            PenaltyGate::Neutral
        );
    }

    #[test]
    fn reduce_only_params_classify_reduce_only() {
        assert_eq!(
            classify_penalties(&params(1.05, 0.0, 0.0, 0.0, 0.0)),
            PenaltyGate::ReduceOnly
        );
        assert_eq!(
            classify_penalties(&params(1.0, 0.5, 0.2, 0.0, 0.0)),
            PenaltyGate::ReduceOnly
        );
    }

    #[test]
    fn raising_or_pattern_penalties_block() {
        // 2026-09-25: rep < 1.0 raises positive logits of history tokens.
        assert_eq!(
            classify_penalties(&params(0.9, 0.0, 0.0, 0.0, 0.0)),
            PenaltyGate::Blocked
        );
        // 2026-09-25: negative presence/frequency raise history tokens.
        assert_eq!(
            classify_penalties(&params(1.0, -0.1, 0.0, 0.0, 0.0)),
            PenaltyGate::Blocked
        );
        // 2026-09-25: LZ and DRY penalize pattern-extending tokens, not
        // history members, so they fall outside the bound.
        assert_eq!(
            classify_penalties(&params(1.0, 0.0, 0.0, 0.2, 0.0)),
            PenaltyGate::Blocked
        );
        assert_eq!(
            classify_penalties(&params(1.0, 0.0, 0.0, 0.0, 0.8)),
            PenaltyGate::Blocked
        );
    }

    #[test]
    fn logit_bias_blocks() {
        let mut p = params(1.0, 0.0, 0.0, 0.0, 0.0);
        p.logit_bias.push((42, -8.0));
        assert_eq!(classify_penalties(&p), PenaltyGate::Blocked);
    }

    #[test]
    fn bias_keeps_argmax_only_with_a_strict_margin() {
        // 2026-10-21: raw logits: argmax 7 at 10.0, biased competitor 3.
        let raw = |id: u32| Some(if id == 7 { 10.0 } else { 8.0 });
        assert!(bias_keeps_argmax(7, &[], |_| unreachable!("no read without bias")));
        assert!(bias_keeps_argmax(7, &[(3, 1.5)], raw), "8.0 + 1.5 < 10.0");
        assert!(!bias_keeps_argmax(7, &[(3, 2.0)], raw), "a tie is not kept");
        assert!(!bias_keeps_argmax(7, &[(3, 3.0)], raw), "8.0 + 3.0 > 10.0");
        assert!(bias_keeps_argmax(7, &[(3, 3.0), (7, 1.5)], raw), "the argmax's own bias counts");
        assert!(bias_keeps_argmax(7, &[(3, 3.0), (3, -2.0)], raw), "entries net per id");
        assert!(!bias_keeps_argmax(7, &[(7, -0.5)], raw), "a lowered argmax may lose");
        assert!(bias_keeps_argmax(7, &[(3, -8.0)], |_| unreachable!("negative bias needs no read")));
        assert!(!bias_keeps_argmax(7, &[(3, 1.0)], |_| None), "a failed read declines");
    }

    #[test]
    fn immunity_requires_absence_and_positivity() {
        assert!(argmax_immune(7, &[1, 2, 3], || true));
        assert!(!argmax_immune(2, &[1, 2, 3], || true));
        assert!(!argmax_immune(7, &[1, 2, 3], || false));
    }

    #[test]
    fn positivity_read_is_lazy_after_membership_fail() {
        let mut read = false;
        assert!(!argmax_immune(2, &[1, 2, 3], || {
            read = true;
            true
        }));
        assert!(!read);
    }
}
