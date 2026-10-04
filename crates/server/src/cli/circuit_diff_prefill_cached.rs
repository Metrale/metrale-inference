// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The cached-prefix leg of `met circuit diff --prefill`: the prefix-cache path
//! exercised on purpose, circuit against legacy in that same mode. For each length `t` whose
//! primed part the restore can serve (at least [`MIN_RESTORE_TOKENS`]), a run primes a fresh
//! prefix-cache namespace with the prompt's first `t - TAIL` tokens, then prefills the whole
//! `t`-token prompt in the same namespace, so the second pass restores the primed snapshot and
//! recomputes only the [`TAIL`]. The warm pass is exactly the strict leg's prompt, so it fits
//! every path the strict leg ran (2026-10-03: a prompt-plus-tail warm pass overran the
//! single-pass arena at the longest length). Legacy runs it twice
//! (the reference and a repeat, each in a namespace of its own) and every circuit forward once;
//! the warm passes are compared as the strict legs are (logits, state, and the repeat's path).
//!
//! Owner: server CLI.
//! Invariants:
//! - A leg whose reference's warm pass ran no exact-replay recurrence did not exercise the
//!   restore and fails as such: the leg cannot pass vacuously.
//! - Each run (prime and warm) owns its namespace; no two runs share a cached prefix.

use anyhow::Result;
use metrale_model_engine::traits::{ForwardSelect, Model};
use serde::Serialize;

use super::{
    PrefillComparison, PrefillPath, compare, first_op_diff, first_state_diff, prefill_once,
    prompt_of_len,
};

/// 2026-10-03: The shortest restore the prefix cache serves (`mtp_carry::marconi_min_tokens`'s
/// default, 256 tokens).
pub(super) const MIN_RESTORE_TOKENS: usize = 256;

/// 2026-10-03: Tokens the warm pass recomputes after the primed part.
pub(super) const TAIL: usize = 37;

/// 2026-10-03: The kernel only the after-restore GatedDeltaNet arm launches; its presence in the
/// warm reference's trace proves the leg ran the cached path.
const RESTORE_MARK: &str = "gated_delta_rule_prefill_regresident";

/// 2026-10-03: The comparisons of one length and path, in the shape the verdict reads.
pub(super) type Flat = (usize, PrefillPath, Vec<PrefillComparison>);

/// 2026-10-03: One length and path of the cached leg.
#[derive(Debug, Serialize)]
pub(super) struct CachedReport {
    tokens: usize,
    path: PrefillPath,
    restored: bool,
    comparisons: Vec<PrefillComparison>,
}

/// 2026-10-03: Prime all of `prompt` but its [`TAIL`] in namespace `ns`, then prefill the whole
/// prompt there; the warm run.
fn warm_once(
    model: &dyn Model,
    prompt: &[u32],
    path: PrefillPath,
    ns: u64,
) -> Result<super::PrefillRun> {
    prefill_once(model, &prompt[..prompt.len() - TAIL], path, ns)?;
    prefill_once(model, prompt, path, ns)
}

/// 2026-10-03: The cached leg over `lens` and `paths`; its reports, and its comparisons in the
/// shape the verdict reads (a leg that did not restore is a path difference at op 0).
pub(super) fn cached_report(
    model: &dyn Model,
    lens: &[usize],
    paths: &[PrefillPath],
    forwards: &[(&'static str, ForwardSelect)],
    next_ns: &mut dyn FnMut() -> u64,
) -> Result<(Vec<CachedReport>, Vec<Flat>)> {
    let (mut reports, mut flat) = (Vec::new(), Vec::new());
    for &t in lens.iter().filter(|&&t| t >= MIN_RESTORE_TOKENS + TAIL) {
        let prompt = prompt_of_len(t, model.vocab_size());
        for &path in paths {
            model.set_forward(&ForwardSelect::Legacy)?;
            let reference = warm_once(model, &prompt, path, next_ns())?;
            let restored = reference.ops.iter().any(|o| o.op.ends_with(RESTORE_MARK));
            let mut cs = Vec::new();
            if !restored {
                let mut c = compare("legacy warm pass (no restore ran)", &[], &[]);
                c.path_diff = Some(0);
                cs.push(c);
            }
            for (name, sel) in forwards {
                model.set_forward(sel)?;
                let run = warm_once(model, &prompt, path, next_ns())?;
                let mut c = compare(
                    &format!("{name}, cached prefix"),
                    &reference.logits,
                    &run.logits,
                );
                c.state_diff = first_state_diff(&reference.state, &run.state);
                if matches!(sel, ForwardSelect::Legacy) {
                    c.path_diff = first_op_diff(&reference.ops, &run.ops);
                }
                tracing::info!(
                    "circuit diff --prefill cached: {t} tokens ({TAIL} recomputed) {path:?} {name}: {c:?}"
                );
                cs.push(c);
            }
            flat.push((t, path, cs.clone()));
            reports.push(CachedReport {
                tokens: t,
                path,
                restored,
                comparisons: cs,
            });
        }
    }
    model.set_forward(&ForwardSelect::Legacy)?;
    Ok((reports, flat))
}
