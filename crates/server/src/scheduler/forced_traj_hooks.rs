// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Where the scheduler's steps meet the forced-trajectory mode
//! (`forced_traj`): a step's coordinates from its sequence, and the fed-token replacement.
//!
//! Owner: scheduler.
//! Invariants: every function is a no-op (returns its input) when the mode is off.

use super::forced_traj::{At, ForcedTraj};
use super::*;

/// 2026-10-23: `a`'s next step's coordinates.
pub(super) fn at(f: &ForcedTraj, a: &ActiveSeq) -> At {
    let prompt = &a.seq.tokens[..a.seq.prompt_len.min(a.seq.tokens.len())];
    At {
        key: f.key(a.seq.slot_idx, a.request_start, prompt),
        out: a.output_tokens.len(),
        kv: a.seq.seq_len,
    }
}

/// 2026-10-23: Before a step that feeds `a.last_token` (a K=2 verify when `k2`, else a
/// bootstrap decode): under replay the fed token becomes the recorded one, and so does the
/// output token it was emitted as. Returns the step's coordinates.
pub(super) fn feed(f: Option<&ForcedTraj>, a: &mut ActiveSeq, k2: bool) -> Result<Option<At>> {
    let Some(f) = f else {
        return Ok(None);
    };
    let at = at(f, a);
    let mut last = a.last_token;
    if f.feed(at, k2, &mut last)? {
        tracing::warn!(
            "PAIRED_TRAJ repair key={:016x} out={} fed {} -> recorded {last}",
            at.key,
            at.out,
            a.last_token
        );
        if let Some(t) = a.output_tokens.last_mut()
            && *t == a.last_token
        {
            *t = last;
        }
        a.last_token = last;
    }
    Ok(Some(at))
}

/// 2026-10-23: A K=2 verify's forced-trajectory state between its feed and its verdict.
pub(super) struct K2Traj {
    pub at: At,
    pub last: u32,
    /// 2026-10-23: The draft the verify feeds (the record's under replay).
    pub draft: u32,
    /// 2026-10-23: The draft this serve proposed (`DEVICE_DRAFT` when it is on the device).
    pub own: u32,
}

/// 2026-10-23: Before a K=2 verify of `a` whose own draft is `own`: [`feed`], then the draft.
pub(super) fn k2_begin(
    f: Option<&ForcedTraj>,
    a: &mut ActiveSeq,
    own: u32,
) -> Result<Option<K2Traj>> {
    let (Some(f), Some(at)) = (f, feed(f, a, true)?) else {
        return Ok(None);
    };
    Ok(Some(K2Traj {
        at,
        last: a.last_token,
        draft: f.k2_draft(at, own)?,
        own,
    }))
}

/// 2026-10-23: After the verify `t` fed `draft0` and picked `picks`: the `(draft, v0, v1)` the
/// step goes on with ([`ForcedTraj::k2_verdict`]). A replayed step's own draft still on the
/// device is read back (the verify's readback already waited for it).
pub(super) fn k2_verdict(
    f: &ForcedTraj,
    t: &K2Traj,
    model: &dyn Model,
    a: &mut ActiveSeq,
    draft0: u32,
    picks: [u32; 2],
) -> Result<[u32; 3]> {
    let own = match t.own {
        metrale_model_layers::speculative::DEVICE_DRAFT if f.replaying() => {
            model.read_device_draft(&mut a.seq)?
        }
        metrale_model_layers::speculative::DEVICE_DRAFT => draft0,
        d => d,
    };
    f.k2_verdict(t.at, t.last, draft0, own, picks)
}
