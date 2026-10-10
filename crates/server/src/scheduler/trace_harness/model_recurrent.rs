// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The recording fake's recurrent state, on when `ModelCfg::recurrent_fed_rows` is
//! set: a stand-in for a model (GLM-5.3's KDA layers) whose per-sequence state advances with
//! every token it consumes and cannot be rewound by lowering `seq_len`.
//!
//! Each sequence remembers the next position its state expects. A forward that consumes
//! positions starting anywhere else (a step re-run, or a continuing row fed twice) corrupts the
//! state for good, and every later output of that sequence is [`GARBAGE`], so a client stream
//! shows it. A prefill from position 0 starts fresh state.
//!
//! Owner: scheduler.
//! Invariants: with the option off, nothing is tracked and every forward is clean.

use metrale_model_engine::traits::SequenceState;

use super::model::{GARBAGE, RecordingModel};

impl RecordingModel {
    /// 2026-10-10: The key the recurrent state is kept under: the request, not the slot, so a
    /// swapped-out sequence comes back to its own state.
    fn recurrent_key(seq: &SequenceState) -> u64 {
        if seq.session_hash != 0 {
            seq.session_hash
        } else {
            u64::from(seq.tokens.first().copied().unwrap_or(0))
        }
    }

    /// 2026-10-10: Consume positions `start..start + len` of `seq`; `false` once the state is
    /// corrupted. A `start` of 0 begins fresh state.
    pub(super) fn recurrent_consume(&self, seq: &SequenceState, start: usize, len: usize) -> bool {
        if self.cfg.recurrent_fed_rows.is_none() {
            return true;
        }
        let key = Self::recurrent_key(seq);
        let (clean, reused) = {
            let mut map = self.recurrent.lock().unwrap();
            let entry = map.entry(key).or_insert((0, false));
            if start == 0 {
                *entry = (0, false);
            }
            let reused = (entry.0 != start && !entry.1).then_some(entry.0);
            entry.1 |= reused.is_some();
            entry.0 = start + len;
            (!entry.1, reused)
        };
        if let Some(expected) = reused {
            self.rec(format!(
                "recurrent state of s{key} consumed again at {start} (expected {expected})"
            ));
        }
        clean
    }

    /// 2026-10-10: `out` when the state that produced it is clean, else [`GARBAGE`].
    pub(super) fn recurrent_out(&self, clean: bool, out: u32) -> u32 {
        if clean { out } else { GARBAGE }
    }
}
