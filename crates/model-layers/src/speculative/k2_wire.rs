// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The fused multi-rank K=2 MTP step (`METRALE_EP_K2_FUSED=1`, v3 wire only): one
//! command block announces the verify, and one verdict broadcast both commits it and starts the
//! workers' propose.
//!
//! Owner: model-layers (speculative).
//! Invariants:
//! - [`EP_CMD_VERIFY_K2_FUSED`] is above the decode-token range and differs from every other
//!   worker opcode (asserted where the opcodes are listed, `decode_checkpoint/plan.rs`).
//! - A worker derives the propose's position (its own `seq_len` after the verdict) and its
//!   hidden row (`accepted`); the verdict's `position` word only checks the derivation.
//!
//! Wire shape, against the unfused step's eleven broadcasts under v2 (nine under v3):
//!
//! ```text
//! rank 0                                         worker
//! block [slot, F_K2, t0, 0, t1, NO_MASK]    ->   tokens (t0, t1)
//! 2-row verify forward (collectives)       <->   the same 2-row verify forward
//! verdict [accepted, next, drafts, pos]     ->   commit; when drafts > 0, save hidden row
//!                                                `accepted` and propose from (next, pos)
//! propose forward (collectives)            <->   the same propose forward
//! ```
//!
//! Rank 0 sends the verdict after it emits the step's tokens: an emit only sets flags on the
//! host, and a sequence the emit finished is sent `drafts = 0`, so no rank proposes for it.

use anyhow::{Result, bail};

use crate::ep_wire::{self, CMD_BLOCK_BYTES, CmdBlock, NO_MASK};

/// 2026-10-11: EP worker command: the fused K=2 verify (see the module doc).
pub const EP_CMD_VERIFY_K2_FUSED: u32 = 0xFFFF_FFFC;

/// 2026-10-11: Words of the verdict broadcast.
pub const K2_VERDICT_WORDS: usize = 4;

/// 2026-10-11: The lever. `1` asks for the fused step; read once per process.
pub const K2_FUSED_ENV: &str = "METRALE_EP_K2_FUSED";

/// 2026-10-11: Whether `METRALE_EP_K2_FUSED=1` asks for the fused step. Whether a serve runs it
/// is the model's decision (`ModelEp::k2_fused_wire`): the wire must be v3, and the proposer
/// must be the sharded one every rank runs.
pub fn k2_fused_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var(K2_FUSED_ENV).ok().as_deref() == Some("1"))
}

/// 2026-10-11: The fused step needs the v3 block (its tokens ride in `arg` and `mask`). A serve
/// that asks for it on another wire fails at startup instead of silently running unfused.
pub fn check_k2_fused_wire(requested: bool, wire: ep_wire::EpWire) -> Result<()> {
    if requested && !wire.one_block() {
        bail!(
            "{K2_FUSED_ENV}=1 needs {}=v3 (the fused K=2 verify rides in the v3 block); got {wire:?}",
            ep_wire::ENV
        );
    }
    Ok(())
}

/// 2026-10-11: Rank 0's announcement of a fused K=2 verify of `tokens` in slot `seq_id`.
pub fn announce_block(seq_id: u32, tokens: [u32; 2]) -> [u8; CMD_BLOCK_BYTES] {
    ep_wire::block(
        seq_id,
        EP_CMD_VERIFY_K2_FUSED,
        tokens[0],
        0,
        [tokens[1], NO_MASK],
    )
}

/// 2026-10-11: The two verify tokens of a received announcement. Errors on another opcode, a
/// flag, or a second mask word, so a block of another shape is never read as one.
pub fn announced_tokens(block: &CmdBlock) -> Result<[u32; 2]> {
    if block.cmd != EP_CMD_VERIFY_K2_FUSED || block.flags != 0 || block.mask[1] != NO_MASK {
        bail!("not a fused K=2 verify block: {block:?}");
    }
    Ok([block.arg, block.mask[0]])
}

/// 2026-10-11: Rank 0's verdict on a fused K=2 verify.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct K2Verdict {
    /// 2026-10-11: The draft was accepted: both rows are kept, and the propose reads hidden
    /// row 1. Otherwise row 0 alone is kept and the propose reads hidden row 0.
    pub accepted: bool,
    /// 2026-10-11: The token the propose starts from (the bonus token, or the correction).
    pub next_token: u32,
    /// 2026-10-11: Drafts to propose; 0 when the step finished the sequence, and no rank
    /// proposes.
    pub num_drafts: u32,
    /// 2026-10-11: The propose position rank 0 uses: its `seq_len` after the verdict.
    pub position: u32,
}

impl K2Verdict {
    pub fn to_words(self) -> [u32; K2_VERDICT_WORDS] {
        [
            u32::from(self.accepted),
            self.next_token,
            self.num_drafts,
            self.position,
        ]
    }

    /// 2026-10-11: Errors on an `accepted` word other than 0 or 1, or a wrong word count.
    pub fn from_words(w: &[u32]) -> Result<Self> {
        let [accepted, next_token, num_drafts, position] = w else {
            bail!(
                "fused K=2 verdict: {} words, expected {K2_VERDICT_WORDS}",
                w.len()
            );
        };
        let accepted = match accepted {
            0 => false,
            1 => true,
            other => bail!("fused K=2 verdict: accepted word {other} is not 0 or 1"),
        };
        Ok(Self {
            accepted,
            next_token: *next_token,
            num_drafts: *num_drafts,
            position: *position,
        })
    }

    /// 2026-10-11: The hidden row the propose reads: row 1 after an accept, row 0 after a
    /// reject (what rank 0 passes to `save_hidden_for_mtp`).
    pub fn hidden_idx(self) -> usize {
        usize::from(self.accepted)
    }

    /// 2026-10-11: A worker's propose position: its own `seq_len` after it applied the verdict,
    /// checked against rank 0's. A mismatch means the ranks' sequences diverged.
    pub fn worker_position(self, worker_seq_len: usize) -> Result<usize> {
        if worker_seq_len != self.position as usize {
            bail!(
                "fused K=2 verdict: worker seq_len {worker_seq_len} != rank 0's propose \
                 position {}",
                self.position
            );
        }
        Ok(worker_seq_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: An announcement round-trips through the block's bytes, for every token
    /// value, including 0 and ids above the vocab.
    #[test]
    fn an_announcement_round_trips_through_the_block_bytes() {
        for tokens in [[0, 0], [785, 154_855], [u32::MAX - 1, 7], [3, 0]] {
            let b = ep_wire::parse_block(&announce_block(5, tokens));
            assert_eq!(b.seq_id, 5);
            assert_eq!(announced_tokens(&b).unwrap(), tokens);
        }
    }

    /// 2026-10-11: A block of another shape is refused: another opcode (a host command, a
    /// decode step), a flag, or a second mask word.
    #[test]
    fn a_block_of_another_shape_is_not_read_as_an_announcement() {
        let host = ep_wire::parse_block(&ep_wire::host_block(0, 0xFFFF_FFF2));
        assert!(announced_tokens(&host).is_err());
        let decode = ep_wire::parse_block(&ep_wire::decode_block(
            0,
            EP_CMD_VERIFY_K2_FUSED,
            ep_wire::FLAG_DEVICE_ARGMAX,
            [1, NO_MASK],
        ));
        assert!(announced_tokens(&decode).is_err());
        let two_mask = ep_wire::parse_block(&ep_wire::block(
            0,
            EP_CMD_VERIFY_K2_FUSED,
            1,
            0,
            [2, 3],
        ));
        assert!(announced_tokens(&two_mask).is_err());
    }

    /// 2026-10-11: A verdict round-trips through its words, both ways and with no propose.
    #[test]
    fn a_verdict_round_trips_through_its_words() {
        for v in [
            K2Verdict {
                accepted: true,
                next_token: 42,
                num_drafts: 1,
                position: 260,
            },
            K2Verdict {
                accepted: false,
                next_token: 154_855,
                num_drafts: 1,
                position: 259,
            },
            K2Verdict {
                accepted: true,
                next_token: 0,
                num_drafts: 0,
                position: 589,
            },
        ] {
            assert_eq!(K2Verdict::from_words(&v.to_words()).unwrap(), v);
        }
    }

    /// 2026-10-11: The hidden row is the accept bit, as rank 0's `save_hidden_for_mtp(1)` after
    /// an accept and `(0)` after a reject.
    #[test]
    fn the_hidden_row_is_the_accept_bit() {
        let v = |accepted| K2Verdict {
            accepted,
            next_token: 1,
            num_drafts: 1,
            position: 10,
        };
        assert_eq!(v(true).hidden_idx(), 1);
        assert_eq!(v(false).hidden_idx(), 0);
    }

    /// 2026-10-11: Malformed words are refused: a short or long verdict, an accept word that is
    /// not a bit.
    #[test]
    fn malformed_verdict_words_are_refused() {
        assert!(K2Verdict::from_words(&[1, 2, 3]).is_err());
        assert!(K2Verdict::from_words(&[1, 2, 3, 4, 5]).is_err());
        assert!(K2Verdict::from_words(&[2, 0, 1, 10]).is_err());
    }

    /// 2026-10-11: The lever is refused on a wire without the block, and accepted on v3 or
    /// when it is off.
    #[test]
    fn the_fused_step_needs_the_v3_wire() {
        use ep_wire::EpWire;
        assert!(check_k2_fused_wire(true, EpWire::V3).is_ok());
        assert!(check_k2_fused_wire(true, EpWire::V2).is_err());
        assert!(check_k2_fused_wire(true, EpWire::V1).is_err());
        for w in [EpWire::V1, EpWire::V2, EpWire::V3] {
            assert!(check_k2_fused_wire(false, w).is_ok());
        }
    }

    /// 2026-10-11: A worker whose length disagrees with rank 0's position fails instead of
    /// proposing at a different position.
    #[test]
    fn a_diverged_worker_length_fails_the_position_check() {
        let v = K2Verdict {
            accepted: false,
            next_token: 9,
            num_drafts: 1,
            position: 100,
        };
        assert_eq!(v.worker_position(100).unwrap(), 100);
        assert!(v.worker_position(101).is_err());
        assert!(v.worker_position(99).is_err());
    }
}
