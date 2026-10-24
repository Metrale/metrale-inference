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

/// 2026-10-11: The draft slot of a K=2 verify whose draft is still on the device
/// (`METRALE_GLM_MTP_DEVICE_STEP`): every rank takes row 1's token from its own drafter's device
/// cell (`DraftProposer::device_draft`) instead of a host id. Never a token id: vocabularies are
/// far below `u32::MAX`.
pub const DEVICE_DRAFT: u32 = u32::MAX;

/// 2026-10-11: Words of the device verdict (`mtp_k2_verdict.cu`): accepted, next token, draft,
/// argmax 0, argmax 1.
pub const K2_DEVICE_VERDICT_WORDS: usize = 5;

/// 2026-10-11: The host specification of the device verdict `mtp_k2_verdict` writes, for a K=2
/// verify of `draft` whose rows' argmaxes are `argmax`: the verdict `step_verify_k2` reaches for
/// a greedy request whose picks are its argmaxes. Accepted when the draft is row 0's argmax; the
/// next token is row 1's argmax after an accept, row 0's after a reject.
pub fn k2_device_verdict(argmax: [u32; 2], draft: u32) -> [u32; K2_DEVICE_VERDICT_WORDS] {
    let accepted = draft == argmax[0];
    [
        u32::from(accepted),
        if accepted { argmax[1] } else { argmax[0] },
        draft,
        argmax[0],
        argmax[1],
    ]
}

/// 2026-10-11: A worker's cross-check of a device-verdict K=2 step (`FLAG_K2_DEVICE_VERDICT`):
/// rank 0's host verdict, sent after the propose, must be the one the worker committed and
/// proposed from (its device verdict and its own length after the commit). A mismatch means the
/// host picks were not the argmax; the ranks' states then differ and the step must fail.
pub fn device_verdict_agrees(
    rank0: K2Verdict,
    accepted: bool,
    next_token: u32,
    position: usize,
) -> bool {
    rank0.accepted == accepted
        && rank0.next_token == next_token
        && rank0.position as usize == position
}

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

/// 2026-10-11: `flags` bit of a fused K=2 announcement (`METRALE_GLM_MTP_DEVICE_STEP`): every rank
/// takes this step's verdict from its own device verdict and proposes right after its verify;
/// rank 0's verdict follows the propose and only cross-checks it ([`announce_block`]). Distinct
/// from `ep_wire::FLAG_DEVICE_ARGMAX`, which decode blocks carry.
pub const FLAG_K2_DEVICE_VERDICT: u32 = 2;

/// 2026-10-21: `flags` bit of a fused K=2 announcement (`METRALE_SCREEN_HEAD`): every rank ends
/// this verify in the screened head (`model-engine screen_head.rs`) instead of the full head and
/// its logits all-reduce. Rank 0 sets it only for a request whose picks need nothing but the
/// argmaxes and single logits; the collectives then differ from a full-head verify, so every
/// rank must follow the bit.
pub const FLAG_K2_SCREEN_HEAD: u32 = 4;

/// 2026-10-21: The flags a fused K=2 announcement carries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct K2Flags {
    /// 2026-10-11: [`FLAG_K2_DEVICE_VERDICT`].
    pub device_verdict: bool,
    /// 2026-10-21: [`FLAG_K2_SCREEN_HEAD`].
    pub screen_head: bool,
}

/// 2026-10-11: Rank 0's announcement of a fused K=2 verify of `tokens` in slot `seq_id` under
/// `flags`.
pub fn announce_block(seq_id: u32, tokens: [u32; 2], flags: K2Flags) -> [u8; CMD_BLOCK_BYTES] {
    let bits = if flags.device_verdict {
        FLAG_K2_DEVICE_VERDICT
    } else {
        0
    } | if flags.screen_head {
        FLAG_K2_SCREEN_HEAD
    } else {
        0
    };
    ep_wire::block(
        seq_id,
        EP_CMD_VERIFY_K2_FUSED,
        tokens[0],
        bits,
        [tokens[1], NO_MASK],
    )
}

/// 2026-10-11: The two verify tokens of a received announcement, and its flags. Errors on
/// another opcode, an unknown flag, or a second mask word, so a block of another shape is never
/// read as one.
pub fn announced_tokens(block: &CmdBlock) -> Result<([u32; 2], K2Flags)> {
    if block.cmd != EP_CMD_VERIFY_K2_FUSED
        || block.flags & !(FLAG_K2_DEVICE_VERDICT | FLAG_K2_SCREEN_HEAD) != 0
        || block.mask[1] != NO_MASK
    {
        bail!("not a fused K=2 verify block: {block:?}");
    }
    Ok((
        [block.arg, block.mask[0]],
        K2Flags {
            device_verdict: block.flags & FLAG_K2_DEVICE_VERDICT != 0,
            screen_head: block.flags & FLAG_K2_SCREEN_HEAD != 0,
        },
    ))
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
        for tokens in [
            [0, 0],
            [785, 154_855],
            [u32::MAX - 1, 7],
            [3, 0],
            [9, DEVICE_DRAFT],
        ] {
            for bits in 0..4u32 {
                let flags = K2Flags {
                    device_verdict: bits & 1 != 0,
                    screen_head: bits & 2 != 0,
                };
                let b = ep_wire::parse_block(&announce_block(5, tokens, flags));
                assert_eq!(b.seq_id, 5);
                assert_eq!(announced_tokens(&b).unwrap(), (tokens, flags));
            }
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
        // 2026-10-21: The device-verdict and screened-head bits are the only flags an
        // announcement may carry.
        let other_flag = ep_wire::parse_block(&ep_wire::block(
            0,
            EP_CMD_VERIFY_K2_FUSED,
            1,
            FLAG_K2_DEVICE_VERDICT | FLAG_K2_SCREEN_HEAD | 8,
            [2, NO_MASK],
        ));
        assert!(announced_tokens(&other_flag).is_err());
        let two_mask =
            ep_wire::parse_block(&ep_wire::block(0, EP_CMD_VERIFY_K2_FUSED, 1, 0, [2, 3]));
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

    /// 2026-10-11: The device verdict's words: an accept keeps the draft and moves to row 1's
    /// argmax; a reject (any other draft, the sentinel included) moves to row 0's argmax. Ids 0
    /// and the vocabulary's top are ordinary ids.
    #[test]
    fn the_device_verdict_accepts_exactly_row_0s_argmax() {
        assert_eq!(k2_device_verdict([7, 9], 7), [1, 9, 7, 7, 9]);
        assert_eq!(k2_device_verdict([7, 9], 9), [0, 7, 9, 7, 9]);
        assert_eq!(k2_device_verdict([0, 0], 0), [1, 0, 0, 0, 0]);
        assert_eq!(
            k2_device_verdict([154_855, 3], DEVICE_DRAFT),
            [0, 154_855, DEVICE_DRAFT, 154_855, 3]
        );
        assert_eq!(
            k2_device_verdict([154_855, 154_855], 154_855),
            [1, 154_855, 154_855, 154_855, 154_855]
        );
    }

    /// 2026-10-11: The cross-check passes only for the verdict the worker acted on: the same
    /// accept bit, next token and position; any one differing fails it.
    #[test]
    fn a_device_verdict_step_agrees_only_with_its_own_verdict() {
        let v = K2Verdict {
            accepted: true,
            next_token: 9,
            num_drafts: 1,
            position: 102,
        };
        assert!(device_verdict_agrees(v, true, 9, 102));
        assert!(!device_verdict_agrees(v, false, 9, 102), "accept bit");
        assert!(!device_verdict_agrees(v, true, 8, 102), "next token");
        assert!(!device_verdict_agrees(v, true, 9, 101), "position");
    }
}
