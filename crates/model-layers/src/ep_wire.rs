// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The multi-rank command wire, `METRALE_EP_PROTOCOL` (read here only).
//!
//! - `v1` (unset or any other value): one `u32` per command, every command targets slot 0.
//! - `v2`: a `seq_id` word, then the command word: two broadcasts.
//! - `v3`: the v2 preamble as one fixed block `[seq_id, cmd, arg, flags, m0, m1]` (24 bytes) in
//!   one broadcast. `arg` is 0 for a command from the host; a fed decode puts its token there
//!   on the device (`decode_fed_ep.rs`). `flags` and the row mask `(m0, m1)` qualify a decode
//!   step (2026-10-10: [`FLAG_DEVICE_ARGMAX`]); other commands send 0 and no mask. A
//!   command's later words follow as in v2.
//!
//! Each broadcast of the preamble crosses the rank ring with a host detect on every worker,
//! so v3 removes one ring round trip per decode step (a 3-rank GLM-5.3 serve spent about
//! 140 us per step on rank 1 waiting for v2's second word).
//!
//! Owner: model-engine.
//! Invariants:
//! - Every rank reads the same wire (checked when the model is built).
//! - The command buffer holds a whole block, so the 4-byte words of every wire fit in it.

/// 2026-10-10: The variable.
pub const ENV: &str = "METRALE_EP_PROTOCOL";

/// 2026-10-10: Bytes of the v3 command block, and so of the command buffer.
pub const CMD_BLOCK_BYTES: usize = 24;

/// 2026-10-10: Byte offset of `arg` in the block.
pub const CMD_BLOCK_ARG: usize = 8;

/// 2026-10-10: Byte offset of the decode step's row mask `(m0, m1)` in the block.
pub const CMD_BLOCK_MASK: usize = 16;

/// 2026-10-10: `flags` bit: the decode step ends in the sharded device argmax
/// (`device_argmax.rs` in model-engine) under the block's row mask, on every rank.
pub const FLAG_DEVICE_ARGMAX: u32 = 1;

/// 2026-10-10: "No id" in a row mask.
pub const NO_MASK: u32 = u32::MAX;

/// 2026-10-10: The fields of a received block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CmdBlock {
    pub seq_id: u32,
    pub cmd: u32,
    pub arg: u32,
    pub flags: u32,
    pub mask: [u32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpWire {
    V1,
    V2,
    V3,
}

impl EpWire {
    /// 2026-10-10: `v2` and `v3` by name; anything else is v1, as `METRALE_EP_PROTOCOL` was read
    /// before v3 existed.
    pub fn parse(value: Option<&str>) -> Self {
        match value {
            Some("v2") => Self::V2,
            Some("v3") => Self::V3,
            _ => Self::V1,
        }
    }

    /// 2026-10-10: The process's wire, read once.
    pub fn get() -> Self {
        static WIRE: std::sync::OnceLock<EpWire> = std::sync::OnceLock::new();
        *WIRE.get_or_init(|| Self::parse(std::env::var(ENV).ok().as_deref()))
    }

    /// 2026-10-10: Whether every command carries the slot it runs in (`ep_protocol_v2`).
    pub fn has_seq_preamble(self) -> bool {
        self != Self::V1
    }

    /// 2026-10-10: Whether the preamble travels as one block.
    pub fn one_block(self) -> bool {
        self == Self::V3
    }

    /// 2026-10-10: The value the ranks compare at construction.
    pub fn agreement_value(self) -> u64 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
            Self::V3 => 3,
        }
    }
}

/// 2026-10-10: The block's bytes for a command from the host: `arg` 0, no flags, no mask.
pub fn host_block(seq_id: u32, cmd: u32) -> [u8; CMD_BLOCK_BYTES] {
    decode_block(seq_id, cmd, 0, [NO_MASK, NO_MASK])
}

/// 2026-10-10: The block's bytes for a decode step with `flags` and its row `mask` (`arg` 0).
pub fn decode_block(seq_id: u32, cmd: u32, flags: u32, mask: [u32; 2]) -> [u8; CMD_BLOCK_BYTES] {
    block(seq_id, cmd, 0, flags, mask)
}

/// 2026-10-11: The block's bytes with every field given; a host command that carries its own
/// argument words in `arg` and `mask` (the fused K=2 verify, `speculative::k2_wire`).
pub fn block(seq_id: u32, cmd: u32, arg: u32, flags: u32, mask: [u32; 2]) -> [u8; CMD_BLOCK_BYTES] {
    let mut b = [0u8; CMD_BLOCK_BYTES];
    for (i, w) in [seq_id, cmd, arg, flags, mask[0], mask[1]]
        .into_iter()
        .enumerate()
    {
        b[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    b
}

/// 2026-10-10: The fields of a received block.
pub fn parse_block(b: &[u8; CMD_BLOCK_BYTES]) -> CmdBlock {
    let word = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    CmdBlock {
        seq_id: word(0),
        cmd: word(4),
        arg: word(CMD_BLOCK_ARG),
        flags: word(12),
        mask: [word(CMD_BLOCK_MASK), word(CMD_BLOCK_MASK + 4)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wire_names_parse_and_every_other_value_is_v1() {
        assert_eq!(EpWire::parse(Some("v2")), EpWire::V2);
        assert_eq!(EpWire::parse(Some("v3")), EpWire::V3);
        for v1 in [None, Some(""), Some("v1"), Some("V3"), Some("3")] {
            assert_eq!(EpWire::parse(v1), EpWire::V1, "{v1:?}");
        }
        assert!(EpWire::V3.has_seq_preamble() && EpWire::V2.has_seq_preamble());
        assert!(!EpWire::V1.has_seq_preamble());
        assert!(EpWire::V3.one_block() && !EpWire::V2.one_block());
    }

    #[test]
    fn blocks_round_trip_and_a_host_block_carries_no_flags_or_mask() {
        let b = host_block(7, 0xFFFF_FFE0);
        let f = parse_block(&b);
        assert_eq!((f.seq_id, f.cmd, f.arg, f.flags), (7, 0xFFFF_FFE0, 0, 0));
        assert_eq!(f.mask, [NO_MASK, NO_MASK]);
        let d = parse_block(&decode_block(3, 42, FLAG_DEVICE_ARGMAX, [9, 11]));
        assert_eq!(
            (d.seq_id, d.cmd, d.arg, d.flags, d.mask),
            (3, 42, 0, 1, [9, 11])
        );
    }
}
