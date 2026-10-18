// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Host specifications of the device-step kernels (`mtp_device_step.cu`) for the K=2
//! MTP step under `METRALE_GLM_MTP_DEVICE_STEP`: the moves that wait for the host verdict today,
//! made on the device from the device verdict (`k2_wire::k2_device_verdict`).
//!
//! Owner: model-layers (speculative).
//! Invariants:
//! - [`k2_verify_meta`] is also what the host K=2 verify uploads
//!   (`TransformerModel::decode_verify_graphed`), so the device metadata and the host metadata are
//!   one rule.

/// 2026-10-11: The 2-row verify metadata of a sequence of length `len`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct K2VerifyMeta {
    /// 2026-10-11: RoPE positions `len` and `len + 1`.
    pub positions: [u32; 2],
    /// 2026-10-11: KV slot of each row: its physical block times `block_size` plus the offset.
    pub slots: [i64; 2],
    /// 2026-10-11: Attention length of each row (its position plus one).
    pub seq_lens: [i32; 2],
}

/// 2026-10-11: [`K2VerifyMeta`] for length `len`. `block(i)` is the physical block of logical
/// block `i`, `None` past the table, which writes into `dummy_block` (as the host path does).
pub fn k2_verify_meta(
    len: usize,
    block_size: usize,
    dummy_block: u32,
    block: impl Fn(usize) -> Option<u32>,
) -> K2VerifyMeta {
    let mut m = K2VerifyMeta {
        positions: [0; 2],
        slots: [0; 2],
        seq_lens: [0; 2],
    };
    for t in 0..2 {
        let pos = len + t;
        let phys = block(pos / block_size).unwrap_or(dummy_block);
        m.positions[t] = pos as u32;
        m.slots[t] = phys as i64 * block_size as i64 + (pos % block_size) as i64;
        m.seq_lens[t] = (pos + 1) as i32;
    }
    m
}

/// 2026-10-11: The sequence length after a K=2 verdict: the anchor row plus the draft when
/// accepted (`k2_step_advance`).
pub fn k2_step_advance(len: usize, accepted: bool) -> usize {
    len + 1 + usize::from(accepted)
}

/// 2026-10-11: Whether `predicated_copy_2d` copies: the flag word equals `match_value`. The K=2
/// commit copies the row-0 SSM intermediate over the live state on a reject (`match_value` 0);
/// the hidden-row select copies row 0 on a reject and row 1 on an accept.
pub fn predicated_copy_applies(flag: u32, match_value: u32) -> bool {
    flag == match_value
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: Rows at a block boundary land in two blocks; lengths are positions plus one.
    #[test]
    fn verify_rows_follow_the_block_table_across_a_boundary() {
        let table = [7u32, 3, 9];
        let m = k2_verify_meta(31, 16, 999, |i| table.get(i).copied());
        assert_eq!(m.positions, [31, 32]);
        assert_eq!(m.slots, [3 * 16 + 15, 9 * 16]);
        assert_eq!(m.seq_lens, [32, 33]);
    }

    /// 2026-10-11: A position past the table writes into the dummy block, never block 0.
    #[test]
    fn a_row_past_the_table_goes_to_the_dummy_block() {
        let m = k2_verify_meta(47, 16, 999, |i| [7u32, 3, 9].get(i).copied());
        assert_eq!(m.slots, [9 * 16 + 15, 999 * 16]);
    }

    /// 2026-10-11: A reject commits the anchor alone, an accept the anchor and the draft.
    #[test]
    fn the_length_advances_by_the_committed_rows() {
        assert_eq!(k2_step_advance(100, false), 101);
        assert_eq!(k2_step_advance(100, true), 102);
    }

    /// 2026-10-11: The copy runs only on its own verdict: the reject commit never on an accept,
    /// and the two row selects never both.
    #[test]
    fn exactly_one_of_the_row_selects_copies() {
        for accepted in [0u32, 1] {
            assert_eq!(predicated_copy_applies(accepted, 0), accepted == 0);
            assert_ne!(
                predicated_copy_applies(accepted, 0),
                predicated_copy_applies(accepted, 1)
            );
        }
    }
}
