// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The multi-rank DFlash K=γ verify command: its opcode, its width bound and the
//! committed-row arithmetic both ranks apply after the verdict.
//!
//! Owner: model-layers (speculative).
//! Invariants:
//! - [`EP_CMD_VERIFY_KGAMMA`] is above the decode-token range and differs from every other
//!   worker opcode (asserted where the opcodes are listed, `decode_checkpoint/plan.rs`).
//!
//! Wire shape, after the `(seq_id, cmd)` preamble:
//!
//! ```text
//! rank 0                                   worker
//! K                    (one word)      ->  K, checked by kgamma_width
//! tokens[K]            (one bulk)      ->  tokens
//! K-row verify forward (collectives)  <->  the same K-row verify forward
//! committed rows       (one word)      ->  trim to kgamma_committed_len, commit the state
//! ```
//!
//! The committed count includes the anchor row (`num_accepted + 1`, in `1..=K`). Rank 0 sends it
//! before it emits any token, because an emit can finish the sequence.

use anyhow::{Result, bail};

/// 2026-10-08: EP worker command: the DFlash K=γ verify of one sequence (see the module doc).
pub const EP_CMD_VERIFY_KGAMMA: u32 = 0xFFFF_FFF6;

/// 2026-10-08: The widest K=γ verify: the K=γ forward stages one LoRA routing word per row in
/// the 128-byte gap before its slot metadata (`verify_d.rs`), so K is at most 32.
pub const KGAMMA_MAX_ROWS: usize = 32;

/// 2026-10-08: `k` as a K=γ verify width, or an error for 0 or more than [`KGAMMA_MAX_ROWS`].
pub fn kgamma_width(k: u32) -> Result<usize> {
    let k = k as usize;
    if k == 0 || k > KGAMMA_MAX_ROWS {
        bail!("K=γ verify width {k} outside 1..={KGAMMA_MAX_ROWS}");
    }
    Ok(k)
}

/// 2026-10-08: The sequence length after committing `committed` of the `k` rows a verify
/// appended (`seq_len_after` includes all `k`). Errors when `committed` is 0 or above `k`, or
/// when the verify appended fewer than `k` positions.
pub fn kgamma_committed_len(seq_len_after: usize, k: usize, committed: u32) -> Result<usize> {
    let committed = committed as usize;
    if committed == 0 || committed > k {
        bail!(
            "K=γ verify verdict: {committed} committed rows of {k} (the anchor row is always kept)"
        );
    }
    let Some(pre) = seq_len_after.checked_sub(k) else {
        bail!("K=γ verify verdict: seq_len {seq_len_after} is shorter than the {k} verified rows");
    };
    Ok(pre + committed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-08: Rank 0 keeps `pre_verify_len + num_accepted + 1` positions
    /// (`verify_dflash_step.rs`); the worker's length from the committed count is the same.
    #[test]
    fn the_worker_keeps_what_rank_zero_keeps() {
        let (pre, k) = (100usize, 9usize);
        for num_accepted in 0..k {
            let committed = (num_accepted + 1) as u32;
            assert_eq!(
                kgamma_committed_len(pre + k, k, committed).unwrap(),
                pre + num_accepted + 1
            );
        }
    }

    #[test]
    fn an_impossible_verdict_or_width_is_refused() {
        assert!(kgamma_committed_len(109, 9, 0).is_err());
        assert!(kgamma_committed_len(109, 9, 10).is_err());
        assert!(kgamma_committed_len(5, 9, 3).is_err());
        assert!(kgamma_width(0).is_err());
        assert!(kgamma_width(33).is_err());
        assert_eq!(kgamma_width(32).unwrap(), 32);
    }
}
