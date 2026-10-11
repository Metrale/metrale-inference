// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The batched-verify catch-up: the drafter rows of each sequence's accepted drafts,
//! which no propose writes, from the verify rows the model stashed (`catchup_batch`).
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - Sequence i's accepted drafts `tokens[i]` sit at positions `first_pos[i]..`; the draft at
//!   position p pairs with the target hidden of position p - 1, which is verify row
//!   `p - first_pos[i]` and stash row `hiddens[i][p - first_pos[i]]`. So its pair key is p - 1
//!   and the rows are keys `first_pos[i] - 1..`, appended only where the dense row space ends
//!   (`rows_impl`), never over or past it.
//! - Drafter-only, and no collective: every rank runs it on the same stash rows and tokens.

use super::*;

/// 2026-10-11: `(window, row_base)` for one sequence, or `None`: the token window `rows_impl`
/// reads (a leading slot it skips, then the accepted drafts) and the first pair key. `None` for
/// no drafts, a first position of 0, or stash rows that are not one contiguous run.
pub(super) fn plan_catchup(
    tokens: &[u32],
    hiddens: &[DevicePtr],
    first_pos: usize,
    row_bytes: usize,
) -> Option<(Vec<u32>, usize)> {
    if tokens.is_empty() || hiddens.len() != tokens.len() {
        return None;
    }
    let base = hiddens[0];
    if hiddens
        .iter()
        .enumerate()
        .any(|(k, p)| *p != base.offset(k * row_bytes))
    {
        return None;
    }
    let row_base = first_pos.checked_sub(1)?;
    let window = std::iter::once(0).chain(tokens.iter().copied()).collect();
    Some((window, row_base))
}

impl Glm5NextMtpHead {
    /// 2026-10-11: Rows written over all sequences. A sequence whose drafter does not end at
    /// its first missing key writes nothing (`rows_impl`).
    pub(super) fn catchup_rows(
        &self,
        tokens: &[Vec<u32>],
        hiddens: &[Vec<DevicePtr>],
        first_pos: &[usize],
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let mut written = 0;
        for (i, st) in states.iter_mut().enumerate() {
            let (Some(t), Some(h), Some(&p)) = (tokens.get(i), hiddens.get(i), first_pos.get(i))
            else {
                continue;
            };
            let Some((window, row_base)) = plan_catchup(t, h, p, self.hidden * 2) else {
                continue;
            };
            let n = self.rows_impl(&window, h[0], row_base, &mut **st, ctx, stream)?;
            if n > 0 {
                super::hidden_half::note_applied("ACCEPTED_ROWS (batched verify)");
            }
            written += n;
        }
        Ok(written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROW: usize = 8192;

    fn rows(base: u64, n: usize) -> Vec<DevicePtr> {
        (0..n).map(|k| DevicePtr(base).offset(k * ROW)).collect()
    }

    /// 2026-10-11: Two accepted drafts at positions 41 and 42: keys 40 and 41, the window
    /// leading with a skipped slot so row r reads draft r.
    #[test]
    fn accepted_drafts_become_keys_from_first_pos_minus_one() {
        assert_eq!(
            plan_catchup(&[7, 9], &rows(0x1000, 2), 41, ROW),
            Some((vec![0, 7, 9], 40))
        );
    }

    /// 2026-10-11: Nothing to write, a position-0 draft (no key before it), and stash rows that
    /// are not one contiguous run (the per-row offsets `rows_impl` assumes) are refused.
    #[test]
    fn empty_position_zero_and_scattered_rows_are_refused() {
        assert_eq!(plan_catchup(&[], &[], 41, ROW), None);
        assert_eq!(plan_catchup(&[7], &rows(0x1000, 1), 0, ROW), None);
        let mut h = rows(0x1000, 2);
        h[1] = DevicePtr(0x9000);
        assert_eq!(plan_catchup(&[7, 9], &h, 41, ROW), None);
        assert_eq!(
            plan_catchup(&[7, 9], &rows(0x1000, 1), 41, ROW),
            None,
            "row count"
        );
    }
}
