// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Drafts left on the device (`METRALE_GLM_MTP_DEVICE_STEP`, `DEVICE_DRAFT`) at the
//! start of an MTP step: kept for the one route that verifies them from the device, read back
//! for every other.
//!
//! Owner: scheduler.
//! Invariants:
//! - Only a lone, grammarless sequence on the serial MTP path keeps a device draft: its K=2
//!   verify reads row 1 from the device. A batch, a prompt-lookup copy, a grammar truncation or a
//!   DFlash step reads drafts on the host, so they get the id first.
//! - A draft that cannot be read back is dropped (the sequence bootstraps a new one): the
//!   sentinel is never verified as a token.

use metrale_model_layers::speculative::DEVICE_DRAFT;

use super::*;

/// 2026-10-11: Whether a sequence's drafts are a device draft.
pub(super) fn holds_device_draft(drafts: &[u32]) -> bool {
    drafts.first() == Some(&DEVICE_DRAFT)
}

/// 2026-10-11: Whether this step may keep a device draft: one sequence, no prompt lookup, not
/// DFlash, no grammar.
pub(super) fn keeps_device_draft(
    active: usize,
    prompt_lookup: bool,
    dflash: bool,
    grammar: bool,
) -> bool {
    active == 1 && !prompt_lookup && !dflash && !grammar
}

/// 2026-10-11: Read back every device draft this step does not keep ([`keeps_device_draft`]).
pub(super) fn resolve_device_drafts(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    prompt_lookup: bool,
    dflash: bool,
) {
    let n = active.len();
    for a in active.iter_mut() {
        if !holds_device_draft(&a.pending_drafts)
            || keeps_device_draft(n, prompt_lookup, dflash, a.grammar_state.is_some())
        {
            continue;
        }
        match model.read_device_draft(&mut a.seq) {
            Ok(id) => a.pending_drafts = vec![id],
            Err(e) => {
                tracing::warn!("device draft read-back failed; the draft is dropped: {e:#}");
                a.pending_drafts.clear();
                a.pending_draft_conf.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: The sentinel in the first slot marks a device draft; a host draft (any real
    /// id, 0 included) and no draft do not.
    #[test]
    fn only_the_sentinel_marks_a_device_draft() {
        assert!(holds_device_draft(&[DEVICE_DRAFT]));
        assert!(!holds_device_draft(&[0]));
        assert!(!holds_device_draft(&[154_855]));
        assert!(!holds_device_draft(&[]));
    }

    /// 2026-10-11: A lone grammarless MTP sequence without prompt lookup keeps it; a batch, a
    /// prompt-lookup serve, DFlash or a grammar read it back.
    #[test]
    fn only_a_lone_plain_mtp_sequence_keeps_its_device_draft() {
        assert!(keeps_device_draft(1, false, false, false));
        assert!(!keeps_device_draft(2, false, false, false), "a batch");
        assert!(!keeps_device_draft(1, true, false, false), "prompt lookup");
        assert!(!keeps_device_draft(1, false, true, false), "DFlash");
        assert!(!keeps_device_draft(1, false, false, true), "a grammar");
    }
}
