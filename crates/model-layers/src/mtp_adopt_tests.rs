// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tests for `mtp_adopt`: the capture record's bookkeeping and the adoption claim,
//! walked through the request orders the model drives.
//!
//! Owner: model-layers (MTP drafter).
//! Invariants: none beyond the types.

use super::{AdoptClaim, CaptureRecord, claims_adoption};

/// 2026-10-14: The 1k probe: a 1073-token prompt whose hit matches 1072 rows.
fn probe_prompt() -> Vec<u32> {
    (0..1073u32)
        .map(|i| i.wrapping_mul(2_654_435_761) % 151_552)
        .collect()
}

/// 2026-10-14: A record left by the consumed miss of `prompt`.
fn after_miss(prompt: &[u32]) -> CaptureRecord {
    let mut r = CaptureRecord::default();
    r.note_write(0);
    r.note_consumed(prompt);
    r
}

/// 2026-10-14: The probe's order: the miss consumes, the hit writes row 1072 under its claim,
/// and the claim is covered, so the hit's drafter is prefilled over the whole prompt.
#[test]
fn a_hit_after_its_own_miss_is_covered() {
    let p = probe_prompt();
    let mut r = after_miss(&p);
    assert!(claims_adoption(true, false, 1072, r.tokens.len()));
    r.note_write(1072);
    r.note_claim(7, 1072);
    assert!(r.covers(7, &p));
    r.note_consumed(&p);
    assert_eq!(r.tokens, p);
    assert_eq!(r.adopted, None);
}

/// 2026-10-14: The negative control: one changed token inside the matched prefix (a different
/// prompt whose hiddens the buffer holds) refuses the claim; one changed token past it does not.
#[test]
fn a_record_of_other_tokens_does_not_cover_the_claim() {
    let p = probe_prompt();
    let mut other = p.clone();
    other[500] ^= 1;
    let mut r = after_miss(&other);
    r.note_write(1072);
    r.note_claim(7, 1072);
    assert!(!r.covers(7, &p));

    let mut tail_differs = p.clone();
    tail_differs[1072] ^= 1;
    let mut r = after_miss(&tail_differs);
    r.note_write(1072);
    r.note_claim(7, 1072);
    assert!(r.covers(7, &p), "row 1072 is the hit's own write");
}

/// 2026-10-14: Another sequence's capture between the miss and the hit (its chunk 0 writes from
/// row 0) empties the record, so the hit's claim is refused.
#[test]
fn an_intervening_capture_from_row_zero_refuses_the_claim() {
    let p = probe_prompt();
    let mut r = after_miss(&p);
    r.note_write(0);
    r.note_write(1072);
    r.note_claim(9, 1072);
    assert!(!r.covers(9, &p));
}

/// 2026-10-14: A write that overlaps the matched rows (a carried warm turn writing from row 300)
/// cuts the record there, so a claim matching 1072 rows is refused and one matching 300 is not.
#[test]
fn a_write_inside_the_matched_rows_cuts_the_record() {
    let p = probe_prompt();
    let mut r = after_miss(&p);
    r.note_write(300);
    assert_eq!(r.tokens.len(), 300);
    r.note_claim(3, 1072);
    assert!(!r.covers(3, &p));
    r.note_claim(4, 300);
    assert!(r.covers(4, &p));
}

/// 2026-10-14: A hit whose matched prefix is longer than the consumed prompt (a later turn whose
/// match runs into decoded tokens, which the capture never holds) is refused.
#[test]
fn a_match_past_the_recorded_rows_is_refused() {
    let p = probe_prompt();
    let mut r = after_miss(&p[..800]);
    r.note_write(1072);
    r.note_claim(5, 1072);
    assert!(!r.covers(5, &p));
}

/// 2026-10-14: A capture that is not an adoption claim is covered whatever the record holds, so
/// the cold path decides as before; so is a claim under an older generation.
#[test]
fn a_capture_that_is_not_the_current_claim_is_always_covered() {
    let p = probe_prompt();
    let mut r = CaptureRecord::default();
    assert!(r.covers(1, &p));
    r.adopted = Some(AdoptClaim {
        generation: 2,
        matched: 1072,
    });
    assert!(!r.covers(2, &p));
    assert!(r.covers(3, &p));
}

/// 2026-10-14: A claim is drawn only with the lever on, without the drafter carry, for a write
/// past row 0, and while the record holds the rows below it.
#[test]
fn only_the_lever_without_the_carry_claims_a_write_the_record_reaches() {
    for adopt in [false, true] {
        for carry in [false, true] {
            for start in [0usize, 1, 1072] {
                for recorded in [0usize, 1, 1071, 1072, 1073] {
                    assert_eq!(
                        claims_adoption(adopt, carry, start, recorded),
                        adopt && !carry && start > 0 && recorded >= start,
                        "adopt {adopt} carry {carry} start {start} recorded {recorded}"
                    );
                }
            }
        }
    }
    assert!(!claims_adoption(true, false, 1072, 1071));
    assert!(claims_adoption(true, false, 1072, 1072));
}
