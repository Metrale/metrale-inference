// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Piece mode: the lever's reading, the call plan against the `P`-chunked serve's
//! sub-chunks, and the load-time budget check.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: the lever reads as its replaced environment reader (`old_parse`, frozen here)
//! did, except the listed probes: a padded count is now read, and a count past `i64::MAX` is
//! refused.

use super::*;
use metrale_model_layers::lever_equivalence::settings_from;

/// 2026-10-17: The replaced `parse`, unchanged.
fn old_parse(v: Option<&str>) -> Result<Option<usize>, String> {
    let Some(s) = v else {
        return Ok(None);
    };
    let p: usize = s
        .parse()
        .map_err(|_| format!("{PIECE_ROWS_ENV}={s:?}: expected a row count"))?;
    if p == 0 || !p.is_multiple_of(PIECE_ALIGN) {
        return Err(format!(
            "{PIECE_ROWS_ENV}={p}: must be a positive multiple of {PIECE_ALIGN} (the KDA scan's \
             slab), or unset"
        ));
    }
    Ok(Some(p))
}

/// 2026-10-17: The lever at `probe` through the settings; `None` when the settings or the check
/// refuse it.
fn new_at(probe: Option<&str>) -> Option<Option<usize>> {
    let pairs: Vec<(&str, &str)> = probe.map(|v| (PIECE_ROWS_ENV, v)).into_iter().collect();
    settings_from(&pairs).ok().and_then(|s| piece_rows(&s).ok())
}

/// 2026-10-14: Unset is off; a positive multiple of 256 is taken; zero, an off-grid count and
/// text are refused. 2026-10-17: As the replaced reader, but for the listed probes.
#[test]
fn reads_as_before_except_padding_and_past_i64() {
    for probe in [
        None,
        Some("8192"),
        Some("256"),
        Some("+512"),
        Some("0"),
        Some("8208"),
        Some("64"),
        Some("-8192"),
        Some(""),
        Some("8k"),
        Some("1"),
        Some("true"),
        Some("off"),
        Some("garbage"),
    ] {
        assert_eq!(new_at(probe), old_parse(probe).ok(), "{probe:?}");
    }
    let past_i64 = (1u64 << 63).to_string();
    for (probe, was, now) in [
        (" 8192", None, Some(Some(8192))),
        ("256 ", None, Some(Some(256))),
        (past_i64.as_str(), Some(Some(1usize << 63)), None),
    ] {
        assert_eq!(
            (old_parse(Some(probe)).ok(), new_at(Some(probe))),
            (was, now),
            "{probe:?}"
        );
    }
    // 2026-10-17: A negative count resolves as settings and is refused here with the
    // positive-multiple text, where the old parse said "expected a row count".
    let s = settings_from(&[(PIECE_ROWS_ENV, "-8192")]).expect("an Int resolves");
    let err = format!(
        "{:#}",
        piece_rows(&s).expect_err("a negative count is refused")
    );
    assert!(err.contains("must be a positive multiple of 256"), "{err}");
}

fn plan(k: usize, p: usize, share: impl Fn(usize) -> Option<u8>) -> Vec<Vec<usize>> {
    plan_calls::<u8>(k, p, |n| Ok(share(n))).unwrap()
}

/// 2026-10-14: The serve's case: a 16384-row chunk at P 8192 is one call of two pieces, and the
/// last chunk of a 32000-row prompt is one call of 8192 + 7424.
#[test]
fn a_wide_chunk_is_one_call_of_p_pieces() {
    let all = |_: usize| Some(0u8);
    assert_eq!(plan(16384, 8192, all), vec![vec![8192, 8192]]);
    assert_eq!(plan(15616, 8192, all), vec![vec![8192, 7424]]);
    assert_eq!(plan(8192, 8192, all), vec![vec![8192]]);
    assert_eq!(plan(100, 8192, all), vec![vec![100]]);
}

/// 2026-10-14: A piece whose decisions differ from the call's first piece, or that would change
/// the call total's decisions, starts a call of its own; a `None` piece always runs alone.
#[test]
fn differing_decisions_split_the_call() {
    // 2026-10-14: SP-mHC from 192 rows: a 100-row last piece is not SP and runs alone.
    let sp = |n: usize| Some(u8::from(n >= 192));
    assert_eq!(plan(16484, 8192, sp), vec![vec![8192, 8192], vec![100]]);
    // 2026-10-14: The total's decision differs above 12000 rows: every piece runs alone.
    let total = |n: usize| Some(u8::from(n > 12000));
    assert_eq!(
        plan(24576, 8192, total),
        vec![vec![8192], vec![8192], vec![8192]]
    );
    let refused = |n: usize| (n != 4096).then_some(0u8);
    assert_eq!(
        plan(3 * 8192 + 4096, 8192, refused),
        vec![vec![8192, 8192, 8192], vec![4096]]
    );
    let off = |n: usize| (n <= 8192).then_some(0u8);
    assert_eq!(plan(16384, 8192, off), vec![vec![8192], vec![8192]]);
}

/// 2026-10-14: For assorted shapes and predicates, the calls flattened are exactly the
/// `P`-chunked serve's sub-chunks of the same rows: `P` rows from row 0, the last shorter.
#[test]
fn pieces_are_the_p_chunked_sub_chunks() {
    let preds: [&dyn Fn(usize) -> Option<u8>; 4] = [
        &|_| Some(0),
        &|n| Some(u8::from(n >= 192)),
        &|n| (n % 3 != 0).then_some(1),
        &|_| None,
    ];
    for k in [
        1usize, 255, 256, 8191, 8192, 8193, 16384, 16400, 32768, 40000,
    ] {
        for p in [256usize, 512, 8192] {
            for pred in preds {
                let calls = plan(k, p, pred);
                let flat: Vec<usize> = calls.iter().flatten().copied().collect();
                let mut want = Vec::new();
                let mut t = 0;
                while t < k {
                    want.push(p.min(k - t));
                    t += p;
                }
                assert_eq!(flat, want, "k {k} p {p}");
                assert!(calls.iter().all(|c| !c.is_empty()));
            }
        }
    }
}

/// 2026-10-14: A share error comes back unchanged; a zero piece is refused.
#[test]
fn errors_propagate() {
    let r = plan_calls::<u8>(16384, 8192, |_| Err(anyhow::anyhow!("boom")));
    assert_eq!(r.unwrap_err().to_string(), "boom");
    assert!(plan_calls::<u8>(10, 0, |_| Ok(Some(0))).is_err());
}

/// 2026-10-14: The load check: a budget that is a multiple of the piece and wider than it, with
/// `prefill_rows` covering the batch, passes; each other case is refused.
#[test]
fn budget_check() {
    assert_eq!(check_piece_budget(8192, 16400, 16384, 16400), Ok(()));
    assert_eq!(check_piece_budget(8192, 16400, 24576, 16400), Ok(()));
    assert!(check_piece_budget(8192, 8208, 8192, 8208).is_err());
    assert!(check_piece_budget(8192, 16400, 12288, 12304).is_err());
    assert!(check_piece_budget(8192, 8208, 16384, 16400).is_err());
    assert!(check_piece_budget(8192, 16384, 16384, 16400).is_err());
}
