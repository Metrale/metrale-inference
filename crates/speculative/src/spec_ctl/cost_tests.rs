// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Cost sources, the layering order, and the calibration.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use super::super::source::DraftCost;
use super::*;

#[test]
fn the_step_table_parses_consecutive_positive_entries_only() {
    let t = StepTable::parse("0:32, 1:44,2:50").unwrap();
    assert_eq!(
        (t.max_drafts(), t.ms(0), t.ms(2), t.ms(3)),
        (2, Some(32.0), Some(50.0), None)
    );
    let t = StepTable::parse("1:44,2:50").unwrap();
    assert_eq!((t.ms(0), t.ms(1)), (None, Some(44.0)));
    for bad in [
        "",
        "0:32",
        "2:50",
        "0:32,2:50",
        "1:-1",
        "1:0",
        "1:x",
        "a:1",
        "1-44",
    ] {
        assert!(StepTable::parse(bad).is_err(), "{bad:?} accepted");
    }
}

#[test]
fn a_depth_ratio_prices_exactly_its_depths() {
    let r = CostSource::DepthRatio(DepthRatio::new(vec![(2, 1.32), (1, 1.0)]).unwrap());
    assert_eq!(r.cost(16, 2).unwrap().ms, 1.32);
    assert_eq!(r.cost(16, 3), None);
    assert_eq!(r.max_k(), Some(2));
    assert!(DepthRatio::new(vec![]).is_none());
    assert!(DepthRatio::new(vec![(1, 1.0), (1, 2.0)]).is_none());
    assert!(DepthRatio::new(vec![(1, 0.0)]).is_none());
}

/// 2026-10-10: The envelope prices verify rows by interpolation and adds the drafter's shape:
/// a block drafter's propose cost does not grow with depth, a chained one's does.
#[test]
fn the_envelope_interpolates_rows_and_adds_the_draft_shape() {
    let block = EnvelopeCurve {
        points: vec![(1, 28.0, 2.0), (5, 48.0, 3.0), (8, 57.0, 4.0)],
        draft: DraftCost::PerBlock { ms: 10.0, j: 0.5 },
    };
    let e = CostSource::Envelope(block.clone());
    assert_eq!(e.cost(1, 0).unwrap().ms, 28.0);
    assert_eq!(e.cost(1, 2).unwrap().ms, 38.0 + 10.0);
    assert_eq!(e.cost(1, 4).unwrap().ms, 48.0 + 10.0);
    assert_eq!(e.cost(2, 3).unwrap().ms, 57.0 + 10.0, "8 rows");
    assert_eq!(
        e.cost(4, 3).unwrap().ms,
        57.0 + 10.0,
        "past the last point: the last"
    );
    let chained = CostSource::Envelope(EnvelopeCurve {
        draft: DraftCost::PerDraft { ms: 5.0, j: 0.2 },
        ..block
    });
    assert_eq!(chained.cost(1, 4).unwrap().ms, 48.0 + 20.0);
    assert_eq!(chained.max_k(), None);
}

/// 2026-10-10: Measured beats the envelope beats the cold-start prior; every skipped layer's
/// reason is kept; no source at all is an error, never an invented cost.
#[test]
fn layering_is_measured_then_envelope_then_cold_start_with_reasons() {
    let cold = || {
        Some(CostSource::StepTable(
            StepTable::parse("0:30,1:40").unwrap(),
        ))
    };
    let env = EnvelopeCurve {
        points: vec![(1, 1.0, 1.0)],
        draft: DraftCost::Free,
    };
    let (_, p) = layer(None, Ok(env), cold()).unwrap();
    assert_eq!(p, Provenance::Envelope);
    let (s, p) = layer(None, Err("envelope plan not wired".into()), cold()).unwrap();
    assert!(matches!(s, CostSource::StepTable(_)));
    assert_eq!(
        p,
        Provenance::ColdStart {
            skipped: vec![
                "no measured spec-cost table".into(),
                "envelope plan not wired".into()
            ]
        }
    );
    let err = layer(None, Err("envelope plan not wired".into()), None).unwrap_err();
    assert!(err.contains("envelope plan not wired"), "{err}");
}

/// 2026-10-10: Calibration off leaves source costs bit for bit; on, it scales per width
/// bucket and leaves other buckets alone.
#[test]
fn calibration_is_per_width_bucket_and_off_at_alpha_zero() {
    let src = CostSource::StepTable(StepTable::parse("0:30.3,1:40.7").unwrap());
    let mut off = CostModel {
        source: src.clone(),
        calib: Calibration::new(0.0),
    };
    off.observe(1, 1, 99.0, None);
    assert_eq!(off.cost(1, 1).unwrap().ms.to_bits(), 40.7f64.to_bits());
    let mut on = CostModel {
        source: src,
        calib: Calibration::new(0.5),
    };
    on.observe(4, 1, 81.4, None);
    assert!(
        (on.cost(3, 1).unwrap().ms - 40.7 * 1.5).abs() < 1e-9,
        "3 shares 4's bucket"
    );
    assert_eq!(on.cost(1, 1).unwrap().ms, 40.7);
    assert_eq!(on.calib.samples(4), 1);
}
