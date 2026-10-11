// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Ladders as derived priors. A static draft ladder is what the controller's choice
//! gives for one acceptance prior and one cost model at each width; [`derive_cold_start`]
//! computes it, so a ladder need not be a hand-set recipe table. An explicit ladder
//! (`--mtp-k-ladder`, `--dflash-draft-ladder`) stays an operator override, read through the
//! same [`RungTable`].
//!
//! Owner: speculative.
//! Invariants: the derived table answers, at every width it was derived at, exactly the
//! controller's choice there.

pub use metrale_model_layers::speculative::RungTable;

use super::chain::expected_tokens;
use super::cost::CostModel;
use super::decide::{Candidate, Margins, Objective, choose};

/// 2026-10-10: The ladder the controller would run with no live data: at each width in
/// `widths` (ascending), the choice over depths `0..=k_cap` for `n` streams at the conditional
/// `prior` rates. Consecutive widths with the same choice share a rung. `None` when `widths`
/// is empty or a width has no priced depth.
pub fn derive_cold_start(
    cost: &CostModel,
    prior: &[f64],
    obj: &Objective,
    k_cap: usize,
    widths: &[usize],
) -> Option<RungTable> {
    let mut rungs: Vec<(usize, usize)> = Vec::new();
    for &n in widths {
        let cands: Vec<Candidate> = (0..=k_cap)
            .filter_map(|k| {
                let e = expected_tokens(prior, k);
                let cost = cost.cost(n, k)?;
                Some(Candidate {
                    k,
                    tokens: n as f64 * e,
                    slowest: e,
                    cost,
                })
            })
            .collect();
        let k = choose(obj, &Margins::NONE, None, &cands)?;
        if rungs.last().map(|r| r.1) != Some(k) {
            rungs.push((n, k));
        }
    }
    RungTable::from_lower_bounds(rungs)
}

#[cfg(test)]
mod tests {
    use super::super::calib::Calibration;
    use super::super::cost::{CostSource, StepTable};
    use super::*;

    /// 2026-10-10: A width-independent step table: the derived ladder is one rung at the
    /// single-stream optimum; a poorer prior derives plain decode.
    #[test]
    fn a_flat_cost_derives_one_rung_at_the_optimum() {
        let cost = CostModel {
            source: CostSource::StepTable(StepTable::parse("0:32,1:44,2:50,3:56").unwrap()),
            calib: Calibration::new(0.0),
        };
        let t = Objective::Throughput;
        let l = derive_cold_start(&cost, &[0.8], &t, 3, &[1, 2, 4, 8]).unwrap();
        assert_eq!(l.rungs(), &[(1, 3)]);
        let l = derive_cold_start(&cost, &[0.1], &t, 3, &[1, 2, 4, 8]).unwrap();
        assert_eq!(l.rungs(), &[(1, 0)]);
        assert!(derive_cold_start(&cost, &[0.8], &t, 3, &[]).is_none());
    }
}
