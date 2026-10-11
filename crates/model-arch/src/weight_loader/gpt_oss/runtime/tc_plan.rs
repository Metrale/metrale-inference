// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Complete expert grouping and inverse slot-major permutation for TC diagnostics.
use super::expert_plan::ExpertTokenPlan;
use anyhow::{Result, ensure};
pub(super) struct TcPlan {
    pub offsets: Vec<u32>,
    pub gate_rows: Vec<u32>,
    pub down_rows: Vec<u32>,
    pub inverse: Vec<u32>,
    pub max_rows: u32,
}
impl TcPlan {
    pub fn new(ids: &[u32]) -> Result<Self> {
        let _validated = ExpertTokenPlan::new(ids)?;
        let tokens = ids.len() / 4;
        let mut offsets = vec![0];
        let mut gate_rows = Vec::with_capacity(ids.len());
        let mut down_rows = Vec::with_capacity(ids.len());
        let mut inverse = vec![u32::MAX; ids.len()];
        let mut max_rows = 0;
        for expert in 0..32 {
            let start = gate_rows.len();
            for (entry, &selected) in ids.iter().enumerate() {
                if selected != expert {
                    continue;
                }
                let token = entry / 4;
                let slot = entry % 4;
                let slot_major = slot * tokens + token;
                ensure!(inverse[slot_major] == u32::MAX, "TC repeated token slot");
                inverse[slot_major] = gate_rows.len() as u32;
                gate_rows.push(token as u32);
                down_rows.push(slot_major as u32);
            }
            max_rows = max_rows.max((gate_rows.len() - start) as u32);
            offsets.push(gate_rows.len() as u32);
        }
        ensure!(
            gate_rows.len() == ids.len() && inverse.iter().all(|&v| v < ids.len() as u32),
            "TC incomplete token slot coverage"
        );
        Ok(Self {
            offsets,
            gate_rows,
            down_rows,
            inverse,
            max_rows,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouping_gathers_and_inverts_distinct_per_token_slots() {
        for tokens in [1, 2, 15, 16, 17, 64, 127, 128] {
            let ids: Vec<_> = (0..tokens)
                .flat_map(|t| [31, (3 * t) % 31, (3 * t + 7) % 31, (3 * t + 19) % 31])
                .collect();
            let p = TcPlan::new(&ids).unwrap();
            assert_eq!(p.offsets.len(), 33);
            assert_eq!(p.offsets[32], 4 * tokens);
            assert_eq!(p.max_rows, tokens);
            for e in 0..32 {
                for packed in p.offsets[e]..p.offsets[e + 1] {
                    let slot_major = p.down_rows[packed as usize] as usize;
                    let token = slot_major % tokens as usize;
                    let slot = slot_major / tokens as usize;
                    assert_eq!(ids[token * 4 + slot], e as u32);
                    assert_eq!(p.gate_rows[packed as usize], token as u32);
                    assert_eq!(p.inverse[slot_major], packed);
                }
            }
            let mut sorted = p.inverse.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..4 * tokens).collect::<Vec<_>>());
            if tokens > 1 {
                assert_ne!(p.gate_rows, p.down_rows);
            }
        }
        for bad in [
            vec![],
            vec![0, 1, 2],
            vec![0, 0, 1, 2],
            vec![0, 1, 2, 32],
            vec![0, 1, 2, u32::MAX],
            (0..129).flat_map(|_| [0, 1, 2, 3]).collect(),
        ] {
            assert!(TcPlan::new(&bad).is_err());
        }
    }
}
