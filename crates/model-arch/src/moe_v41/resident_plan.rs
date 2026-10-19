// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The host half of the resident-expert MoE: from one call's routing, the expert-sorted
//! row plan the grouped pointer-table GEMMs read, and the single upload that carries it.
//!
//! A routed (token, expert) pair is a "position". Positions are grouped by ascending expert id and,
//! within an expert, by token order; `offsets` bounds each active expert's run. `pos[t * topk + j]`
//! is token `t`'s `j`-th position in ascending expert order, the order the reference adds the
//! experts' outputs in.
//!
//! Owner: model-arch, DeepSeek-V4.1 MoE.
//! Invariants:
//! - The blob layout depends only on the capacities (`n_routed`, `max_pos`), so its device
//!   addresses are fixed for the life of the scratch.

use anyhow::{Result, ensure};

/// 2026-10-19: One call's expert-sorted plan.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidentPlan {
    /// 2026-10-19: The active experts, ascending.
    pub experts: Vec<u32>,
    /// 2026-10-19: `experts.len() + 1` cumulative position counts.
    pub offsets: Vec<i32>,
    /// 2026-10-19: The token of each position.
    pub token_of_pos: Vec<i32>,
    /// 2026-10-19: The routing weight of each position.
    pub weight_of_pos: Vec<f32>,
    /// 2026-10-19: `[m * topk]`: each token's positions in ascending expert order.
    pub pos: Vec<i32>,
    /// 2026-10-19: The most positions any one expert holds.
    pub max_rows: usize,
}

impl ResidentPlan {
    pub fn n_pos(&self) -> usize {
        self.token_of_pos.len()
    }
}

/// 2026-10-19: Plan `m` tokens' routing (`indices`, `weights`: `[m, topk]` each, the selection's
/// order). Refuses an expert id outside `n_routed` and a token that picks one expert twice.
pub fn resident_plan(
    indices: &[usize],
    weights: &[f32],
    m: usize,
    topk: usize,
    n_routed: usize,
) -> Result<ResidentPlan> {
    ensure!(
        indices.len() == m * topk && weights.len() == m * topk,
        "routing of {} indices and {} weights for {m} tokens of top-{topk}",
        indices.len(),
        weights.len()
    );
    let mut count = vec![0usize; n_routed];
    for (i, &e) in indices.iter().enumerate() {
        ensure!(
            e < n_routed,
            "token {} routed to expert {e} of {n_routed}",
            i / topk
        );
        count[e] += 1;
    }
    for t in 0..m {
        let picks = &indices[t * topk..(t + 1) * topk];
        for (j, e) in picks.iter().enumerate() {
            ensure!(
                !picks[..j].contains(e),
                "token {t} routed to expert {e} twice"
            );
        }
    }
    let experts: Vec<u32> = (0..n_routed)
        .filter(|&e| count[e] > 0)
        .map(|e| e as u32)
        .collect();
    let mut start = vec![0usize; n_routed];
    let mut offsets = Vec::with_capacity(experts.len() + 1);
    let mut acc = 0usize;
    offsets.push(0);
    for &e in &experts {
        start[e as usize] = acc;
        acc += count[e as usize];
        offsets.push(acc as i32);
    }
    let mut token_of_pos = vec![0i32; acc];
    let mut weight_of_pos = vec![0f32; acc];
    let mut fill = start.clone();
    let mut pos = vec![0i32; m * topk];
    for t in 0..m {
        let mut mine: Vec<(usize, usize)> = (0..topk).map(|j| (indices[t * topk + j], j)).collect();
        mine.sort_unstable();
        for (slot, &(e, j)) in mine.iter().enumerate() {
            let p = fill[e];
            fill[e] += 1;
            token_of_pos[p] = t as i32;
            weight_of_pos[p] = weights[t * topk + j];
            pos[t * topk + slot] = p as i32;
        }
    }
    Ok(ResidentPlan {
        max_rows: experts
            .iter()
            .map(|&e| count[e as usize])
            .max()
            .unwrap_or(0),
        experts,
        offsets,
        token_of_pos,
        weight_of_pos,
        pos,
    })
}

/// 2026-10-19: Per active expert, the device addresses of its three projections' packed weights
/// and scales: w1, w3 (`[inter, dim]`) and w2 (`[dim, inter]`).
pub const PTR_TABLES: usize = 6;

/// 2026-10-19: Byte offsets of the blob's arrays: six `[n_routed]` u64 pointer tables, the
/// `[n_routed]` f32 second-level scales (unused by MXFP4, read by the kernel), the
/// `[n_routed + 1]` i32 offsets, then `[max_pos]` i32 tokens, f32 weights and i32 positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlobLayout {
    pub n_routed: usize,
    pub max_pos: usize,
}

impl BlobLayout {
    pub fn ptrs(&self, table: usize) -> usize {
        table * self.n_routed * 8
    }
    pub fn scale2(&self) -> usize {
        PTR_TABLES * self.n_routed * 8
    }
    pub fn offsets(&self) -> usize {
        self.scale2() + self.n_routed * 4
    }
    pub fn tokens(&self) -> usize {
        self.offsets() + (self.n_routed + 1) * 4
    }
    pub fn weights(&self) -> usize {
        self.tokens() + self.max_pos * 4
    }
    pub fn pos(&self) -> usize {
        self.weights() + self.max_pos * 4
    }
    pub fn bytes(&self) -> usize {
        self.pos() + self.max_pos * 4
    }

    /// 2026-10-19: The blob for `plan`; `addr(table, expert)` is that expert's device address
    /// for one of the [`PTR_TABLES`] tables. Refuses a plan past the capacities.
    pub fn encode(&self, plan: &ResidentPlan, addr: impl Fn(usize, u32) -> u64) -> Result<Vec<u8>> {
        ensure!(
            plan.experts.len() <= self.n_routed && plan.n_pos() <= self.max_pos,
            "plan of {} experts and {} positions exceeds {} experts and {} positions",
            plan.experts.len(),
            plan.n_pos(),
            self.n_routed,
            self.max_pos
        );
        let mut b = vec![0u8; self.bytes()];
        let mut put = |at: usize, v: &[u8]| b[at..at + v.len()].copy_from_slice(v);
        for table in 0..PTR_TABLES {
            for (i, &e) in plan.experts.iter().enumerate() {
                put(self.ptrs(table) + i * 8, &addr(table, e).to_le_bytes());
            }
        }
        let words = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        put(self.offsets(), &words(&plan.offsets));
        put(self.tokens(), &words(&plan.token_of_pos));
        let w: Vec<u8> = plan
            .weight_of_pos
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        put(self.weights(), &w);
        put(self.pos(), &words(&plan.pos));
        Ok(b)
    }
}
