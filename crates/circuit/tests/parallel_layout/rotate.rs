// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Pricing a hypothetical `Remainder::Rotate { offset }` on a deployment: which
//! weight node each split part slices (found by replicating the part alone and seeing which of
//! rank 0's weight reads change), then, per segment between two collectives, the best per-part
//! rotation of the slices ([`rotate_floor`]) against today's placement. Priced only: nothing
//! here changes a layout.
//!
//! Owner: metrale-circuit tests.
//! Invariants: every part priced splits over the whole world (its slice index is the rank).

use std::collections::BTreeMap;

use metrale_circuit::hardware::estimate::main_weight_bytes;
use metrale_circuit::ir::{Circuit, OpKind};
use metrale_circuit::parallel_cost::{rotate_floor, segment_of_nodes};

/// 2026-10-23: The rotation's price on a layout.
pub struct Priced {
    /// 2026-10-23: Σ over segments of the slowest rank's weight time, today.
    pub today_us: f64,
    /// 2026-10-23: The same with each segment's best rotation.
    pub rotated_us: f64,
    /// 2026-10-23: Per closing block: segments, segments a rotation improves, us it saves.
    pub per_block: BTreeMap<String, (usize, usize, f64)>,
}

/// 2026-10-23: Rank 0's weight bytes by node id.
pub fn bytes_by_id(c: &Circuit) -> BTreeMap<String, f64> {
    let b = main_weight_bytes(c).unwrap();
    c.nodes
        .iter()
        .zip(b)
        .filter(|(_, x)| *x > 0.0)
        .map(|(n, x)| (n.id.clone(), x))
        .collect()
}

/// 2026-10-23: Node id to the part that slices it: a node whose rank-0 bytes change when
/// `replicated[part]` (that part alone at degree 1) is instantiated.
pub fn attribute(
    today: &Circuit,
    replicated: &BTreeMap<String, Circuit>,
) -> Result<BTreeMap<String, String>, String> {
    let now = bytes_by_id(today);
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    for (part, c) in replicated {
        let alt = bytes_by_id(c);
        for (id, b) in &now {
            if alt.get(id).is_some_and(|a| a != b)
                && let Some(prev) = out.insert(id.clone(), part.clone())
            {
                return Err(format!(
                    "node `{id}` is sliced by both `{prev}` and `{part}`"
                ));
            }
        }
    }
    Ok(out)
}

/// 2026-10-23: Price the rotation over `today`'s ranks at `dram_gbps`.
pub fn price(
    today: &[Circuit],
    parts_of: &BTreeMap<String, String>,
    dram_gbps: f64,
) -> Result<Priced, String> {
    let n = today.len();
    let to_us = |b: f64| b / (dram_gbps * 1e3);
    // 2026-10-23: segment -> (base per rank, part -> per rank)
    type Seg = (Vec<f64>, BTreeMap<String, Vec<f64>>);
    let mut segs: Vec<Seg> = Vec::new();
    let mut closing: Vec<String> = Vec::new();
    for (r, c) in today.iter().enumerate() {
        let bytes = main_weight_bytes(c)?;
        let seg = segment_of_nodes(c);
        let ends: Vec<String> = c
            .nodes
            .iter()
            .filter(|x| {
                matches!(
                    x.op,
                    OpKind::AllReduce | OpKind::AllGather | OpKind::Broadcast
                )
            })
            .map(|x| x.block.clone())
            .chain(std::iter::once("(tail)".to_string()))
            .collect();
        if r == 0 {
            segs = vec![(vec![0.0; n], BTreeMap::new()); ends.len()];
            closing = ends;
        } else if ends != closing {
            return Err(format!("rank {r}: another segment sequence than rank 0"));
        }
        for (i, node) in c.nodes.iter().enumerate() {
            if bytes[i] == 0.0 {
                continue;
            }
            let s = &mut segs[seg[i]];
            match parts_of.get(&node.id) {
                Some(p) => s.1.entry(p.clone()).or_insert_with(|| vec![0.0; n])[r] += bytes[i],
                None => s.0[r] += bytes[i],
            }
        }
    }
    let mut p = Priced {
        today_us: 0.0,
        rotated_us: 0.0,
        per_block: BTreeMap::new(),
    };
    for (k, ((base, parts), block)) in segs.iter().zip(&closing).enumerate() {
        if base.iter().any(|b| *b != base[0]) {
            return Err(format!(
                "segment {k} (`{block}`): its unattributed reads differ across ranks {base:?}: a sliced node is attributed to no part"
            ));
        }
        // 2026-10-23: A debug dump, chosen when the test compiles (`ROTATE_DUMP=1 cargo test`):
        // this pure crate's tests read no process environment (`env-ban`).
        if option_env!("ROTATE_DUMP").is_some() {
            eprintln!("seg {k} {block} base {:.0} parts {:?}", base[0], parts);
        }
        let parts: Vec<Vec<f64>> = parts.values().cloned().collect();
        let now = (0..n)
            .map(|r| base[r] + parts.iter().map(|q| q[r]).sum::<f64>())
            .fold(f64::MIN, f64::max);
        let (best, _) = rotate_floor(base, &parts, 3u64.pow(10))?;
        let e = p.per_block.entry(block.clone()).or_insert((0, 0, 0.0));
        e.0 += 1;
        e.1 += usize::from(best < now);
        e.2 += to_us(now - best);
        p.today_us += to_us(now);
        p.rotated_us += to_us(best);
    }
    Ok(p)
}
