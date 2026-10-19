// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-19: Prints the expert-union curve of the CPU reference's prefill routes
//! (`deepseek_v41_ref_cpu` writes `routes.u16`): for each concurrency `C` and draft count `k`,
//! the mean distinct routed experts one layer reads, next to uniform routing's count.
//!
//! Inputs: `METRALE_DSV41_CKPT` (the snapshot directory, for the layer, expert and top-k
//! counts) and `METRALE_DSV41_REF_OUT` (the reference's output directory).

use std::path::PathBuf;

use metrale_model_arch::deepseek_v41_ref::routes::{Routes, uniform_union};

/// 2026-10-19: The concurrency rungs and draft counts the curve is printed at.
const CONCURRENCY: [usize; 6] = [1, 2, 4, 8, 12, 16];
const DRAFTS: [usize; 4] = [0, 1, 3, 7];

#[test]
#[ignore = "needs METRALE_DSV41_CKPT and METRALE_DSV41_REF_OUT"]
fn expert_union_curve_of_the_reference_routes() {
    let env = |k: &str| PathBuf::from(std::env::var(k).unwrap_or_else(|_| panic!("{k} unset")));
    let (dir, out) = (env("METRALE_DSV41_CKPT"), env("METRALE_DSV41_REF_OUT"));
    let cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let mc = metrale_config::parse_config(&cfg).expect("engine config");
    let (layers, topk, experts) = (mc.num_hidden_layers, mc.num_experts_per_tok, mc.num_experts);
    let bytes = std::fs::read(out.join("routes.u16")).expect("routes.u16");
    let ids: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    let tokens = ids.len() / (layers * topk);
    let routes = Routes::new(layers, tokens, topk, experts, ids).expect("routes");
    println!("tokens {tokens} layers {layers} topk {topk} experts {experts}");
    for c in CONCURRENCY {
        for k in DRAFTS {
            let Ok(u) = routes.union_mean(c, k) else {
                continue;
            };
            let rows = c * (k + 1);
            let flat = uniform_union(experts, topk, rows);
            println!(
                "C {c} k {k} rows {rows} union {u:.1} uniform {flat:.1} ratio {:.3}",
                u / flat
            );
        }
    }
}
