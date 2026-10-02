// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: KV admission never binds on the certified gates that run more than one request at
//! a time: at each gate's widest rung, that many requests, each reserving the full
//! `--max-seq-len` (more than any request of the gate's shape reserves), are admitted at once
//! into the pool its serve builds. The pools are the KV blocks the gate serves booted with on
//! GB10 on 2026-10-01 under the reserve plan of `serve_phases/preflight/reserve_plan.rs`. A gate
//! at C=1 cannot bind: a lone request is always admitted
//! (`liveness_oversized_lone_request_still_admits`).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-10-01: One gate's serve: its KV pool, `--max-seq-len`, block size and widest rung.
struct GateServe {
    gate: &'static str,
    kv_blocks: usize,
    max_seq_len: usize,
    block_size: usize,
    widest_rung: usize,
}

/// 2026-10-01: The gates with a rung above C=1 (kernels/gb10 BENCH.toml), with the KV blocks
/// their serves booted with (main before this change in brackets): concurrency-sweep on the
/// throughput recipe (57,088 [55,393]), concurrency-sweep-moe on the nvfp4-head MoE serve at
/// 128 slots (109,132 [111,075]), concurrency-sweep-dflash2 on the DFlash2 recipe (14,900
/// [14,240]).
const GATES: [GateServe; 3] = [
    GateServe {
        gate: "concurrency-sweep",
        kv_blocks: 57_088,
        max_seq_len: 2048,
        block_size: 16,
        widest_rung: 128,
    },
    GateServe {
        gate: "concurrency-sweep-moe",
        kv_blocks: 109_132,
        max_seq_len: 2048,
        block_size: 16,
        widest_rung: 16,
    },
    GateServe {
        gate: "concurrency-sweep-dflash2",
        kv_blocks: 14_900,
        max_seq_len: 4096,
        block_size: 16,
        widest_rung: 16,
    },
];

#[test]
fn every_request_of_the_widest_rung_is_admitted_at_once() {
    for g in &GATES {
        // 2026-10-01: A prompt of the full length with output to spare: the reservation is
        // clamped to `--max-seq-len`, the most any request can reserve. The watermark is its
        // default, `--max-seq-len` (`resolve_admit_watermark` with the variable unset).
        let reqs = vec![(g.max_seq_len, g.max_seq_len); g.widest_rung];
        let (n, forced) = admit_count(
            g.kv_blocks,
            0,
            &reqs,
            g.max_seq_len,
            g.max_seq_len,
            g.block_size,
        );
        assert_eq!((n, forced), (g.widest_rung, false), "{}", g.gate);
    }
}

#[test]
fn the_bound_is_tight_enough_to_fail() {
    // 2026-10-01: The same check refuses a pool one full-length sequence short of the rung, so
    // it is not vacuous.
    for g in &GATES {
        let full = blocks_for_tokens(g.max_seq_len, g.block_size);
        let reqs = vec![(g.max_seq_len, g.max_seq_len); g.widest_rung];
        let short = g.widest_rung * full - 1;
        let (n, _) = admit_count(short, 0, &reqs, g.max_seq_len, g.max_seq_len, g.block_size);
        assert_eq!(n, g.widest_rung - 1, "{}", g.gate);
    }
}
