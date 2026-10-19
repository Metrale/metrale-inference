// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: Every routed expert table lands in its pointer-table slot as exactly the middle
//! rank's rows (w1, w3 and their scales) or columns (w2 and its scales), read from a real file;
//! a table of the wrong shape or a missing one is refused. Two layers of two experts, width 64,
//! expert width 96 (rank 1 keeps 32..64).
//!
//! The GPU is `MockGpuBackend`, which holds the uploaded bytes in host memory: the reader's only
//! device work is allocations and uploads, and the test reads back exactly what landed.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::super::hf::tests::{back, pattern, shard};
use super::*;

const DIM: usize = 64;
const INTER: usize = 96;

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.hidden_size = DIM;
    c.moe_intermediate_size = INTER;
    c.num_experts = 2;
    c.num_hidden_layers = 2;
    c
}

fn local() -> Local {
    Local {
        world: 3,
        heads: 0..1,
        groups: 0..1,
        inter: 32..64,
        head_dim: 32,
        o_rank: 32,
    }
}

/// 2026-10-19: Every table of every expert at its published shape, each with its own bytes.
fn tables(c: &ModelConfig) -> Vec<(String, WeightDtype, Vec<usize>, Vec<u8>)> {
    let mut v = Vec::new();
    for l in 0..2 {
        for e in 0..2 {
            for t in 0..6 {
                let (dtype, [r, k]) = expert_table_shape(c, t);
                v.push((
                    expert_name(l, e, t),
                    dtype,
                    vec![r, k],
                    pattern(r * k, 36 * l + 6 * e + t),
                ));
            }
        }
    }
    v
}

fn rows(b: &[u8], width: usize, r: std::ops::Range<usize>) -> Vec<u8> {
    b[r.start * width..r.end * width].to_vec()
}

fn cols(b: &[u8], width: usize, c: std::ops::Range<usize>) -> Vec<u8> {
    b.chunks_exact(width)
        .flat_map(|row| row[c.clone()].to_vec())
        .collect()
}

#[test]
fn each_table_lands_in_its_slot_as_the_ranks_slice() {
    let c = config();
    let t = tables(&c);
    let f = shard(t.clone());
    let g = MockGpuBackend::new();
    let layers = upload_experts(&g, &f.store, &c, &local(), 3).unwrap();
    assert_eq!(layers.len(), 2);
    for (i, (_, _, shape, bytes)) in t.iter().enumerate() {
        let (l, e, table) = (i / 12, (i / 6) % 2, i % 6);
        let ex = &layers[l];
        let want = match table {
            0 | 2 => rows(bytes, shape[1], 32..64),
            1 | 3 => rows(bytes, shape[1], 32..64),
            4 => cols(bytes, shape[1], 16..32),
            _ => cols(bytes, shape[1], 1..2),
        };
        let got = back(&g, DevicePtr(ex.addr(table, e as u32)), want.len());
        assert_eq!(got, want, "layer {l} expert {e} table {table}");
        let slot = if table % 2 == 0 {
            ex.packed_bytes()
        } else {
            ex.scale_bytes()
        };
        assert_eq!(want.len(), slot);
    }
}

#[test]
fn a_table_of_the_wrong_shape_is_refused() {
    let c = config();
    let mut t = tables(&c);
    t[4].2 = vec![DIM, INTER];
    t[4].3 = pattern(DIM * INTER, 0);
    let f = shard(t);
    let e = upload_experts(&MockGpuBackend::new(), &f.store, &c, &local(), 2)
        .unwrap_err()
        .to_string();
    assert!(e.contains("layers.0.ffn.experts.0.w2.weight"), "{e}");
}

#[test]
fn a_missing_table_is_refused() {
    let c = config();
    let mut t = tables(&c);
    t.pop();
    let f = shard(t);
    let e = upload_experts(&MockGpuBackend::new(), &f.store, &c, &local(), 1)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("layers.1.ffn.experts.1.w2.scale was not deferred"),
        "{e}"
    );
}
