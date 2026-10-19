// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The resident-expert MoE on the host and the mock. The plan: positions grouped by
//! ascending expert and token order, each token's positions in ascending expert order with its own
//! weights, and the refusals; the blob: every array lands at its layout offset; the forward: the
//! launch chain reads the plan it uploaded, w2 and the shared slice write the reduction's partial
//! rows, and a mismatched layer or an oversized call is refused.
//!
//! Owner: model-arch, DeepSeek-V4.1 MoE.
//! Invariants: none beyond the types.

use super::resident::{ResidentExperts, ResidentMoe};
use super::resident_plan::{BlobLayout, PTR_TABLES, ResidentPlan, resident_plan};
use super::{MoeV41, MoeV41Cfg, MoeV41LayerWeights};
use crate::deepseek_v41_layer::levers::tests::at_defaults;
use crate::fp8_block_v41::Fp8StageSize;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops::ResidentMat;

#[test]
fn positions_group_by_ascending_expert_and_each_token_lists_its_own() {
    let p = resident_plan(
        &[5, 1, 1, 3, 3, 5],
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        3,
        2,
        8,
    )
    .unwrap();
    assert_eq!(
        p,
        ResidentPlan {
            experts: vec![1, 3, 5],
            offsets: vec![0, 2, 4, 6],
            token_of_pos: vec![0, 1, 1, 2, 0, 2],
            weight_of_pos: vec![0.2, 0.3, 0.4, 0.5, 0.1, 0.6],
            pos: vec![0, 4, 1, 2, 3, 5],
            max_rows: 2,
        }
    );
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

#[test]
fn every_routed_pair_has_exactly_one_position_on_random_routing() {
    let (m, topk, n) = (57usize, 6usize, 384usize);
    let mut s = 7u64;
    let mut indices = Vec::new();
    for _ in 0..m {
        let mut picks: Vec<usize> = Vec::new();
        while picks.len() < topk {
            let e = (lcg(&mut s) % 40) as usize * 9 % n;
            if !picks.contains(&e) {
                picks.push(e);
            }
        }
        indices.extend(picks);
    }
    let weights: Vec<f32> = (0..m * topk).map(|i| i as f32 * 0.25).collect();
    let p = resident_plan(&indices, &weights, m, topk, n).unwrap();
    assert_eq!(p.n_pos(), m * topk);
    assert!(p.experts.windows(2).all(|w| w[0] < w[1]));
    assert!(p.offsets.windows(2).all(|w| w[0] < w[1]));
    let expert_of = |pos: usize| {
        let i = p.offsets.iter().rposition(|&o| o as usize <= pos).unwrap();
        p.experts[i] as usize
    };
    let mut seen = vec![false; m * topk];
    for t in 0..m {
        let mine = &p.pos[t * topk..(t + 1) * topk];
        let es: Vec<usize> = mine.iter().map(|&q| expert_of(q as usize)).collect();
        assert!(es.windows(2).all(|w| w[0] < w[1]), "token {t}: {es:?}");
        for &q in mine {
            let q = q as usize;
            assert!(!seen[q], "position {q} used twice");
            seen[q] = true;
            assert_eq!(p.token_of_pos[q] as usize, t);
            let j = (0..topk)
                .find(|&j| indices[t * topk + j] == expert_of(q))
                .unwrap();
            assert_eq!(p.weight_of_pos[q], weights[t * topk + j]);
        }
    }
    for i in 0..p.experts.len() {
        let run = &p.token_of_pos[p.offsets[i] as usize..p.offsets[i + 1] as usize];
        assert!(
            run.windows(2).all(|w| w[0] < w[1]),
            "token order within an expert"
        );
    }
    let most = (0..n)
        .map(|e| indices.iter().filter(|&&x| x == e).count())
        .max()
        .unwrap();
    assert_eq!(p.max_rows, most);
}

#[test]
fn a_bad_routing_is_refused() {
    let e = resident_plan(&[1, 8], &[0.5, 0.5], 1, 2, 8).unwrap_err();
    assert!(e.to_string().contains("expert 8 of 8"), "{e}");
    let e = resident_plan(&[3, 3], &[0.5, 0.5], 1, 2, 8).unwrap_err();
    assert!(e.to_string().contains("twice"), "{e}");
    let e = resident_plan(&[1, 2, 3], &[0.5, 0.5], 1, 2, 8).unwrap_err();
    assert!(e.to_string().contains("3 indices and 2 weights"), "{e}");
}

fn read_words<T>(b: &[u8], at: usize, n: usize, f: impl Fn([u8; 4]) -> T) -> Vec<T> {
    (0..n)
        .map(|i| f(b[at + 4 * i..at + 4 * i + 4].try_into().unwrap()))
        .collect()
}

#[test]
fn the_blob_holds_each_array_at_its_layout_offset() {
    let p = resident_plan(
        &[5, 1, 1, 3, 3, 5],
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        3,
        2,
        8,
    )
    .unwrap();
    let l = BlobLayout {
        n_routed: 8,
        max_pos: 8,
    };
    let b = l
        .encode(&p, |t, e| ((t as u64) << 32) | u64::from(e))
        .unwrap();
    assert_eq!(b.len(), l.bytes());
    for t in 0..PTR_TABLES {
        let got: Vec<u64> = (0..3)
            .map(|i| u64::from_le_bytes(b[l.ptrs(t) + 8 * i..][..8].try_into().unwrap()))
            .collect();
        let want: Vec<u64> = [1u64, 3, 5]
            .iter()
            .map(|&e| ((t as u64) << 32) | e)
            .collect();
        assert_eq!(got, want, "table {t}");
    }
    assert_eq!(read_words(&b, l.scale2(), 3, f32::from_le_bytes), [0.0; 3]);
    assert_eq!(
        read_words(&b, l.offsets(), 4, i32::from_le_bytes),
        p.offsets
    );
    assert_eq!(
        read_words(&b, l.tokens(), 6, i32::from_le_bytes),
        p.token_of_pos
    );
    assert_eq!(
        read_words(&b, l.weights(), 6, f32::from_le_bytes),
        p.weight_of_pos
    );
    assert_eq!(read_words(&b, l.pos(), 6, i32::from_le_bytes), p.pos);
    let small = BlobLayout {
        n_routed: 8,
        max_pos: 5,
    };
    assert!(small.encode(&p, |_, _| 0).is_err());
}

const DIM: usize = 256;
const INTER: usize = 256;
const NR: usize = 8;
const TOPK: usize = 2;

fn cfg() -> MoeV41Cfg {
    MoeV41Cfg {
        dim: DIM,
        inter: INTER,
        n_routed: NR,
        topk: TOPK,
        gate_temp: 1.0,
        norm_topk_prob: true,
        route_scale: 1.0,
        swiglu_limit: 10.0,
        max_tokens: 4,
        fp8_stage: Fp8StageSize::NONE,
    }
}

fn layer(g: &MockGpuBackend) -> MoeV41LayerWeights {
    // 2026-10-19: The mock's logits read back as zeros, so every score ties and the bias alone
    // picks: experts 7 and 6 for every token.
    let bias: Vec<f32> = (0..NR).map(|e| e as f32 * 0.1).collect();
    MoeV41LayerWeights {
        layer: 0,
        gate_w: g.alloc(NR * DIM * 2).unwrap(),
        gate_bias_dev: MoeV41::upload_bias(g, &bias).unwrap(),
        gate_bias: bias,
        shared_w1: ResidentMat::Bf16(DevicePtr(0x5100)),
        shared_w2: ResidentMat::Bf16(DevicePtr(0x5200)),
        shared_w3: ResidentMat::Bf16(DevicePtr(0x5300)),
    }
}

fn experts(inter: usize) -> ResidentExperts {
    ResidentExperts {
        n_experts: NR,
        inter,
        dim: DIM,
        w1: DevicePtr(0x10_0000),
        w1_s: DevicePtr(0x20_0000),
        w3: DevicePtr(0x30_0000),
        w3_s: DevicePtr(0x40_0000),
        w2: DevicePtr(0x50_0000),
        w2_s: DevicePtr(0x60_0000),
    }
}

fn ptr(a: &MockArg) -> DevicePtr {
    match a {
        MockArg::Buffer(p) => *p,
        other => panic!("{other:?}"),
    }
}

fn u32_arg(a: &MockArg) -> u32 {
    match a {
        MockArg::Bytes(b) => u32::from_le_bytes(b[..4].try_into().unwrap()),
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_forward_reads_the_plan_it_uploaded_and_fills_the_partial_rows() {
    let g = MockGpuBackend::new();
    g.set_kernel_n_tile(KernelHandle(0xDEAD), 64);
    let moe = MoeV41::new(&g, cfg(), at_defaults()).unwrap();
    let rx = ResidentMoe::new(&g, &moe, 1).unwrap();
    let w = layer(&g);
    let ex = experts(INTER);
    let x = g.alloc(3 * DIM * 2).unwrap();
    let base = g.launches_snapshot().len();
    let (out, weights, indices) = moe
        .forward_resident(&g, &w, &rx, &ex, None, x, 3, 0)
        .unwrap();
    assert_eq!(out, moe.out_ptr());
    assert_eq!(indices, [7, 6, 7, 6, 7, 6]);
    assert_eq!(weights, [0.5; 6]);
    let l = &g.launches_snapshot()[base..];
    assert_eq!(
        l.len(),
        12,
        "router, quant, w1, w3, swiglu, quant, w2, shared x4, combine"
    );
    let (w1, w3, w2) = (&l[2], &l[3], &l[6]);
    for (k, n) in [(w1, INTER), (w3, INTER), (w2, DIM)] {
        assert_eq!(
            k.grid,
            [(n / 64) as u32, 1, 2],
            "two experts, one 64-row tile each"
        );
        assert_eq!(u32_arg(&k.args[7]), 2);
    }
    assert_eq!(
        ptr(&w1.args[0]),
        ptr(&w3.args[0]),
        "w1 and w3 read the rounded input"
    );
    assert_eq!(ptr(&l[1].args[0]), ptr(&w1.args[0]));
    assert_ne!(
        ptr(&w1.args[0]),
        x,
        "the caller's input is not rounded in place"
    );
    assert_eq!(
        ptr(&w2.args[6]),
        DevicePtr::NULL,
        "w2 reads the hidden rows by position"
    );
    assert_ne!(ptr(&w1.args[6]), DevicePtr::NULL);
    assert_eq!(ptr(&w2.args[4]), rx.reduce().partial());

    // 2026-10-19: The uploaded tables: experts 6 then 7; tokens 0..3 under each.
    let mut b = vec![0u8; 2 * 8];
    for (k, table) in [(w1, 0usize), (w3, 2), (w2, 4)] {
        for (arg, t) in [(1usize, table), (2, table + 1)] {
            g.copy_d2h(ptr(&k.args[arg]), &mut b).unwrap();
            let got: Vec<u64> = b
                .chunks(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(got, [ex.addr(t, 6), ex.addr(t, 7)], "table {t}");
        }
    }
    let mut words = vec![0u8; 6 * 4];
    g.copy_d2h(ptr(&w1.args[6]), &mut words).unwrap();
    assert_eq!(
        read_words(&words, 0, 6, i32::from_le_bytes),
        [0, 1, 2, 0, 1, 2]
    );
    let mut off = vec![0u8; 3 * 4];
    g.copy_d2h(ptr(&w1.args[5]), &mut off).unwrap();
    assert_eq!(read_words(&off, 0, 3, i32::from_le_bytes), [0, 3, 6]);

    // 2026-10-19: The shared slice's w2 writes the rows after the six expert rows.
    let shared_w2 = &l[10];
    assert_eq!(
        ptr(&shared_w2.args[2]).0,
        rx.reduce().partial().0 + (6 * DIM * 4) as u64
    );
    let combine = &l[11];
    assert_eq!(combine.grid, [1, 3, 1]);
    assert_eq!(
        (u32_arg(&combine.args[4]), u32_arg(&combine.args[5])),
        (2, 6)
    );
    let mut pos = vec![0u8; 6 * 4];
    g.copy_d2h(ptr(&combine.args[3]), &mut pos).unwrap();
    assert_eq!(
        read_words(&pos, 0, 6, i32::from_le_bytes),
        [0, 3, 1, 4, 2, 5],
        "each token: expert 6's row, then expert 7's"
    );
}

#[test]
fn a_mismatched_layer_or_an_oversized_call_is_refused() {
    let g = MockGpuBackend::new();
    g.set_kernel_n_tile(KernelHandle(0xDEAD), 64);
    let moe = MoeV41::new(&g, cfg(), at_defaults()).unwrap();
    let rx = ResidentMoe::new(&g, &moe, 1).unwrap();
    let w = layer(&g);
    let x = g.alloc(5 * DIM * 2).unwrap();
    let e = moe
        .forward_resident(&g, &w, &rx, &experts(2 * INTER), None, x, 3, 0)
        .unwrap_err();
    assert!(e.to_string().contains("resident experts"), "{e}");
    let e = moe
        .forward_resident(&g, &w, &rx, &experts(INTER), None, x, 5, 0)
        .unwrap_err();
    assert!(e.to_string().contains("exceed"), "{e}");
}
