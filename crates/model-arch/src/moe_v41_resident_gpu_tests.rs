// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The resident-expert MoE on a GPU against the CPU reference `moe()`: synthetic MXFP4
//! routed experts (the input rounded to E4M3 blocks) and an FP8-block shared expert. At one rank the
//! device runs the whole layer; at three ranks one GPU holds three hand-sliced ranks and a local
//! gather stands in for the communicator, so the third rank's output is the full sum.
//!
//! Owner: model-arch, DeepSeek-V4.1 MoE.
//! Invariants: none beyond the types.

use std::ops::Range;

use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::ResidentMat;

use super::resident::{ResidentExperts, ResidentMoe};
use super::{MoeV41, MoeV41Cfg, MoeV41LayerWeights};
use crate::deepseek_v41_layer::levers::tests::at_defaults;
use crate::deepseek_v41_ref::mat::{InputQuant, Mat, Store};
use crate::deepseek_v41_ref::moe::{MoeCfg, MoeWeights, moe};
use crate::fp8_block_v41::Fp8StageSize;
use crate::tp_slice_v41::{rank_share, slice2d};

const DIM: usize = 256;
const NR: usize = 8;
const TOPK: usize = 3;
const M: usize = 5;
const LIMIT: f32 = 10.0;

fn backend() -> metrale_gpu_runtime::cuda_backend::MetraleCudaBackend {
    let set = metrale_kernels::ptx_for_exact_target("deepseek-v4-flash", "nvfp4")
        .expect("deepseek-v4-flash/nvfp4 not in this build");
    metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend")
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
    }
    fn bytes(&mut self, n: usize, f: impl Fn(u32) -> u8) -> Vec<u8> {
        (0..n).map(|_| f(self.next())).collect()
    }
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_val(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// 2026-10-19: One MXFP4 matrix: packed codes and E8M0 scales near 2^-4 (never code 0 or 255).
struct Mx {
    w: Vec<u8>,
    s: Vec<u8>,
}

fn mx(r: &mut Rng, rows: usize, cols: usize) -> Mx {
    Mx {
        w: r.bytes(rows * cols / 2, |v| v as u8),
        s: r.bytes(rows * cols / 32, |v| 120 + (v % 6) as u8),
    }
}

/// 2026-10-19: One FP8 block matrix: E4M3 codes with exponents 1..=8 (no NaN) and one E8M0
/// scale per 32 x 32 tile.
fn fp8(r: &mut Rng, rows: usize, cols: usize) -> Mx {
    Mx {
        w: r.bytes(rows * cols, |v| {
            ((v & 0x80) | ((1 + (v >> 8) % 8) << 3) | ((v >> 12) & 7)) as u8
        }),
        s: r.bytes(rows * cols / 1024, |v| 121 + (v % 4) as u8),
    }
}

/// 2026-10-19: The full layer at `inter` columns.
struct Layer {
    inter: usize,
    x: Vec<u16>,
    gate: Vec<u16>,
    bias: Vec<f32>,
    experts: Vec<[Mx; 3]>,
    shared: [Mx; 3],
}

fn layer(inter: usize) -> Layer {
    let mut r = Rng(0x00C0_FFEE);
    let x = (0..M * DIM).map(|_| bf16_bits(r.unit())).collect();
    let gate = (0..NR * DIM).map(|_| bf16_bits(r.unit() * 0.25)).collect();
    let bias = (0..NR).map(|_| r.unit() * 0.05).collect();
    let experts = (0..NR)
        .map(|_| {
            [
                mx(&mut r, inter, DIM),
                mx(&mut r, DIM, inter),
                mx(&mut r, inter, DIM),
            ]
        })
        .collect();
    let shared = [
        fp8(&mut r, inter, DIM),
        fp8(&mut r, DIM, inter),
        fp8(&mut r, inter, DIM),
    ];
    Layer {
        inter,
        x,
        gate,
        bias,
        experts,
        shared,
    }
}

fn reference(l: &Layer) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let inter = l.inter;
    let gate_b: Vec<u8> = l.gate.iter().flat_map(|h| h.to_le_bytes()).collect();
    let gate = Mat::new(Store::Bf16(&gate_b), NR, DIM, InputQuant::Exact).unwrap();
    fn e2m1(m: &Mx, rows: usize, cols: usize) -> Mat<'_> {
        Mat::new(
            Store::E2m1 {
                w: &m.w,
                scale: &m.s,
            },
            rows,
            cols,
            InputQuant::Fp8Block,
        )
        .unwrap()
    }
    fn e4m3(m: &Mx, rows: usize, cols: usize) -> Mat<'_> {
        let store = Store::E4m3 {
            w: &m.w,
            scale: &m.s,
            block_rows: 32,
        };
        Mat::new(store, rows, cols, InputQuant::Fp8Block).unwrap()
    }
    let ex: Vec<[Mat; 3]> = l
        .experts
        .iter()
        .map(|[a, b, c]| {
            [
                e2m1(a, inter, DIM),
                e2m1(b, DIM, inter),
                e2m1(c, inter, DIM),
            ]
        })
        .collect();
    let sh = [
        e4m3(&l.shared[0], inter, DIM),
        e4m3(&l.shared[1], DIM, inter),
        e4m3(&l.shared[2], inter, DIM),
    ];
    let w = MoeWeights {
        gate_w: &gate,
        gate_bias: &l.bias,
        experts: ex.iter().map(|[a, b, c]| (a, b, c)).collect(),
        shared: (&sh[0], &sh[1], &sh[2]),
    };
    let c = MoeCfg {
        dim: DIM,
        inter,
        n_routed: NR,
        topk: TOPK,
        gate_temp: 1.0,
        norm_topk_prob: true,
        route_scale: 1.5,
        swiglu_limit: LIMIT,
    };
    let x: Vec<f32> = l.x.iter().map(|&h| bf16_val(h)).collect();
    moe(&x, M, &w, &c)
}

/// 2026-10-19: Rank `rank` of `world`'s slice of the layer on the device.
struct Rank {
    moe: MoeV41,
    rx: ResidentMoe,
    w: MoeV41LayerWeights,
    ex: ResidentExperts,
    held: Vec<DevicePtr>,
}

fn up(g: &dyn GpuBackend, held: &mut Vec<DevicePtr>, b: &[u8]) -> DevicePtr {
    let p = g.alloc(b.len()).unwrap();
    g.copy_h2d(b, p).unwrap();
    held.push(p);
    p
}

/// 2026-10-19: The row slice `rows` of a `[*, cols]` matrix whose bytes per value are `num /
/// den`, or the column slice when `by_cols`.
fn cut(
    src: &[u8],
    cols: usize,
    num: usize,
    den: usize,
    keep: &Range<usize>,
    by_cols: bool,
) -> Vec<u8> {
    let row_bytes = cols * num / den;
    let n_rows = src.len() / row_bytes;
    if by_cols {
        slice2d(
            src,
            row_bytes,
            0..n_rows,
            keep.start * num / den..keep.end * num / den,
        )
        .unwrap()
    } else {
        slice2d(src, row_bytes, keep.clone(), 0..row_bytes).unwrap()
    }
}

fn rank(g: &dyn GpuBackend, l: &Layer, world: usize, rank: usize) -> Rank {
    let keep = rank_share(l.inter, 32, world, rank).unwrap();
    let inter = keep.len();
    let tile = keep.start / 32..keep.end / 32;
    let mut held = Vec::new();
    let cfg = MoeV41Cfg {
        dim: DIM,
        inter,
        n_routed: NR,
        topk: TOPK,
        gate_temp: 1.0,
        norm_topk_prob: true,
        route_scale: 1.5,
        swiglu_limit: LIMIT,
        max_tokens: M,
        fp8_stage: Fp8StageSize {
            weight_elems: inter * DIM,
            act_elems: M * DIM.max(inter),
        },
    };
    let moe = MoeV41::new(g, cfg, at_defaults()).unwrap();
    let rx = ResidentMoe::new(g, &moe, world).unwrap();
    // 2026-10-19: w1 and w3 keep rows; w2 keeps columns (packed 1/2 byte, scales 1/32 byte).
    let mut cat = |pick: usize, w: bool| -> DevicePtr {
        let mut all = Vec::new();
        for e in &l.experts {
            let m = &e[pick];
            let (cols, by_cols) = if pick == 1 {
                (l.inter, true)
            } else {
                (DIM, false)
            };
            all.extend(if w {
                cut(&m.w, cols, 1, 2, &keep, by_cols)
            } else {
                cut(&m.s, cols, 1, 32, &keep, by_cols)
            });
        }
        up(g, &mut held, &all)
    };
    let ex = ResidentExperts {
        n_experts: NR,
        inter,
        dim: DIM,
        w1: cat(0, true),
        w1_s: cat(0, false),
        w3: cat(2, true),
        w3_s: cat(2, false),
        w2: cat(1, true),
        w2_s: cat(1, false),
    };
    // 2026-10-19: The shared expert's scale planes are `[rows / 32, cols / 32]` tiles.
    let mut shared = |pick: usize| -> ResidentMat {
        let m = &l.shared[pick];
        let (w, s) = if pick == 1 {
            (
                cut(&m.w, l.inter, 1, 1, &keep, true),
                slice2d(&m.s, l.inter / 32, 0..DIM / 32, tile.clone()).unwrap(),
            )
        } else {
            (
                cut(&m.w, DIM, 1, 1, &keep, false),
                slice2d(&m.s, DIM / 32, tile.clone(), 0..DIM / 32).unwrap(),
            )
        };
        ResidentMat::Fp8Block {
            w: up(g, &mut held, &w),
            scale: up(g, &mut held, &s),
            qdq: true,
        }
    };
    let (s1, s2, s3) = (shared(0), shared(1), shared(2));
    let gate_b: Vec<u8> = l.gate.iter().flat_map(|h| h.to_le_bytes()).collect();
    let w = MoeV41LayerWeights {
        layer: 0,
        gate_w: up(g, &mut held, &gate_b),
        gate_bias_dev: MoeV41::upload_bias(g, &l.bias).unwrap(),
        gate_bias: l.bias.clone(),
        shared_w1: s1,
        shared_w2: s2,
        shared_w3: s3,
    };
    Rank {
        moe,
        rx,
        w,
        ex,
        held,
    }
}

/// 2026-10-19: A one-GPU stand-in for the communicator: the gather copies every rank's partial.
struct LocalGather<'a> {
    gpu: &'a dyn GpuBackend,
    partials: Vec<DevicePtr>,
}

impl CommBackend for LocalGather<'_> {
    fn all_reduce(&self, _p: u64, _b: usize) -> anyhow::Result<()> {
        anyhow::bail!("not used")
    }
    fn all_gather(&self, _send: u64, recv: u64, bytes: usize) -> anyhow::Result<()> {
        let s = self.gpu.default_stream();
        for (r, p) in self.partials.iter().enumerate() {
            self.gpu
                .copy_d2d_async(*p, DevicePtr(recv + (r * bytes) as u64), bytes, s)?;
        }
        self.gpu.synchronize(s)
    }
    fn reduce_scatter(&self, _s: u64, _r: u64, _b: usize) -> anyhow::Result<()> {
        anyhow::bail!("not used")
    }
    fn broadcast(&self, _p: u64, _b: usize, _root: usize) -> anyhow::Result<()> {
        anyhow::bail!("not used")
    }
    fn send_to(&self, _p: u64, _b: usize, _peer: usize, _s: u64) -> anyhow::Result<()> {
        anyhow::bail!("not used")
    }
    fn recv_from(&self, _p: u64, _b: usize, _peer: usize, _s: u64) -> anyhow::Result<()> {
        anyhow::bail!("not used")
    }
    fn barrier(&self) -> anyhow::Result<()> {
        Ok(())
    }
    fn rank(&self) -> usize {
        self.partials.len() - 1
    }
    fn world_size(&self) -> usize {
        self.partials.len()
    }
}

/// 2026-10-19: The device result against the reference: identical routing, weights within f32
/// noise, every output within two bf16 ulps of its magnitude and a cosine of at least 0.9999.
fn check(got: &[u8], weights: &[f32], indices: &[usize], want: &(Vec<f32>, Vec<f32>, Vec<usize>)) {
    assert_eq!(indices, want.2.as_slice(), "routing");
    let mut used = want.2.clone();
    used.sort_unstable();
    used.dedup();
    let peak = want.0.iter().fold(0f32, |a, v| a.max(v.abs()));
    assert!(
        used.len() > TOPK && peak > 0.1,
        "a trivial layer: experts {used:?}, peak {peak}"
    );
    for (a, b) in weights.iter().zip(&want.1) {
        assert!(
            (a - b).abs() <= 1e-5 * b.abs().max(1.0),
            "weight {a} vs {b}"
        );
    }
    let (mut dot, mut na, mut nb, mut exact) = (0f64, 0f64, 0f64, 0usize);
    for (i, c) in got.chunks_exact(2).enumerate() {
        let a = bf16_val(u16::from_le_bytes([c[0], c[1]]));
        let b = want.0[i];
        assert!(a.is_finite(), "out[{i}] = {a}");
        let tol = b.abs().max(0.05) * (2.0 / 128.0);
        assert!(
            (a - b).abs() <= tol,
            "out[{}, {}]: {a} vs {b}",
            i / DIM,
            i % DIM
        );
        exact += usize::from(a == b);
        dot += f64::from(a) * f64::from(b);
        na += f64::from(a) * f64::from(a);
        nb += f64::from(b) * f64::from(b);
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    eprintln!(
        "bit-exact {exact} of {}; cosine {cos:.8}; experts {used:?}; peak {peak}",
        want.0.len()
    );
    assert!(cos >= 0.9999, "cosine {cos}");
}

fn x_dev(g: &dyn GpuBackend, l: &Layer, held: &mut Vec<DevicePtr>) -> DevicePtr {
    let b: Vec<u8> = l.x.iter().flat_map(|h| h.to_le_bytes()).collect();
    up(g, held, &b)
}

fn release(g: &dyn GpuBackend, r: Rank) {
    for p in r.held {
        g.free(p).unwrap();
    }
    r.rx.free(g).unwrap();
}

#[test]
#[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
fn one_rank_matches_the_reference_moe() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let l = layer(256);
    let want = reference(&l);
    let mut r = rank(g, &l, 1, 0);
    let x = x_dev(g, &l, &mut r.held);
    let s = g.default_stream();
    let (out, weights, indices) = r
        .moe
        .forward_resident(g, &r.w, &r.rx, &r.ex, None, x, M, s)
        .unwrap();
    g.synchronize(s).unwrap();
    let mut b = vec![0u8; M * DIM * 2];
    g.copy_d2h(out, &mut b).unwrap();
    check(&b, &weights, &indices, &want);
    release(g, r);
}

#[test]
#[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
fn three_hand_sliced_ranks_sum_to_the_reference_moe() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let l = layer(768);
    let want = reference(&l);
    let mut ranks: Vec<Rank> = (0..3).map(|i| rank(g, &l, 3, i)).collect();
    let x = x_dev(g, &l, &mut ranks[0].held);
    let comm = LocalGather {
        gpu: g,
        partials: ranks.iter().map(|r| r.rx.reduce().partial()).collect(),
    };
    let s = g.default_stream();
    let mut last = None;
    for r in &ranks {
        last = Some(
            r.moe
                .forward_resident(g, &r.w, &r.rx, &r.ex, Some(&comm), x, M, s)
                .unwrap(),
        );
    }
    g.synchronize(s).unwrap();
    let (out, weights, indices) = last.unwrap();
    let mut b = vec![0u8; M * DIM * 2];
    g.copy_d2h(out, &mut b).unwrap();
    check(&b, &weights, &indices, &want);
    for r in ranks {
        release(g, r);
    }
}
