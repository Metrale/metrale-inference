// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Routings, widths and launchers of the `glm5next_moe_epi_bench` example: a Zipf
//! routing over a seeded expert permutation (exponent 0 = uniform), the two expert widths the
//! fused gate/up launch must serve (704, the TP=3 slice with the `_k64` down, and 640, a plain
//! multiple of 128) and the separate and fused gate/up segments.
//! 2026-10-15: and the down GEMM on any entry (the `_n` twins).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: none beyond the types.

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_arch::glm5next_mlp::forward::{GroupedEntry, W4a4GuCall, launch_w4a4_gu};
use metrale_model_arch::glm5next_mlp::weights::Glm5NextExpertPtrTable;

use crate::device::Rng;
use crate::launch::{Bufs, EXPERTS, GS_DOWN, GS_GATE_UP, HIDDEN, Kern, TOP_K, Table};
use crate::launch::{grouped, grouped_entry, quant};

/// 2026-10-14: GLM-5.3's `swiglu_limit`.
pub const LIMIT: f32 = 10.0;

/// 2026-10-14: `rows` rows of `TOP_K` distinct experts each, expert popularity proportional to
/// 1 / (rank + 1)^s over a seeded permutation of the experts.
pub fn zipf_routing(rng: &mut Rng, rows: usize, s: f64) -> Vec<i32> {
    let mut perm: Vec<i32> = (0..EXPERTS as i32).collect();
    for i in (1..EXPERTS).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    let w: Vec<f64> = (0..EXPERTS)
        .map(|r| 1.0 / ((r + 1) as f64).powf(s))
        .collect();
    let total: f64 = w.iter().sum();
    let mut acc = 0.0;
    let cdf: Vec<f64> = w
        .iter()
        .map(|x| {
            acc += x / total;
            acc
        })
        .collect();
    let mut ids = Vec::with_capacity(rows * TOP_K);
    for _ in 0..rows {
        let mut row: Vec<i32> = Vec::with_capacity(TOP_K);
        while row.len() < TOP_K {
            let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
            let e = perm[cdf.partition_point(|&c| c < u).min(EXPERTS - 1)];
            if !row.contains(&e) {
                row.push(e);
            }
        }
        ids.extend(row);
    }
    ids
}

/// 2026-10-14: The largest number of slots routed to one expert.
pub fn max_load(ids: &[i32]) -> usize {
    let mut n = vec![0usize; EXPERTS];
    for &e in ids {
        n[e as usize] += 1;
    }
    n.into_iter().max().unwrap_or(0)
}

/// 2026-10-14: One expert width: its tables, the down input's stored width, the down
/// quantizer and the down entry per tile (the plain ones at `mi % 128 == 0`, else the `_k64`
/// twins), as `w4a4_grouped::down_entry` picks them; 2026-10-14: and the m128 down's `_r` twin.
pub struct Width {
    pub mi: usize,
    pub k_store: usize,
    pub down_quant: KernelHandle,
    pub down: [KernelHandle; 3],
    pub down_r: KernelHandle,
    /// 2026-10-15: The m128 down's `_n` twin (`w4a4_grouped_mx_m128[_k64]_n`).
    pub down_n: KernelHandle,
    pub gate: Table,
    pub up: Table,
    pub dn: Table,
}

impl Width {
    /// 2026-10-14: `down_r` = `w4a4_grouped_mx_m128_r` and `_m128_k64_r`; 2026-10-15: `down_n`
    /// = `w4a4_grouped_mx_m128_n` and `_m128_k64_n`.
    pub fn new(
        kn: &Kern,
        mi: usize,
        t: [Table; 3],
        down_r: [KernelHandle; 2],
        down_n: [KernelHandle; 2],
    ) -> Self {
        let plain = mi.is_multiple_of(128);
        let [gate, up, dn] = t;
        Self {
            mi,
            k_store: mi.next_multiple_of(128),
            down_quant: if plain { kn.quant } else { kn.quant_k64 },
            down: if plain {
                [kn.mx[0], kn.mx[1], kn.mx[2]]
            } else {
                [kn.mx_k64[0], kn.mx_k64[1], kn.mx_k64[2]]
            },
            down_r: down_r[usize::from(!plain)],
            down_n: down_n[usize::from(!plain)],
            gate,
            up,
            dn,
        }
    }

    /// 2026-10-14: The down input's code, scale and global bytes of `slots` rows.
    pub fn dq_bytes(&self, slots: usize) -> [usize; 3] {
        [
            slots * self.k_store / 2,
            slots * self.k_store / 16,
            slots * 4,
        ]
    }
}

/// 2026-10-14: The fused call's view of a bench table.
fn ptr_table(t: &Table) -> Glm5NextExpertPtrTable {
    Glm5NextExpertPtrTable {
        packed_ptrs: t.packed,
        scale_ptrs: t.scale,
        scale2_vals: t.scale2,
        scale_bases: DevicePtr::NULL,
    }
}

/// 2026-10-14: The separate segment the fused launch replaces, after the gate/up quantization:
/// gate, up, the SwiGLU into `b.act`, and the down quantization into the W4A4 scratch.
pub fn separate(
    gpu: &dyn GpuBackend,
    kn: &Kern,
    w: &Width,
    b: &Bufs,
    ti: usize,
    limit: f32,
    s: u64,
) -> Result<()> {
    let (slots, mi) = (b.rows * TOP_K, w.mi);
    grouped(
        gpu,
        kn.mx[ti],
        ti,
        b,
        &w.gate,
        b.gate,
        b.sorted,
        (mi, HIDDEN),
        s,
    )?;
    grouped(
        gpu,
        kn.mx[ti],
        ti,
        b,
        &w.up,
        b.upo,
        b.sorted,
        (mi, HIDDEN),
        s,
    )?;
    KernelLaunch::new(gpu, kn.swiglu)
        .grid([div_ceil((slots * mi) as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(b.gate)
        .arg_ptr(b.upo)
        .arg_ptr(b.act)
        .arg_u32((slots * mi) as u32)
        .arg_f32(limit)
        .launch(s)?;
    quant(gpu, w.down_quant, b, b.act, slots, mi, GS_DOWN, s)
}

/// 2026-10-14: The fused launch on entry `ti`, its down input into `dq`.
#[allow(clippy::too_many_arguments)]
pub fn fused(
    gpu: &dyn GpuBackend,
    gu: GroupedEntry,
    w: &Width,
    b: &Bufs,
    dq: [DevicePtr; 3],
    ti: usize,
    limit: f32,
    s: u64,
) -> Result<()> {
    let call = W4a4GuCall {
        ti,
        act: [b.aq, b.as_, b.ag],
        gate: ptr_table(&w.gate),
        up: ptr_table(&w.up),
        out: dq,
        offsets: b.offsets,
        tile_off: b.tile_off,
        gather: b.sorted,
        num_experts: EXPERTS,
        slots: b.rows * TOP_K,
        n: w.mi,
        k: HIDDEN,
        k_store: w.k_store,
        gs: GS_DOWN,
        limit,
    };
    launch_w4a4_gu(gpu, gu, &call, s)
}

/// 2026-10-14: The down GEMM on entry `ti` reading its input from `act` into `out`; with a
/// raster group, the m128 `_r` twin (`ti` must be 2).
#[allow(clippy::too_many_arguments)]
pub fn down(
    gpu: &dyn GpuBackend,
    w: &Width,
    b: &Bufs,
    act: [DevicePtr; 3],
    out: DevicePtr,
    ti: usize,
    raster: Option<u32>,
    s: u64,
) -> Result<()> {
    anyhow::ensure!(
        raster.is_none() || ti == 2,
        "the raster twins are m128 only"
    );
    let e = match raster {
        Some(g) => GroupedEntry::raster(w.down_r, g),
        None => GroupedEntry::plain(w.down[ti]),
    };
    down_entry(gpu, w, b, act, out, ti, e, s)
}

/// 2026-10-15: [`down`] on a given entry of tile `ti` (an `_n` twin is m128).
#[allow(clippy::too_many_arguments)]
pub fn down_entry(
    gpu: &dyn GpuBackend,
    w: &Width,
    b: &Bufs,
    act: [DevicePtr; 3],
    out: DevicePtr,
    ti: usize,
    e: GroupedEntry,
    s: u64,
) -> Result<()> {
    let v = Bufs {
        aq: act[0],
        as_: act[1],
        ag: act[2],
        ..*b
    };
    grouped_entry(
        gpu,
        e,
        ti,
        &v,
        &w.dn,
        out,
        DevicePtr::NULL,
        (HIDDEN, w.mi),
        s,
    )
}

/// 2026-10-14: The gate/up quantization of the token rows into the W4A4 scratch.
pub fn quant_x(gpu: &dyn GpuBackend, kn: &Kern, b: &Bufs, s: u64) -> Result<()> {
    quant(gpu, kn.quant, b, b.x, b.rows, HIDDEN, GS_GATE_UP, s)
}

/// 2026-10-14: The 3/4 quantile of |v| over BF16 bytes: a SwiGLU limit that clamps about a
/// quarter of the gate values from above and of the up values on both sides.
pub fn quartile_limit(bytes: &[u8]) -> f32 {
    let mut v: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32().abs())
        .filter(|x| x.is_finite())
        .collect();
    v.sort_by(f32::total_cmp);
    v.get(v.len() * 3 / 4).copied().unwrap_or(1.0).max(1e-3)
}
