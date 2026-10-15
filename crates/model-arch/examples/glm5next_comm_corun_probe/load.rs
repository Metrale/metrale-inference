// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The two compute loads the probe co-runs a transfer with, at their serve
//! footprints on GB10 (48 SMs):
//! - `w4a4_grouped_gu_m128` (fused MoE gate/up, 256 threads, 128 registers, ~42.6 KB dynamic
//!   shared memory: two CTAs per SM, neither leaving room for a 98.5 KB NCCL CTA), through the
//!   production launcher `launch_w4a4_gu`, on a uniform random routing;
//! - `glm5next_dsa_mla_prefill_tc_phi` (prefill DSA attend, 384 threads, 87.6 KB: one CTA per
//!   SM, ~170 waves at 8208 rows), on the prefill twin gates' synthetic 32k-token case.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: each load's inputs are prepared once on the null stream and synchronized before
//! its first timed launch; a launch only reads them.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_dsa::attend::{DSA_PREFILL_TC_MODULE, TC_BLOCK, TC_SMEM_BYTES};
use metrale_model_arch::glm5next_mlp::forward::{GroupedEntry, W4a4GuCall, launch_w4a4_gu};
use metrale_model_arch::glm5next_mlp::weights::Glm5NextExpertPtrTable;

use crate::device::Rng;
use crate::fixture::{Lcg, Pool, View, hb_launch, make_pool};
use crate::launch::{
    Bufs, EXPERTS, GS_DOWN, GS_GATE_UP, HIDDEN, KP, Kern, MI, TOP_K, Table, bufs, quant, sort,
    table, tiles,
};
use crate::prefill::make_prefill_case;

/// 2026-10-15: The m128 tile entry (`W4A4_GROUPED_MT[2]`).
const M128: usize = 2;
const M128_ROWS: usize = 128;
/// 2026-10-15: GLM-5.3's `swiglu_limit`.
const LIMIT: f32 = 10.0;
/// 2026-10-15: Per-rank heads at TP=3 and the rows of a 32k prefill sub-chunk, as the twin
/// gate's prefill case.
const PHI_HEADS: usize = 22;
const PHI_ROWS: usize = 8208;

pub trait Load {
    fn name(&self) -> &'static str;
    fn launch(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()>;
}

fn ptrs(t: &Table) -> Glm5NextExpertPtrTable {
    Glm5NextExpertPtrTable {
        packed_ptrs: t.packed,
        scale_ptrs: t.scale,
        scale2_vals: t.scale2,
    }
}

pub struct Gu {
    kern: KernelHandle,
    b: Bufs,
    gate: Table,
    up: Table,
    dq: [DevicePtr; 3],
}

impl Gu {
    /// 2026-10-15: `rows` token rows routed top 8 over 288 experts at the TP=3 expert width;
    /// sorts, tiles and quantizes them once.
    pub fn new(gpu: &dyn GpuBackend, kn: &Kern, gu: KernelHandle, rows: usize) -> Result<Self> {
        let mut rng = Rng(0x13198a2e_03707344);
        let gate = table(gpu, &mut rng, MI, HIDDEN)?;
        let up = table(gpu, &mut rng, MI, HIDDEN)?;
        let b = bufs(gpu, &mut rng, rows)?;
        // 2026-10-15: The gate/up launch writes only `dq`; drop the down outputs early to stay
        // well inside the window's device-memory budget.
        gpu.free(b.out)?;
        gpu.free(b.out_ref)?;
        let slots = rows * TOP_K;
        let dq = [
            gpu.alloc(slots * KP / 2)?,
            gpu.alloc(slots * KP / 16)?,
            gpu.alloc(slots * 4)?,
        ];
        sort(gpu, kn, &b, 0)?;
        tiles(gpu, kn, &b, M128_ROWS, 0)?;
        quant(gpu, kn.quant, &b, b.x, rows, HIDDEN, GS_GATE_UP, 0)?;
        gpu.synchronize(0)?;
        Ok(Self {
            kern: gu,
            b,
            gate,
            up,
            dq,
        })
    }
}

impl Load for Gu {
    fn name(&self) -> &'static str {
        "w4a4_grouped_gu_m128"
    }

    fn launch(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let b = &self.b;
        let call = W4a4GuCall {
            ti: M128,
            act: [b.aq, b.as_, b.ag],
            gate: ptrs(&self.gate),
            up: ptrs(&self.up),
            out: self.dq,
            offsets: b.offsets,
            tile_off: b.tile_off,
            gather: b.sorted,
            num_experts: EXPERTS,
            slots: b.rows * TOP_K,
            n: MI,
            k: HIDDEN,
            k_store: KP,
            gs: GS_DOWN,
            limit: LIMIT,
        };
        launch_w4a4_gu(gpu, GroupedEntry::plain(self.kern), &call, stream)
    }
}

pub struct Phi {
    kern: KernelHandle,
    pool: Pool,
    view: View,
    out: DevicePtr,
}

impl Phi {
    pub fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        let mut rng = Lcg(0xa409_3822_299f_31d0);
        let pool = make_pool(gpu, &mut rng, 8)?;
        let case = make_prefill_case(gpu, &mut rng, PHI_HEADS, PHI_ROWS, 1, 1.0)?;
        let view = case.view(0, 0, PHI_ROWS);
        let out = gpu.alloc(PHI_ROWS * PHI_HEADS * crate::fixture::KVL * 2)?;
        Ok(Self {
            kern: gpu.kernel(DSA_PREFILL_TC_MODULE, "glm5next_dsa_mla_prefill_tc_phi")?,
            pool,
            view,
            out,
        })
    }
}

impl Load for Phi {
    fn name(&self) -> &'static str {
        "glm5next_dsa_mla_prefill_tc_phi"
    }

    fn launch(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let none = (DevicePtr::NULL, DevicePtr::NULL);
        hb_launch(
            gpu,
            self.kern,
            [1, PHI_ROWS as u32, 1],
            TC_BLOCK,
            TC_SMEM_BYTES,
            self.pool.dev,
            self.view,
            self.out,
            none,
            1,
        )
        .launch(stream)
    }
}
