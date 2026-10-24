// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The top-k launch of a selection pass (moved from `launch::run`): the tiled
//! bitonic walk (`dsa_topk_pools`), the one-CTA radix select (`dsa_topk_pools_radix`), or for a
//! ceiling launch under `METRALE_GLM_DSA_TOPK_MC` the multi-CTA radix select
//! (`dsa_topk_mc_pass` x 4 + `dsa_topk_mc_collect`, `dsa_topk_mc.cu`) and the workspace it needs.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Every path writes the same selected pools in the same order (score descending, pool
//!   ascending): `dsa_topk_pools_radix` gated against `dsa_topk_pools`, the multi-CTA select
//!   against `dsa_topk_pools_radix` (examples/glm5next_dsa_topk_mc_gate.rs).
//! - `--glm-dsa-topk-seg` (R4, `DsaRowsKernels::topk_seg`): where the multi-CTA select runs, its
//!   five launches run as the one persistent segment `dsa_topk_mc_seg` ([`launch_mc_seg`]; the
//!   five launches' bytes at the same CTA count). The lever on and the kernel absent refuses the
//!   load ([`DsaRowsKernels::with_topk_seg`]), never the five launches.
//! - The multi-CTA select's CTA count is a function of the launch's pool count alone
//!   ([`topk_mc_ctas`]), so a captured graph keeps its grid; the kernels read each row's live
//!   pools and select_k from the device geometry. It runs only where its workspace holds that
//!   many CTAs for every row of the pass.

use super::*;
use crate::glm5next_dsa::DsaRowsKernels;
use anyhow::ensure;
use metrale_gpu_runtime::gpu::KernelHandle;

/// 2026-10-21: `DSA_MC_CMAX` and `DSA_MC_WS_WORDS(C)` of `dsa_topk_mc.cu`: the most CTAs per row,
/// and one row's workspace in u32 words at `c` CTAs.
pub(crate) const TOPK_MC_CTAS_MAX: usize = 64;
pub(crate) fn topk_mc_ws_words(c: usize) -> usize {
    c * 256 + 768 + 2 + 2 * TOPK_RADIX_MAX + c + c * TOPK_RADIX_MAX + 16
}

/// 2026-10-21: CTAs per row of the multi-CTA select at `pools` pools (a ceiling launch's
/// `max_pools`): one per 4,096 pools, between 16 and 32 (GB10: 16 is fastest up to a 128k
/// context, 32 at 512k; more CTAs add per-CTA histogram reads).
pub(crate) fn topk_mc_ctas(pools: usize) -> usize {
    pools.div_ceil(4_096).clamp(TOPK_MC_CTAS.0, TOPK_MC_CTAS.1)
}
pub(crate) const TOPK_MC_CTAS: (usize, usize) = (16, 32);
const _: () = assert!(TOPK_MC_CTAS.1 <= TOPK_MC_CTAS_MAX);

/// 2026-10-21: The most rows a multi-CTA select runs (a captured verify's; more keep the one-CTA
/// radix).
pub(crate) const TOPK_MC_ROWS: usize = 16;

/// 2026-10-21: The multi-CTA select's workspace, allocated with the scratch when its lever and
/// the decode radix are on (empty otherwise): the scratch's rows up to [`TOPK_MC_ROWS`] at the
/// most CTAs [`topk_mc_ctas`] returns, so any ceiling the scratch serves fits (16 rows x 32
/// CTAs = 1.6 MB).
#[derive(Debug, Clone, Copy)]
pub(crate) struct DsaTopkMcRegion {
    ws: DevicePtr,
    rows: usize,
    ctas: usize,
    /// 2026-10-24: The segment's control block ([`TOPK_MC_SEG_CTL_BYTES`], zeroed at allocation
    /// and left zero by every launch), allocated with the workspace (the paired A/B can switch the
    /// segment on per pass).
    ctl: DevicePtr,
}

/// 2026-10-24: Bytes of `dsa_topk_mc_seg`'s `SegCtlN<5>` (ticket, exited, gated, done[5]).
pub(crate) const TOPK_MC_SEG_CTL_BYTES: usize = 4 * (3 + 5);

impl DsaTopkMcRegion {
    pub(crate) fn alloc(
        gpu: &dyn GpuBackend,
        geom: &DsaSelectGeometry,
        levers: &DsaSelectLevers,
    ) -> Result<Self> {
        if !(levers.topk_mc && levers.decode_topk_radix) {
            return Ok(Self {
                ws: DevicePtr::NULL,
                rows: 0,
                ctas: 0,
                ctl: DevicePtr::NULL,
            });
        }
        let ctas = TOPK_MC_CTAS.1;
        let rows = geom.q_rows.min(TOPK_MC_ROWS);
        let ws = gpu.alloc(rows * topk_mc_ws_words(ctas) * 4)?;
        let ctl = gpu.alloc(TOPK_MC_SEG_CTL_BYTES)?;
        gpu.memset(ctl, 0, TOPK_MC_SEG_CTL_BYTES)?;
        Ok(Self {
            ws,
            rows,
            ctas,
            ctl,
        })
    }

    pub(crate) fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.ws.0 != 0 {
            gpu.free(self.ws)?;
        }
        if self.ctl.0 != 0 {
            gpu.free(self.ctl)?;
        }
        Ok(())
    }

    /// 2026-10-21: Whether `rows` rows at `ctas` CTAs fit (a row's stride is
    /// `topk_mc_ws_words(ctas)`, at most the allocation's).
    fn holds(&self, rows: usize, ctas: usize) -> bool {
        self.ws.0 != 0 && rows <= self.rows && ctas <= self.ctas
    }
}

/// 2026-10-21: Whether a pass's top-k runs over several CTAs: a ceiling launch that takes the
/// radix select (`radix`), the lever on, both kernels resolved, and a workspace for its rows
/// at [`topk_mc_ctas`] of the launch's `pools`.
pub(crate) fn topk_mc_for(
    kernels: &Glm5NextDsaKernels,
    scratch: &DsaSelectScratch,
    ceiling: bool,
    radix: bool,
    rows: usize,
    pools: usize,
) -> bool {
    ceiling
        && radix
        && scratch.levers.topk_mc
        && kernels.rows.topk_mc_pass.0 != 0
        && kernels.rows.topk_mc_collect.0 != 0
        && scratch.topk_mc.holds(rows, topk_mc_ctas(pools))
}

/// 2026-10-21: The histogram passes of the multi-CTA select, each a launch of `dsa_topk_mc_pass`.
pub(crate) const TOPK_MC_PASSES: u32 = 4;

/// 2026-10-23: The multi-CTA select of `rows` rows at `c` CTAs per row: [`TOPK_MC_PASSES`] launches
/// of `pass`, then `collect` (the layer's kernels, or a plan's). Refuses a workspace that does
/// not hold that many rows and CTAs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_mc(
    gpu: &dyn GpuBackend,
    (pass, collect): (KernelHandle, KernelHandle),
    c: usize,
    rows: usize,
    scratch: &DsaSelectScratch,
    (npools_a, selk_a): (usize, usize),
    gd: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        scratch.topk_mc.holds(rows, c),
        "DSA select: the multi-CTA top-k workspace does not hold {rows} rows at {c} CTAs"
    );
    let (c, rows) = (c as u32, rows as u32);
    for p in 0..TOPK_MC_PASSES {
        KernelLaunch::new(gpu, pass)
            .grid([c, rows, 1])
            .block([ROW_BLOCK, 1, 1])
            .arg_ptr(scratch.scores)
            .arg_ptr(scratch.topk_mc.ws)
            .arg_u32(rows)
            .arg_u32(npools_a as u32)
            .arg_u32(selk_a as u32)
            .arg_ptr(gd)
            .arg_u32(p)
            .launch(stream)?;
    }
    KernelLaunch::new(gpu, collect)
        .grid([c, rows, 1])
        .block([ROW_BLOCK, 1, 1])
        .arg_ptr(scratch.scores)
        .arg_ptr(scratch.topk_mc.ws)
        .arg_ptr(scratch.selected)
        .arg_u32(rows)
        .arg_u32(npools_a as u32)
        .arg_u32(selk_a as u32)
        .arg_ptr(gd)
        .launch(stream)
}

impl DsaRowsKernels {
    /// 2026-10-24: These kernels with the multi-CTA top-k as one segment ([`Self::topk_seg`]);
    /// refused unless `dsa_topk_mc_seg` resolved.
    pub fn with_topk_seg(mut self) -> Result<Self> {
        ensure!(
            self.topk_mc_seg.0 != 0,
            "--glm-dsa-topk-seg needs dsa_topk_mc_seg on this target"
        );
        self.topk_seg = true;
        Ok(self)
    }
}

/// 2026-10-24: The multi-CTA select of `rows` rows at `c` CTAs per row as one launch of `seg`
/// (`dsa_topk_mc_seg`) on `c x rows` blocks. Refuses an unresolved kernel, a workspace that does
/// not hold that many rows and CTAs, or a region without the segment's control block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_mc_seg(
    gpu: &dyn GpuBackend,
    seg: KernelHandle,
    c: usize,
    rows: usize,
    scratch: &DsaSelectScratch,
    (npools_a, selk_a): (usize, usize),
    gd: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        seg.0 != 0,
        "--glm-dsa-topk-seg needs dsa_topk_mc_seg on this target"
    );
    let region = &scratch.topk_mc;
    ensure!(
        region.holds(rows, c) && region.ctl.0 != 0,
        "DSA select: the multi-CTA top-k workspace does not hold {rows} rows at {c} CTAs \
         with a segment control block"
    );
    KernelLaunch::new(gpu, seg)
        .grid([(c * rows) as u32, 1, 1])
        .block([ROW_BLOCK, 1, 1])
        .arg_ptr(scratch.scores)
        .arg_ptr(region.ws)
        .arg_ptr(scratch.selected)
        .arg_u32(rows as u32)
        .arg_u32(npools_a as u32)
        .arg_u32(selk_a as u32)
        .arg_ptr(gd)
        .arg_u32(c as u32)
        .arg_ptr(region.ctl)
        .launch(stream)?;
    // 2026-10-24: Once per process: the serve log's proof that R4 reached the select.
    static TAKEN: std::sync::Once = std::sync::Once::new();
    TAKEN.call_once(|| {
        tracing::info!("GLM DSA: multi-CTA top-k as one segment ({rows} rows x {c} CTAs)")
    });
    Ok(())
}

/// 2026-10-21: The pass's top-k over `scratch.scores` into `scratch.selected`; `(npools_a,
/// np2_a, selk_a)` are the pass's launch arguments (`launch::run`), `gd` its device geometry.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_topk(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextDsaKernels,
    geom: &DsaSelectGeometry,
    scratch: &DsaSelectScratch,
    ceiling: bool,
    (npools_a, np2_a, selk_a): (usize, usize, usize),
    gd: DevicePtr,
    stream: u64,
) -> Result<()> {
    let rows = geom.q_rows as u32;
    // 2026-10-12: `METRALE_GLM_DSA_TOPK_RADIX=1`: the same selection by radix select.
    // 2026-10-21: a ceiling launch by `METRALE_GLM_DSA_DECODE_TOPK_RADIX` (default on), over
    // the live pools in `gd`.
    let radix = topk_radix_for(
        ceiling,
        selk_a,
        kernels.rows.topk_radix,
        scratch.levers.topk_radix,
        scratch.levers.decode_topk_radix,
    );
    if topk_mc_for(kernels, scratch, ceiling, radix, geom.q_rows, npools_a) {
        let (pass, collect) = (kernels.rows.topk_mc_pass, kernels.rows.topk_mc_collect);
        let c = topk_mc_ctas(npools_a);
        if kernels.rows.topk_seg {
            return launch_mc_seg(
                gpu,
                kernels.rows.topk_mc_seg,
                c,
                geom.q_rows,
                scratch,
                (npools_a, selk_a),
                gd,
                stream,
            );
        }
        return launch_mc(
            gpu,
            (pass, collect),
            c,
            geom.q_rows,
            scratch,
            (npools_a, selk_a),
            gd,
            stream,
        );
    }
    if radix {
        return KernelLaunch::new(gpu, kernels.rows.topk_radix)
            .grid([rows, 1, 1])
            .block([ROW_BLOCK, 1, 1])
            .arg_ptr(scratch.scores)
            .arg_ptr(scratch.selected)
            .arg_u32(rows)
            .arg_u32(npools_a as u32)
            .arg_u32(selk_a as u32)
            .arg_ptr(gd)
            .launch(stream);
    }
    // 2026-09-25: `np2_a` is at most `topk_tile()`, so the request is at most
    // `topk_smem_for_tile(topk_tile())`, within `TOPK_SMEM_CEILING`.
    KernelLaunch::new(gpu, kernels.topk_pools)
        .grid([rows, 1, 1])
        .block([ROW_BLOCK, 1, 1])
        .shared_mem(topk_smem_for_tile(np2_a) as u32)
        .arg_ptr(scratch.scores)
        .arg_ptr(scratch.selected)
        .arg_u32(rows)
        .arg_u32(npools_a as u32)
        .arg_u32(np2_a as u32)
        .arg_u32(selk_a as u32)
        .arg_ptr(gd)
        .launch(stream)
}
