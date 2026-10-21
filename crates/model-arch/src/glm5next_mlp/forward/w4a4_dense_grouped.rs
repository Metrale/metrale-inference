// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_DENSE_W4A4_GROUPED_MIN_ROWS=<r>`: the dense SwiGLU MLP of GLM-5.3's
//! layers 0-2 (the checkpoint's NVFP4 W4A4 Linears) at `r` rows or more runs each projection as
//! ONE group of the grouped W4A4 GEMM (`w4a4_grouped_mx_m128`: expert_offsets [0, m], a
//! one-entry pointer table, row r reading activation row r) instead of `w4a4_gemv_mx32` per
//! 32-row chunk, which re-reads the weight once per chunk (128 launches per projection and a
//! 4096-row prompt). The quantization runs once over all rows per projection input.
//!
//! Numerics: BIT-IDENTICAL to the chunk path it replaces. The chunk GEMV
//! (`w4a4_gemv_mx_tok_impl<4, 2>`) and the slot GEMV (`<1, 4>`) both give warp w the k128 chunks
//! w, w + 8, .. in ascending order with two block-scale MMAs per chunk on the same fragments, add
//! the 8 warp sums in order and write `bf16_rn(v * (Ag * scale2))`; the grouped kernel reproduces
//! that order per output (`w4a4_grouped_mx.cu`); and the static quantizer is one CTA per row
//! under a fixed global scale, so quantizing all rows at once writes each row's chunk bytes.
//! Gated byte for byte against the chunk path by `examples/glm5next_moe_prefill_w4a4_bench`
//! (part 3). So the choice depends on M only through the threshold and never changes a byte.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The lever is resolved once at startup (`MlpLevers::dense_grouped_min_rows`), and
//!   [`DenseGroupedTables::build`] refuses it without `METRALE_GLM_MOE_W4A4_GROUPED=1` (the
//!   workspace then quantizes every row of a sub-chunk at once) or without the grouped kernels
//!   in the target's PTX; a launch wider than the workspace is an error, never a fallback.
//! - No host buffer in a forward: `w4a4_grouped_one_group` writes the offsets [0, m] and the
//!   tile offsets from m passed by value, so a captured launch replays its own m.
//! - Unset or `0` launches exactly what the dense site launched before this lever.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::Glm5NextMlpWorkspace;
use super::w4a4_grouped::{W4A4_GROUPED_MT, grouped_launch, smem_bytes_of};
use super::w4a4_grouped_raster::GroupedEntry;
use crate::glm5next_mlp::weights::{Glm5NextDenseNvfp4Weights, Glm5NextExpertPtrTable, Nvfp4Proj};
use crate::glm5next_mlp::{DENSE_W4A4_CHUNK_ROWS, Glm5NextMlpKernels};

/// 2026-10-12: The narrowest value the lever takes: above one mx32 chunk, where the chunk path
/// reads each weight once anyway.
pub const MIN_ROWS_FLOOR: usize = DENSE_W4A4_CHUNK_ROWS + 1;

/// 2026-10-12: The grouped kernel's tile entry the dense site uses (the 128-slot tile).
pub(super) const TILE: usize = 2;

/// 2026-10-13: The widest K the grouped kernel takes (`w4a4_grouped_mx.cu`: the class-order
/// table).
const MAX_K: usize = 8192;

/// 2026-10-13: Whether the lever value `min` can run on this load: `grouped_ws` is
/// `METRALE_GLM_MOE_W4A4_GROUPED` (the workspace quantizes every row of a sub-chunk), `ready`
/// whether the grouped kernels the dense site launches resolved.
pub(crate) fn check_ready(min: usize, grouped_ws: bool, ready: bool) -> Result<(), String> {
    if !grouped_ws {
        return Err(format!(
            "METRALE_GLM_DENSE_W4A4_GROUPED_MIN_ROWS={min} needs METRALE_GLM_MOE_W4A4_GROUPED=1"
        ));
    }
    if !ready {
        return Err(format!(
            "METRALE_GLM_DENSE_W4A4_GROUPED_MIN_ROWS={min}: w4a4_grouped_one_group or \
             w4a4_grouped_mx_m128 is absent from this target's PTX"
        ));
    }
    Ok(())
}

/// 2026-10-12: One dense site's device-side inputs to the grouped kernel: a one-entry pointer
/// table per projection, the `[2]` i32 expert offsets and the `[2]` i32 tile offsets.
/// 2026-10-13: `min_rows` is the lever value read at load; a launch of `m >= min_rows` rows
/// takes the one-group GEMM ([`Self::takes`]).
#[derive(Debug, Clone, Copy)]
pub struct DenseGroupedTables {
    pub min_rows: usize,
    pub gate: Glm5NextExpertPtrTable,
    pub up: Glm5NextExpertPtrTable,
    pub down: Glm5NextExpertPtrTable,
    pub offsets: DevicePtr,
    pub tile_off: DevicePtr,
}

/// 2026-10-12: A one-entry table of `p`.
fn one_table(gpu: &dyn GpuBackend, p: &Nvfp4Proj) -> Result<Glm5NextExpertPtrTable> {
    let up = |bytes: &[u8]| -> Result<DevicePtr> {
        let d = gpu.alloc(bytes.len())?;
        gpu.copy_h2d(bytes, d)?;
        Ok(d)
    };
    Ok(Glm5NextExpertPtrTable {
        packed_ptrs: up(&p.packed.0.to_le_bytes())?,
        scale_ptrs: up(&p.scale.0.to_le_bytes())?,
        scale2_vals: up(&p.scale_2.to_le_bytes())?,
        scale_bases: DevicePtr::NULL,
    })
}

impl DenseGroupedTables {
    /// 2026-10-12: Build the tables for `w` at load when the lever is on; `None` when it is off.
    /// 2026-10-13: Refuses a lever this load cannot run ([`check_ready`]).
    pub fn build(
        gpu: &dyn GpuBackend,
        k: &Glm5NextMlpKernels,
        w: &Glm5NextDenseNvfp4Weights,
    ) -> Result<Option<Self>> {
        let Some(min_rows) = k.levers.mlp.dense_grouped_min_rows else {
            return Ok(None);
        };
        let g = k.w4a4_grouped;
        check_ready(
            min_rows,
            k.levers.mlp.w4a4_grouped,
            g.one_group.0 != 0 && g.mx[TILE].0 != 0,
        )
        .map_err(anyhow::Error::msg)?;
        Ok(Some(Self {
            min_rows,
            gate: one_table(gpu, &w.gate_proj)?,
            up: one_table(gpu, &w.up_proj)?,
            down: one_table(gpu, &w.down_proj)?,
            offsets: gpu.alloc(8)?,
            tile_off: gpu.alloc(8)?,
        }))
    }

    /// 2026-10-13: Whether a dense launch of `m` rows takes the one-group GEMM.
    pub fn takes(&self, m: usize) -> bool {
        m >= self.min_rows
    }
}

/// 2026-10-13: Check a dense launch of `m` rows, hidden `h` and width `inter` against the
/// kernels' contract (each is a K and an N: K % 128 == 0 and K <= 8192, which covers N % 8 == 0)
/// and the quantized scratch, before anything launches; then the offsets
/// `[0, m]` and the tile offsets of one group of `m` rows, on the device from `m` by value
/// (`w4a4_grouped_one_group`).
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    ws: &Glm5NextMlpWorkspace,
    t: &DenseGroupedTables,
    m: usize,
    h: usize,
    inter: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        [h, inter]
            .iter()
            .all(|&kk| kk.is_multiple_of(128) && kk <= MAX_K)
            && m <= ws.w4a4_rows,
        "GLM dense W4A4 grouped: hidden {h} and width {inter} must be multiples of 128 up to \
         {MAX_K}, and {m} rows must fit the {} quantized rows (needs \
         METRALE_GLM_MOE_W4A4_GROUPED=1)",
        ws.w4a4_rows
    );
    KernelLaunch::new(gpu, k.w4a4_grouped.one_group)
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(t.offsets)
        .arg_ptr(t.tile_off)
        .arg_u32(m as u32)
        .arg_u32(W4A4_GROUPED_MT[TILE] as u32)
        .launch(stream)
}

/// 2026-10-12: `c[m, n] = quantized rows x table^T` as one group of the 128-slot grouped GEMM
/// (shapes checked by [`prepare`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn proj(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    ws: &Glm5NextMlpWorkspace,
    t: &DenseGroupedTables,
    table: &Glm5NextExpertPtrTable,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    grouped_launch(
        gpu,
        GroupedEntry::plain(k.w4a4_grouped.mx[TILE]),
        TILE,
        smem_bytes_of(TILE),
        [ws.w4a4_aq, ws.w4a4_as, ws.w4a4_ag],
        table,
        c,
        t.offsets,
        t.tile_off,
        DevicePtr::NULL,
        1,
        m,
        n,
        kk,
        stream,
    )
}

#[cfg(test)]
#[path = "w4a4_dense_grouped_tests.rs"]
mod launch_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-13: The lever needs the grouped workspace and the kernels; each missing one is
    /// named.
    #[test]
    fn the_lever_is_refused_where_it_cannot_run() {
        assert_eq!(check_ready(64, true, true), Ok(()));
        let e = check_ready(64, false, true).unwrap_err();
        assert!(e.contains("METRALE_GLM_MOE_W4A4_GROUPED=1"), "{e}");
        let e = check_ready(64, true, false).unwrap_err();
        assert!(e.contains("w4a4_grouped_one_group"), "{e}");
    }
}
