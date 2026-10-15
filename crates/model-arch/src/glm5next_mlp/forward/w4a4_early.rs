// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The routed experts' gate/up quantization of a grouped W4A4 launch as a value
//! ([`GateUpCodes`]): `METRALE_GLM_MOE_QUANT_EARLY` (`glm5next_moe_quant_early`) quantizes a
//! rank's shard of the rows with it ahead of the MoE and all-gathers the codes, and the grouped
//! launch (`w4a4_grouped`) checks that the codes it waits for are the ones it would write.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - [`gate_up_codes`] is `Some` exactly when `forward_moe_pieces` of those rows runs the
//!   grouped W4A4 GEMM, and [`GateUpCodes::of`] is the one constructor it and the grouped
//!   launch read.
//! - The quantizer writes each row from that row alone under a static global scale, so rows
//!   `[r0, r1)` launched at pointers `r0` rows on are the bytes of a launch of every row there.

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Glm5NextMlpWorkspace;
use super::route::{ExpertRoute, expert_route};
use super::w4a4::launch_quant;
use super::w4a4_grouped::takes_grouped;
use crate::glm5next_mlp::weights::{Glm5NextMoeWeights, W4a4ActScales};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-10-15: The gate/up quantization of a grouped W4A4 launch: the quantizer, the static
/// `input_scale`, the row width, and the codes, group scales and global scales of row 0 in a
/// scratch of `rows_cap` rows.
#[derive(Clone, Copy, Debug)]
pub struct GateUpCodes {
    pub kern: KernelHandle,
    pub gs: f32,
    pub k: usize,
    pub out: [DevicePtr; 3],
    pub rows_cap: usize,
}

impl GateUpCodes {
    /// 2026-10-15: The quantization the grouped launch makes of `h`-wide rows.
    pub(super) fn of(
        k: &Glm5NextMlpKernels,
        ws: &Glm5NextMlpWorkspace,
        h: usize,
        scales: W4a4ActScales,
    ) -> Self {
        Self {
            kern: k.w4a4_quant_static,
            gs: scales.gate_up,
            k: h,
            out: [ws.w4a4_aq, ws.w4a4_as, ws.w4a4_ag],
            rows_cap: ws.w4a4_rows,
        }
    }

    /// 2026-10-15: Bytes of one row of each output: E2M1 codes, E4M3 group scales, F32 global.
    pub fn row_bytes(&self) -> [usize; 3] {
        [self.k / 2, self.k / 16, 4]
    }

    /// 2026-10-15: The outputs' pointers `r` rows on.
    pub fn at(&self, r: usize) -> [DevicePtr; 3] {
        let b = self.row_bytes();
        [0, 1, 2].map(|i| self.out[i].offset(r * b[i]))
    }

    /// 2026-10-15: Whether `o` is the same quantization into the same scratch.
    pub fn same(&self, o: &Self) -> bool {
        self.kern.0 == o.kern.0
            && self.gs.to_bits() == o.gs.to_bits()
            && self.k == o.k
            && self.out == o.out
            && self.rows_cap == o.rows_cap
    }

    /// 2026-10-15: Quantize rows `[r0, r1)` of the BF16 rows at `x` into their rows of the
    /// scratch on `stream`.
    pub fn launch_rows(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        r0: usize,
        r1: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            r0 < r1 && r1 <= self.rows_cap,
            "GLM W4A4 gate/up codes: rows {r0}..{r1} outside the scratch of {} rows",
            self.rows_cap
        );
        launch_quant(
            gpu,
            self.kern,
            x.offset(r0 * self.k * 2),
            self.at(r0),
            r1 - r0,
            self.k,
            self.gs,
            stream,
        )
    }
}

/// 2026-10-15: The gate/up quantization a `forward_moe_pieces` of `rows` rows makes when it runs
/// the grouped W4A4 GEMM; `None` on every other path.
pub fn gate_up_codes(
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    ws: &Glm5NextMlpWorkspace,
    rows: usize,
) -> Result<Option<GateUpCodes>> {
    if expert_route(k, cfg, w, ws, rows) != ExpertRoute::W4a4 || !takes_grouped(rows) {
        return Ok(None);
    }
    let scales = w
        .act_scales
        .context("GLM MoE: the plan runs W4A4 but no scales were bound")?;
    let h = cfg.hidden;
    ensure!(
        rows <= ws.w4a4_rows && h <= ws.w4a4_k && h.is_multiple_of(128),
        "GLM W4A4 gate/up codes: {rows} rows of {h} do not fit the scratch of {} x {}",
        ws.w4a4_rows,
        ws.w4a4_k
    );
    Ok(Some(GateUpCodes::of(k, ws, h, scales)))
}
