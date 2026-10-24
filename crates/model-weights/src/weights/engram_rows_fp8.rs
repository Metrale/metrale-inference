// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: Row reads of the DeepSeek-V4.1 safetensors checkpoint's engram tables: one E4M3
//! code per value (`layers.L.engram.embed.weight`, `[rows, head_dim]`) and one E8M0 scale per
//! `head_dim / scale_cols` values (`layers.L.engram.embed.scale`, `[rows, scale_cols]`). The
//! tables are far larger than device memory, so a step reads only the rows its n-gram hashes
//! name, by position, and the caller decodes them.
//!
//! Owner: model-weights, DeepSeek-V4.1 engram.
//! Invariants:
//! - A table is admitted only with an E4M3 `[rows, head_dim]` code tensor and an E8M0
//!   `[rows, scale_cols]` scale tensor of the same row count, `scale_cols` dividing `head_dim`.
//! - `read_rows` refuses a row id past the table and a destination of the wrong size before
//!   any read.

use std::fs::File;

use anyhow::{Context, Result, ensure};

use super::expert_stream::pread;
use super::{DeferredTensor, WeightDtype};

/// 2026-10-19: One layer's table: open shard files and the two tensors' absolute offsets.
pub struct Fp8EngramTable {
    pub layer: usize,
    pub rows: usize,
    pub head_dim: usize,
    pub scale_cols: usize,
    codes_file: File,
    codes_at: u64,
    scales_file: File,
    scales_at: u64,
}

/// 2026-10-19: Every engram layer's table, sorted by layer.
pub struct Fp8EngramRows {
    tables: Vec<Fp8EngramTable>,
}

impl Fp8EngramRows {
    /// 2026-10-19: Admit `(layer, codes, scales)` per engram layer. Refuses a dtype or shape
    /// outside the invariants, a repeated layer, and an empty list.
    pub fn new(tables: Vec<(usize, DeferredTensor, DeferredTensor)>) -> Result<Self> {
        ensure!(!tables.is_empty(), "no engram tables");
        let mut out = Vec::with_capacity(tables.len());
        for (layer, codes, scales) in tables {
            ensure!(
                codes.dtype == WeightDtype::FP8E4M3 && codes.shape.len() == 2,
                "engram layer {layer}: codes are {:?} {:?}, expected E4M3 [rows, head_dim]",
                codes.dtype,
                codes.shape
            );
            ensure!(
                scales.dtype == WeightDtype::FP8E8M0 && scales.shape.len() == 2,
                "engram layer {layer}: scales are {:?} {:?}, expected E8M0 [rows, cols]",
                scales.dtype,
                scales.shape
            );
            let (rows, head_dim, scale_cols) = (codes.shape[0], codes.shape[1], scales.shape[1]);
            ensure!(
                scales.shape[0] == rows && scale_cols > 0 && head_dim % scale_cols == 0,
                "engram layer {layer}: scales {:?} do not tile codes {:?}",
                scales.shape,
                codes.shape
            );
            ensure!(
                out.iter().all(|t: &Fp8EngramTable| t.layer != layer),
                "engram layer {layer} given twice"
            );
            let open = |d: &DeferredTensor| {
                File::open(&d.path).with_context(|| format!("open {}", d.path.display()))
            };
            out.push(Fp8EngramTable {
                layer,
                rows,
                head_dim,
                scale_cols,
                codes_file: open(&codes)?,
                codes_at: codes.offset,
                scales_file: open(&scales)?,
                scales_at: scales.offset,
            });
        }
        out.sort_by_key(|t| t.layer);
        Ok(Fp8EngramRows { tables: out })
    }

    pub fn tables(&self) -> &[Fp8EngramTable] {
        &self.tables
    }

    pub fn table(&self, layer: usize) -> Option<&Fp8EngramTable> {
        self.tables.iter().find(|t| t.layer == layer)
    }

    /// 2026-10-19: Read rows `ids` of layer `layer`'s table: codes into `codes`
    /// (`ids.len() * head_dim` bytes) and scales into `scales` (`ids.len() * scale_cols`
    /// bytes), both in `ids` order.
    pub fn read_rows(
        &self,
        layer: usize,
        ids: &[u64],
        codes: &mut [u8],
        scales: &mut [u8],
    ) -> Result<()> {
        let t = self
            .table(layer)
            .with_context(|| format!("layer {layer} has no engram table"))?;
        let (hd, sc) = (t.head_dim, t.scale_cols);
        ensure!(
            codes.len() == ids.len() * hd && scales.len() == ids.len() * sc,
            "engram layer {layer}: {} code and {} scale bytes for {} rows",
            codes.len(),
            scales.len(),
            ids.len()
        );
        if let Some(&bad) = ids.iter().find(|&&id| id >= t.rows as u64) {
            anyhow::bail!("engram row {bad} >= {} (layer {layer})", t.rows);
        }
        for (i, &id) in ids.iter().enumerate() {
            pread(
                &t.codes_file,
                t.codes_at + id * hd as u64,
                &mut codes[i * hd..(i + 1) * hd],
            )
            .with_context(|| format!("engram layer {layer} row {id} codes"))?;
            pread(
                &t.scales_file,
                t.scales_at + id * sc as u64,
                &mut scales[i * sc..(i + 1) * sc],
            )
            .with_context(|| format!("engram layer {layer} row {id} scales"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "engram_rows_fp8_tests.rs"]
mod tests;
