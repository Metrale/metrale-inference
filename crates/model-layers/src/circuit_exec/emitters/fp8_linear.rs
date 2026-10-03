// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The FP8 block-scaled (128 x 128) W8A16 projections of the attention and
//! GatedDeltaNet mixers, as the canonical row tiers run them: `w8a16_tc_rows` (the row-tile
//! kernel, `ops/w8a16_tc_rows.rs`: `_16`, `_32` or `_64` by rows, `_64c` past 64 rows in one
//! launch).
//!
//! Owner: model-layers (MoE) circuit emitters; the projections are the Qwen3.6 FP8 checkpoints'.
//! Invariants:
//! - Each launch's entry is checked against the one the launcher picks for its row count: the
//!   attention projections run in chunks of 64 rows past 64 (`qkv_fp8_batch.rs`,
//!   `attn/o_proj.rs`, each chunk's entry by its own rows), the GDN projections in one launch
//!   (`row_tier_proj.rs`).
//! - Input and output are read with their row strides (a row pack is fine), as the kernels
//!   take `lda` and `ldc`.

use anyhow::{Result, anyhow, ensure};
use metrale_circuit::{LinearRole, OpKind};

use super::super::bindings::{BoundWeight, WeightSlot};
use super::super::compile::{Cx, OpEmitter};
use super::rows;
use crate::layers::ops;
use crate::weight_map::Fp8Weight;

/// 2026-10-03: The one projection of the group, its role and its FP8 weight.
fn fp8_proj(cx: &Cx<'_>) -> Result<(LinearRole, Fp8Weight)> {
    let OpKind::Linear(role) = cx.g.node(0).op else {
        return Err(anyhow!("`{}` is no projection", cx.g.node(0).id));
    };
    ensure!(
        cx.g.group.nodes.len() == 1,
        "an FP8 projection group of {} nodes",
        cx.g.group.nodes.len()
    );
    match cx.weight(0, WeightSlot::Linear(role))? {
        BoundWeight::Fp8(w) => Ok((role, w)),
        other => Err(anyhow!(
            "`{}`: expected an FP8 W8A16 weight, the layer holds {}",
            cx.g.node(0).id,
            other.family()
        )),
    }
}

/// 2026-10-03: The row-tile entry `ops::w8a16_tc_rows` launches for `m` rows.
fn tc_rows_entry(m: u32) -> &'static str {
    match m {
        0..=16 => "w8a16_tc_rows_16",
        17..=32 => "w8a16_tc_rows_32",
        33..=64 => "w8a16_tc_rows_64",
        _ => "w8a16_tc_rows_64c",
    }
}

/// 2026-10-03: The attention projections, which run past 64 rows in 64-row chunks.
fn chunked(role: LinearRole) -> bool {
    matches!(
        role,
        LinearRole::Q | LinearRole::K | LinearRole::V | LinearRole::O
    )
}

/// 2026-10-03: `w8a16_tc_rows`.
pub(crate) struct W8a16TcRows;

impl OpEmitter for W8a16TcRows {
    fn id(&self) -> &'static str {
        "w8a16_tc_rows"
    }

    fn emit(&self, cx: &mut Cx<'_>) -> Result<()> {
        let (role, w) = fp8_proj(cx)?;
        let (x, lda) = cx.strided(cx.g.input(0, 0)?)?;
        let (y, ldc) = cx.strided(cx.g.output(0, 0)?)?;
        let (m, n, k) = (rows(cx)?, w.n, w.k);
        ensure!(
            ops::w8a16_tc_rows_shape_ok(m, n, k, lda, ldc),
            "`{}`: m={m} n={n} k={k} lda={lda} ldc={ldc} is outside the row-tile kernel",
            cx.g.node(0).id
        );
        let step = if chunked(role) {
            ops::W8A16_TC_ROWS_MAX_M
        } else {
            m
        };
        let chunks: Vec<u32> = (0..m.div_ceil(step))
            .map(|c| (m - c * step).min(step))
            .collect();
        ensure!(
            cx.g.group.kernels.len() == chunks.len(),
            "group {}: the plan lists {} kernels for {} launches of {m} rows",
            cx.g.index,
            cx.g.group.kernels.len(),
            chunks.len()
        );
        for (i, &rows_i) in chunks.iter().enumerate() {
            super::expect_kernel(cx, i, tc_rows_entry(rows_i))?;
            let done = i * step as usize;
            let (xi, yi) = (
                x.offset(done * lda as usize * 2),
                y.offset(done * ldc as usize * 2),
            );
            cx.push(
                i,
                Box::new(move |e| {
                    ops::w8a16_tc_rows(
                        e.gpu,
                        xi,
                        w.weight,
                        w.row_scale,
                        yi,
                        rows_i,
                        n,
                        k,
                        lda,
                        ldc,
                        e.stream,
                    )
                }),
            )?;
        }
        Ok(())
    }
}

/// 2026-10-03: The FP8 projection emitters.
pub(super) static ALL: &[&dyn OpEmitter] = &[&W8a16TcRows];
