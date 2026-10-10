// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Byte comparison of a tensor-parallel drafter rank's quantized weight copies
//! (`--draft-weight-quantization`, `dflash_head::draft_weight`) with the one-rank drafter's:
//! each rank copy must equal its rows or columns of the one-rank copy, scales included.
//!
//! Owner: model-arch examples.
//! Invariants: a copy in another format than its reference, or of another size, compares
//! unequal; nothing is written to the device.

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::dflash_head::tp::TpPlan;
use metrale_model_arch::dflash_head::{BlockDiffusionDraftHead, DraftWeight};

/// 2026-10-10: A copy and its `[rows, cols]` shape.
#[derive(Clone, Copy)]
pub struct WeightCopy<'a> {
    pub w: &'a DraftWeight,
    pub rows: usize,
    pub cols: usize,
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, len: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; len];
    gpu.copy_d2h(p, &mut v)?;
    Ok(v)
}

/// 2026-10-10: Rows `r0..r0 + n`, bytes `b0..b0 + bw` of each, from a row-major buffer whose
/// rows are `row_bytes` long.
fn region(buf: &[u8], row_bytes: usize, r0: usize, n: usize, b0: usize, bw: usize) -> Vec<u8> {
    (r0..r0 + n)
        .flat_map(|r| {
            buf[r * row_bytes + b0..r * row_bytes + b0 + bw]
                .iter()
                .copied()
        })
        .collect()
}

/// 2026-10-10: Whether the `[n, k]` block at `(r0, c0)` of `a` equals the one at `(s0, d0)` of
/// `b`, codes and scales. FP8 row scales cover whole rows, so a column block compares the rows'
/// scales; NVFP4 group scales cover 16 columns, so `c0`, `d0` and `k` must be multiples of 16.
pub fn same_block(
    gpu: &dyn GpuBackend,
    a: WeightCopy<'_>,
    (r0, c0): (usize, usize),
    b: WeightCopy<'_>,
    (s0, d0): (usize, usize),
    (n, k): (usize, usize),
) -> Result<bool> {
    ensure!(r0 + n <= a.rows && c0 + k <= a.cols && s0 + n <= b.rows && d0 + k <= b.cols);
    Ok(match (a.w, b.w) {
        (DraftWeight::Fp8(x), DraftWeight::Fp8(y)) => {
            let (xa, ya) = (
                read(gpu, x.weight, a.rows * a.cols)?,
                read(gpu, y.weight, b.rows * b.cols)?,
            );
            let (xs, ys) = (
                read(gpu, x.row_scale, a.rows * 4)?,
                read(gpu, y.row_scale, b.rows * 4)?,
            );
            region(&xa, a.cols, r0, n, c0, k) == region(&ya, b.cols, s0, n, d0, k)
                && xs[r0 * 4..(r0 + n) * 4] == ys[s0 * 4..(s0 + n) * 4]
        }
        (DraftWeight::Nvfp4(x), DraftWeight::Nvfp4(y)) => {
            ensure!(c0 % 16 == 0 && d0 % 16 == 0 && k % 16 == 0);
            let (xa, ya) = (
                read(gpu, x.weight, a.rows * a.cols / 2)?,
                read(gpu, y.weight, b.rows * b.cols / 2)?,
            );
            let (xs, ys) = (
                read(gpu, x.weight_scale, a.rows * a.cols / 16)?,
                read(gpu, y.weight_scale, b.rows * b.cols / 16)?,
            );
            x.weight_scale_2.to_bits() == y.weight_scale_2.to_bits()
                && region(&xa, a.cols / 2, r0, n, c0 / 2, k / 2)
                    == region(&ya, b.cols / 2, s0, n, d0 / 2, k / 2)
                && region(&xs, a.cols / 16, r0, n, c0 / 16, k / 16)
                    == region(&ys, b.cols / 16, s0, n, d0 / 16, k / 16)
        }
        _ => false,
    })
}

/// 2026-10-10: One copy of a comparison and its `[rows, cols]` shape.
type Side<'a> = (Option<&'a DraftWeight>, usize, usize);
/// 2026-10-10: A comparison: its name, the reference and its block origin, the rank copy and
/// its block origin, and the block's `[rows, cols]`.
type Case<'a> = (
    String,
    Side<'a>,
    (usize, usize),
    Side<'a>,
    (usize, usize),
    (usize, usize),
);

/// 2026-10-10: Compare every copy of rank `p.rank`'s drafter `local` with its part of the
/// one-rank drafter `whole` (head dim `hd`): the seven layer GEMMs, the LM head and, when the
/// tier copies them, `fc`, the fused K/V (each layer's K and V rows) and the conv and selector
/// projections. Returns the number of blocks compared; errors on the first that differs.
pub fn check_rank_copies<'h>(
    gpu: &dyn GpuBackend,
    whole: &'h BlockDiffusionDraftHead,
    local: &'h BlockDiffusionDraftHead,
    p: &TpPlan,
    hd: usize,
) -> Result<usize> {
    let h = whole.hidden_size;
    let (nq, nkv, inter) = (
        whole.num_q_heads * hd,
        whole.num_kv_heads * hd,
        whole.intermediate_size,
    );
    let (lq, lkv, li) = (p.q_heads.1 * hd, p.kv_heads.1 * hd, p.inter.1);
    let mut n = 0;
    let mut cmp = |what: String,
                   a: Side<'_>,
                   at: (usize, usize),
                   b: Side<'_>,
                   bt: (usize, usize),
                   size: (usize, usize)|
     -> Result<()> {
        let (Some(aw), Some(bw)) = (a.0, b.0) else {
            bail!("{what}: a copy is missing");
        };
        let ca = WeightCopy {
            w: aw,
            rows: a.1,
            cols: a.2,
        };
        let cb = WeightCopy {
            w: bw,
            rows: b.1,
            cols: b.2,
        };
        ensure!(
            same_block(gpu, ca, at, cb, bt, size)?,
            "{what}: bytes differ"
        );
        n += 1;
        Ok(())
    };
    for (l, (w, r)) in whole.layers.iter().zip(&local.layers).enumerate() {
        let rows = |what: &str,
                    a: &'h Option<DraftWeight>,
                    b: &'h Option<DraftWeight>,
                    full: usize,
                    r0: usize,
                    len: usize|
         -> Case<'h> {
            (
                format!("layer {l} {what}"),
                (a.as_ref(), full, h),
                (r0, 0),
                (b.as_ref(), len, h),
                (0, 0),
                (len, h),
            )
        };
        for (what, a, at, b, bt, size) in [
            rows("q_proj", &w.q_proj_q, &r.q_proj_q, nq, p.q_heads.0 * hd, lq),
            rows(
                "k_proj",
                &w.k_proj_q,
                &r.k_proj_q,
                nkv,
                p.kv_heads.0 * hd,
                lkv,
            ),
            rows(
                "v_proj",
                &w.v_proj_q,
                &r.v_proj_q,
                nkv,
                p.kv_heads.0 * hd,
                lkv,
            ),
            rows(
                "gate_proj",
                &w.gate_proj_q,
                &r.gate_proj_q,
                inter,
                p.inter.0,
                li,
            ),
            rows("up_proj", &w.up_proj_q, &r.up_proj_q, inter, p.inter.0, li),
            (
                format!("layer {l} o_proj"),
                (w.o_proj_q.as_ref(), h, nq),
                (0, p.q_heads.0 * hd),
                (r.o_proj_q.as_ref(), h, lq),
                (0, 0),
                (h, lq),
            ),
            (
                format!("layer {l} down_proj"),
                (w.down_proj_q.as_ref(), h, inter),
                (0, p.inter.0),
                (r.down_proj_q.as_ref(), h, li),
                (0, 0),
                (h, li),
            ),
        ] {
            cmp(what, a, at, b, bt, size)?;
        }
    }
    let vocab = whole.vocab_size;
    cmp(
        "lm_head".into(),
        (whole.lm_head_q.as_ref(), vocab, h),
        (p.vocab.0, 0),
        (local.lm_head_q.as_ref(), p.vocab.1, h),
        (0, 0),
        (p.vocab.1, h),
    )?;
    let (Some(wa), Some(ra)) = (whole.aux_q.as_ref(), local.aux_q.as_ref()) else {
        ensure!(
            whole.aux_q.is_none() && local.aux_q.is_none(),
            "auxiliary copies on one drafter only"
        );
        return Ok(n);
    };
    let fc_k = whole.target_layer_ids.len() * whole.target_hidden_size;
    cmp(
        "fc".into(),
        (Some(&wa.fc), h, fc_k),
        (0, 0),
        (Some(&ra.fc), h, fc_k),
        (0, 0),
        (h, fc_k),
    )?;
    let layers = whole.num_layers;
    for l in 0..layers {
        for (part, what) in [(0, "K"), (1, "V")] {
            cmp(
                format!("fused K/V layer {l} {what}"),
                (Some(&wa.fused_kv), layers * 2 * nkv, h),
                ((2 * l + part) * nkv + p.kv_heads.0 * hd, 0),
                (Some(&ra.fused_kv), layers * 2 * lkv, h),
                ((2 * l + part) * lkv, 0),
                (lkv, h),
            )?;
        }
    }
    if let (Some(a), Some(b)) = (&wa.selector_hidden_proj, &ra.selector_hidden_proj) {
        let sr = whole.selector_rank;
        cmp(
            "selector projection".into(),
            (Some(a), sr, h),
            (0, 0),
            (Some(b), sr, h),
            (0, 0),
            (sr, h),
        )?;
    }
    let conv = 2 * whole.conv_kernel_size * (h / whole.conv_group_size.max(1));
    for (l, (w, r)) in whole.layers.iter().zip(&local.layers).enumerate() {
        for (what, a, b) in [
            (
                "attention conv",
                &w.attention_conv_proj_q,
                &r.attention_conv_proj_q,
            ),
            ("mlp conv", &w.mlp_conv_proj_q, &r.mlp_conv_proj_q),
        ] {
            cmp(
                format!("layer {l} {what} projection"),
                (a.as_ref(), conv, h),
                (0, 0),
                (b.as_ref(), conv, h),
                (0, 0),
                (conv, h),
            )?;
        }
    }
    Ok(n)
}
