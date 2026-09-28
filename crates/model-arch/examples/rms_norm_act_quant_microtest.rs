// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The three `rms_norm_act_quant` entries against `rms_norm` followed by the
//! quantizer each replaces, byte for byte:
//!   rms_norm_quant_fp8_row  vs quant_rowwise_fp8          (E4M3, FP32 scale per row)
//!   rms_norm_quant_fp8_g128 vs per_token_group_quant_fp8  (E4M3, FP32 scale per 128)
//!   rms_norm_quant_nvfp4    vs w4a4_quant_rows            (E2M1, E4M3 group-16 scales in
//!                                                          fragment order, FP32 row scale)
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exits with an error unless, on every compiled circuit target, both hidden sizes and every
//!   row count 1..=128, every output byte (values and scales) equals the chain's and rows past
//!   the launch are untouched.
//!
//! Run (GB10, the default all-target kernel build):
//!   cargo run -p metrale-model-arch --features cuda,gpu-examples \
//!     --example rms_norm_act_quant_microtest

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

#[path = "common/norm_fusion_fixture.rs"]
mod fixture;
use fixture::{EPS, HIDDENS, MAX_ROWS, Rng, backends, bytes, compare, matrix, sentinel, upload};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Fp8Row,
    Fp8G128,
    Nvfp4,
}

impl Kind {
    /// 2026-09-28: Bytes per row of each output: values, then one or two scale arrays.
    fn outputs(self, h: usize) -> Vec<usize> {
        match self {
            Kind::Fp8Row => vec![h, 4],
            Kind::Fp8G128 => vec![h, h / 128 * 4],
            Kind::Nvfp4 => vec![h / 2, h / 16, 4],
        }
    }
}

struct Kernels {
    norm: KernelHandle,
    row: KernelHandle,
    g128: KernelHandle,
    nvfp4: KernelHandle,
    fused_row: KernelHandle,
    fused_g128: KernelHandle,
    fused_nvfp4: KernelHandle,
}

/// 2026-09-28: The chain: rms_norm into a BF16 buffer, then the quantizer, as the engine
/// launches each (ops::rms_norm, ops::per_token_group_quant_fp8, quant_rowwise_fp8,
/// ops::w4a4_proj).
fn unfused(
    g: &dyn GpuBackend,
    k: &Kernels,
    kind: Kind,
    x: DevicePtr,
    w: DevicePtr,
    outs: &[DevicePtr],
    rows: usize,
    h: usize,
) -> Result<()> {
    let s = g.default_stream();
    let y = g.alloc(rows * h * 2)?;
    KernelLaunch::new(g, k.norm)
        .grid([rows as u32, 1, 1])
        .block([(h as u32).min(1024), 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(y)
        .arg_u32(h as u32)
        .arg_f32(EPS)
        .launch(s)?;
    let (r, hu) = (rows as u32, h as u32);
    match kind {
        Kind::Fp8Row => KernelLaunch::new(g, k.row)
            .grid([r, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(y)
            .arg_ptr(outs[0])
            .arg_ptr(outs[1])
            .arg_u32(r)
            .arg_u32(hu)
            .launch(s)?,
        Kind::Fp8G128 => KernelLaunch::new(g, k.g128)
            .grid([r, hu / 128, 1])
            .block([128, 1, 1])
            .arg_ptr(y)
            .arg_ptr(outs[0])
            .arg_ptr(outs[1])
            .arg_u32(r)
            .arg_u32(hu)
            .launch(s)?,
        Kind::Nvfp4 => KernelLaunch::new(g, k.nvfp4)
            .grid([r, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(y)
            .arg_ptr(outs[0])
            .arg_ptr(outs[1])
            .arg_ptr(outs[2])
            .arg_u32(hu)
            .launch(s)?,
    }
    g.synchronize(s)?;
    g.free(y)
}

fn fused(
    g: &dyn GpuBackend,
    k: &Kernels,
    kind: Kind,
    x: DevicePtr,
    w: DevicePtr,
    outs: &[DevicePtr],
    rows: usize,
    h: usize,
) -> Result<()> {
    let kernel = match kind {
        Kind::Fp8Row => k.fused_row,
        Kind::Fp8G128 => k.fused_g128,
        Kind::Nvfp4 => k.fused_nvfp4,
    };
    let mut l = KernelLaunch::new(g, kernel)
        .grid([rows as u32, 1, 1])
        .block([(h as u32).min(1024), 1, 1])
        .shared_mem((h * 2) as u32)
        .arg_ptr(x)
        .arg_ptr(w);
    for &p in outs {
        l = l.arg_ptr(p);
    }
    l.arg_u32(h as u32).arg_f32(EPS).launch(g.default_stream())
}

fn run_rows(
    g: &dyn GpuBackend,
    k: &Kernels,
    kind: Kind,
    x: DevicePtr,
    w: DevicePtr,
    rows: usize,
    h: usize,
) -> Result<usize> {
    let per_row = kind.outputs(h);
    let alloc = |_: &usize| -> Result<Vec<DevicePtr>> {
        per_row.iter().map(|b| sentinel(g, b * MAX_ROWS)).collect()
    };
    let (a, b) = (alloc(&0)?, alloc(&0)?);
    unfused(g, k, kind, x, w, &a, rows, h)?;
    fused(g, k, kind, x, w, &b, rows, h)?;
    let mut bad = 0;
    for ((pa, pb), n) in a.iter().zip(&b).zip(&per_row) {
        bad += compare(g, *pa, *pb, n * rows, n * MAX_ROWS)?;
    }
    for p in a.into_iter().chain(b) {
        g.free(p)?;
    }
    Ok(bad)
}

fn main() -> Result<()> {
    let mut failures = 0usize;
    for (target, backend) in backends()? {
        let g: &dyn GpuBackend = &backend;
        let q = "rms_norm_act_quant";
        let k = Kernels {
            norm: g.kernel("norm", "rms_norm")?,
            row: g.kernel("quant_rowwise_fp8", "quant_rowwise_fp8")?,
            g128: g.kernel("per_token_group_quant_fp8", "per_token_group_quant_fp8")?,
            nvfp4: g.kernel("w4a4_gemv_mx", "w4a4_quant_rows")?,
            fused_row: g.kernel(q, "rms_norm_quant_fp8_row")?,
            fused_g128: g.kernel(q, "rms_norm_quant_fp8_g128")?,
            fused_nvfp4: g.kernel(q, "rms_norm_quant_nvfp4")?,
        };
        for h in HIDDENS {
            let mut rng = Rng(0x0a4a_0000 + h as u64);
            let x = upload(g, &bytes(&matrix(&mut rng, MAX_ROWS, h, 7)))?;
            let w = fixture::weight(g, &mut rng, h)?;
            for kind in [Kind::Fp8Row, Kind::Fp8G128, Kind::Nvfp4] {
                let mut bad = 0usize;
                for rows in 1..=MAX_ROWS {
                    let b = run_rows(g, &k, kind, x, w, rows, h)?;
                    if b > 0 {
                        println!("  {target} H={h} {kind:?} rows={rows}: {b} mismatching bytes");
                    }
                    bad += b;
                }
                println!(
                    "{target:<16} H={h:<5} {kind:<8?} rows 1..={MAX_ROWS}: {bad} mismatching bytes"
                );
                failures += bad;
            }
            g.free(x)?;
            g.free(w)?;
        }
    }
    ensure!(failures == 0, "{failures} mismatching bytes");
    println!("PASS: rms_norm_act_quant is byte-identical to rms_norm + each quantizer");
    Ok(())
}
