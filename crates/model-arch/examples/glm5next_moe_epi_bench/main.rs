// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Byte gate and microbench of the fused gate/up epilogue
//! (`METRALE_GLM_MOE_W4A4_GU_FUSED=1`, `w4a4_grouped_gu_m{32,64,128}` in
//! `kernels/gb10/common/w4a4_grouped_core.cuh`) against the separate launches it replaces
//! (grouped gate, grouped up, `glm5next_swiglu_clamp`, the static down quantizer), at GLM-5.3's
//! TP=3 `tp` expert layout (hidden 4096, all 288 experts, top 8).
//!
//! 1. Byte gate, at `GLM_BENCH_ROWS` rows (default 17, 64, 512, 1036, 8208) of Zipf-skewed
//!    routing (`GLM_BENCH_ZIPF` exponents, default 0 = uniform and 1.2), on every tile entry,
//!    at two expert widths (704: the TP=3 slice, stored at 768 for the `_k64` down; 640: a plain
//!    multiple of 128) and two SwiGLU limits (GLM-5.3's 10.0, and the 3/4 quantile of the
//!    routing's |gate|, |up| so that a quarter of the values clamp): the fused launch's down
//!    input (codes, block scales, globals of every sorted slot row, pad included) against the
//!    separate launches', then the down GEMM's BF16 output from each. The fused outputs are
//!    poisoned first, so a row it fails to write fails too.
//! 2. Timing (CUDA-graph replay, median with p10 and max) on the 32- and 128-slot entries: the
//!    replaced segment (gate/up quantization, gate, up, SwiGLU, down quantization) against the
//!    gate/up quantization plus the fused launch, the whole routed sequence of each (sort, tiles
//!    and down included), and the down GEMM alone; with the saving per layer and per 32k prefill
//!    (`MOE_LAYERS` routed layers, 32768 / rows chunks).
//!
//! 3. 2026-10-14: The raster twins (`METRALE_GLM_MOE_W4A4_RASTER`, `w4a4_grouped_gu_m128_r` and
//!    `w4a4_grouped_mx_m128[_k64]_r`): in part 1 at the m128 entry, the fused down input and the
//!    down output of each group in [`RASTER_GATE`] against the plain entries' bytes (outputs
//!    poisoned first); in part 2, the fused launch and the down GEMM at the `GLM_BENCH_RASTER`
//!    groups (`<gu>,<down>`, default 16,4) against the plain entries, interleaved.
//!    Negative control (part 1, m128): the raster down on the reference input with every
//!    per-slot global doubled must differ from the reference output, or the bench fails.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error on the first byte mismatch.
//!
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//! GLM_BENCH_GPU_ORDINAL=0 cargo run -p metrale-model-arch --release \
//!     --example glm5next_moe_epi_bench --features cuda,gpu-examples
//! ```

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_mlp::forward::{GroupedEntry, w4a4_grouped_tile_index};

#[allow(dead_code)]
#[path = "../glm5next_moe_wide_bench/device.rs"]
mod device;
use device::{Rng, fmt, read, time_set};

#[allow(dead_code)]
#[path = "../glm5next_moe_prefill_w4a4_bench/launch.rs"]
mod launch;
use launch::{Bufs, EXPERTS, HIDDEN, KP, Kern, MI, MT, TOP_K, bufs, sort, table, tiles};

mod route;
use route::{LIMIT, Width, down, fused, max_load, quant_x, quartile_limit, separate, zipf_routing};

/// 2026-10-14: GLM-5.3's routed layers (45 layers, the first 3 dense).
const MOE_LAYERS: f64 = 42.0;
const PREFILL: f64 = 32768.0;
/// 2026-10-14: Raster groups of the byte gate: 1 (column-major), a non-divisor, the measured
/// down and gate/up picks, and one past every tile count (a single band).
const RASTER_GATE: [u32; 5] = [1, 3, 4, 16, 4096];

/// 2026-10-14: The fused entries per tile and the m128 fused entry's `_r` twin.
struct GuKerns {
    plain: [KernelHandle; 3],
    r: KernelHandle,
}

impl GuKerns {
    fn entry(&self, ti: usize, raster: Option<u32>) -> GroupedEntry {
        match raster {
            Some(g) => GroupedEntry {
                kern: self.r,
                raster: Some(g),
            },
            None => GroupedEntry::plain(self.plain[ti]),
        }
    }
}

/// 2026-10-14: The fused launch's down input buffers, `rows` rows at the widest stored width.
fn dq_alloc(gpu: &dyn GpuBackend, slots: usize) -> Result<[DevicePtr; 3]> {
    Ok([
        gpu.alloc(slots * KP / 2)?,
        gpu.alloc(slots * KP / 16)?,
        gpu.alloc(slots * 4)?,
    ])
}

/// 2026-10-14: `got` against `want` row by row (`row` bytes a row, `slots` rows).
fn same_rows(name: &str, got: &[u8], want: &[u8], row: usize, at: &str) -> Result<()> {
    ensure!(got.len() == want.len(), "{at}: {name} lengths differ");
    for (r, (g, w)) in got.chunks(row).zip(want.chunks(row)).enumerate() {
        ensure!(
            g == w,
            "{at}: {name} of sorted slot row {r} differs from the separate launches"
        );
    }
    Ok(())
}

/// 2026-10-14: The separate launches' gate and up values of this routing (tile-independent):
/// the second SwiGLU limit of part 1.
fn routing_limit(gpu: &dyn GpuBackend, kn: &Kern, w: &Width, b: &Bufs) -> Result<f32> {
    let slots = b.rows * TOP_K;
    sort(gpu, kn, b, 0)?;
    tiles(gpu, kn, b, MT[0], 0)?;
    quant_x(gpu, kn, b, 0)?;
    separate(gpu, kn, w, b, 0, LIMIT, 0)?;
    let mut v = read(gpu, b.gate, slots * w.mi * 2)?;
    v.extend(read(gpu, b.upo, slots * w.mi * 2)?);
    Ok(quartile_limit(&v))
}

/// 2026-10-14: Part 1 at one routing, width, tile entry and limit.
#[allow(clippy::too_many_arguments)]
fn byte_gate(
    gpu: &dyn GpuBackend,
    kn: &Kern,
    gu: &GuKerns,
    w: &Width,
    b: &Bufs,
    dq: [DevicePtr; 3],
    ti: usize,
    limit: f32,
) -> Result<()> {
    let slots = b.rows * TOP_K;
    let at = format!("rows={} n={} mt={} limit={limit}", b.rows, w.mi, MT[ti]);
    let bytes = w.dq_bytes(slots);
    sort(gpu, kn, b, 0)?;
    tiles(gpu, kn, b, MT[ti], 0)?;
    quant_x(gpu, kn, b, 0)?;
    separate(gpu, kn, w, b, ti, limit, 0)?;
    let act = [b.aq, b.as_, b.ag];
    let mut want = Vec::with_capacity(3);
    for (p, n) in act.into_iter().zip(bytes) {
        want.push(read(gpu, p, n)?);
    }
    down(gpu, w, b, act, b.out, ti, None, 0)?;
    let want_out = read(gpu, b.out, slots * HIDDEN * 2)?;
    quant_x(gpu, kn, b, 0)?;
    let groups: &[u32] = if ti == 2 { &RASTER_GATE } else { &[] };
    for raster in std::iter::once(None).chain(groups.iter().copied().map(Some)) {
        let at = format!("{at} raster={raster:?}");
        for (p, n) in dq.into_iter().zip(bytes) {
            gpu.copy_h2d(&vec![0xA5u8; n], p)?;
        }
        fused(gpu, gu.entry(ti, raster), w, b, dq, ti, limit, 0)?;
        let names = ["down codes", "down block scales", "down globals"];
        for ((p, n), (name, want)) in dq.into_iter().zip(bytes).zip(names.iter().zip(&want)) {
            same_rows(name, &read(gpu, p, n)?, want, n / slots, &at)?;
        }
        gpu.copy_h2d(&vec![0xA5u8; slots * HIDDEN * 2], b.out_ref)?;
        down(gpu, w, b, dq, b.out_ref, ti, raster, 0)?;
        let got_out = read(gpu, b.out_ref, slots * HIDDEN * 2)?;
        same_rows("down output", &got_out, &want_out, HIDDEN * 2, &at)?;
    }
    if ti == 2 {
        raster_control(gpu, w, b, dq, &want, &want_out, &at)?;
    }
    ensure!(
        want_out.iter().any(|&x| x != 0) && want[0].iter().any(|&x| x != 0),
        "{at}: the separate launches wrote nothing"
    );
    Ok(())
}

/// 2026-10-14: The raster gate's negative control: the raster down (group 16) on the separate
/// launches' down input with every per-slot global doubled must NOT match their down output, so
/// a raster-path difference is visible to the comparison above (a blind gate fails here).
fn raster_control(
    gpu: &dyn GpuBackend,
    w: &Width,
    b: &Bufs,
    dq: [DevicePtr; 3],
    want: &[Vec<u8>],
    want_out: &[u8],
    at: &str,
) -> Result<()> {
    let globals: Vec<u8> = want[2]
        .chunks_exact(4)
        .flat_map(|c| (2.0 * f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes())
        .collect();
    for (p, v) in dq.into_iter().zip([&want[0], &want[1], &globals]) {
        gpu.copy_h2d(v, p)?;
    }
    gpu.copy_h2d(&vec![0xA5u8; want_out.len()], b.out_ref)?;
    down(gpu, w, b, dq, b.out_ref, 2, Some(RASTER_GATE[3]), 0)?;
    let got = read(gpu, b.out_ref, want_out.len())?;
    ensure!(
        same_rows("down output", &got, want_out, HIDDEN * 2, at).is_err(),
        "{at}: negative control (globals x2 into the raster down) matched the reference: the \
         raster byte gate cannot see a difference"
    );
    Ok(())
}

/// 2026-10-14: Part 2 at one routing and width, on tile entry `ti`.
#[allow(clippy::too_many_arguments)]
fn timing(
    gpu: &dyn GpuBackend,
    kn: &Kern,
    gu: &GuKerns,
    w: &Width,
    b: &Bufs,
    dq: [DevicePtr; 3],
    ti: usize,
    s: u64,
) -> Result<()> {
    let (rows, mt) = (b.rows, MT[ti]);
    let act = [b.aq, b.as_, b.ag];
    sort(gpu, kn, b, 0)?;
    tiles(gpu, kn, b, mt, 0)?;
    gpu.synchronize(0)?;
    let sep = || -> Result<()> {
        quant_x(gpu, kn, b, s)?;
        separate(gpu, kn, w, b, ti, LIMIT, s)
    };
    let fus = || -> Result<()> {
        quant_x(gpu, kn, b, s)?;
        fused(gpu, gu.entry(ti, None), w, b, dq, ti, LIMIT, s)
    };
    let whole_sep = || -> Result<()> {
        sort(gpu, kn, b, s)?;
        tiles(gpu, kn, b, mt, s)?;
        sep()?;
        down(gpu, w, b, act, b.out, ti, None, s)
    };
    let whole_fus = || -> Result<()> {
        sort(gpu, kn, b, s)?;
        tiles(gpu, kn, b, mt, s)?;
        fus()?;
        down(gpu, w, b, dq, b.out, ti, None, s)
    };
    let down_only = || down(gpu, w, b, act, b.out, ti, None, s);
    let r = time_set(gpu, s, &[&sep, &fus, &whole_sep, &whole_fus, &down_only])?;
    let at = format!("rows={rows:4} n={} m{mt:<3}", w.mi);
    for (name, x) in [
        ("separate gate/up segment", r[0]),
        ("fused gate/up segment", r[1]),
        ("whole routed, separate", r[2]),
        ("whole routed, fused", r[3]),
        ("down GEMM alone", r[4]),
    ] {
        println!("{at} {name:26} {}", fmt(x));
    }
    for (name, a, f) in [("segment", r[0][0], r[1][0]), ("whole", r[2][0], r[3][0])] {
        let d = a - f;
        println!(
            "{at} saving ({name}): {d:.1} us per layer ({:.1}%), {:.3} s per {PREFILL} prefill",
            100.0 * d / a,
            d * 1e-6 * MOE_LAYERS * PREFILL / rows as f64
        );
    }
    if ti == 2 {
        raster_timing(gpu, gu, w, b, dq, &at, s)?;
    }
    Ok(())
}

/// 2026-10-14: Part 3's timing on the m128 entry: the fused launch and the down GEMM, plain
/// against the `GLM_BENCH_RASTER` groups, interleaved.
fn raster_timing(
    gpu: &dyn GpuBackend,
    gu: &GuKerns,
    w: &Width,
    b: &Bufs,
    dq: [DevicePtr; 3],
    at: &str,
    s: u64,
) -> Result<()> {
    let v = std::env::var("GLM_BENCH_RASTER").unwrap_or_else(|_| "16,4".to_string());
    let (gg, gd) = v
        .split_once(',')
        .and_then(|(a, d)| Some((a.parse::<u32>().ok()?, d.parse::<u32>().ok()?)))
        .with_context(|| format!("GLM_BENCH_RASTER={v:?}: expected <gu>,<down>"))?;
    let act = [b.aq, b.as_, b.ag];
    let fp = || fused(gpu, gu.entry(2, None), w, b, dq, 2, LIMIT, s);
    let fr = || fused(gpu, gu.entry(2, Some(gg)), w, b, dq, 2, LIMIT, s);
    let dp = || down(gpu, w, b, act, b.out, 2, None, s);
    let dr = || down(gpu, w, b, act, b.out, 2, Some(gd), s);
    let r = time_set(gpu, s, &[&fp, &fr, &dp, &dr])?;
    let chunks = PREFILL / b.rows as f64;
    for (name, p, x) in [
        (format!("fused gate/up raster {gg}"), r[0], r[1]),
        (format!("down GEMM raster {gd}"), r[2], r[3]),
    ] {
        println!("{at} {name:26} plain {} raster {}", fmt(p), fmt(x));
        let d = p[0] - x[0];
        println!(
            "{at} saving ({name}): {d:.1} us per layer ({:.1}%), {:.3} s per {PREFILL} prefill",
            100.0 * d / p[0],
            d * 1e-6 * MOE_LAYERS * chunks
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let ordinal = std::env::var("GLM_BENCH_GPU_ORDINAL")
        .context("GLM_BENCH_GPU_ORDINAL names the GPU to run on")?
        .parse()?;
    let target = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    let gpu = MetraleCudaBackend::new(ordinal, &target.modules)?;
    let stream = gpu.create_stream()?;
    let g: &dyn GpuBackend = &gpu;
    let m = "w4a4_grouped_mx";
    let q = "w4a4_gemv_mx_moe";
    let kn = Kern {
        sort: g.kernel("moe", "moe_sort_by_expert")?,
        tiles: g.kernel(m, "w4a4_grouped_tiles")?,
        quant: g.kernel(q, "w4a4_quant_rows_static")?,
        quant_k64: g.kernel(q, "w4a4_quant_rows_static_k64")?,
        swiglu: g.kernel("glm5next_ffn", "glm5next_swiglu_clamp")?,
        slots: g.kernel(q, "w4a4_gemv_mx8_moe_slots")?,
        slots_k64: g.kernel(q, "w4a4_gemv_mx8_moe_slots_k64")?,
        mx: [
            g.kernel(m, "w4a4_grouped_mx_m32")?,
            g.kernel(m, "w4a4_grouped_mx_m64")?,
            g.kernel(m, "w4a4_grouped_mx_m128")?,
            g.kernel(m, "w4a4_grouped_mx_m128_s6")?,
        ],
        mx_k64: [
            g.kernel(m, "w4a4_grouped_mx_m32_k64")?,
            g.kernel(m, "w4a4_grouped_mx_m64_k64")?,
            g.kernel(m, "w4a4_grouped_mx_m128_k64")?,
            g.kernel(m, "w4a4_grouped_mx_m128_k64")?,
        ],
        mx32: g.kernel("w4a4_gemv_mx", "w4a4_gemv_mx32")?,
        w4a16: g.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128")?,
    };
    let gu = GuKerns {
        plain: [
            g.kernel(m, "w4a4_grouped_gu_m32")?,
            g.kernel(m, "w4a4_grouped_gu_m64")?,
            g.kernel(m, "w4a4_grouped_gu_m128")?,
        ],
        r: g.kernel(m, "w4a4_grouped_gu_m128_r")?,
    };
    let down_r = [
        g.kernel(m, "w4a4_grouped_mx_m128_r")?,
        g.kernel(m, "w4a4_grouped_mx_m128_k64_r")?,
    ];
    let mut rng = Rng(0x243f_6a88_85a3_08d3);
    let mut widths = Vec::with_capacity(2);
    for mi in [MI, 640] {
        let t = [
            table(g, &mut rng, mi, HIDDEN)?,
            table(g, &mut rng, mi, HIDDEN)?,
            table(g, &mut rng, HIDDEN, mi)?,
        ];
        widths.push(Width::new(&kn, mi, t, down_r));
    }
    let list = |name: &str, default: &str| -> Vec<String> {
        std::env::var(name)
            .unwrap_or_else(|_| default.to_string())
            .split(',')
            .map(|x| x.trim().to_string())
            .collect()
    };
    let rows_list = list("GLM_BENCH_ROWS", "17,64,512,1036,8208");
    let zipf_list = list("GLM_BENCH_ZIPF", "0,1.2");
    for rows in rows_list {
        let rows: usize = rows
            .parse()
            .with_context(|| format!("GLM_BENCH_ROWS {rows}"))?;
        for z in &zipf_list {
            let z: f64 = z.parse().with_context(|| format!("GLM_BENCH_ZIPF {z}"))?;
            let b = bufs(g, &mut rng, rows)?;
            let ids = zipf_routing(&mut rng, rows, z);
            let ib: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
            g.copy_h2d(&ib, b.ids)?;
            let slots = rows * TOP_K;
            let prod = w4a4_grouped_tile_index(slots, EXPERTS);
            println!(
                "rows={rows} zipf={z}: max {} slots on one expert (mean {:.1}); production tile m{}",
                max_load(&ids),
                slots as f64 / EXPERTS as f64,
                MT[prod]
            );
            let dq = dq_alloc(g, slots)?;
            for w in &widths {
                let q = routing_limit(g, &kn, w, &b)?;
                for ti in 0..3 {
                    for limit in [LIMIT, q] {
                        byte_gate(g, &kn, &gu, w, &b, dq, ti, limit)?;
                    }
                }
                println!(
                    "byte gate rows={rows} zipf={z} n={}: fused down input and down output \
                     identical on m32/m64/m128 and the m128 raster groups {RASTER_GATE:?} at \
                     limits {LIMIT} and {q}; negative control (globals x2, raster {}) differs",
                    w.mi, RASTER_GATE[3]
                );
                for ti in [0, 2] {
                    timing(g, &kn, &gu, w, &b, dq, ti, stream)?;
                }
            }
            for p in dq.into_iter().chain([
                b.ids, b.sorted, b.sorted_e, b.offsets, b.perm, b.tile_off, b.x, b.aq, b.as_, b.ag,
                b.gate, b.upo, b.act, b.out, b.out_ref,
            ]) {
                g.free(p)?;
            }
        }
    }
    Ok(())
}
