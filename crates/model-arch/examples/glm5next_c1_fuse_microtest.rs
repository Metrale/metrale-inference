// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Byte gate and microbench for GLM-5.3's small per-layer decode kernels at the
//! per-rank shapes of the TP=3 layout, at 1 and 16 rows: each current kernel (old) against its
//! cheaper launch (new).
//!
//! 1. Router top-k (288 experts, top 8): `glm5next_router_topk` against
//!    `glm5next_router_topk_rank` (`METRALE_GLM_ROUTER_TOPK_RANK=1`), on drawn logits with many
//!    exact ties and on edge rows (NaN and infinite logits, a choice of exactly -1e30, fewer
//!    participating experts than slots), with and without the BF16 ladder and the
//!    renormalisation. ids and weights must be byte-identical.
//! 2. MoE combine (hidden 4096, top 8): one block per row against the row spread over several
//!    blocks (`METRALE_GLM_MOE_COMBINE_SPLIT=1`), slot order and expert-sorted order. Outputs must
//!    be byte-identical, and equal to a host reference. 2026-10-14: its second arm,
//!    eight elements per thread (`METRALE_GLM_MOE_COMBINE_VEC8=1`), at 1, 16 and 8192 rows,
//!    byte-equal to the split combine and the host reference, with a negative control
//!    (`common/glm5next_c1_combine_vec8.rs`; alone with `C1_FUSE_ONLY=combine_vec8`).
//! 3. KDA recurrence of one row (22 heads of 128): `kda_recurrent_decode_bf16_smem` against
//!    `kda_recurrent_decode_bf16_seq_reg` at T = 1 (`METRALE_GLM_KDA_ROW_REG=1`) for several
//!    block widths, sixteen consecutive tokens on one state. Every output and the final state must
//!    be byte-identical (`common/glm5next_c1_kda.rs`).
//! 4. KDA gate chain of one token: act-quant + f_b + kda_gate + beta sigmoid + act-quant + g_b
//!    against `kda_gates_w8a8_k128` (`METRALE_GLM_KDA_GATES_FUSED=1`), byte for byte
//!    (`common/glm5next_c1_kda_gates.rs`).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error on the first byte mismatch; prints GPU time per launch otherwise,
//!   measured on a captured CUDA graph of one launch per layer (`common/graph_timing.rs`).
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_c1_fuse_microtest \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_c1_combine_vec8.rs"]
mod c1_combine_vec8;
#[path = "common/glm5next_c1_kda.rs"]
mod c1_kda;
#[path = "common/glm5next_c1_kda_gates.rs"]
mod c1_kda_gates;
#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Context, Result, bail};
use fixture::{Lcg, same, up, up_f32};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use timing::report;

const ROWS: [usize; 2] = [1, 16];
/// 2026-10-10: GLM-5.3: 288 routed experts, top 8, hidden 4096, 42 MoE layers.
const EXPERTS: usize = 288;
const TOP_K: usize = 8;
const HIDDEN: usize = 4096;
const MOE_LAYERS: usize = 42;

fn zeros(g: &dyn GpuBackend, n: usize) -> Result<DevicePtr> {
    let p = g.alloc(n.max(1))?;
    g.memset(p, 0, n)?;
    Ok(p)
}

fn read(g: &dyn GpuBackend, s: u64, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(s)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-10: `timing::time_graph` of `copies` copies of `f`.
fn tg(
    g: &dyn GpuBackend,
    s: u64,
    copies: usize,
    f: &dyn Fn(u64) -> Result<()>,
) -> Result<(f64, f64, f64)> {
    timing::time_graph(g, s, copies, &mut |s| f(s))
}

/// 2026-10-10: Logits on a coarse grid (so sigmoid scores tie exactly), and from row 1 on the
/// edge rows: NaN and infinite logits, a row where only six experts take part (the sixth,
/// expert 11, at a choice of exactly -1e30; expert 13, below it, does not), then a row of
/// equal logits, where the choices tie in runs of about 70 experts and the cut at top_k falls
/// inside a tie (a reversed tie-break inside a run of 32 shows there and nowhere else).
fn router_logits(rng: &mut Lcg, rows: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..rows * EXPERTS)
        .map(|_| (rng.next() * 32.0).round() / 8.0)
        .collect();
    if rows > 1 {
        for e in (0..EXPERTS).step_by(7) {
            v[EXPERTS + e] = f32::NAN;
        }
        v[EXPERTS + 3] = f32::INFINITY;
        v[EXPERTS + 5] = f32::INFINITY;
        v[EXPERTS + 9] = f32::NEG_INFINITY;
    }
    if rows > 2 {
        for e in 0..EXPERTS {
            if e % 60 != 0 && e != 11 && e != 13 {
                v[2 * EXPERTS + e] = f32::NAN;
            }
        }
    }
    if rows > 3 {
        v[3 * EXPERTS..4 * EXPERTS].fill(0.0);
    }
    v
}

fn router(g: &dyn GpuBackend, s: u64, rng: &mut Lcg) -> Result<()> {
    let old = g.kernel("glm5next_ffn", "glm5next_router_topk")?;
    let new = g.kernel("glm5next_ffn", "glm5next_router_topk_rank")?;
    // 2026-10-10: Bias from a small set (more exact ties in the choice); expert 11 at -1e30, so
    // its choice is exactly the old kernel's seed, and expert 13 below it.
    let mut bias: Vec<f32> = (0..EXPERTS)
        .map(|_| [0.0f32, 0.015625, -0.015625, 0.03125][(rng.next().abs() * 3.99) as usize])
        .collect();
    bias[11] = -1e30;
    bias[13] = -2e30;
    let bias = up_f32(g, &bias)?;
    for rows in ROWS {
        let logits = up_f32(g, &router_logits(rng, rows))?;
        let (ib, wb) = (rows * TOP_K * 4, rows * TOP_K * 4);
        let bufs = [zeros(g, ib)?, zeros(g, wb)?, zeros(g, ib)?, zeros(g, wb)?];
        let launch = |k: KernelHandle, block: u32, ids, wts, ladder: u32, renorm: u32, s| {
            KernelLaunch::new(g, k)
                .grid([rows as u32, 1, 1])
                .block([block, 1, 1])
                .arg_ptr(logits)
                .arg_ptr(bias)
                .arg_ptr(ids)
                .arg_ptr(wts)
                .arg_u32(EXPERTS as u32)
                .arg_u32(TOP_K as u32)
                .arg_u32(1)
                .arg_f32(2.5)
                .arg_u32(renorm)
                .arg_u32(ladder)
                .launch(s)
        };
        let rank_block = EXPERTS.div_ceil(32) as u32 * 32;
        for (ladder, renorm) in [(0u32, 1u32), (1, 1), (0, 0), (1, 0)] {
            for p in bufs {
                g.memset(p, 0xA5, ib)?;
            }
            launch(old, 256, bufs[0], bufs[1], ladder, renorm, s)?;
            let what = format!("router top-k R={rows} ladder={ladder} renorm={renorm}");
            let ids = read(g, s, bufs[0], ib)?;
            let wts = read(g, s, bufs[1], wb)?;
            // 2026-10-10: The serve's block (one thread per expert) and a narrower one, where
            // the experts stride over the block (256 threads for 288 experts).
            for block in [rank_block, 256] {
                g.memset(bufs[2], 0xA5, ib)?;
                g.memset(bufs[3], 0xA5, wb)?;
                launch(new, block, bufs[2], bufs[3], ladder, renorm, s)?;
                same(
                    &format!("{what} block={block} ids"),
                    &ids,
                    &read(g, s, bufs[2], ib)?,
                )?;
                same(
                    &format!("{what} block={block} weights"),
                    &wts,
                    &read(g, s, bufs[3], wb)?,
                )?;
            }
            if rows > 2 {
                // 2026-10-10: The six-expert row fills a slot per participating expert and leaves
                // the rest empty, in both kernels. Expert 11's choice is exactly -1e30, which
                // takes part; under the ladder it is rounded to BF16, below -1e30, and does not.
                let id = |r: usize, k: usize| {
                    i32::from_le_bytes(ids[(r * TOP_K + k) * 4..][..4].try_into().unwrap())
                };
                let seed = if ladder == 1 {
                    bf16::from_f32(-1e30).to_f32()
                } else {
                    -1e30
                };
                let filled = if seed >= -1e30 { 6 } else { 5 };
                if (filled..TOP_K).any(|k| id(2, k) != -1)
                    || (0..filled).any(|k| id(2, k) < 0)
                    || (filled == 6 && id(2, 5) != 11)
                {
                    bail!("{what}: the six-expert row does not fill exactly {filled} slots");
                }
            }
        }
        report(
            "router top-k",
            rows,
            tg(g, s, MOE_LAYERS, &|s| {
                launch(old, 256, bufs[0], bufs[1], 0, 1, s)
            })?,
            tg(g, s, MOE_LAYERS, &|s| {
                launch(new, rank_block, bufs[2], bufs[3], 0, 1, s)
            })?,
        );
        for p in bufs.into_iter().chain([logits]) {
            g.free(p)?;
        }
    }
    g.free(bias)?;
    Ok(())
}

/// 2026-10-10: Host reference of one combined element: per slot in slot order the FP32 product
/// rounded, then added to the sum (the kernel's build does not contract the two), the shared
/// value added, one rounding to BF16.
fn combine_ref(eo: &[bf16], w: &[f32], shared: &[bf16], perm: &[usize]) -> Vec<u8> {
    let rows = shared.len() / HIDDEN;
    let mut out = Vec::with_capacity(rows * HIDDEN * 2);
    for t in 0..rows {
        for d in 0..HIDDEN {
            let mut acc = 0.0f32;
            for k in 0..TOP_K {
                let e = eo[perm[t * TOP_K + k] * HIDDEN + d].to_f32();
                acc += w[t * TOP_K + k] * e;
            }
            acc += shared[t * HIDDEN + d].to_f32();
            out.extend_from_slice(&bf16::from_f32(acc).to_le_bytes());
        }
    }
    out
}

fn combine(g: &dyn GpuBackend, s: u64, rng: &mut Lcg) -> Result<()> {
    let plain = g.kernel("glm5next_ffn", "glm5next_moe_combine")?;
    let indexed = g.kernel("glm5next_ffn", "glm5next_moe_combine_indexed")?;
    for rows in ROWS {
        let n = rows * TOP_K;
        let eo: Vec<bf16> = rng
            .vec(n * HIDDEN, 2.0)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let shared: Vec<bf16> = rng
            .vec(rows * HIDDEN, 2.0)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let w = rng.vec(n, 1.5);
        // 2026-10-10: A reversed permutation stands in for the expert-sorted order.
        let perm: Vec<usize> = (0..n).rev().collect();
        let to_bytes = |v: &[bf16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let (eo_d, sh_d, w_d) = (
            up(g, &to_bytes(&eo))?,
            up(g, &to_bytes(&shared))?,
            up_f32(g, &w)?,
        );
        let perm_i: Vec<u8> = perm
            .iter()
            .flat_map(|p| (*p as i32).to_le_bytes())
            .collect();
        let perm_d = up(g, &perm_i)?;
        let ob = rows * HIDDEN * 2;
        let launch = |idx: bool, block: u32, y: u32, out: DevicePtr, s: u64| {
            let l = KernelLaunch::new(g, if idx { indexed } else { plain })
                .grid([rows as u32, y, 1])
                .block([block, 1, 1])
                .arg_ptr(eo_d);
            let l = if idx { l.arg_ptr(perm_d) } else { l };
            l.arg_ptr(w_d)
                .arg_ptr(sh_d)
                .arg_ptr(out)
                .arg_u32(HIDDEN as u32)
                .arg_u32(TOP_K as u32)
                .launch(s)
        };
        let identity: Vec<usize> = (0..n).collect();
        for idx in [false, true] {
            let order = if idx { &perm } else { &identity };
            let base = zeros(g, ob)?;
            launch(idx, 256, 1, base, s)?;
            let want = read(g, s, base, ob)?;
            same(
                &format!("combine R={rows} indexed={idx} vs host reference"),
                &combine_ref(&eo, &w, &shared, order),
                &want,
            )?;
            let arm = zeros(g, ob)?;
            for (block, y) in [(256u32, 16u32), (128, 32), (64, 64), (256, 4)] {
                g.memset(arm, 0xA5, ob)?;
                launch(idx, block, y, arm, s)?;
                same(
                    &format!("combine R={rows} indexed={idx} block={block} y={y}"),
                    &want,
                    &read(g, s, arm, ob)?,
                )?;
                if !idx {
                    report(
                        &format!("combine y={y} block={block}"),
                        rows,
                        tg(g, s, MOE_LAYERS, &|s| launch(false, 256, 1, base, s))?,
                        tg(g, s, MOE_LAYERS, &|s| launch(false, block, y, arm, s))?,
                    );
                }
            }
            g.free(base)?;
            g.free(arm)?;
        }
        for p in [eo_d, sh_d, w_d, perm_d] {
            g.free(p)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let s = g.create_stream()?;
    let mut rng = Lcg(0x6c6d_c1f5_0001);
    let only = std::env::var("C1_FUSE_ONLY").unwrap_or_default();
    if only.is_empty() || only == "router" {
        router(g, s, &mut rng)?;
    }
    if only.is_empty() || only == "combine" {
        combine(g, s, &mut rng)?;
    }
    if only.is_empty() || only == "combine" || only == "combine_vec8" {
        c1_combine_vec8::combine_vec8(g, s, &mut rng)?;
    }
    if only.is_empty() || only == "kda" {
        c1_kda::kda_one_row(g, s, &mut rng)?;
    }
    if only.is_empty() || only == "gates" {
        c1_kda_gates::kda_gates(g, s, &mut rng)?;
    }
    println!("PASS: every new launch is byte-identical to the current kernel");
    Ok(())
}
