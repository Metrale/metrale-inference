// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Bit-identity and speed test of `dense_gemv_bf16_batchm` against M separate
//! `dense_gemv_bf16` launches on the same rows. 2026-10-14: where the target compiles the
//! direct-schedule entries (`METRALE_BATCHM_DIRECT_R`), they are checked and timed too,
//! through the production launcher `ops::dense_gemv_batchm_direct`.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - The run exits with an error when any (shape, M) output differs in any bit from the
//!   M=1 reference; timings are printed for every case.
//!
//! Run:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=laguna-s-2.1 METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dense_gemv_bf16_batchm_microtest

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-09-25: (N, K, label). The first three rows are Laguna-S-2.1 shapes: hidden 3072,
/// 72 q heads x 128 = 9216 (q_proj N, o_proj K), 8 kv heads x 128 and the shared
/// expert's 1024.
const SHAPES: &[(usize, usize, &str)] = &[
    (9216, 3072, "q_proj"),
    (3072, 9216, "o_proj"),
    (1024, 3072, "k/v/shared"),
    (4096, 3072, "glm N4096"),
    (4096, 16384, "glm N4096 K16k"),
    (4096, 4096, "glm KDA qkv/o"),
    (16384, 1536, "glm DSA q_absorb"),
    (1024, 4096, "glm shared gate/up"),
    (4096, 1024, "glm shared down"),
    (4096, 128, "glm KDA f_b (shallow K)"),
    (32, 4096, "glm KDA b_proj (8 blocks)"),
    // 2026-10-14: The Qwen3.8-Flash-Next GDN BF16 decode projections (hidden 2560): the fused
    // qkvz input projection and the output projection from the 6144-wide value heads.
    (16384, 2560, "fn GDN qkvz"),
    (2560, 6144, "fn GDN out"),
    // 2026-09-25: N % 4 != 0 (the kernel's N_PER_BLOCK is 4), so the last block has
    // lanes with n >= N. The kernel masks them instead of returning, so they still
    // reach the shared-memory staging barriers; this shape exercises that path.
    (4098, 3072, "N not mult of 4"),
];
// 2026-09-25: Widths up to the kernel's MAX_M (16). The GLM prefill sub-chunk
// (`glm5next_layer::PREFILL_ROWS`) is 16 rows.
const MS: &[usize] = &[1, 2, 4, 8, 9, 12, 15, 16];
const ITERS: usize = 50;
const WARMUP: usize = 10;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self) -> f32 {
        -1.0 + 2.0 * self.f()
    }
}

fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn_bits(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn launch_m1(
    g: &dyn GpuBackend,
    kern: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    n: usize,
    k_dim: usize,
) -> Result<()> {
    KernelLaunch::new(g, kern)
        .grid([div_ceil(n as u32, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(n as u32)
        .arg_u32(k_dim as u32)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn launch_batchm(
    g: &dyn GpuBackend,
    kern: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k_dim: usize,
    out_stride: usize,
) -> Result<()> {
    KernelLaunch::new(g, kern)
        .grid([div_ceil(n as u32, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k_dim as u32)
        .arg_u32(out_stride as u32)
        .launch(0)
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let m1_k = g.kernel("gemv", "dense_gemv_bf16")?;
    let bm_k = g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?;
    let direct = ops::DenseGemvBatchmDirect::load(g);
    println!("direct entries: {direct:?}");

    let mut rng = Lcg(0x5eed_1234);
    let mut fail = 0usize;

    println!(
        "{:<12} {:>5} {:>10} {:>12} {:>12} {:>9} {:>9} {:>12} {:>9}  bit-identical (batchm / direct)",
        "shape", "M", "N", "Mx M=1 (ms)", "batchm (ms)", "speedup", "GB/s", "direct (ms)", "GB/s"
    );

    for &(n, k_dim, label) in SHAPES {
        let w: Vec<bf16> = (0..n * k_dim).map(|_| bf16::from_f32(rng.r())).collect();
        let wd = up_bf16(g, &w)?;
        let wdense = DenseWeight { weight: wd };

        for &m in MS {
            let a: Vec<bf16> = (0..m * k_dim).map(|_| bf16::from_f32(rng.r())).collect();
            let ad = up_bf16(g, &a)?;
            let c_ref = g.alloc(m * n * 2)?;
            let c_bat = g.alloc(m * n * 2)?;

            for t in 0..m {
                launch_m1(
                    g,
                    m1_k,
                    ad.offset(t * k_dim * 2),
                    wd,
                    c_ref.offset(t * n * 2),
                    n,
                    k_dim,
                )?;
            }
            launch_batchm(g, bm_k, ad, wd, c_bat, m, n, k_dim, n)?;
            g.synchronize(0)?;

            let r = dn_bits(g, c_ref, m * n)?;
            let b = dn_bits(g, c_bat, m * n)?;
            let identical = r == b;
            let ndiff = r.iter().zip(&b).filter(|(x, y)| x != y).count();
            if !identical {
                fail += 1;
            }

            for _ in 0..WARMUP {
                for t in 0..m {
                    launch_m1(
                        g,
                        m1_k,
                        ad.offset(t * k_dim * 2),
                        wd,
                        c_ref.offset(t * n * 2),
                        n,
                        k_dim,
                    )?;
                }
                launch_batchm(g, bm_k, ad, wd, c_bat, m, n, k_dim, n)?;
            }
            g.synchronize(0)?;

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                for t in 0..m {
                    launch_m1(
                        g,
                        m1_k,
                        ad.offset(t * k_dim * 2),
                        wd,
                        c_ref.offset(t * n * 2),
                        n,
                        k_dim,
                    )?;
                }
            }
            g.synchronize(0)?;
            let t_ref = t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

            let t1 = std::time::Instant::now();
            for _ in 0..ITERS {
                launch_batchm(g, bm_k, ad, wd, c_bat, m, n, k_dim, n)?;
            }
            g.synchronize(0)?;
            let t_bat = t1.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
            // 2026-10-14: Effective bandwidth of the batched launch: weight, activations and
            // outputs each moved once.
            let bytes = 2 * (n * k_dim + m * k_dim + m * n);
            let gbps = bytes as f64 / (t_bat * 1e6);

            let (t_dir, dir_bits) = match (&direct, m >= 2) {
                (Some(d), true) => {
                    let run = || {
                        ops::dense_gemv_batchm_direct(
                            g,
                            d,
                            ad,
                            &wdense,
                            c_bat,
                            m as u32,
                            n as u32,
                            k_dim as u32,
                            n as u32,
                            0,
                        )
                    };
                    g.memset(c_bat, 0, m * n * 2)?;
                    run()?;
                    g.synchronize(0)?;
                    let dbits = dn_bits(g, c_bat, m * n)?;
                    let dd = r.iter().zip(&dbits).filter(|(x, y)| x != y).count();
                    if dd != 0 {
                        fail += 1;
                    }
                    for _ in 0..WARMUP {
                        run()?;
                    }
                    g.synchronize(0)?;
                    let t2 = std::time::Instant::now();
                    for _ in 0..ITERS {
                        run()?;
                    }
                    g.synchronize(0)?;
                    let t = t2.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
                    let v = if dd == 0 {
                        "yes".to_string()
                    } else {
                        format!("NO ({dd})")
                    };
                    (Some(t), v)
                }
                _ => (None, "-".to_string()),
            };
            let (t_dir_s, gbps_dir_s) = match t_dir {
                Some(t) => (
                    format!("{t:.4}"),
                    format!("{:.0}", bytes as f64 / (t * 1e6)),
                ),
                None => ("-".to_string(), "-".to_string()),
            };

            println!(
                "{:<12} {:>5} {:>10} {:>12.4} {:>12.4} {:>8.2}x {:>9.0} {:>12} {:>9}  {} / {}",
                label,
                m,
                n,
                t_ref,
                t_bat,
                t_ref / t_bat,
                gbps,
                t_dir_s,
                gbps_dir_s,
                if identical {
                    "yes".to_string()
                } else {
                    format!("NO ({ndiff} elems differ)")
                },
                dir_bits
            );

            g.free(ad)?;
            g.free(c_ref)?;
            g.free(c_bat)?;
        }
        g.free(wd)?;
    }

    if fail > 0 {
        anyhow::bail!("{fail} shape/M combinations were NOT bit-identical");
    }
    println!("\nAll bit-identical. PASS criterion: speedup should approach Mx at M=2 and M=4.");
    Ok(())
}
