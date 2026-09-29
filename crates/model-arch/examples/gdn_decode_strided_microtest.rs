// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The batched GDN decode recurrence (`gated_delta_rule_decode_f32_strided`, launched
//! by `ops::gdn_decode_f32_strided` with its occupancy cap) against the per-sequence kernel the
//! multi-sequence decode launches once per sequence (`gated_delta_rule_decode_f32`), at
//! Qwen3.6-35B-A3B GDN dimensions.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for B = 1..=128 sequences (powers of two), the strided launch's output
//!   and final state equal the per-sequence launches' byte for byte, with one head of one
//!   sequence driven past the state-norm clamp (`SSM_state()_MAX_NORM`) so the clamp branch is
//!   compared too.
//! - Before the sweep, a flipped output bit must be refused by the same check.
//!
//! Prints the mean time of each form (20 launches) at every B, the strided one at several
//! dynamic shared-memory reservations (the occupancy cap `GDN_DECODE_STRIDED_SMEM_CAP_BYTES`
//! was chosen from this sweep).
//!
//! Arguments: value heads, 32 (Qwen3.6-35B-A3B, default) or 48 (Qwen3.8-27B); optionally a
//! model target (e.g. `qwen3.6-35b-a3b`) whose gated_delta_rule shadow to test. Or
//! `--ab old.ptx,new.ptx`: two nvcc builds of one gated_delta_rule source pair (same flags),
//! whose per-sequence and strided entries must agree byte for byte at B = 1 and 128.
//!
//! Run (GB10):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example gdn_decode_strided_microtest

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const NK: usize = 16;
const KD: usize = 128;
const VD: usize = 128;
const MAX_B: usize = 128;

/// 2026-09-29: Value heads, the first argument: 32 (Qwen3.6-35B-A3B, the default) or 48
/// (Qwen3.8-27B).
fn nv() -> usize {
    static NV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *NV.get_or_init(|| {
        std::env::args()
            .nth(1)
            .and_then(|a| a.parse().ok())
            .unwrap_or(32)
    })
}

fn state() -> usize {
    nv() * KD * VD
}

struct Lcg(u64);
impl Lcg {
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lo + (hi - lo) * ((((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32)
    }
    fn v(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.r(lo, hi)).collect()
    }
}

fn bytes(d: &[f32]) -> Vec<u8> {
    d.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn up(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b = bytes(d);
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-09-29: The first differing byte, as an error.
fn same(what: &str, a: &[u8], b: &[u8]) -> Result<()> {
    ensure!(
        a.len() == b.len(),
        "{what}: length {} vs {}",
        a.len(),
        b.len()
    );
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        anyhow::bail!("{what}: byte {i} differs ({} vs {})", a[i], b[i]);
    }
    Ok(())
}

/// 2026-09-29: One sequence's inputs: q/k `[NK, KD]`, v `[NV, VD]`, gate and beta `[NV]`, all
/// FP32, rows `b` apart by the strided kernel's strides.
struct Inputs {
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    h0: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn per_seq(
    g: &dyn GpuBackend,
    kernel: KernelHandle,
    x: &Inputs,
    h: DevicePtr,
    out: DevicePtr,
    b_count: usize,
) -> Result<()> {
    for b in 0..b_count {
        KernelLaunch::new(g, kernel)
            .grid([nv() as u32, 1, 1])
            .block([VD as u32, 1, 1])
            .arg_ptr(h.offset(b * state() * 4))
            .arg_ptr(x.q.offset(b * NK * KD * 4))
            .arg_ptr(x.k.offset(b * NK * KD * 4))
            .arg_ptr(x.v.offset(b * nv() * VD * 4))
            .arg_ptr(x.gate.offset(b * nv() * 4))
            .arg_ptr(x.beta.offset(b * nv() * 4))
            .arg_ptr(out.offset(b * nv() * VD * 4))
            .arg_u32(1)
            .arg_u32(NK as u32)
            .arg_u32(nv() as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .launch(0)?;
    }
    Ok(())
}

fn strided(
    g: &dyn GpuBackend,
    kernel: KernelHandle,
    x: &Inputs,
    h: DevicePtr,
    out: DevicePtr,
    b_count: usize,
    smem: u32,
) -> Result<()> {
    KernelLaunch::new(g, kernel)
        .grid([nv() as u32, b_count as u32, 1])
        .block([VD as u32, 1, 1])
        .shared_mem(smem)
        .arg_ptr(h)
        .arg_ptr(x.q)
        .arg_ptr(x.k)
        .arg_ptr(x.v)
        .arg_ptr(x.gate)
        .arg_ptr(x.beta)
        .arg_ptr(out)
        .arg_u32(b_count as u32)
        .arg_u32(NK as u32)
        .arg_u32(nv() as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32((NK * KD) as u32)
        .arg_u32((nv() * VD) as u32)
        .arg_u32(nv() as u32)
        .arg_u32((nv() * VD) as u32)
        .launch(0)
}

/// 2026-09-29: Two PTX builds of gated_delta_rule (`old,new`): the per-sequence and strided
/// entries of `new` must reproduce `old`'s output and state bytes at B = 1 and 128, with the
/// state-norm clamp head driven as in the main sweep. A flipped bit is refused first.
fn ab_ptx(pair: &str) -> Result<()> {
    let (old, new) = pair
        .split_once(',')
        .ok_or_else(|| anyhow::anyhow!("--ab wants old.ptx,new.ptx"))?;
    let leak =
        |p: &str| -> Result<&'static [u8]> { Ok(Box::leak(std::fs::read(p)?.into_boxed_slice())) };
    let modules = vec![("gdr_old", leak(old)?), ("gdr_new", leak(new)?)];
    let gpu = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &gpu;
    let mut rng = Lcg(0x6764_6e2d_6162);
    let mut h0 = rng.v(MAX_B * state(), -0.05, 0.05);
    for e in &mut h0[state() + 5 * KD * VD..state() + 6 * KD * VD] {
        *e *= 400.0;
    }
    let x = Inputs {
        q: up(g, &rng.v(MAX_B * NK * KD, -0.1, 0.1))?,
        k: up(g, &rng.v(MAX_B * NK * KD, -0.1, 0.1))?,
        v: up(g, &rng.v(MAX_B * nv() * VD, -1.0, 1.0))?,
        gate: up(g, &rng.v(MAX_B * nv(), 0.6, 0.999))?,
        beta: up(g, &rng.v(MAX_B * nv(), 0.05, 0.95))?,
        h0,
    };
    let (h_a, h_b) = (up(g, &x.h0)?, up(g, &x.h0)?);
    let (out_a, out_b) = (
        up(g, &vec![0.0; MAX_B * nv() * VD])?,
        up(g, &vec![0.0; MAX_B * nv() * VD])?,
    );
    let mut control_done = false;
    for entry in [
        "gated_delta_rule_decode_f32",
        "gated_delta_rule_decode_f32_strided",
    ] {
        let (ka, kb) = (g.kernel("gdr_old", entry)?, g.kernel("gdr_new", entry)?);
        for b_count in [1usize, 128] {
            g.copy_h2d(&bytes(&x.h0), h_a)?;
            g.copy_h2d(&bytes(&x.h0), h_b)?;
            if entry.ends_with("strided") {
                strided(g, ka, &x, h_a, out_a, b_count, 0)?;
                strided(g, kb, &x, h_b, out_b, b_count, 0)?;
            } else {
                per_seq(g, ka, &x, h_a, out_a, b_count)?;
                per_seq(g, kb, &x, h_b, out_b, b_count)?;
            }
            let (oa, ob) = (
                down(g, out_a, b_count * nv() * VD)?,
                down(g, out_b, b_count * nv() * VD)?,
            );
            if !control_done {
                let mut bad = ob.clone();
                bad[7] ^= 1;
                ensure!(
                    same("control", &oa, &bad).is_err(),
                    "a flipped bit was admitted"
                );
                println!("KNOWN_BAD flipped-bit: refused");
                control_done = true;
            }
            same(&format!("{entry} B={b_count} output"), &oa, &ob)?;
            let (sa, sb) = (
                down(g, h_a, b_count * state())?,
                down(g, h_b, b_count * state())?,
            );
            same(&format!("{entry} B={b_count} state"), &sa, &sb)?;
            println!("{entry} B={b_count}: old == new, output and state byte for byte");
        }
    }
    println!("ALL PASS: the two builds agree byte for byte");
    Ok(())
}

fn main() -> Result<()> {
    // 2026-09-29: `--ab old.ptx,new.ptx`: compare two builds of a gated_delta_rule module
    // (same nvcc flags, two sources) byte for byte and exit.
    if std::env::args().nth(1).as_deref() == Some("--ab") {
        let pair = std::env::args()
            .nth(2)
            .ok_or_else(|| anyhow::anyhow!("--ab wants old.ptx,new.ptx"))?;
        return ab_ptx(&pair);
    }
    // 2026-09-29: The second argument names a model target (`kernels/gb10/<model>`), whose
    // shadow of the gated_delta_rule module then runs; without it, the default target.
    let modules = match std::env::args().nth(2) {
        Some(model) => {
            metrale_kernels::all_ptx_sets()
                .into_iter()
                .find(|s| s.target.model == model)
                .ok_or_else(|| anyhow::anyhow!("no kernel target for model {model}"))?
                .modules
        }
        None => metrale_kernels::ptx_modules(),
    };
    let gpu = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &gpu;
    let per_k = g.kernel("gated_delta_rule", "gated_delta_rule_decode_f32")?;
    let str_k = g.kernel("gated_delta_rule", "gated_delta_rule_decode_f32_strided")?;
    let mut rng = Lcg(0x6764_6e2d_7374);
    let mut h0 = rng.v(MAX_B * state(), -0.05, 0.05);
    // 2026-09-29: Head 5 of sequence 1 starts far past the clamp norm (1000): 16384 entries of
    // magnitude ~20 give a norm of ~2600.
    for e in &mut h0[state() + 5 * KD * VD..state() + 6 * KD * VD] {
        *e *= 400.0;
    }
    let x = Inputs {
        q: up(g, &rng.v(MAX_B * NK * KD, -0.1, 0.1))?,
        k: up(g, &rng.v(MAX_B * NK * KD, -0.1, 0.1))?,
        v: up(g, &rng.v(MAX_B * nv() * VD, -1.0, 1.0))?,
        gate: up(g, &rng.v(MAX_B * nv(), 0.6, 0.999))?,
        beta: up(g, &rng.v(MAX_B * nv(), 0.05, 0.95))?,
        h0,
    };
    let h_a = up(g, &x.h0)?;
    let h_b = up(g, &x.h0)?;
    let out_a = up(g, &vec![0.0; MAX_B * nv() * VD])?;
    let out_b = up(g, &vec![0.0; MAX_B * nv() * VD])?;
    let reset = |h: DevicePtr| g.copy_h2d(&bytes(&x.h0), h);

    let mut control_done = false;
    for b_count in [1usize, 2, 4, 8, 16, 32, 64, 128] {
        reset(h_a)?;
        reset(h_b)?;
        per_seq(g, per_k, &x, h_a, out_a, b_count)?;
        metrale_model_layers::layers::ops::gdn_decode_f32_strided(
            g,
            str_k,
            h_b,
            x.q,
            x.k,
            x.v,
            x.gate,
            x.beta,
            out_b,
            b_count as u32,
            NK as u32,
            nv() as u32,
            KD as u32,
            VD as u32,
            (NK * KD) as u32,
            (nv() * VD) as u32,
            nv() as u32,
            (nv() * VD) as u32,
            0,
        )?;
        let (oa, ob) = (
            down(g, out_a, b_count * nv() * VD)?,
            down(g, out_b, b_count * nv() * VD)?,
        );
        let (sa, sb) = (
            down(g, h_a, b_count * state())?,
            down(g, h_b, b_count * state())?,
        );
        if !control_done {
            let mut bad = ob.clone();
            bad[7] ^= 1;
            ensure!(
                same("control", &oa, &bad).is_err(),
                "a flipped bit was admitted"
            );
            println!("KNOWN_BAD flipped-bit: refused");
            control_done = true;
        }
        let ulp = |a: &[u8], b: &[u8]| -> (usize, u32) {
            let mut n = 0usize;
            let mut worst = 0u32;
            for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
                let (x, y) = (
                    u32::from_le_bytes([x[0], x[1], x[2], x[3]]),
                    u32::from_le_bytes([y[0], y[1], y[2], y[3]]),
                );
                if x != y {
                    n += 1;
                    worst = worst.max((x as i64 - y as i64).unsigned_abs() as u32);
                }
            }
            (n, worst)
        };
        let (so, uo) = ulp(&oa, &ob);
        let (ss, us) = ulp(&sa, &sb);
        println!(
            "B={b_count}: output {so} words differ (max {uo} ulp), state {ss} differ (max {us} ulp)"
        );
        if so > 0 {
            let f = |v: &[u8], i: usize| {
                f32::from_le_bytes([v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]])
            };
            for i in [0usize, 1, 2, 3, 128, 4095] {
                println!(
                    "  out[{i}] per-seq {:e} strided {:e} ratio {:.9}",
                    f(&oa, i),
                    f(&ob, i),
                    f(&oa, i) as f64 / f(&ob, i) as f64
                );
            }
        }
        same(&format!("B={b_count} state"), &sa, &sb)?;
        same(&format!("B={b_count} output"), &oa, &ob)?;
        // 2026-09-29: The common module clamps the state norm; a model shadow may not (the
        // qwen3.6-35b-a3b decode kernels do not), and byte equality covers both.
        if b_count >= 2 && std::env::args().nth(2).is_none() {
            // 2026-09-29: The clamp fired on the driven head: its state norm is at most 1000.
            let head = &sb[(state() + 5 * KD * VD) * 4..(state() + 6 * KD * VD) * 4];
            let norm: f64 = head
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            ensure!(norm <= 1000.5, "clamp head norm {norm}");
        }
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            g.synchronize(0)?;
            let t = std::time::Instant::now();
            for _ in 0..20 {
                f()?;
            }
            g.synchronize(0)?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / 20.0)
        };
        let us_a = time(&|| per_seq(g, per_k, &x, h_a, out_a, b_count))?;
        let gbs = |us: f64| (b_count * state() * 8) as f64 / us / 1e3;
        let mut line = format!(
            "B={b_count:3} output+state bit-identical | per-seq {us_a:8.1}us ({:5.1} GB/s R+W)",
            gbs(us_a)
        );
        // 2026-09-29: Dynamic shared memory per CTA caps the CTAs resident per SM, and with
        // them the bytes of H in flight between a CTA's two passes over its state.
        for smem in [
            0u32,
            8 << 10,
            12 << 10,
            16 << 10,
            24 << 10,
            32 << 10,
            44 << 10,
        ] {
            let us_b = time(&|| strided(g, str_k, &x, h_b, out_b, b_count, smem))?;
            line += &format!(" | smem {:2}K {us_b:7.1}us ({:5.1})", smem >> 10, gbs(us_b));
        }
        println!("{line}  PASS");
    }
    println!("ALL PASS: strided GDN decode == per-sequence decode, byte for byte");
    Ok(())
}
