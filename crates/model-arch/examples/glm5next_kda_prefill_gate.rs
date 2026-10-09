// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The GLM-5.3 KDA prefill on real checkpoint weights (one KDA block, all 64 heads),
//! three ways over a T-token prompt chunk from a carried state:
//!
//! * per token: `decode_k` row by row (two launches per token), the serve's default;
//! * token kernels: `decode_k_with(.., true)` (`METRALE_GLM_KDA_SEQ_TOKENS=1`), one conv and one
//!   recurrent launch for the chunk; must be byte-identical to per token;
//! * chunked scan: `prefill` (`METRALE_GLM_KDA_CHUNK_PREFILL=1`), a different summation order;
//!   reported as its error against per token, on the chunk's outputs, on the carried state, and
//!   on 16 decode tokens run after it.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error unless the token kernels match per token byte for byte (outputs,
//!   recurrent state, conv state) at every T; prints the chunked scan's errors and the GPU time of
//!   each way (captured graphs, `common/graph_timing.rs`).
//!
//!   GLM_CKPT=<checkpoint dir> METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash \
//!   METRALE_TARGET_QUANT=nvfp4 cargo run -p metrale-model-arch --release \
//!       --example glm5next_kda_prefill_gate --features cuda,gpu-examples

#[path = "common/glm_ckpt.rs"]
mod glm_ckpt;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace, KdaSeqState,
};
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-10-09: Two KDA layers (3 KDA then 1 DSA, repeating): an early and a middle one.
const LAYERS: [usize; 2] = [0, 21];
const TS: [usize; 3] = [77, 198, 512];
const HISTORY: usize = 40;
const DECODE_AFTER: usize = 16;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    /// 2026-10-09: Unit-RMS BF16 rows (a sum of three uniforms, scaled).
    fn rows(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                let x = (self.next() + self.next() + self.next()) * 1.0;
                bf16::from_f32(x).to_le_bytes()
            })
            .collect()
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn read(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(g.default_stream())?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn copy(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<DevicePtr> {
    let q = g.alloc(n)?;
    g.copy_d2d(p, q, n)?;
    Ok(q)
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}
/// 2026-10-09: (RMS of the difference / RMS of `a`, max |difference| / RMS of `a`).
fn err(a: &[f32], b: &[f32]) -> (f64, f64) {
    let rms = |v: &mut dyn Iterator<Item = f64>, n: usize| {
        (v.map(|x| x * x).sum::<f64>() / n as f64).sqrt()
    };
    let n = a.len();
    let ra = rms(&mut a.iter().map(|&x| x as f64), n).max(1e-30);
    let rd = rms(&mut a.iter().zip(b).map(|(&x, &y)| (x - y) as f64), n);
    let md = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| ((x - y) as f64).abs())
        .fold(0.0, f64::max);
    (rd / ra, md / ra)
}

fn load(
    g: &dyn GpuBackend,
    dir: &str,
    idx: usize,
    c: &Glm5NextKdaConfig,
) -> Result<Glm5NextKdaWeights> {
    let name = |t: &str| format!("model.language_model.layers.{idx}.self_attn.{t}");
    let raw = |t: &str| glm_ckpt::tensor_raw(dir, &name(t));
    let dense = |t: &str| -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: up(g, &raw(t)?.bytes)?,
        })
    };
    let mut conv = Vec::new();
    // 2026-10-09: The checkpoint stores the conv weights F32; the loader casts them to the BF16
    // the kernels read (round to nearest even), as here.
    for t in ["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"] {
        let r = raw(t)?;
        match r.dtype.as_str() {
            "BF16" => conv.extend(r.bytes),
            "F32" => conv.extend(r.bytes.chunks(4).flat_map(|b| {
                bf16::from_f32(f32::from_le_bytes([b[0], b[1], b[2], b[3]])).to_le_bytes()
            })),
            other => bail!("layer {idx}: {t} is {other}"),
        }
    }
    if conv.len() != c.conv_dim() * c.conv_kernel * 2 {
        bail!("layer {idx}: conv weights are {} bytes", conv.len());
    }
    Ok(Glm5NextKdaWeights {
        q_proj: dense("q_proj.weight")?,
        k_proj: dense("k_proj.weight")?,
        v_proj: dense("v_proj.weight")?,
        conv: DenseWeight {
            weight: up(g, &conv)?,
        },
        f_a: dense("f_a_proj.weight")?,
        f_b: dense("f_b_proj.weight")?,
        dt_bias: up(g, &raw("dt_bias")?.bytes)?,
        a_log: up(g, &raw("A_log")?.bytes)?,
        b_proj: dense("b_proj.weight")?,
        g_a: dense("g_a_proj.weight")?,
        g_b: dense("g_b_proj.weight")?,
        o_norm: dense("o_norm.weight")?,
        o_proj: dense("o_proj.weight")?,
    })
}

fn main() -> Result<()> {
    let dir = std::env::var("GLM_CKPT").context("GLM_CKPT names the checkpoint directory")?;
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let s = g.default_stream();
    let c = Glm5NextKdaConfig {
        hidden: 4096,
        heads: 64,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: -5.0,
        rms_norm_eps: 1e-5,
        l2_eps: 1e-6,
        chunk: 32,
    };
    let (hb, cb) = (c.recurrent_state_elems() * 4, c.conv_state_elems() * 4);
    let ws = Glm5NextKdaWorkspace::new(g, &c, *TS.iter().max().unwrap_or(&1))?;
    let mut rng = Lcg(0x6c6d_4b44_0001);
    println!(
        "GATE: KDA prefill on real weights: token kernels vs per token (bytes), chunked scan (error)"
    );
    for idx in LAYERS {
        let l = Glm5NextKdaLayer::new(
            idx,
            c,
            load(g, &dir, idx, &c)?,
            Glm5NextKdaKernels::resolve(g)?,
        )?;
        // 2026-10-09: A carried state: HISTORY tokens per token from zero.
        let st0 = KdaSeqState {
            conv: up(g, &vec![0u8; cb])?,
            recurrent: up(g, &vec![0u8; hb])?,
        };
        let hist = up(g, &rng.rows(HISTORY * c.hidden))?;
        l.decode_k_with(g, hist, HISTORY, &st0, &ws, &[], s, false)?;
        for t in TS {
            let x = rng.rows(t * c.hidden);
            let after = rng.rows(DECODE_AFTER * c.hidden);
            let fork = || -> Result<KdaSeqState> {
                Ok(KdaSeqState {
                    conv: copy(g, st0.conv, cb)?,
                    recurrent: copy(g, st0.recurrent, hb)?,
                })
            };
            let out_bytes = t * c.hidden * 2;
            // 2026-10-09: Runs one way; returns (chunk outputs, recurrent, conv, decode-after outputs).
            let run = |way: u8, st: &KdaSeqState| -> Result<[Vec<u8>; 4]> {
                let h = up(g, &x)?;
                match way {
                    0 => l.decode_k_with(g, h, t, st, &ws, &[], s, false)?,
                    1 => l.decode_k_with(g, h, t, st, &ws, &[], s, true)?,
                    _ => l.prefill(g, h, t, st, &ws, s)?,
                }
                let o = read(g, ws.final_out, out_bytes)?;
                let (rh, rc) = (read(g, st.recurrent, hb)?, read(g, st.conv, cb)?);
                let ha = up(g, &after)?;
                l.decode_k_with(g, ha, DECODE_AFTER, st, &ws, &[], s, false)?;
                let d = read(g, ws.final_out, DECODE_AFTER * c.hidden * 2)?;
                Ok([o, rh, rc, d])
            };
            let (sa, sb, sc) = (fork()?, fork()?, fork()?);
            let a = run(0, &sa)?;
            let b = run(1, &sb)?;
            let ch = run(2, &sc)?;
            for (i, what) in [
                "outputs",
                "recurrent state",
                "conv state",
                "decode-after outputs",
            ]
            .iter()
            .enumerate()
            {
                if a[i] != b[i] {
                    bail!("layer {idx} T={t}: token kernels differ from per token in {what}");
                }
            }
            let (eo, mo) = err(&bf16s(&a[0]), &bf16s(&ch[0]));
            let (es, ms) = err(&f32s(&a[1]), &f32s(&ch[1]));
            let (ed, md) = err(&bf16s(&a[3]), &bf16s(&ch[3]));
            let conv_same = a[2] == ch[2];
            // 2026-10-09: Timing from the carried state each call (the work does not depend on it).
            let h = up(g, &x)?;
            let st = fork()?;
            let tm = |way: u8| {
                timing::time_graph(g, s, 1, &mut |s| match way {
                    0 => l.decode_k_with(g, h, t, &st, &ws, &[], s, false),
                    1 => l.decode_k_with(g, h, t, &st, &ws, &[], s, true),
                    _ => l.prefill(g, h, t, &st, &ws, s),
                })
            };
            let (t0, t1, t2) = (tm(0)?, tm(1)?, tm(2)?);
            println!(
                "  layer {idx:>2} T={t:>3}: token kernels byte-identical | chunked: out rms {eo:.2e} \
                 max {mo:.2e}, state rms {es:.2e} max {ms:.2e}, conv {}, decode-after rms {ed:.2e} \
                 max {md:.2e} | us/layer per-token {:.0} [{:.0}..{:.0}], tokens {:.0} [{:.0}..{:.0}], \
                 chunked {:.0} [{:.0}..{:.0}]",
                if conv_same { "identical" } else { "DIFFERS" },
                t0.0,
                t0.1,
                t0.2,
                t1.0,
                t1.1,
                t1.2,
                t2.0,
                t2.1,
                t2.2
            );
        }
    }
    println!("PASS: the token kernels are byte-identical to the per-token walk");
    Ok(())
}
