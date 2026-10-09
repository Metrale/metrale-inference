// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: GPU gate for the KDA verify rollback by replay (`glm5next_kda/replay.rs`), at the
//! GLM-5.3 Flash KDA geometry (hidden 4096, head_dim 128, conv 4) for the full head count and
//! the largest TP=3 share.
//!
//! Owner: model-arch examples (GLM-5.3 KDA layer).
//! Invariants: the run exits 1 when any comparison below is not bit-exact.
//!
//! From one random state and K random hidden rows it runs:
//! (a) K sequential single-token `decode`s, keeping the state and output after each;
//! (b) a K-row verify as the replay mode runs it: `checkpoint_state`, `decode_k` in place, then
//!     `record_verify_rows`;
//! (c) a K-row verify with per-row snapshots (the snapshot mode).
//! It checks, for every accepted count n in 1..K, that `commit_replay(n, K)` rebuilds the state
//! (a) held after n tokens, that (c)'s snapshot n - 1 equals it too, and that the verify's
//! final state and its output rows equal (a)'s. The last check is the lossless-verify premise
//! (the batched projections give each row the bits of the M = 1 GEMV); it is reported
//! separately because it holds only up to `DENSE_GEMV_BATCHM_MAX_M` rows.
//!
//!   cargo run -p metrale-model-arch --release --example kda_replay_microtest \
//!       --features cuda,gpu-examples

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace, KdaSeqState, KdaVerifyRecord,
};
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-10-08: Verify width: the GLM DFlash2 drafter's block of 8 plus the anchor row.
const K: usize = 9;

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 40) as f32) / ((1u32 << 24) as f32)) * 2.0 - 1.0
    }
    fn scaled(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.u() * s).collect()
    }
}

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn dw(g: &dyn GpuBackend, rng: &mut Lcg, n: usize, s: f32) -> Result<DenseWeight> {
    Ok(DenseWeight {
        weight: up_bf16(g, &rng.scaled(n, s))?,
    })
}

struct Fixture<'a> {
    gpu: &'a dyn GpuBackend,
    cfg: Glm5NextKdaConfig,
    layer: Glm5NextKdaLayer,
    ws: Glm5NextKdaWorkspace,
    hidden: DevicePtr,
    s0: (Vec<u8>, Vec<u8>),
}

impl Fixture<'_> {
    fn bytes(&self) -> (usize, usize) {
        (
            self.cfg.recurrent_state_elems() * 4,
            self.cfg.conv_state_elems() * 4,
        )
    }

    /// 2026-10-08: A fresh state holding the start state's bytes.
    fn fresh(&self) -> Result<KdaSeqState> {
        let (h, c) = self.bytes();
        let st = KdaSeqState {
            conv: self.gpu.alloc(c)?,
            recurrent: self.gpu.alloc(h)?,
        };
        self.gpu.copy_h2d(&self.s0.0, st.recurrent)?;
        self.gpu.copy_h2d(&self.s0.1, st.conv)?;
        Ok(st)
    }

    fn read(&self, st: &KdaSeqState) -> Result<(Vec<u8>, Vec<u8>)> {
        let (h, c) = self.bytes();
        Ok((
            down(self.gpu, st.recurrent, h)?,
            down(self.gpu, st.conv, c)?,
        ))
    }
}

fn fixture(gpu: &dyn GpuBackend, heads: usize, seed: u64) -> Result<Fixture<'_>> {
    let cfg = Glm5NextKdaConfig {
        hidden: 4096,
        heads,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: -5.0,
        rms_norm_eps: 1e-5,
        l2_eps: 1e-6,
        chunk: 32,
    };
    let (hid, qkv, hd) = (cfg.hidden, cfg.qkv_dim(), cfg.head_dim);
    let mut rng = Lcg(seed);
    let s = 1.0 / (hid as f32).sqrt();
    let weights = Glm5NextKdaWeights {
        q_proj: dw(gpu, &mut rng, qkv * hid, s)?,
        k_proj: dw(gpu, &mut rng, qkv * hid, s)?,
        v_proj: dw(gpu, &mut rng, qkv * hid, s)?,
        conv: dw(gpu, &mut rng, cfg.conv_dim() * cfg.conv_kernel, 0.5)?,
        f_a: dw(gpu, &mut rng, hd * hid, s)?,
        f_b: dw(gpu, &mut rng, qkv * hd, 0.1)?,
        dt_bias: up_f32(gpu, &rng.scaled(qkv, 0.5))?,
        a_log: up_f32(gpu, &rng.scaled(heads, 0.5))?,
        b_proj: dw(gpu, &mut rng, heads * hid, s)?,
        g_a: dw(gpu, &mut rng, hd * hid, s)?,
        g_b: dw(gpu, &mut rng, qkv * hd, 0.1)?,
        o_norm: dw(gpu, &mut rng, hd, 1.0)?,
        o_proj: dw(gpu, &mut rng, hid * qkv, 1.0 / (qkv as f32).sqrt())?,
    };
    let kernels = Glm5NextKdaKernels::resolve(gpu)?;
    let layer = Glm5NextKdaLayer::new(0, cfg, weights, kernels)?;
    let ws = Glm5NextKdaWorkspace::new(gpu, &cfg, K)?;
    let hidden = up_bf16(gpu, &rng.scaled(K * hid, 1.0))?;
    let h0: Vec<u8> = rng
        .scaled(cfg.recurrent_state_elems(), 0.05)
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let c0: Vec<u8> = rng
        .scaled(cfg.conv_state_elems(), 0.5)
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    Ok(Fixture {
        gpu,
        cfg,
        layer,
        ws,
        hidden,
        s0: (h0, c0),
    })
}

/// 2026-10-08: One geometry; returns the number of failed checks.
fn run(gpu: &dyn GpuBackend, heads: usize) -> Result<usize> {
    let f = fixture(gpu, heads, 0x5EED_0000 + heads as u64)?;
    let s = gpu.default_stream();
    let out_bytes = f.cfg.hidden * 2;
    let mut failures = 0usize;
    let mut check = |name: String, ok: bool| {
        println!("  [{}] {name}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            failures += 1;
        }
    };

    // 2026-10-08: (a) sequential decode.
    let a = f.fresh()?;
    let mut seq_states = Vec::with_capacity(K);
    let mut seq_out = Vec::with_capacity(K);
    for t in 0..K {
        f.layer
            .decode(gpu, f.hidden.offset(t * f.cfg.hidden * 2), &a, &f.ws, s)?;
        gpu.synchronize(s)?;
        seq_out.push(down(gpu, f.ws.final_out, out_bytes)?);
        seq_states.push(f.read(&a)?);
    }

    // 2026-10-08: (b) replay-mode verify.
    let b = f.fresh()?;
    let (hb, cb) = f.bytes();
    let ckpt = KdaSeqState {
        conv: gpu.alloc(cb)?,
        recurrent: gpu.alloc(hb)?,
    };
    let rec_bytes = (K - 1) * f.cfg.replay_row_bytes();
    let rec = KdaVerifyRecord::new(&f.cfg, gpu.alloc(rec_bytes)?, rec_bytes);
    f.layer.checkpoint_state(gpu, &b, &ckpt, s)?;
    f.layer.decode_k(gpu, f.hidden, K, &b, &f.ws, &[], s)?;
    f.layer.record_verify_rows(gpu, &f.ws, K - 1, &rec, s)?;
    gpu.synchronize(s)?;
    let ver_out = down(gpu, f.ws.final_out, K * out_bytes)?;
    check(
        format!("heads={heads}: verify final state == {K} sequential decodes"),
        f.read(&b)? == seq_states[K - 1],
    );
    let rows_equal = (0..K).all(|t| ver_out[t * out_bytes..(t + 1) * out_bytes] == seq_out[t]);
    check(
        format!("heads={heads}: verify output rows == sequential outputs (lossless premise)"),
        rows_equal,
    );

    // 2026-10-08: (c) snapshot-mode verify.
    let c = f.fresh()?;
    let snaps: Vec<(DevicePtr, DevicePtr)> = (0..K - 1)
        .map(|_| Ok((gpu.alloc(hb)?, gpu.alloc(cb)?)))
        .collect::<Result<_>>()?;
    f.layer.decode_k(gpu, f.hidden, K, &c, &f.ws, &snaps, s)?;
    gpu.synchronize(s)?;

    for n in 1..K {
        f.layer
            .commit_replay(gpu, &b, &ckpt, &rec, n, K, &f.ws, s)?;
        gpu.synchronize(s)?;
        let replayed = f.read(&b)?;
        check(
            format!("heads={heads} n={n}: replayed state == {n} sequential decodes"),
            replayed == seq_states[n - 1],
        );
        let snap = (
            down(gpu, snaps[n - 1].0, hb)?,
            down(gpu, snaps[n - 1].1, cb)?,
        );
        check(
            format!("heads={heads} n={n}: replayed state == snapshot {}", n - 1),
            replayed == snap,
        );
    }
    Ok(failures)
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    println!("GLM-5.3 KDA verify rollback by replay, K = {K}");
    // 2026-10-08: 64 heads (TP = 1) and 22 (the largest TP = 3 share of 64).
    let failures = run(gpu, 64)? + run(gpu, 22)?;
    if failures > 0 {
        println!("{failures} check(s) failed");
        std::process::exit(1);
    }
    println!("all checks passed");
    Ok(())
}
