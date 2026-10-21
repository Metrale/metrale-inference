// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: A GLM-5.3-Flash KDA layer with random weights at the serving geometry (hidden
//! 4096, head_dim 128, conv 4) for a chosen head count, and the host helpers the KDA gates share
//! (`kda_replay_microtest`, `kda_commit_swap_gate`).
//!
//! Owner: gpu-examples.
//! Invariants: the same seed builds the same weights and start state.

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace,
};
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-10-08: A 64-bit LCG; `u` is uniform in [-1, 1).
pub struct Lcg(pub u64);

impl Lcg {
    pub fn u(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 40) as f32) / ((1u32 << 24) as f32)) * 2.0 - 1.0
    }

    pub fn scaled(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.u() * s).collect()
    }
}

/// 2026-10-08: `d` as FP32 bytes in a fresh device buffer.
pub fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

/// 2026-10-08: `d` as BF16 in a fresh device buffer.
pub fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

/// 2026-10-08: `bytes` bytes at `p`, read back.
pub fn down(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn dw(g: &dyn GpuBackend, rng: &mut Lcg, n: usize, s: f32) -> Result<DenseWeight> {
    Ok(DenseWeight {
        weight: up_bf16(g, &rng.scaled(n, s))?,
    })
}

/// 2026-10-21: The layer, a workspace for `max_rows` rows, and a start state (h, conv bytes).
pub struct KdaFixture {
    pub cfg: Glm5NextKdaConfig,
    pub layer: Glm5NextKdaLayer,
    pub ws: Glm5NextKdaWorkspace,
    pub s0: (Vec<u8>, Vec<u8>),
}

impl KdaFixture {
    /// 2026-10-21: (h, conv) state bytes.
    pub fn state_bytes(&self) -> (usize, usize) {
        (
            self.cfg.recurrent_state_elems() * 4,
            self.cfg.conv_state_elems() * 4,
        )
    }
}

/// 2026-10-21: The fixture for `heads` heads from `rng`, with the process's GLM levers; the
/// caller draws its hidden rows from `rng` afterwards.
pub fn kda_fixture(
    gpu: &dyn GpuBackend,
    heads: usize,
    max_rows: usize,
    rng: &mut Lcg,
) -> Result<KdaFixture> {
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
    let s = 1.0 / (hid as f32).sqrt();
    let weights = Glm5NextKdaWeights {
        q_proj: dw(gpu, rng, qkv * hid, s)?,
        k_proj: dw(gpu, rng, qkv * hid, s)?,
        v_proj: dw(gpu, rng, qkv * hid, s)?,
        conv: dw(gpu, rng, cfg.conv_dim() * cfg.conv_kernel, 0.5)?,
        f_a: dw(gpu, rng, hd * hid, s)?,
        f_b: dw(gpu, rng, qkv * hd, 0.1)?,
        dt_bias: up_f32(gpu, &rng.scaled(qkv, 0.5))?,
        a_log: up_f32(gpu, &rng.scaled(heads, 0.5))?,
        b_proj: dw(gpu, rng, heads * hid, s)?,
        g_a: dw(gpu, rng, hd * hid, s)?,
        g_b: dw(gpu, rng, qkv * hd, 0.1)?,
        o_norm: dw(gpu, rng, hd, 1.0)?,
        o_proj: dw(gpu, rng, hid * qkv, 1.0 / (qkv as f32).sqrt())?,
    };
    let kernels = Glm5NextKdaKernels::resolve(gpu)?;
    let layer = Glm5NextKdaLayer::new(0, cfg, weights, kernels, crate::glm_levers()?)?;
    let ws = Glm5NextKdaWorkspace::new(gpu, &cfg, max_rows)?;
    Ok(KdaFixture {
        cfg,
        layer,
        ws,
        s0: (Vec::new(), Vec::new()),
    })
}

/// 2026-10-21: [`kda_fixture`] plus `rows` random hidden rows (BF16, uploaded) and a random start
/// state, drawn in that order after the weights.
pub fn kda_fixture_with_inputs(
    gpu: &dyn GpuBackend,
    heads: usize,
    max_rows: usize,
    rows: usize,
    seed: u64,
) -> Result<(KdaFixture, DevicePtr)> {
    let mut rng = Lcg(seed);
    let mut f = kda_fixture(gpu, heads, max_rows, &mut rng)?;
    let hidden = up_bf16(gpu, &rng.scaled(rows * f.cfg.hidden, 1.0))?;
    let bytes = |v: Vec<f32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    f.s0 = (
        bytes(rng.scaled(f.cfg.recurrent_state_elems(), 0.05)),
        bytes(rng.scaled(f.cfg.conv_state_elems(), 0.5)),
    );
    Ok((f, hidden))
}
