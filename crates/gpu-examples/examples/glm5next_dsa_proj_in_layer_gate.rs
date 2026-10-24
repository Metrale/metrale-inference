// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Layer-level byte gate for `METRALE_GLM_DSA_PROJ_IN_MERGED` under
//! `--dsa-quantization w8a8`: a DSA layer's decodes and captured verifies with the input
//! projections merged (`project_in` + `indexer_project_rows` taking one split launch) against the
//! same layer with the lever off; every output must be byte-identical. The kernel-level gate and
//! the microbench are `glm5next_dsa_proj_in_gate`; this one checks the host wiring (which
//! buffers the merged launch writes, the indexer skipping only its projections).
//!
//! Owner: gpu examples (GLM-5.3 DSA).
//! Invariants:
//! - Two paged arms run the same prefill chunk, then the same 1..=4-row steps (1 row: a decode,
//!   q_a and kv_a merged; 2..=4: a verify, wk and the gate merged as well), each followed by a
//!   rewind to a varying accepted length. Arm A runs with the lever on, arm B off.
//! - An engagement control: the gate fails unless arm A's steps took the merged launch
//!   (`Glm5NextDsaLayer::proj_in_merged_ready`) and arm B's did not.
//!
//!   METRALE_GLM_DSA_VERIFY_ROWS=1 \
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-gpu-examples --release --example glm5next_dsa_proj_in_layer_gate \
//!       --features cuda,gpu-examples

use anyhow::{Context, Result, ensure};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::glm5next_dsa::layer::Glm5NextDsaLayer;
use metrale_model_arch::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_layers::layers::DsaQuantization;
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};

#[path = "common/glm5next_dsa_paged_rig.rs"]
pub(crate) mod rig;
use rig::*;

struct Arm {
    st: Glm5NextDsaState,
    kv: PagedKvCache,
    bt: Vec<u32>,
}

impl Arm {
    fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            st: Glm5NextDsaState::paged(&cfg())?,
            kv: kv(gpu, 3 * MB)?,
            bt: (0..MB).map(|b| ((b * 7 + 3) % (3 * MB)) as u32).collect(),
        })
    }

    /// 2026-10-21: `k` rows of `x` at `pos..pos + k` on the replay-safe path (one metadata
    /// row per row, as the serve lays a decode or verify out); returns the layer output.
    fn step(
        &mut self,
        l: &Glm5NextDsaLayer,
        fwd: &Fwd,
        gpu: &dyn GpuBackend,
        x: &[u8],
        pos: usize,
        k: usize,
    ) -> Result<Vec<u8>> {
        let h = up(gpu, x)?;
        let rows: Vec<(usize, &[u32])> = (0..k).map(|t| (pos + t, self.bt.as_slice())).collect();
        let m = meta(gpu, &rows)?;
        let ctx = fwd.ctx(gpu, true, true, Some(m));
        l.decode_k(
            h,
            k,
            &mut self.st,
            &mut self.kv,
            pos,
            &mut self.bt,
            &ctx,
            stream(gpu),
            false,
        )?;
        read(gpu, h, k * HIDDEN * 2)
    }
}

fn main() -> Result<()> {
    ensure!(
        metrale_model_layers::layers::set_dsa_quantization_from_cli(DsaQuantization::W8a8)
            == DsaQuantization::W8a8,
        "the DSA tier was already published"
    );
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 128;
    config.intermediate_size = 128;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 128;
    config.vocab_size = 128;
    let fwd = Fwd {
        buffers: BufferArena::new(&config, 8, 16, 16, 8, &gpu)?,
        config,
        dispatch: GemmDispatch::defaults(),
        derived: DerivedWeights::new(),
        levers: ModelLevers::defaults(),
        settings: rig::process_settings()?,
        stats: ModelStats::new(),
    };
    let mut rng = Lcg(0x6473_7072_6f6a);
    let mut l = layer(&gpu, &cfg(), &mut rng)?;
    ensure!(
        l.levers.dsa_verify_rows,
        "set METRALE_GLM_DSA_VERIFY_ROWS=1: the indexer projections join only on the batched verify rows"
    );
    // 2026-10-21: The loader's `--dsa-quantization w8a8` registration (`dense_tiers`).
    let projs = l.dense_projections();
    let max_k = projs.iter().map(|p| p.2).max().unwrap_or(0);
    metrale_model_arch::glm5next_fp8_dense::prepare(&gpu, max_k, None)?;
    let quantize = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
    for (w, n, k, name) in projs {
        metrale_model_arch::glm5next_fp8_dense::register(&gpu, quantize, w, n, k, name, 0)?;
    }
    println!("GATE: DSA input projections merged into one W8A8 launch vs one launch each");
    let (mut a, mut b) = (Arm::new(&gpu)?, Arm::new(&gpu)?);
    let chunk = 12;
    let x = rng.bf16_bytes(chunk * HIDDEN, 1.0);
    for arm in [&mut a, &mut b] {
        let h = up(&gpu, &x)?;
        let ctx = fwd.ctx(&gpu, false, false, None);
        l.decode_k(
            h,
            chunk,
            &mut arm.st,
            &mut arm.kv,
            0,
            &mut arm.bt,
            &ctx,
            stream(&gpu),
            true,
        )?;
    }
    let (mut pos, mut steps, mut i) = (chunk, 0usize, 0usize);
    while pos + 4 <= LEN {
        let k = 1 + i % 4;
        let x = rng.bf16_bytes(k * HIDDEN, 1.0);
        l.levers.dsa_proj_in_merged = true;
        ensure!(
            l.proj_in_merged_ready(k, k > 1)?,
            "engagement: the {k}-row step at {pos} would not take the merged launch"
        );
        let ra = a.step(&l, &fwd, &gpu, &x, pos, k)?;
        l.levers.dsa_proj_in_merged = false;
        ensure!(!l.proj_in_merged_ready(k, k > 1)?, "lever off still merges");
        let rb = b.step(&l, &fwd, &gpu, &x, pos, k)?;
        same_rows(&format!("step k={k} at {pos}"), &ra, &rb, HIDDEN * 2)?;
        let accepted = 1 + (i * 5 + pos) % k;
        pos += accepted;
        for arm in [&mut a, &mut b] {
            arm.st.rewind_to(pos)?;
        }
        steps += 1;
        i += 1;
    }
    println!("  {steps} steps (k = 1..4, rewound between) byte-identical, merged on arm A only");
    println!("PASS");
    Ok(())
}
