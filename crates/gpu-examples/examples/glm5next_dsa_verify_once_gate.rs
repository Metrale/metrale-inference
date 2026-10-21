// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Byte-identity gate for `METRALE_GLM_DSA_VERIFY_SELECT_ONCE`: a captured k-row
//! verify of one sequence over persisted pool keys selects all its rows in one ceiling pass;
//! every output must equal the per-row passes' on the same tokens.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Two paged pooled arms run the same prefill chunk, then the same verifies of k = 2, 3, 4
//!   rows at every position residue mod `index_kpool` (so some verifies complete a pool inside
//!   the group and their rows' live pool counts differ), each followed by a rewind to a
//!   varying accepted length, as the serve does after a rejected draft. Arm A runs with the
//!   lever on, arm B with it off; each verify's output rows must be byte-identical.
//! - An engagement control: the gate fails unless arm A's verifies took the one pass
//!   (`Glm5NextDsaLayer::verify_select_once_for`).
//!
//!   METRALE_PAGED_PARITY_POOLED=1 METRALE_GLM_DSA_VERIFY_ROWS=1 \
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-gpu-examples --release --example glm5next_dsa_verify_once_gate \
//!       --features cuda,gpu-examples

use anyhow::{Context, Result, ensure};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::glm5next_dsa::layer::Glm5NextDsaLayer;
use metrale_model_arch::glm5next_dsa::state::Glm5NextDsaState;
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
    /// row per verify row, as the serve lays a verify out); returns the layer output.
    fn verify(
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
        pooled(),
        "set METRALE_PAGED_PARITY_POOLED=1: the one pass runs only over persisted pool keys"
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
    let mut rng = Lcg(0x6c6d_5e9d_0021);
    let mut l = layer(&gpu, &cfg(), &mut rng)?;
    ensure!(
        l.levers.dsa_verify_rows,
        "set METRALE_GLM_DSA_VERIFY_ROWS=1: the one pass runs only on the batched verify rows"
    );
    println!("GATE: one selection pass for a captured verify's rows vs one pass per row");
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
    let (mut pos, mut verifies, mut i) = (chunk, 0usize, 0usize);
    while pos + 4 <= LEN {
        let k = 2 + i % 3;
        let x = rng.bf16_bytes(k * HIDDEN, 1.0);
        l.levers.dsa_verify_select_once = true;
        let bt_dev = up_i32(&gpu, &a.bt.iter().map(|b| *b as i32).collect::<Vec<_>>())?;
        let rows = l.indexer_rows(&a.st, &a.kv, bt_dev)?;
        ensure!(
            l.verify_select_once_for(&a.st, rows, k)?,
            "engagement: the {k}-row verify at {pos} would not take the one pass"
        );
        let ra = a.verify(&l, &fwd, &gpu, &x, pos, k)?;
        l.levers.dsa_verify_select_once = false;
        let rb = b.verify(&l, &fwd, &gpu, &x, pos, k)?;
        same_rows(&format!("verify k={k} at {pos}"), &ra, &rb, HIDDEN * 2)?;
        // 2026-10-21: Accept 1..=k rows, varying, then rewind both arms there.
        let accepted = 1 + (i * 5 + pos) % k;
        pos += accepted;
        for arm in [&mut a, &mut b] {
            arm.st.rewind_to(pos)?;
        }
        verifies += 1;
        i += 1;
    }
    println!("  {verifies} verifies (k = 2..4, rewound between) byte-identical, one pass each");
    println!("PASS");
    Ok(())
}
