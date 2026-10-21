// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Byte-identity gate for persisted pool keys on a flat DSA indexer cache (the MTP
//! drafter's, `IndexerRowFormat::drafter` under `METRALE_GLM_DSA_POOL_KEYS`): one synthetic DSA
//! layer runs the same token stream over a plain flat state (the captured selection
//! recompresses every pool, `dsa_kpool_compress`) and a pooled flat state (it scores the keys
//! the row writers persisted), through the real kernels.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error unless every compared output is byte-identical. The stream goes
//!   through every writer of a drafter's flat rows: `write_kv_rows_flat` (the drafter's
//!   context rows, two passes ending off a pool boundary), eager `decode_k` (host-placed
//!   rows), `write_kv_row`, and replay-safe `decode_k` (device-placed rows, the captured
//!   selection that reads the keys), with rewinds to off-boundary lengths between captured
//!   steps; each captured step's output is compared.
//! - An engagement control: overwriting the pooled state's keys must change its next captured
//!   output, or the selection did not read them.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-gpu-examples --release --example glm5next_dsa_flat_pooled_gate \
//!       --features cuda,gpu-examples

use anyhow::{Context, Result, ensure};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::glm5next_dsa::fp8_rows::IndexerRowFormat;
use metrale_model_arch::glm5next_dsa::layer::Glm5NextDsaLayer;
use metrale_model_arch::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_layers::layer::LayerState;
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};

#[path = "common/glm5next_dsa_paged_rig.rs"]
pub(crate) mod rig;
use rig::*;

/// 2026-10-21: One flat state in `format`, its KV cache and identity block table.
struct Arm {
    st: Box<dyn LayerState>,
    kv: PagedKvCache,
    bt: Vec<u32>,
}

impl Arm {
    fn new(gpu: &dyn GpuBackend, format: IndexerRowFormat) -> Result<Self> {
        Ok(Self {
            st: Box::new(Glm5NextDsaState::alloc_as(gpu, &cfg(), format)?),
            kv: kv(gpu, MB)?,
            bt: (0..MB as u32).collect(),
        })
    }

    fn state(&mut self) -> Result<&mut Glm5NextDsaState> {
        self.st
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .context("DSA state")
    }

    /// 2026-10-21: `m` drafter context rows from `x` at `pos` (`write_kv_rows_flat`).
    fn context_rows(
        &mut self,
        l: &Glm5NextDsaLayer,
        fwd: &Fwd,
        gpu: &dyn GpuBackend,
        x: &[u8],
        pos: usize,
        m: usize,
    ) -> Result<()> {
        let h = up(gpu, x)?;
        let slots: Vec<u8> = (pos..pos + m)
            .flat_map(|p| ((self.bt[p / BLOCK] as usize * BLOCK + p % BLOCK) as i64).to_le_bytes())
            .collect();
        let slots = up(gpu, &slots)?;
        let ctx = fwd.ctx(gpu, false, false, None);
        l.write_kv_rows_flat(h, m, self.st.as_mut(), &mut self.kv, pos, slots, &ctx, stream(gpu))
    }

    /// 2026-10-21: One row at `pos` without its output (`write_kv_row`).
    fn kv_row(
        &mut self,
        l: &Glm5NextDsaLayer,
        fwd: &Fwd,
        gpu: &dyn GpuBackend,
        x: &[u8],
        pos: usize,
    ) -> Result<()> {
        let h = up(gpu, x)?;
        let ctx = fwd.ctx(gpu, false, false, None);
        l.write_kv_row(h, self.st.as_mut(), &mut self.kv, pos, &mut self.bt, &ctx, stream(gpu))
    }

    /// 2026-10-21: One decode step at `pos` (`decode_k`, k = 1), replay-safe when `capture`.
    fn step(
        &mut self,
        l: &Glm5NextDsaLayer,
        fwd: &Fwd,
        gpu: &dyn GpuBackend,
        x: &[u8],
        pos: usize,
        capture: bool,
    ) -> Result<Vec<u8>> {
        let h = up(gpu, x)?;
        let m = capture
            .then(|| meta(gpu, &[(pos, self.bt.as_slice())]))
            .transpose()?;
        let ctx = fwd.ctx(gpu, capture, true, m);
        l.decode_k(h, 1, self.st.as_mut(), &mut self.kv, pos, &mut self.bt, &ctx, stream(gpu), false)?;
        read(gpu, h, HIDDEN * 2)
    }
}

fn gate(gpu: &dyn GpuBackend, fwd: &Fwd, l: &Glm5NextDsaLayer, rng: &mut Lcg) -> Result<()> {
    let mut a = Arm::new(gpu, IndexerRowFormat::Bf16)?;
    let mut b = Arm::new(gpu, IndexerRowFormat::Bf16Pooled)?;
    ensure!(
        !b.state()?.pool_keys.is_null() && a.state()?.pool_keys.is_null(),
        "the pooled arm must hold persisted keys and the plain arm none"
    );
    // 2026-10-21: Context rows 0..13 and 13..23 (the second pass ends inside a pool).
    for (pos, m) in [(0usize, 13usize), (13, 10)] {
        let x = rng.bf16_bytes(m * HIDDEN, 1.0);
        a.context_rows(l, fwd, gpu, &x, pos, m)?;
        b.context_rows(l, fwd, gpu, &x, pos, m)?;
    }
    let mut pos = 23;
    let mut captured = 0;
    let cmp = |a: &mut Arm, b: &mut Arm, x: &[u8], pos: usize, capture: bool| -> Result<()> {
        let (ra, rb) = (
            a.step(l, fwd, gpu, x, pos, capture)?,
            b.step(l, fwd, gpu, x, pos, capture)?,
        );
        same(&format!("capture={capture} pos {pos} output"), &ra, &rb)
    };
    for _ in 0..9 {
        cmp(&mut a, &mut b, &rng.bf16_bytes(HIDDEN, 1.0), pos, false)?;
        pos += 1;
    }
    for _ in 0..5 {
        let x = rng.bf16_bytes(HIDDEN, 1.0);
        a.kv_row(l, fwd, gpu, &x, pos)?;
        b.kv_row(l, fwd, gpu, &x, pos)?;
        pos += 1;
    }
    // 2026-10-21: Captured steps, rewound by 3 rows every 7 steps (a rejected draft's rows are
    // rewritten, and the pools they complete again get their keys again).
    let mut since = 0;
    while pos < LEN - 1 {
        cmp(&mut a, &mut b, &rng.bf16_bytes(HIDDEN, 1.0), pos, true)?;
        captured += 1;
        pos += 1;
        since += 1;
        if since == 7 {
            pos -= 3;
            a.state()?.rewind_to(pos)?;
            b.state()?.rewind_to(pos)?;
            since = 0;
        }
    }
    println!(
        "  context rows (2 passes), 9 eager steps, 5 kv rows, {captured} captured steps with \
         rewinds: byte-identical"
    );
    // 2026-10-21: Engagement: the pooled arm's keys overwritten, its next captured output must
    // differ from the plain arm's.
    let (pk, n) = {
        let s = b.state()?;
        (s.pool_keys, (s.capacity() / cfg().index_kpool) * cfg().index_head_dim * 4)
    };
    gpu.memset(pk, 0x3f, n)?;
    let x = rng.bf16_bytes(HIDDEN, 1.0);
    let (ra, rb) = (
        a.step(l, fwd, gpu, &x, pos, true)?,
        b.step(l, fwd, gpu, &x, pos, true)?,
    );
    ensure!(
        ra != rb,
        "flat pooled engagement: overwriting the persisted keys left the captured output \
         unchanged, so the selection did not read them"
    );
    println!("  engagement: overwritten flat pool keys change the captured output");
    Ok(())
}

fn main() -> Result<()> {
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
    let mut rng = Lcg(0x5eed_f1a7_0021);
    let l = layer(&gpu, &cfg(), &mut rng)?;
    println!("GATE: persisted pool keys on a flat DSA indexer cache vs recompression");
    gate(&gpu, &fwd, &l, &mut rng)?;
    println!("PASS");
    Ok(())
}
