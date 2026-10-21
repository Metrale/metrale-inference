// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Byte-identity and no-host-wait gate for `METRALE_GLM_MTP_ROW_NOSYNC`: one synthetic
//! DSA layer writes the same drafter rows through `write_kv_row` with the blocking slot upload
//! (arm A) and the stream-ordered one (arm B, `SlotUpload::StreamOrdered`), each row behind a
//! 256 MiB fill that keeps the stream busy, then both arms run the same decode steps over those
//! rows, eager and captured, whose outputs must be byte-identical.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error unless every compared output is byte-identical, and unless arm B's
//!   `write_kv_row` returns while the fill ahead of it is still running (median host time under
//!   a quarter of arm A's, which waits for it).
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-gpu-examples --release --example glm5next_dsa_row_nosync_gate \
//!       --features cuda,gpu-examples

use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::fp8_rows::IndexerRowFormat;
use metrale_model_arch::glm5next_dsa::layer::{Glm5NextDsaLayer, SlotUpload};
use metrale_model_arch::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_layers::layer::LayerState;
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};

#[path = "common/glm5next_dsa_paged_rig.rs"]
pub(crate) mod rig;
use rig::*;

/// 2026-10-21: The fill ahead of every row: ~1 ms of stream work on GB10.
const FILL: usize = 256 << 20;

struct Arm {
    st: Box<dyn LayerState>,
    kv: PagedKvCache,
    bt: Vec<u32>,
    upload: SlotUpload,
}

impl Arm {
    fn new(gpu: &dyn GpuBackend, upload: SlotUpload) -> Result<Self> {
        Ok(Self {
            st: Box::new(Glm5NextDsaState::alloc_as(
                gpu,
                &cfg(),
                IndexerRowFormat::Bf16,
            )?),
            kv: kv(gpu, MB)?,
            bt: (0..MB as u32).collect(),
            upload,
        })
    }

    /// 2026-10-21: One row at `pos` behind the fill; returns the host time of `write_kv_row`.
    #[allow(clippy::too_many_arguments)]
    fn kv_row(
        &mut self,
        l: &Glm5NextDsaLayer,
        fwd: &Fwd,
        gpu: &dyn GpuBackend,
        x: &[u8],
        pos: usize,
        fill: DevicePtr,
    ) -> Result<Duration> {
        let h = up(gpu, x)?;
        let ctx = fwd.ctx(gpu, false, false, None);
        gpu.memset_async(fill, pos as u8, FILL, stream(gpu))?;
        let t = Instant::now();
        l.write_kv_row(
            h,
            self.st.as_mut(),
            &mut self.kv,
            pos,
            &mut self.bt,
            self.upload,
            &ctx,
            stream(gpu),
        )?;
        let dt = t.elapsed();
        gpu.synchronize(stream(gpu))?;
        Ok(dt)
    }

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
        l.decode_k(
            h,
            1,
            self.st.as_mut(),
            &mut self.kv,
            pos,
            &mut self.bt,
            &ctx,
            stream(gpu),
            false,
        )?;
        read(gpu, h, HIDDEN * 2)
    }
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn gate(gpu: &dyn GpuBackend, fwd: &Fwd, l: &Glm5NextDsaLayer, rng: &mut Lcg) -> Result<()> {
    let mut a = Arm::new(gpu, SlotUpload::Blocking)?;
    let mut b = Arm::new(gpu, SlotUpload::StreamOrdered)?;
    let fill = gpu.alloc(FILL)?;
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    // 2026-10-21: Rows 0..40 one at a time (the drafter's catch-up writes one per accepted step).
    for pos in 0..40 {
        let x = rng.bf16_bytes(HIDDEN, 1.0);
        ta.push(a.kv_row(l, fwd, gpu, &x, pos, fill)?);
        tb.push(b.kv_row(l, fwd, gpu, &x, pos, fill)?);
    }
    let (ma, mb) = (median(ta), median(tb));
    println!(
        "  write_kv_row host time behind a busy stream: blocking p50 {ma:?}, stream-ordered p50 {mb:?}"
    );
    ensure!(
        mb * 4 < ma,
        "the stream-ordered row did not return before the fill ahead of it ended ({mb:?} vs \
         blocking {ma:?})"
    );
    let mut pos = 40;
    let mut n = 0;
    while pos < LEN - 1 {
        let x = rng.bf16_bytes(HIDDEN, 1.0);
        let capture = n % 2 == 1;
        let (ra, rb) = (
            a.step(l, fwd, gpu, &x, pos, capture)?,
            b.step(l, fwd, gpu, &x, pos, capture)?,
        );
        same(&format!("capture={capture} pos {pos} output"), &ra, &rb)?;
        pos += 1;
        n += 1;
    }
    println!("  40 rows, {n} decode steps over them (eager and captured): byte-identical");
    gpu.free(fill)?;
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
    let mut rng = Lcg(0x5eed_a2a0_0021);
    let l = layer(&gpu, &cfg(), &mut rng)?;
    println!(
        "GATE: stream-ordered drafter row slot upload vs blocking (METRALE_GLM_MTP_ROW_NOSYNC)"
    );
    gate(&gpu, &fwd, &l, &mut rng)?;
    println!("PASS");
    Ok(())
}
