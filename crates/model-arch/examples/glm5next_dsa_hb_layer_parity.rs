// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The head-batched DSA decode through the real layer paths against the per-head
//! kernel: two DSA layers with the same synthetic weights, one per decode kernel, fed the same
//! tokens into their own paged caches (22 heads, top-2048 selection, sequences past 2048
//! tokens): prefill sub-chunks (`decode_k`, k > 1), single-row decode (`decode_k`, k = 1,
//! eager and on the replay-safe metadata path) and the batched multi-sequence decode
//! (`decode_rows`) over ragged lengths.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Run with `METRALE_GLM_DSA_DECODE_HB=<splits>` (the workspace reads it). Exits with an error
//!   when any layer output row differs from the per-head kernel's by more than [`TOL`] of that
//!   row's largest magnitude.
//!
//!   METRALE_GLM_DSA_DECODE_HB=8 METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash \
//!   METRALE_TARGET_QUANT=nvfp4 cargo run -p metrale-model-arch --release \
//!       --example glm5next_dsa_hb_layer_parity --features cuda,gpu-examples

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::glm5next_dsa::Glm5NextDsaConfig;
use metrale_model_arch::glm5next_dsa::attend::{Glm5NextDsaDecodeKernel, hb_splits_lever};
use metrale_model_arch::glm5next_dsa::layer::{Glm5NextDsaLayer, Glm5NextDsaWorkspace};
use metrale_model_arch::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_layers::layer::{AttnMetadataDev, LayerState};
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};

// 2026-10-09: The rig serves the paged-parity gate too; this example uses part of it.
#[allow(dead_code)]
#[path = "common/glm5next_dsa_paged_rig.rs"]
pub(crate) mod rig;
use rig::{BLOCK, Fwd, HIDDEN, Lcg, kv, layer, read, stream, up, up_i32};

/// 2026-10-09: Blocks per sequence: 2,720 tokens.
const SEQ_BLOCKS: usize = 170;
/// 2026-10-09: Allowed difference, as a fraction of the row's largest |output|. Rounding-order
/// differences sit near 2^-8 of an element (one BF16 ulp); a defect is far above it.
const TOL: f64 = 0.02;
const LAT: usize = 22 * 512;

/// 2026-10-09: Bit-equal fraction and worst element difference in BF16 ulps of the element,
/// of two latent attention outputs.
fn attn_stats(what: &str, a: &[u8], b: &[u8]) {
    let f = |x: &[u8]| -> Vec<f32> {
        x.chunks(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    };
    let (a, b) = (f(a), f(b));
    let same = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() == y.to_bits())
        .count();
    let mut worst = 0f64;
    let mut at = 0;
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        let ulp = (x.abs().max(1e-30) as f64).log2().floor().exp2() / 128.0;
        let d = (x - y).abs() as f64 / ulp;
        if d > worst {
            worst = d;
            at = i;
        }
    }
    if std::env::var("HB_PARITY_VERBOSE").is_ok() || worst > 4.0 {
        println!(
            "    attn {what}: {:.2}% bit-equal, worst {worst:.1} ulp at row {} head {} dim {} ({} vs {})",
            same as f64 * 100.0 / a.len() as f64,
            at / LAT,
            (at % LAT) / 512,
            at % 512,
            a[at],
            b[at]
        );
    }
}

fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: HIDDEN,
        index_heads: 8,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 22,
        q_lora_rank: 512,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: SEQ_BLOCKS * BLOCK,
    }
}

/// 2026-10-09: One metadata row per `(position, table)`, tables `SEQ_BLOCKS` wide.
fn meta(gpu: &dyn GpuBackend, seqs: &[(usize, &[u32])]) -> Result<AttnMetadataDev> {
    let positions: Vec<i32> = seqs.iter().map(|(l, _)| *l as i32).collect();
    let slots: Vec<u8> = seqs
        .iter()
        .flat_map(|(l, bt)| ((bt[l / BLOCK] as usize * BLOCK + l % BLOCK) as i64).to_le_bytes())
        .collect();
    let lens: Vec<i32> = seqs.iter().map(|(l, _)| *l as i32 + 1).collect();
    let bts: Vec<i32> = seqs
        .iter()
        .flat_map(|(_, bt)| bt.iter().map(|b| *b as i32))
        .collect();
    let p = up_i32(gpu, &positions)?;
    Ok(AttnMetadataDev {
        positions: p,
        positions_h: p,
        positions_w: p,
        slot: up(gpu, &slots)?,
        seq_len: up_i32(gpu, &lens)?,
        block_table: up_i32(gpu, &bts)?,
        max_blocks_per_seq: SEQ_BLOCKS as u32,
        num_seqs: seqs.len() as u32,
        seq_slot: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        moe_row_adapter: metrale_gpu_runtime::gpu::DevicePtr::NULL,
    })
}

/// 2026-10-09: Worst row difference of `b` against `a`, as a fraction of the row's max |a|.
fn worst(a: &[u8], b: &[u8]) -> f64 {
    let f = |x: &[u8]| -> Vec<f32> {
        x.chunks(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    };
    let (a, b) = (f(a), f(b));
    let mut w = 0f64;
    for (ra, rb) in a.chunks(HIDDEN).zip(b.chunks(HIDDEN)) {
        let scale = ra.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30) as f64;
        let d = ra
            .iter()
            .zip(rb)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs())) as f64;
        w = w.max(d / scale);
    }
    w
}

/// 2026-10-09: One kernel's side: a KV cache holding both sequences, their states and tables.
struct Side {
    kv: PagedKvCache,
    st: Vec<Box<dyn LayerState>>,
    bt: Vec<Vec<u32>>,
}

fn check(what: &str, a: &[u8], b: &[u8], failed: &mut bool) {
    let w = worst(a, b);
    let ok = w <= TOL;
    *failed |= !ok;
    println!(
        "  {what:48} worst {w:.4} of row max  {}",
        if ok { "ok" } else { "FAIL" }
    );
}

fn main() -> Result<()> {
    let splits = hb_splits_lever()?.context("set METRALE_GLM_DSA_DECODE_HB=<splits>")?;
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm target")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
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
        stats: ModelStats::new(),
    };
    let c = cfg();
    let build = |hb: Option<usize>| -> Result<Glm5NextDsaLayer> {
        let mut l = layer(g, &c, &mut Lcg(0x6c6d_5e9d_0009))?;
        l.decode_kernel = Glm5NextDsaDecodeKernel::resolve_with(g, hb)?;
        l.workspace = Glm5NextDsaWorkspace::new(g, &c, 64)?;
        Ok(l)
    };
    let (lo, lh) = (build(None)?, build(Some(splits))?);
    println!("head-batched ({splits} splits) vs per-head DSA decode through the layer:");
    let mut rng = Lcg(0x5eed_0001);
    let mut failed = false;

    // 2026-10-09: Two sequences per side, in scrambled disjoint blocks of one cache, prefilled
    // in sub-chunks of 64 rows to 2,600 and 1,100 tokens (early chunks select fewer than 2,048
    // entries, late ones a full row).
    let lens = [2600usize, 1100];
    let pool = 3 * 2 * SEQ_BLOCKS;
    let mut sides: Vec<Side> = Vec::new();
    for _ in 0..2 {
        sides.push(Side {
            kv: kv(g, pool)?,
            st: vec![
                Box::new(Glm5NextDsaState::paged(&c)?),
                Box::new(Glm5NextDsaState::paged(&c)?),
            ],
            // 2026-10-09: `i * 7 + 5 mod pool` is a bijection (7 and 1,020 are coprime).
            bt: (0..2)
                .map(|s| {
                    (0..SEQ_BLOCKS)
                        .map(|b| (((s * SEQ_BLOCKS + b) * 7 + 5) % pool) as u32)
                        .collect()
                })
                .collect(),
        });
    }
    let ls = [&lo, &lh];
    for (s, &len) in lens.iter().enumerate() {
        let chunk = 64;
        let mut pos = 0;
        while pos < len {
            let k = chunk.min(len - pos);
            let x = rng.bf16_bytes(k * HIDDEN, 1.0);
            let mut outs = Vec::new();
            let mut atts = Vec::new();
            for (side, l) in sides.iter_mut().zip(ls) {
                let h = up(g, &x)?;
                let ctx = fwd.ctx(g, false, false, None);
                l.decode_k(
                    h,
                    k,
                    side.st[s].as_mut(),
                    &mut side.kv,
                    pos,
                    &mut side.bt[s],
                    &ctx,
                    stream(g),
                    true,
                )?;
                outs.push(read(g, h, k * HIDDEN * 2)?);
                atts.push(read(g, l.workspace.attn_out(), k * LAT * 2)?);
            }
            attn_stats(
                &format!("seq {s} prefill rows {pos}..{}", pos + k),
                &atts[0],
                &atts[1],
            );
            if pos % 512 == 0 || pos + k >= len || worst(&outs[0], &outs[1]) > TOL {
                let what = format!("seq {s} prefill rows {pos}..{}", pos + k);
                check(&what, &outs[0], &outs[1], &mut failed);
            }
            pos += k;
        }
    }

    // 2026-10-09: Single-row decode, eager and on the metadata (graph-replay) path.
    for capture in [false, true] {
        for (s, len) in lens.iter().enumerate() {
            let pos = len + capture as usize;
            let x = rng.bf16_bytes(HIDDEN, 1.0);
            let mut outs = Vec::new();
            for (side, l) in sides.iter_mut().zip(ls) {
                let h = up(g, &x)?;
                let m = capture
                    .then(|| meta(g, &[(pos, side.bt[s].as_slice())]))
                    .transpose()?;
                let ctx = fwd.ctx(g, capture, true, m);
                l.decode_k(
                    h,
                    1,
                    side.st[s].as_mut(),
                    &mut side.kv,
                    pos,
                    &mut side.bt[s],
                    &ctx,
                    stream(g),
                    false,
                )?;
                outs.push(read(g, h, HIDDEN * 2)?);
            }
            let what = format!("seq {s} decode pos {pos} capture={capture}");
            check(&what, &outs[0], &outs[1], &mut failed);
        }
    }

    // 2026-10-09: Both sequences in one batched decode (ragged: 2,602 and 1,102 tokens).
    let x = rng.bf16_bytes(2 * HIDDEN, 1.0);
    let pos: Vec<usize> = lens.iter().map(|l| l + 2).collect();
    let mut outs = Vec::new();
    for (side, l) in sides.iter_mut().zip(ls) {
        let h = up(g, &x)?;
        let m = meta(
            g,
            &[
                (pos[0], side.bt[0].as_slice()),
                (pos[1], side.bt[1].as_slice()),
            ],
        )?;
        let ctx = fwd.ctx(g, true, true, Some(m));
        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
            side.st.iter_mut().map(|b| b.as_mut()).collect();
        l.decode_rows(h, &mut refs, &pos, &mut side.kv, &m, 0, &ctx, stream(g))?;
        outs.push(read(g, h, 2 * HIDDEN * 2)?);
    }
    check(
        "decode_rows, 2 ragged sequences",
        &outs[0],
        &outs[1],
        &mut failed,
    );

    if failed {
        bail!("head-batched DSA decode differs from the per-head kernel through the layer");
    }
    println!("PASS");
    Ok(())
}
