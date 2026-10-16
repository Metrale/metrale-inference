// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: MLA-UNABSORB G2, layer level: one GLM-5.3-Flash DSA layer (layer 3, TP=3 rank
//! 0, 22 heads, real checkpoint weights) runs two 1040-row prefill chunks per arm, and each
//! arm's latent query (`q_abs`) and output (`hidden`) are checked against an f64 reference
//! built from that arm's own inputs (`q_resid`, `attn_out`) through the unabsorbed chain in
//! exact arithmetic (`unabsorb::host`).
//!
//! Arms (one layer, its `unabsorb` swapped; each arm a fresh KV cache and DSA state):
//! - ABS: absorbed (`q_absorb`, `o_absorb`), the lever off.
//! - A: `METRALE_GLM_MLA_UNABSORB=dsa`, `q_b_proj` / `o_proj` at `--dsa-quantization`'s tier.
//! - B: `=declared`, `q_b_proj` / `o_proj` at the checkpoint's BF16 (B is also run twice and
//!   must repeat byte for byte).
//! - NEG (negative control, must FIRE): B with heads 0 and 1 of both `kv_b_proj` halves swapped.
//!
//! Owner: model-arch examples (GLM-5.3 DSA).
//! Invariants:
//! - Gate, per chunk and per output (`q_abs`, `hidden`): `rel-RMS(A) <= 1.25 rel-RMS(ABS)` and
//!   `rel-RMS(B) <= 1.25 rel-RMS(ABS)` against the f64 reference; NEG must exceed 1.25 on both.
//!   Raw max-abs and max-ulp are printed beside each.
//! - `q_resid` (`q_a_proj` and its norm, upstream of every arm difference) is byte-identical
//!   across arms, so all arms see one input. Under `--dsa declared` the site table gives A and B
//!   the same precision, and their bytes must be equal.
//! - The projections each arm registers follow the DSA site tier table (`site_rule`), as
//!   `dense_tiers::tier_for` does with `--dense-quantization declared`: under `--dsa w8a8`, ABS
//!   registers `q_a_proj` / `q_absorb` / `o_absorb` FP8 and A adds `q_b_proj` / `o_proj`; B and
//!   NEG add nothing. The other DSA projections stay BF16 in every arm.
//!
//!   GLM_CKPT=/srv/models/glm-5.3-flash-nvfp4 cargo run -p metrale-model-arch --release \
//!       --example glm5next_mla_unabsorb_layer_gate --features cuda,gpu-examples -- --dsa declared
//!   (`--dsa w8a8` with `METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS=1024`, as the recipe runs it)

#[path = "common/glm_ckpt.rs"]
mod glm_ckpt;
#[path = "common/mla_unabsorb_gate.rs"]
mod mla;
#[path = "common/glm5next_dsa_paged_rig.rs"]
mod rig;

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::Glm5NextDsaConfig;
use metrale_model_arch::glm5next_dsa::build::build_dsa_weights;
use metrale_model_arch::glm5next_dsa::layer::Glm5NextDsaLayer;
use metrale_model_arch::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_arch::glm5next_dsa::tp::DsaTpPlan;
use metrale_model_arch::glm5next_dsa::unabsorb::host::unabsorb_for_rank;
use metrale_model_arch::glm5next_dsa::unabsorb::{
    MlaUnabsorb, SiteRule, TextUnabsorb, UnabsorbWeights, site_rule,
};
use metrale_model_arch::glm5next_fp8_dense as fp8;
use metrale_model_arch::glm5next_w4a16_dense::DSA_SITES;
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use metrale_model_layers::layers::{DsaQuantization, set_dsa_quantization_from_cli};
use mla::{ErrStats, Lcg, dense_par, err_stats, f64s, from_bf16_bytes, head_bmm_par, read};

const LAYER: usize = 3;
const FULL_HEADS: usize = 64;
const TP: usize = 3;
const ROWS: usize = 1040;
const CHUNKS: usize = 2;
const KV_BLOCKS: usize = 256;
const BOUND: f64 = 1.25;
/// 2026-10-15: The widest K any DSA projection registers FP8 with (`o_absorb`, 22 x 512).
const FP8_MAX_K: usize = 11264;

fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 22,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: KV_BLOCKS * rig::BLOCK,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Abs,
    A,
    B,
    BRepeat,
    Neg,
}
const ARMS: [Arm; 5] = [Arm::Abs, Arm::A, Arm::B, Arm::BRepeat, Arm::Neg];

/// 2026-10-15: One chunk of one arm: the BF16 bytes of its query input, latent query, attention output
/// and layer output.
struct Chunk {
    q_resid: Vec<u8>,
    q_abs: Vec<u8>,
    attn: Vec<u8>,
    out: Vec<u8>,
}

/// 2026-10-15: Register FP8 the projections of `projs` the site table gives `--dsa-quantization`
/// under `mode` (the process runs `--dsa w8a8`); returns their names.
fn register_dsa_sites(
    g: &dyn GpuBackend,
    projs: Vec<(DevicePtr, usize, usize, &'static str)>,
    mode: MlaUnabsorb,
) -> Result<Vec<&'static str>> {
    let quantize = g.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
    let mut names = Vec::new();
    for (w, n, k, name) in projs {
        if DSA_SITES.contains(&name) && site_rule(name, mode)? == SiteRule::DsaKnob {
            fp8::register(g, quantize, w, n, k, name, g.default_stream())?;
            names.push(name);
        }
    }
    g.synchronize(g.default_stream())?;
    Ok(names)
}

/// 2026-10-15: Two prefill chunks of `inputs` through `layer` from an empty sequence.
fn run_arm(
    g: &dyn GpuBackend,
    fwd: &rig::Fwd,
    layer: &Glm5NextDsaLayer,
    c: &Glm5NextDsaConfig,
    inputs: &[Vec<u8>],
) -> Result<Vec<Chunk>> {
    let mut kv = rig::kv(g, KV_BLOCKS)?;
    let mut st = Glm5NextDsaState::paged(c)?;
    let mut bt: Vec<u32> = (0..KV_BLOCKS as u32).collect();
    let hidden = g.alloc(ROWS * c.hidden * 2)?;
    let ctx = fwd.ctx(g, false, false, None);
    let s = g.default_stream();
    let latent = c.local_heads * c.kv_lora_rank * 2;
    let mut chunks = Vec::new();
    for (i, x) in inputs.iter().enumerate() {
        g.copy_h2d(x, hidden)?;
        layer.decode_k(
            hidden,
            ROWS,
            &mut st,
            &mut kv,
            i * ROWS,
            &mut bt,
            &ctx,
            s,
            true,
        )?;
        let ws = &layer.workspace;
        chunks.push(Chunk {
            q_resid: read(g, ws.q_resid(), ROWS * c.q_lora_rank * 2)?,
            q_abs: read(g, ws.q_abs(), ROWS * latent)?,
            attn: read(g, ws.attn_out(), ROWS * latent)?,
            out: read(g, hidden, ROWS * c.hidden * 2)?,
        });
    }
    g.free(hidden)?;
    Ok(chunks)
}

/// 2026-10-15: `v` with its first two blocks of `per` elements (heads 0 and 1) swapped.
fn swap_heads(v: &[f32], per: usize) -> Vec<f32> {
    let mut w = v.to_vec();
    w[..per].copy_from_slice(&v[per..2 * per]);
    w[per..2 * per].copy_from_slice(&v[..per]);
    w
}

fn verdict(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

/// 2026-10-15: Prints one output's statistics for every arm and returns whether the gate holds
/// (A and B within the bound) and whether NEG fired.
fn judge(what: &str, stats: &[(Arm, ErrStats)]) -> Result<(bool, bool)> {
    let abs = stats
        .iter()
        .find(|(a, _)| *a == Arm::Abs)
        .context("no ABS arm")?
        .1;
    let (mut ok, mut fired) = (true, true);
    for (arm, st) in stats {
        let ratio = st.rel_rms / abs.rel_rms;
        let note = match arm {
            Arm::Abs => "reference arm".to_string(),
            Arm::Neg => {
                let f = ratio > BOUND;
                fired &= f;
                format!(
                    "ratio {ratio:.3} -> {}",
                    if f { "FIRED" } else { "DID NOT FIRE" }
                )
            }
            _ => {
                let pass = ratio <= BOUND;
                ok &= pass;
                format!("ratio {ratio:.3} <= {BOUND} -> {}", verdict(pass))
            }
        };
        println!("    {what:6} {:8} {st}; {note}", format!("{arm:?}"));
    }
    Ok((ok, fired))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dsa = match args
        .iter()
        .position(|a| a == "--dsa")
        .map(|i| args.get(i + 1))
    {
        Some(Some(v)) if v == "declared" => DsaQuantization::Declared,
        Some(Some(v)) if v == "w8a8" => DsaQuantization::W8a8,
        _ => bail!("usage: glm5next_mla_unabsorb_layer_gate --dsa declared|w8a8"),
    };
    ensure!(
        set_dsa_quantization_from_cli(dsa) == dsa,
        "--dsa-quantization already set"
    );
    let dir = std::env::var("GLM_CKPT").context("GLM_CKPT names the checkpoint directory")?;
    let min_rows = std::env::var("METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS").ok();
    let c = cfg();
    let plan = DsaTpPlan::new(0, TP, FULL_HEADS, &c)?;
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let base = mla::used_mb(g)?;
    println!(
        "MLA-UNABSORB layer gate: layer {LAYER}, rank 0 of {TP} ({} heads), {CHUNKS} prefill \
         chunks of {ROWS} rows, --dsa-quantization {dsa:?}, FP8 dequant GEMM min rows {}, \
         --dense-quantization declared, DEQUANT_PERSIST off; device in use {base:.0} MB at start",
        c.local_heads,
        min_rows.as_deref().unwrap_or("unset")
    );

    let cache: RefCell<HashMap<String, Vec<f32>>> = RefCell::new(HashMap::new());
    let load = |name: &str| -> Result<Vec<f32>> {
        if let Some(v) = cache.borrow().get(name) {
            return Ok(v.clone());
        }
        let v = glm_ckpt::tensor_f32(&dir, &format!("model.language_model.layers.{LAYER}.{name}"))?;
        cache.borrow_mut().insert(name.to_string(), v.clone());
        Ok(v)
    };
    let t = Instant::now();
    let weights = build_dsa_weights(g, &c, &plan, &load)?;
    let mut layer = rig::layer_with(g, &c, weights, ROWS)?;
    let host = unabsorb_for_rank(&c, &plan, &load)?;
    println!(
        "  absorbed weights built and uploaded in {:.1} s",
        t.elapsed().as_secs_f64()
    );
    if dsa == DsaQuantization::W8a8 {
        fp8::prepare(g, FP8_MAX_K)?;
        let names = register_dsa_sites(g, layer.dense_projections(), MlaUnabsorb::Off)?;
        println!("  ABS registers FP8: {names:?}");
    }
    let unabsorbed = |mode| -> Result<UnabsorbWeights> {
        TextUnabsorb::new(g, &c, ROWS, true, 1, mode)?
            .weights(g, &c, &plan, &load)?
            .context("the lever is on")
    };
    let a = unabsorbed(MlaUnabsorb::Dsa)?;
    let b = unabsorbed(MlaUnabsorb::Declared)?;
    let per = c.kv_lora_rank * c.qk_nope_head_dim;
    let neg = UnabsorbWeights {
        w_uk_t: mla::up_bytes(g, &mla::to_bf16_bytes(&swap_heads(&host.w_uk_t, per)))?,
        w_uv: mla::up_bytes(g, &mla::to_bf16_bytes(&swap_heads(&host.w_uv, per)))?,
        mode: b.mode,
        q_b: b.q_b,
        o_proj: b.o_proj,
        mid: b.mid,
        mid_rows: b.mid_rows,
        kernels: b.kernels,
    };
    let fwd = fwd(g)?;
    let mut rng = Lcg(0x6d6c_6175_6e61_6273);
    let inputs: Vec<Vec<u8>> = (0..CHUNKS)
        .map(|_| mla::to_bf16_bytes(&rng.bf16_vals(ROWS * c.hidden, 1.0)))
        .collect();

    let mut slots = [Some(a), Some(b), Some(neg)];
    let mut runs: Vec<(Arm, Vec<Chunk>)> = Vec::new();
    let mut peak = mla::used_mb(g)? - base;
    for arm in ARMS {
        layer.weights.unabsorb = match arm {
            Arm::Abs => None,
            Arm::A => slots[0].take(),
            Arm::B => slots[1].take(),
            Arm::BRepeat => layer.weights.unabsorb.take(),
            Arm::Neg => slots[2].take(),
        };
        if dsa == DsaQuantization::W8a8 && arm == Arm::A {
            let mode = layer.weights.unabsorb.as_ref().context("A weights")?.mode;
            let names = register_dsa_sites(g, layer.unabsorb_projections(), mode)?;
            println!("  A registers FP8: {names:?}");
        }
        let t = Instant::now();
        let chunks = run_arm(g, &fwd, &layer, &c, &inputs)?;
        peak = peak.max(mla::used_mb(g)? - base);
        println!(
            "  arm {arm:?}: {CHUNKS} chunks in {:.2} s (eager, with reads)",
            t.elapsed().as_secs_f64()
        );
        runs.push((arm, chunks));
    }
    println!("  peak device memory above start: {peak:.0} MB\n");

    let of = |arm: Arm| &runs.iter().find(|r| r.0 == arm).expect("every arm ran").1;
    let mut ok = true;
    for i in 0..CHUNKS {
        for (arm, chunks) in &runs {
            ensure!(
                chunks[i].q_resid == of(Arm::Abs)[i].q_resid,
                "chunk {i}: q_resid of {arm:?} differs from ABS"
            );
        }
        let rep = of(Arm::B)[i].q_abs == of(Arm::BRepeat)[i].q_abs
            && of(Arm::B)[i].out == of(Arm::BRepeat)[i].out;
        println!(
            "  chunk {i} (positions {}..{}): B repeat byte-identical (q_abs, hidden) -> {}",
            i * ROWS,
            (i + 1) * ROWS,
            verdict(rep)
        );
        ok &= rep;
        if dsa == DsaQuantization::Declared {
            let same = of(Arm::A)[i].q_abs == of(Arm::B)[i].q_abs
                && of(Arm::A)[i].out == of(Arm::B)[i].out;
            println!(
                "  chunk {i}: A == B byte for byte under declared (site table) -> {}",
                verdict(same)
            );
            ok &= same;
        }
    }
    let (q_b, o_proj) = (f64s(&host.q_b), f64s(&host.o_proj));
    let (w_uk_t, w_uv) = (f64s(&host.w_uk_t), f64s(&host.w_uv));
    let (lh, ql, kvl, d) = (
        c.local_heads,
        c.q_lora_rank,
        c.kv_lora_rank,
        c.qk_nope_head_dim,
    );
    let mut fired = true;
    for i in 0..CHUNKS {
        let t = Instant::now();
        let q_resid = f64s(&from_bf16_bytes(&of(Arm::Abs)[i].q_resid));
        let ref_q = head_bmm_par(
            &dense_par(&q_resid, &q_b, ROWS, lh * d, ql),
            &w_uk_t,
            ROWS,
            lh,
            d,
            kvl,
        );
        let mut q_stats = Vec::new();
        let mut o_stats = Vec::new();
        for (arm, chunks) in runs.iter().filter(|r| r.0 != Arm::BRepeat) {
            q_stats.push((*arm, err_stats(&from_bf16_bytes(&chunks[i].q_abs), &ref_q)));
            let attn = f64s(&from_bf16_bytes(&chunks[i].attn));
            let mid = head_bmm_par(&attn, &w_uv, ROWS, lh, kvl, c.v_head_dim);
            let ref_o = dense_par(&mid, &o_proj, ROWS, c.hidden, lh * c.v_head_dim);
            o_stats.push((*arm, err_stats(&from_bf16_bytes(&chunks[i].out), &ref_o)));
        }
        println!(
            "  chunk {i} vs f64 (references in {:.1} s):",
            t.elapsed().as_secs_f64()
        );
        let (q_ok, q_fired) = judge("q_abs", &q_stats)?;
        let (o_ok, o_fired) = judge("hidden", &o_stats)?;
        ok &= q_ok && o_ok;
        fired &= q_fired && o_fired;
    }
    println!(
        "\nNEGCTL heads 0/1 swapped in both kv_b_proj halves (B weights, --dsa {dsa:?}): rel-RMS above {BOUND}x ABS on q_abs and hidden in every chunk -> {}",
        if fired { "FIRED" } else { "DID NOT FIRE" }
    );
    if !(ok && fired) {
        bail!("MLA-UNABSORB layer gate FAILED (--dsa {dsa:?})");
    }
    println!("MLA-UNABSORB layer gate PASSED (--dsa {dsa:?})");
    Ok(())
}

/// 2026-10-15: The forward context's owner, as `glm5next_dsa_rows_microtest` builds it (the DSA
/// layer reads only its own workspace; the arena is sized for a small dense model).
fn fwd(g: &dyn GpuBackend) -> Result<rig::Fwd> {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 128;
    config.intermediate_size = 128;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 128;
    config.vocab_size = 128;
    Ok(rig::Fwd {
        buffers: BufferArena::new(&config, 8, 16, 16, 8, g)?,
        config,
        dispatch: GemmDispatch::defaults(),
        derived: DerivedWeights::new(),
        levers: ModelLevers::defaults(),
        stats: ModelStats::new(),
    })
}
