// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Byte-identity gate for the paged DSA indexer cache: the same token stream through
//! one DSA layer with a flat indexer state and with a paged one (rows in a scrambled set of KV
//! blocks), on synthetic weights through the real kernels; then a prefix-cache hit served from
//! shared paged blocks against the same tokens served cold.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error unless every compared output is byte-identical:
//!   1. decode (`decode_k`, k = 1), eager and on the replay-safe path a captured graph bakes:
//!      the layer output, the indexer row, the KV latent, at every step;
//!   2. a prefill sub-chunk (`decode_k`, k > 1, the batched selector): the layer output;
//!   3. the batched multi-sequence decode (`decode_rows`) over C paged sequences against the
//!      same C flat ones: each row's output;
//!   4. a prefix hit: sequence B's block table starts with A's first blocks and its fresh
//!      paged state adopts them; B's outputs on A's suffix tokens equal A's.
//!
//! Gate 4 covers the DSA layer only. The KDA recurrent state on a hit is restored from an SSM
//! snapshot (the serve-level check is in the campaign report).
//!
//! `METRALE_PAGED_PARITY_VERBOSE=1` prints every differing row of a failed comparison.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_dsa_paged_parity \
//!       --features cuda,gpu-examples

use anyhow::{Context, Result};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::glm5next_dsa::layer::Glm5NextDsaLayer;
use metrale_model_arch::glm5next_dsa::paged::IndexerCache;
use metrale_model_arch::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_layers::layer::LayerState;
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};

#[path = "common/glm5next_dsa_paged_rig.rs"]
pub(crate) mod rig;
use rig::*;

/// 2026-10-09: One sequence's state, KV cache and block table for one arm of the comparison.
struct Arm {
    st: Box<dyn LayerState>,
    kv: PagedKvCache,
    bt: Vec<u32>,
}

impl Arm {
    fn flat(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            st: Box::new(Glm5NextDsaState::alloc(gpu, &cfg())?),
            kv: kv(gpu, MB)?,
            bt: (0..MB as u32).collect(),
        })
    }
    /// 2026-10-09: The same blocks in a scrambled order inside a pool three times larger, so a
    /// path that ignored the table would read rows another sequence could own.
    fn paged(gpu: &dyn GpuBackend, salt: usize) -> Result<Self> {
        Ok(Self {
            st: Box::new(Glm5NextDsaState::paged(&cfg())?),
            kv: kv(gpu, 3 * MB)?,
            bt: (0..MB)
                .map(|b| ((b * 7 + salt) % (3 * MB)) as u32)
                .collect(),
        })
    }
    fn indexer_row(
        &self,
        l: &Glm5NextDsaLayer,
        gpu: &dyn GpuBackend,
        pos: usize,
    ) -> Result<Vec<u8>> {
        let s = self
            .st
            .as_any()
            .downcast_ref::<Glm5NextDsaState>()
            .context("DSA state")?;
        let d = cfg().index_head_dim;
        let (k, g) = match s.cache() {
            IndexerCache::Flat => (
                s.k_normed.offset(s.row_offset(pos)),
                s.gate.offset(s.row_offset(pos)),
            ),
            IndexerCache::Paged => {
                let lay = l.paged_layout(&self.kv)?;
                let pool = self.kv.v_pool_ptr(0);
                let off = lay.key_row_bytes(&self.bt, pos)?;
                (pool.offset(off), pool.offset(off + lay.gate_offset_bytes()))
            }
        };
        let mut v = read(gpu, k, d * 2)?;
        v.extend(read(gpu, g, d * 2)?);
        Ok(v)
    }
    fn latent(&self, gpu: &dyn GpuBackend, pos: usize) -> Result<Vec<u8>> {
        let slot = self.bt[pos / BLOCK] as usize * BLOCK + pos % BLOCK;
        read(gpu, self.kv.k_pool_ptr(0).offset(slot * 512), 512)
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
        step(
            l,
            fwd,
            gpu,
            self.st.as_mut(),
            &mut self.kv,
            &mut self.bt,
            x,
            pos,
            capture,
        )
    }
}

/// 2026-10-09: Gates 1 and 2: a prefill sub-chunk, then decode token by token, flat against paged.
fn decode_gate(
    gpu: &dyn GpuBackend,
    fwd: &Fwd,
    l: &Glm5NextDsaLayer,
    rng: &mut Lcg,
    capture: bool,
) -> Result<()> {
    let (mut f, mut p) = (Arm::flat(gpu)?, Arm::paged(gpu, 3)?);
    // 2026-10-09: A second flat arm is the control: flat against flat must match before
    // paged against flat means anything.
    let mut control = Arm::flat(gpu)?;
    let chunk = 12;
    let x = rng.bf16_bytes(chunk * HIDDEN, 1.0);
    let mut outs = Vec::new();
    for arm in [&mut f, &mut control, &mut p] {
        let h = up(gpu, &x)?;
        let ctx = fwd.ctx(gpu, false, false, None);
        l.decode_k(
            h,
            chunk,
            arm.st.as_mut(),
            &mut arm.kv,
            0,
            &mut arm.bt,
            &ctx,
            stream(gpu),
            true,
        )?;
        outs.push(read(gpu, h, chunk * HIDDEN * 2)?);
    }
    let tag = format!("capture={capture} prefill sub-chunk output");
    same_rows(
        &format!("{tag} (flat control)"),
        &outs[0],
        &outs[1],
        HIDDEN * 2,
    )?;
    same_rows(
        &format!("{tag} (paged vs flat)"),
        &outs[0],
        &outs[2],
        HIDDEN * 2,
    )?;
    for pos in chunk..LEN {
        let x = rng.bf16_bytes(HIDDEN, 1.0);
        let a = f.step(l, fwd, gpu, &x, pos, capture)?;
        let b = p.step(l, fwd, gpu, &x, pos, capture)?;
        let tag = format!("capture={capture} pos {pos}");
        same(&format!("{tag} output"), &a, &b)?;
        same(
            &format!("{tag} indexer row"),
            &f.indexer_row(l, gpu, pos)?,
            &p.indexer_row(l, gpu, pos)?,
        )?;
        same(
            &format!("{tag} KV latent"),
            &f.latent(gpu, pos)?,
            &p.latent(gpu, pos)?,
        )?;
    }
    println!(
        "  decode capture={capture}: prefill chunk + {} steps byte-identical",
        LEN - chunk
    );
    Ok(())
}

/// 2026-10-09: Gate 3: C sequences of different lengths in one `decode_rows` per arm: the flat
/// arm's rows in their own buffers over a shared KV cache, the paged arm's in scrambled,
/// disjoint blocks of another; each row's output must match.
fn rows_gate(
    gpu: &dyn GpuBackend,
    fwd: &Fwd,
    l: &Glm5NextDsaLayer,
    rng: &mut Lcg,
    capture: bool,
) -> Result<()> {
    let c = 4;
    let lens: Vec<usize> = (0..c).map(|r| 17 + r * 31).collect();
    let n = 3 * c * MB;
    let mut fkv = kv(gpu, c * MB)?;
    let mut pkv = kv(gpu, n)?;
    let fbt: Vec<Vec<u32>> = (0..c)
        .map(|r| (0..MB).map(|b| (r * MB + b) as u32).collect())
        .collect();
    // 2026-10-09: `i * 7 + 5 mod n` is a bijection (7 and 120 are coprime), so the tables are
    // disjoint.
    let pbt: Vec<Vec<u32>> = (0..c)
        .map(|r| {
            (0..MB)
                .map(|b| (((r * MB + b) * 7 + 5) % n) as u32)
                .collect()
        })
        .collect();
    let mut fst: Vec<Box<dyn LayerState>> = Vec::new();
    let mut pst: Vec<Box<dyn LayerState>> = Vec::new();
    for r in 0..c {
        let mut f: Box<dyn LayerState> = Box::new(Glm5NextDsaState::alloc(gpu, &cfg())?);
        let mut p: Box<dyn LayerState> = Box::new(Glm5NextDsaState::paged(&cfg())?);
        let (mut fb, mut pb) = (fbt[r].clone(), pbt[r].clone());
        for pos in 0..lens[r] {
            let x = rng.bf16_bytes(HIDDEN, 1.0);
            let a = step(l, fwd, gpu, f.as_mut(), &mut fkv, &mut fb, &x, pos, false)?;
            let b = step(l, fwd, gpu, p.as_mut(), &mut pkv, &mut pb, &x, pos, false)?;
            same(&format!("row {r} history pos {pos}"), &a, &b)?;
        }
        fst.push(f);
        pst.push(p);
    }
    let x = rng.bf16_bytes(c * HIDDEN, 1.0);
    let mut outs = Vec::new();
    for (states, cache, tables) in [(&mut fst, &mut fkv, &fbt), (&mut pst, &mut pkv, &pbt)] {
        let h = up(gpu, &x)?;
        let rows: Vec<(usize, &[u32])> = (0..c).map(|r| (lens[r], tables[r].as_slice())).collect();
        let m = meta(gpu, &rows)?;
        let ctx = fwd.ctx(gpu, capture, true, Some(m));
        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
            states.iter_mut().map(|b| b.as_mut()).collect();
        l.decode_rows(h, &mut refs, &lens, cache, &m, 0, &ctx, stream(gpu))?;
        outs.push(read(gpu, h, c * HIDDEN * 2)?);
    }
    same_rows(
        &format!("decode_rows capture={capture}"),
        &outs[0],
        &outs[1],
        HIDDEN * 2,
    )?;
    println!("  decode_rows capture={capture}: {c} paged rows byte-identical to flat");
    Ok(())
}

/// 2026-10-09: Decode one token at `pos`; returns the layer output. With `capture`, the
/// replay-safe path a captured graph bakes, launched eagerly with one metadata row.
#[allow(clippy::too_many_arguments)]
fn step(
    l: &Glm5NextDsaLayer,
    fwd: &Fwd,
    gpu: &dyn GpuBackend,
    st: &mut dyn LayerState,
    cache: &mut PagedKvCache,
    bt: &mut Vec<u32>,
    x: &[u8],
    pos: usize,
    capture: bool,
) -> Result<Vec<u8>> {
    let h = up(gpu, x)?;
    let m = capture
        .then(|| meta(gpu, &[(pos, bt.as_slice())]))
        .transpose()?;
    let ctx = fwd.ctx(gpu, capture, true, m);
    l.decode_k(h, 1, st, cache, pos, bt, &ctx, stream(gpu), false)?;
    read(gpu, h, HIDDEN * 2)
}

/// 2026-10-09: Gate 4: A runs LEN tokens. B's table repeats A's first `shared` blocks and owns
/// the rest; its fresh paged state adopts the shared rows at the first step. B then decodes
/// A's suffix tokens: every output must equal A's.
fn prefix_gate(gpu: &dyn GpuBackend, fwd: &Fwd, l: &Glm5NextDsaLayer, rng: &mut Lcg) -> Result<()> {
    let mut a = Arm::paged(gpu, 0)?;
    let shared = 5;
    let start = shared * BLOCK;
    let xs: Vec<Vec<u8>> = (0..LEN).map(|_| rng.bf16_bytes(HIDDEN, 1.0)).collect();
    let mut a_out = Vec::new();
    for (pos, x) in xs.iter().enumerate() {
        a_out.push(a.step(l, fwd, gpu, x, pos, false)?);
    }
    // 2026-10-09: B lives in A's pool: shared blocks by id, the rest in blocks A never used.
    let used: std::collections::BTreeSet<u32> = a.bt.iter().copied().collect();
    let mut free = (0..3 * MB as u32).filter(|b| !used.contains(b));
    let mut bt: Vec<u32> = a.bt[..shared].to_vec();
    bt.extend((shared..MB).map(|_| free.next().expect("a free block")));
    let mut b = Arm {
        st: Box::new(Glm5NextDsaState::paged(&cfg())?),
        kv: std::mem::replace(&mut a.kv, kv(gpu, 1)?),
        bt,
    };
    for pos in start..LEN {
        let out = b.step(l, fwd, gpu, &xs[pos], pos, false)?;
        same(&format!("prefix hit pos {pos}"), &a_out[pos], &out)?;
    }
    println!(
        "  prefix hit at {start} tokens: {} suffix outputs byte-identical to cold",
        LEN - start
    );
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
        stats: ModelStats::new(),
    };
    let mut rng = Lcg(0x6c6d_5e9d_0009);
    let l = layer(&gpu, &cfg(), &mut rng)?;
    println!("GATE: paged DSA indexer cache vs flat, and a prefix hit vs cold");
    for capture in [false, true] {
        decode_gate(&gpu, &fwd, &l, &mut rng, capture)?;
        rows_gate(&gpu, &fwd, &l, &mut rng, capture)?;
    }
    prefix_gate(&gpu, &fwd, &l, &mut rng)?;
    println!("PASS");
    Ok(())
}
