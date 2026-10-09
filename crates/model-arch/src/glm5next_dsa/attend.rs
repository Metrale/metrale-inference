// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3 DSA attention: the launcher for the selected-index NoPE MLA
//! paged decode.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - `decode_attention` launches only after `DsaDecodePaging::validate`
//!   passes, the selection has one query row per decoded sequence, and
//!   `kv_lora_rank` equals `KERNEL_KV_LORA_DIM`.
//!
//! It consumes what [`super::select`] produced: a `[q_rows, out_width]` i32 row
//! of token ids with `-1` holes, and gathers exactly those tokens from the
//! paged FP8 latent cache. `dsa_mla_masked_attn` is an oracle, not this path
//! (see [`super::MASKED_ATTN_MAX_KEYS`]). When the selection row has no
//! duplicates, the gather attends to the same tokens as the reference's masked
//! attention, whose mask (`glm5next_dsa_ref::topk_to_mask`) is 0/1 set
//! membership.
//!
//! # NoPE, and why neither DeepSeek-V4 MLA decode kernel would do
//!
//! GLM-5.3 is NoPE: the `glm5_next` config parser refuses `qk_rope_head_dim != 0`,
//! so the latent is the whole cache token. `deepseek-v4-flash/nvfp4/mla_paged_decode.cu`
//! declares `kv_cache_dim` and never reads it (it hardcodes `ROPE_DIM 64`);
//! `mla_paged_decode_fp8.cu` in the same directory uses the runtime stride but
//! then overwrites dims 448–511 with rope bytes read past the latent, which under
//! NoPE belong to the next token. Hence a GLM-target kernel with no rope arm.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{Glm5NextDsaConfig, select::DsaSelectGeometry};

/// 2026-09-25: Module name the DSA decode kernel resolves from. An unlisted `.cu`
/// takes its file stem, and this one lives in the `glm-5.3-flash` target.
pub const DSA_DECODE_MODULE: &str = "glm5next_dsa_mla_decode";

/// 2026-09-25: Threads per block: `NUM_WARPS * WARP_SIZE` in the kernel. 2026-10-09: 16 × 32: the
/// decode is latency-bound on each warp's serial walk of its selection slice, so twice the warps
/// halve the walk.
const DECODE_BLOCK: u32 = 512;

/// 2026-10-09: Module of the head-batched decode (`glm5next_dsa_mla_decode_hb.cu`): one CTA per
/// (split, row, head group) loads each selected latent token once for all its heads.
pub const DSA_DECODE_HB_MODULE: &str = "glm5next_dsa_mla_decode_hb";

/// 2026-10-09: Rows per head-batched launch. `decode_attention` walks wider row sets in launches
/// of this many rows, so the split workspace is bounded by it, not by the workspace's
/// `max_rows`. A row's output does not depend on the launch it is in.
pub const HB_LAUNCH_ROWS: usize = 16;

/// 2026-10-09: The widest `METRALE_GLM_DSA_DECODE_HB` split count accepted.
pub const HB_MAX_SPLITS: usize = 64;

/// 2026-10-09: Heads per CTA of the head-batched entry; grid z walks groups of this many.
const HB_HEADS: usize = 24;
const HB_BLOCK: u32 = 256;
const HB_MERGE_BLOCK: u32 = 128;

/// 2026-10-09: `METRALE_GLM_DSA_DECODE_HB=N` (1..=[`HB_MAX_SPLITS`]): decode the DSA attention
/// on the head-batched kernel, each row's selection split into `N` slices merged afterwards.
/// Unset keeps the per-head kernel. Opt-in: it changes the summation order (accuracy-gated).
/// Read once; any other value is an error at resolve time.
pub fn hb_splits_lever() -> Result<Option<usize>> {
    static V: std::sync::OnceLock<Result<Option<usize>, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| parse_hb_splits(std::env::var("METRALE_GLM_DSA_DECODE_HB").ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 2026-10-09: [`hb_splits_lever`]'s parse, pure.
fn parse_hb_splits(v: Option<&str>) -> Result<Option<usize>, String> {
    let Some(v) = v else { return Ok(None) };
    match v.trim().parse::<usize>() {
        Ok(n) if (1..=HB_MAX_SPLITS).contains(&n) => Ok(Some(n)),
        _ => Err(format!(
            "METRALE_GLM_DSA_DECODE_HB={v:?}: expected a split count in 1..={HB_MAX_SPLITS}"
        )),
    }
}

/// 2026-10-09: The head-batched entry points and the split count they run with.
#[derive(Clone, Copy)]
struct HeadBatched {
    decode: KernelHandle,
    merge: KernelHandle,
    splits: usize,
}

/// 2026-09-25: The selected-index MLA decode entry point. 2026-10-09: and, under
/// [`hb_splits_lever`], the head-batched one, which `decode_attention` then launches instead.
#[derive(Clone, Copy)]
pub struct Glm5NextDsaDecodeKernel {
    per_head: KernelHandle,
    head_batched: Option<HeadBatched>,
}

impl Glm5NextDsaDecodeKernel {
    /// 2026-09-25: Resolved with `kernel()`, not `try_kernel`: a missing entry
    /// point is an error, with no dense fallback. 2026-10-09: with the head-batched kernel
    /// when [`hb_splits_lever`] names a split count.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Self::resolve_with(gpu, hb_splits_lever()?)
    }

    /// 2026-10-09: [`Self::resolve`] with the split count given (`None`: per-head kernel).
    pub fn resolve_with(gpu: &dyn GpuBackend, hb_splits: Option<usize>) -> Result<Self> {
        let per_head = gpu.kernel(DSA_DECODE_MODULE, "glm5next_dsa_mla_decode_fp8")?;
        let head_batched = match hb_splits {
            None => None,
            Some(splits) => {
                if !(1..=HB_MAX_SPLITS).contains(&splits) {
                    bail!("DSA head-batched decode: {splits} splits, expected 1..={HB_MAX_SPLITS}");
                }
                Some(HeadBatched {
                    decode: gpu.kernel(DSA_DECODE_HB_MODULE, "glm5next_dsa_mla_decode_hb_fp8")?,
                    merge: gpu.kernel(DSA_DECODE_HB_MODULE, "glm5next_dsa_mla_merge")?,
                    splits,
                })
            }
        };
        Ok(Self {
            per_head,
            head_batched,
        })
    }

    /// 2026-10-09: The head-batched split count, `None` on the per-head kernel.
    pub fn hb_splits(&self) -> Option<usize> {
        self.head_batched.map(|h| h.splits)
    }
}

/// 2026-10-09: The head-batched decode's partials: `o` `[rows, splits, heads, kv_lora]` F32 and
/// `ml` `[rows, splits, heads, 2]` F32 (running max, sum), for at most `rows` rows per launch.
#[derive(Debug, Clone, Copy)]
pub struct DsaSplitWorkspace {
    pub o: DevicePtr,
    pub ml: DevicePtr,
    pub rows: usize,
    pub splits: usize,
    pub heads: usize,
}

impl DsaSplitWorkspace {
    /// 2026-10-09: Bytes of `(o, ml)` for `rows` rows, `splits` splits and `heads` heads.
    pub fn bytes(rows: usize, splits: usize, heads: usize, kv_lora: usize) -> (usize, usize) {
        (
            rows * splits * heads * kv_lora * 4,
            rows * splits * heads * 2 * 4,
        )
    }

    /// 2026-10-09: The partials `kernel` needs for up to `max_rows` rows of `cfg`: `None` on the
    /// per-head kernel and at one split (it writes BF16 directly), else allocated for
    /// `min(max_rows, HB_LAUNCH_ROWS)` rows.
    pub fn alloc_for(
        gpu: &dyn GpuBackend,
        cfg: &Glm5NextDsaConfig,
        splits: Option<usize>,
        max_rows: usize,
    ) -> Result<Option<Self>> {
        let Some(splits) = splits.filter(|s| *s > 1) else {
            return Ok(None);
        };
        let rows = max_rows.clamp(1, HB_LAUNCH_ROWS);
        let (o, ml) = Self::bytes(rows, splits, cfg.local_heads, cfg.kv_lora_rank);
        Ok(Some(Self {
            o: gpu.alloc(o)?,
            ml: gpu.alloc(ml)?,
            rows,
            splits,
            heads: cfg.local_heads,
        }))
    }
}

/// 2026-09-25: Everything the decode reads, all caller-owned.
#[derive(Debug, Clone, Copy)]
pub struct DsaDecodeInputs {
    /// 2026-09-25: `[num_q_heads * kv_lora_rank]` BF16: this rank's absorbed
    /// queries.
    pub q: DevicePtr,
    /// 2026-09-25: FP8 paged latent cache. In absorbed NoPE MLA, K and V are the
    /// same buffer; both are taken so a caller may pass them separately.
    pub k_cache: DevicePtr,
    pub v_cache: DevicePtr,
    /// 2026-09-25: `[num_q_heads * kv_lora_rank]` BF16 output.
    pub out: DevicePtr,
    /// 2026-09-25: `[num_seqs, max_blocks_per_seq]` i32.
    pub block_tables: DevicePtr,
    /// 2026-09-25: `[num_seqs]` i32.
    pub seq_lens: DevicePtr,
    /// 2026-09-25: `[num_seqs, out_width]` i32:
    /// [`super::select::DsaSelectScratch::tokens`].
    pub sel_indices: DevicePtr,
    pub k_scale: f32,
    pub v_scale: f32,
    /// 2026-10-09: The head-batched decode's partials; required by it from two splits up,
    /// unused by the per-head kernel.
    pub split_ws: Option<DsaSplitWorkspace>,
}

/// 2026-09-25: Paging geometry the decode needs and the selection does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsaDecodePaging {
    pub num_seqs: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub max_blocks_per_seq: usize,
    pub block_size: usize,
    pub cache_stride_bytes: u64,
}

impl DsaDecodePaging {
    /// 2026-09-25: Refuses zero sequences, heads or `block_size`, a `block_size`
    /// that is not a multiple of `index_kpool`, and any `num_kv_heads` but 1.
    ///
    /// The `index_kpool` rule is the selector's, not this kernel's: pools are
    /// built over absolute positions, so a block that straddles a pool boundary
    /// splits a pool across two pages. The per-token gather would not notice.
    pub fn validate(&self, cfg: &Glm5NextDsaConfig) -> Result<()> {
        if self.num_seqs == 0 || self.num_q_heads == 0 {
            bail!(
                "DSA decode: degenerate launch ({} seqs, {} heads)",
                self.num_seqs,
                self.num_q_heads
            );
        }
        if self.block_size == 0 {
            bail!("DSA decode: block_size must be > 0");
        }
        if !self.block_size.is_multiple_of(cfg.index_kpool) {
            bail!(
                "DSA decode: block_size {} is not a multiple of index_kpool {} — a pool \
                 would straddle a page boundary",
                self.block_size,
                cfg.index_kpool
            );
        }
        if self.num_kv_heads != 1 {
            bail!(
                "DSA decode: MLA carries a single latent KV head, got {}",
                self.num_kv_heads
            );
        }
        Ok(())
    }
}

/// 2026-09-25: Launch the selected-index decode. Enqueued on `stream`, not
/// synchronised.
///
/// `geom.q_rows` must equal `paging.num_seqs`: one query row per sequence. A
/// mismatch would index the selection rows with the wrong stride, so it is an
/// error.
pub fn decode_attention(
    gpu: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    stream: u64,
) -> Result<()> {
    paging.validate(cfg)?;
    if geom.q_rows != paging.num_seqs {
        bail!(
            "DSA decode: selection has {} query rows but {} sequences are being decoded; \
             the selection row stride would be wrong",
            geom.q_rows,
            paging.num_seqs
        );
    }

    // 2026-09-25: The kernel tiles 512 latent dims across 32 lanes at 16 each.
    // `Glm5NextDsaConfig::validate` checks the same bound; it is repeated here
    // because the kernel itself would not notice a mismatch.
    if cfg.kv_lora_rank != super::KERNEL_KV_LORA_DIM {
        bail!(
            "DSA decode: kv_lora_rank {} != kernel tiling {}",
            cfg.kv_lora_rank,
            super::KERNEL_KV_LORA_DIM
        );
    }

    if let Some(hb) = kernel.head_batched {
        decode_head_batched(gpu, hb, cfg, geom, paging, inputs, stream)?;
        if let Some(dir) = check::check_dir()?
            && !gpu.stream_is_capturing(stream)
        {
            check::against_per_head(gpu, kernel.per_head, cfg, geom, paging, inputs, stream, dir)?;
        }
        return Ok(());
    }
    launch_per_head(gpu, kernel.per_head, cfg, geom, paging, inputs, stream)
}

/// 2026-10-09: The per-head kernel's launch (one CTA per head and row).
fn launch_per_head(
    gpu: &dyn GpuBackend,
    per_head: KernelHandle,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, per_head)
        .grid([paging.num_q_heads as u32, paging.num_seqs as u32, 1])
        .block([DECODE_BLOCK, 1, 1])
        .arg_ptr(inputs.q)
        .arg_ptr(inputs.k_cache)
        .arg_ptr(inputs.v_cache)
        .arg_ptr(inputs.out)
        .arg_ptr(inputs.block_tables)
        .arg_ptr(inputs.seq_lens)
        .arg_ptr(inputs.sel_indices)
        .arg_u32(geom.out_width as u32)
        .arg_u32(paging.max_blocks_per_seq as u32)
        .arg_u32(paging.num_q_heads as u32)
        .arg_u32(paging.num_kv_heads as u32)
        .arg_u32(cfg.kv_lora_rank as u32)
        .arg_u32(paging.block_size as u32)
        // 2026-09-25: NoPE: the score scale is over the latent width, which is the
        // whole cache token.
        .arg_f32((cfg.kv_lora_rank as f32).powf(-0.5))
        .arg_f32(inputs.k_scale)
        .arg_f32(inputs.v_scale)
        .arg_u64(paging.cache_stride_bytes)
        .launch(stream)?;
    Ok(())
}

/// 2026-10-09: The head-batched decode of `decode_attention`, in launches of at most
/// [`HB_LAUNCH_ROWS`] rows; from two splits up each launch is followed by the merge.
fn decode_head_batched(
    gpu: &dyn GpuBackend,
    hb: HeadBatched,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    stream: u64,
) -> Result<()> {
    // 2026-10-09: The kernel reads one latent buffer as K and V; distinct ones would need the
    // per-head kernel's second load.
    if inputs.k_cache != inputs.v_cache {
        bail!("DSA head-batched decode: K and V must be one latent buffer (absorbed MLA)");
    }
    let heads = paging.num_q_heads;
    let ws = match (hb.splits, inputs.split_ws) {
        (1, _) => None,
        (_, Some(ws)) if ws.splits == hb.splits && ws.heads == heads => Some(ws),
        (s, ws) => bail!(
            "DSA head-batched decode: {s} splits over {heads} heads needs a split workspace of \
             that shape, got {ws:?}"
        ),
    };
    let launch_rows = ws.map_or(HB_LAUNCH_ROWS, |w| w.rows.min(HB_LAUNCH_ROWS));
    let (ws_o, ws_ml) = ws.map_or((DevicePtr(0), DevicePtr(0)), |w| (w.o, w.ml));
    let latent_row = heads * cfg.kv_lora_rank * 2;
    let mut row0 = 0;
    while row0 < paging.num_seqs {
        let rows = launch_rows.min(paging.num_seqs - row0);
        let q = inputs.q.offset(row0 * latent_row);
        let out = inputs.out.offset(row0 * latent_row);
        let bt = inputs
            .block_tables
            .offset(row0 * paging.max_blocks_per_seq * 4);
        let sl = inputs.seq_lens.offset(row0 * 4);
        KernelLaunch::new(gpu, hb.decode)
            .grid([
                hb.splits as u32,
                rows as u32,
                heads.div_ceil(HB_HEADS) as u32,
            ])
            .block([HB_BLOCK, 1, 1])
            .arg_ptr(q)
            .arg_ptr(inputs.k_cache)
            .arg_ptr(out)
            .arg_ptr(bt)
            .arg_ptr(sl)
            .arg_ptr(inputs.sel_indices.offset(row0 * geom.out_width * 4))
            .arg_ptr(ws_o)
            .arg_ptr(ws_ml)
            .arg_u32(geom.out_width as u32)
            .arg_u32(paging.max_blocks_per_seq as u32)
            .arg_u32(heads as u32)
            .arg_u32(paging.block_size as u32)
            .arg_f32((cfg.kv_lora_rank as f32).powf(-0.5))
            .arg_f32(inputs.k_scale)
            .arg_f32(inputs.v_scale)
            .arg_u64(paging.cache_stride_bytes)
            .arg_u32(hb.splits as u32)
            .launch(stream)?;
        if hb.splits > 1 {
            KernelLaunch::new(gpu, hb.merge)
                .grid([heads as u32, rows as u32, 1])
                .block([HB_MERGE_BLOCK, 1, 1])
                .arg_ptr(ws_o)
                .arg_ptr(ws_ml)
                .arg_ptr(sl)
                .arg_ptr(out)
                .arg_u32(heads as u32)
                .arg_u32(hb.splits as u32)
                .arg_f32(inputs.v_scale)
                .launch(stream)?;
        }
        row0 += rows;
    }
    Ok(())
}

mod check;
#[cfg(test)]
mod tests;
