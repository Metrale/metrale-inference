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

/// 2026-10-14: Module of the pipelined prefill twin of the P-hi entry
/// (`glm5next_dsa_mla_prefill_pipe.cu`).
pub const DSA_PREFILL_PIPE_MODULE: &str = "glm5next_dsa_mla_prefill_pipe";

/// 2026-10-14: Module of the warp-specialized prefill twin of the P-hi entry
/// (`glm5next_dsa_mla_prefill_tc.cu`).
pub const DSA_PREFILL_TC_MODULE: &str = "glm5next_dsa_mla_prefill_tc";

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

mod levers;
pub use levers::{
    hb_prefill_one_split, hb_prefill_p_hi, hb_prefill_pipe, hb_prefill_tc, hb_splits_lever,
};
mod prefill_tc;
pub use prefill_tc::{TC_BLOCK, TC_MAX_WIDTH, TC_SMEM_BYTES};

/// 2026-10-14: Dynamic shared memory of `glm5next_dsa_mla_prefill_pipe_phi`: two bf16 tiles
/// (32 tokens x 512 dims) and four f32 QK chain planes (24 heads x 36), its `PP_SMEM_BYTES`.
/// The kernel traps when launched with less, so a drift between the two fails loudly.
pub const PIPE_SMEM_BYTES: u32 = 2 * (32 * 512 * 2) + 4 * (24 * 36 * 4);

/// 2026-10-09: The head-batched entry points and the split count they run with.
#[derive(Clone, Copy)]
struct HeadBatched {
    decode: KernelHandle,
    /// 2026-10-12: `glm5next_dsa_mla_decode_hb_fp8_phi` (P as bf16 hi only), or 0 when the
    /// target lacks it; [`Glm5NextDsaDecodeKernel::for_prefill_p_hi`] swaps it in.
    decode_p_hi: KernelHandle,
    /// 2026-10-14: `glm5next_dsa_mla_prefill_pipe_phi`, or 0 when the target lacks it;
    /// [`Glm5NextDsaDecodeKernel::for_prefill_pipe`] swaps it in.
    prefill_pipe: KernelHandle,
    /// 2026-10-14: `glm5next_dsa_mla_prefill_tc_phi`, or 0 when the target lacks it;
    /// [`Glm5NextDsaDecodeKernel::for_prefill_tc`] swaps it in.
    prefill_tc: KernelHandle,
    merge: KernelHandle,
    splits: usize,
    /// 2026-10-14: Dynamic shared memory `decode` is launched with (0 but for the pipe entry).
    smem: u32,
    /// 2026-10-14: Threads per block of `decode` ([`HB_BLOCK`] but for the tc entry).
    block: u32,
    /// 2026-10-14: `decode` gathers through a [`TC_MAX_WIDTH`]-slot table of 16-byte cache
    /// offsets (the tc entry), so the launcher refuses a wider row or an unaligned stride.
    offset_table: bool,
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
                tracing::warn!(
                    "METRALE_GLM_DSA_DECODE_HB={splits}: DSA decode on the head-batched kernel"
                );
                Some(HeadBatched {
                    decode: gpu.kernel(DSA_DECODE_HB_MODULE, "glm5next_dsa_mla_decode_hb_fp8")?,
                    decode_p_hi: metrale_model_layers::layers::try_kernel(
                        gpu,
                        DSA_DECODE_HB_MODULE,
                        "glm5next_dsa_mla_decode_hb_fp8_phi",
                    ),
                    prefill_pipe: metrale_model_layers::layers::try_kernel(
                        gpu,
                        DSA_PREFILL_PIPE_MODULE,
                        "glm5next_dsa_mla_prefill_pipe_phi",
                    ),
                    prefill_tc: metrale_model_layers::layers::try_kernel(
                        gpu,
                        DSA_PREFILL_TC_MODULE,
                        "glm5next_dsa_mla_prefill_tc_phi",
                    ),
                    merge: gpu.kernel(DSA_DECODE_HB_MODULE, "glm5next_dsa_mla_merge")?,
                    splits,
                    smem: 0,
                    block: HB_BLOCK,
                    offset_table: false,
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

    /// 2026-10-12: This kernel with the head-batched entry replaced by its P-hi twin when
    /// `p_hi` ([`hb_prefill_p_hi`]) is set; unchanged when it is not. An error when it is set
    /// without the head-batched kernel (`METRALE_GLM_DSA_DECODE_HB`) or without the twin in the
    /// build, never a silent fallback.
    pub fn for_prefill_p_hi(self, p_hi: bool) -> Result<Self> {
        if !p_hi {
            return Ok(self);
        }
        let Some(h) = self.head_batched else {
            bail!("METRALE_GLM_DSA_HB_PREFILL_P_HI=1 needs METRALE_GLM_DSA_DECODE_HB");
        };
        if h.decode_p_hi.0 == 0 {
            bail!(
                "METRALE_GLM_DSA_HB_PREFILL_P_HI=1: glm5next_dsa_mla_decode_hb_fp8_phi is not in \
                 this target's kernels"
            );
        }
        Ok(Self {
            head_batched: Some(HeadBatched {
                decode: h.decode_p_hi,
                ..h
            }),
            ..self
        })
    }

    /// 2026-10-14: This kernel with the P-hi entry replaced by its pipelined twin when `pipe`
    /// ([`hb_prefill_pipe`]) is set; unchanged when it is not. An error when it is set but the
    /// head-batched entry is not the P-hi one ([`Self::for_prefill_p_hi`] first), or the twin is
    /// not in the build.
    pub fn for_prefill_pipe(self, pipe: bool) -> Result<Self> {
        if !pipe {
            return Ok(self);
        }
        let Some(h) = self.head_batched else {
            bail!("METRALE_GLM_DSA_HB_PREFILL_PIPE=1 needs METRALE_GLM_DSA_DECODE_HB");
        };
        if h.decode_p_hi.0 == 0 || h.decode.0 != h.decode_p_hi.0 {
            bail!("METRALE_GLM_DSA_HB_PREFILL_PIPE=1 needs METRALE_GLM_DSA_HB_PREFILL_P_HI=1");
        }
        if h.prefill_pipe.0 == 0 {
            bail!(
                "METRALE_GLM_DSA_HB_PREFILL_PIPE=1: glm5next_dsa_mla_prefill_pipe_phi is not in \
                 this target's kernels"
            );
        }
        Ok(Self {
            head_batched: Some(HeadBatched {
                decode: h.prefill_pipe,
                smem: PIPE_SMEM_BYTES,
                ..h
            }),
            ..self
        })
    }

    /// 2026-10-14: This kernel for a prefill sub-chunk's attend under the prefill levers, in
    /// order: [`hb_prefill_one_split`], [`hb_prefill_p_hi`], [`hb_prefill_pipe`],
    /// [`hb_prefill_tc`].
    /// Each process logs the resolved levers once ([`levers::readback_prefill_levers`]).
    pub fn for_prefill_levers(self) -> Result<Self> {
        let (one, p_hi) = (hb_prefill_one_split(), hb_prefill_p_hi()?);
        let (pipe, tc) = (hb_prefill_pipe()?, hb_prefill_tc()?);
        let k = self
            .for_prefill(one)
            .for_prefill_p_hi(p_hi)?
            .for_prefill_pipe(pipe)?
            .for_prefill_tc(tc)?;
        levers::readback_prefill_levers(one, p_hi, pipe, tc);
        Ok(k)
    }

    /// 2026-10-12: This kernel for a prefill sub-chunk's attend: the head-batched one at one
    /// split when `one_split` ([`hb_prefill_one_split`]) is set, else unchanged.
    pub fn for_prefill(self, one_split: bool) -> Self {
        Self {
            head_batched: self.head_batched.map(|h| {
                if one_split {
                    HeadBatched { splits: 1, ..h }
                } else {
                    h
                }
            }),
            ..self
        }
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
        if let Some(dir) = check::check_dir()? {
            if gpu.stream_is_capturing(stream) {
                check::note_captured_skip(paging.num_seqs);
            } else {
                check::against_per_head(
                    gpu,
                    kernel.per_head,
                    cfg,
                    geom,
                    paging,
                    inputs,
                    stream,
                    dir,
                )?;
            }
        }
        if check::serve_reference()? {
            launch_per_head(gpu, kernel.per_head, cfg, geom, paging, inputs, stream)?;
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

mod check;
mod head_batched;
use head_batched::decode_head_batched;
#[cfg(test)]
mod tests;
