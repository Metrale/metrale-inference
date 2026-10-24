// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: DFlash2 drafter components: the grouped dynamic two-tap convolutions
//! around the attention and MLP sublayers of the paged layer body, and the
//! candidate-selector tail that replaces per-row argmax with a chain walk over each
//! row's top-16 candidates.
//!
//!   Conv(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t-1}
//!   S_t(a,b)  = U_t(b) + ⟨A(a) ⊙ H(h_t), B(b)⟩
//!
//! (A and B are the predecessor and successor codebooks, H the hidden projection, U_t
//! row t's top-16 logits.)
//!
//! Conv: `conv_prepare` runs one GEMM through `kernel_projection` on the pre-conv
//! hidden, producing the dynamic coefficients for both applications, laid out per row
//! as `[application(2), tap(kernel_size), group]`. The prepare application uses slice
//! 0 at once; slice 1 stays in `scratch.conv_dyn` until `conv_finish` reads it after
//! the sublayer. `conv_dyn` is a fixed allocation, so the prepare and finish launches
//! can sit in different captured subgraphs with the eager attention between them.
//!
//! Selector: lm_head logits → per-row top-16 (overwriting the selected logits) →
//! `hidden_projection` GEMM on the post-norm hidden → one chain-walk launch. Each
//! sequence's walk takes its band's row 0 of `draft_tokens_dev` as the first
//! predecessor and writes the chosen tokens into rows 1.. of the band. On the paged
//! path that row 0 holds the sequence's last token (written by `forward_block` before
//! the layers); the walk leaves it as it is, and propose drops it.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{BlockDiffusionDraftHead, DflashLayer};
use metrale_model_layers::layer::ForwardContext;

/// 2026-10-10: `TOPK16_PART_THREADS` of `dflash2.cu`: the block of both top-16 stages.
const TOPK16_THREADS: u32 = 256;
/// 2026-10-10: Vocabulary entries one `dflash2_topk16_part` block covers
/// (`TOPK16_PART_SPAN`: 256 threads x 8 entries).
const TOPK16_PART_SPAN: usize = 2048;
/// 2026-10-10: Candidate count per part and output width of the top-k stages.
const TOPK16_WIDTH: usize = 16;
/// 2026-10-10: Candidates one `dflash2_topk16_merge` block holds (256 threads x 8).
const TOPK16_MERGE_CAPACITY: usize = 2048;

/// 2026-10-10: The `dflash2_topk16_part` blocks per row for a `vocab`-entry row.
pub(super) fn topk16_parts(vocab: usize) -> usize {
    vocab.div_ceil(TOPK16_PART_SPAN).max(1)
}

/// 2026-10-10: Whether the merge block can hold all candidates emitted by `parts` part blocks.
pub(super) fn topk16_merge_fits_parts(parts: usize) -> bool {
    parts
        .checked_mul(TOPK16_WIDTH)
        .is_some_and(|candidates| candidates <= TOPK16_MERGE_CAPACITY)
}

/// 2026-10-10: Whether a vocab row can use the two-stage top-16 merge.
pub(super) fn topk16_merge_fits_vocab(vocab: usize) -> bool {
    topk16_merge_fits_parts(topk16_parts(vocab))
}

/// 2026-10-10: How `dflash2_topk16_merge` reads its candidates and what it writes: the
/// logits row width `n` and first global entry `base` (for the overwrites), `parts` lists of
/// 16 per row at `row * row_stride + part * part_stride`, and `pad` (the one-block kernel's
/// output) or not (raw lists another merge can take).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Topk16Merge {
    pub n: u32,
    pub parts: u32,
    pub base: u32,
    pub part_stride: u32,
    pub row_stride: u32,
    pub pad: bool,
}

/// 2026-09-25: Which sublayer a conv application wraps.
#[derive(Clone, Copy)]
pub(super) enum ConvSite {
    Attention,
    Mlp,
}

impl BlockDiffusionDraftHead {
    /// 2026-09-25: True when the DFlash2 conv and selector path runs: the checkpoint
    /// ships the selector weights with `selector_top_k == 16`, `selector_rank > 0`,
    /// `conv_kernel_size == 2` and `conv_group_size > 0`; the three DFlash2 kernels
    /// resolved; and `METRALE_DFLASH2` is not `0`. A layer without conv weights
    /// skips its convs even then.
    pub fn dflash2_active(&self) -> bool {
        self.selector_pred.is_some()
            && self.selector_succ.is_some()
            && self.selector_hidden_proj.is_some()
            && self.selector_top_k == 16
            && self.selector_rank > 0
            && self.conv_kernel_size == 2
            && self.conv_group_size > 0
            && self.kernels.dflash2_conv2.0 != 0
            && self.kernels.dflash2_topk16.0 != 0
            && self.kernels.dflash2_selector_walk.0 != 0
            && self.levers.dflash2
    }

    /// 2026-09-25: The site's base kernel and kernel projection. 2026-10-10: and the
    /// projection's quantized copy on the tiers that copy the auxiliary GEMMs (`aux_quant.rs`).
    fn conv_weights(
        &self,
        layer: &DflashLayer,
        site: ConvSite,
    ) -> Option<(DevicePtr, DevicePtr, Option<super::DraftWeight>)> {
        match site {
            ConvSite::Attention => Some((
                layer.attention_conv_base.as_ref()?.weight,
                layer.attention_conv_proj.as_ref()?.weight,
                layer.attention_conv_proj_q,
            )),
            ConvSite::Mlp => Some((
                layer.mlp_conv_base.as_ref()?.weight,
                layer.mlp_conv_proj.as_ref()?.weight,
                layer.mlp_conv_proj_q,
            )),
        }
    }

    /// 2026-09-25: One conv application. `x` and `out` must be distinct buffers
    /// (row t reads x[t-1]). `application` is 0 for prepare and 1 for finish;
    /// `n_seq` is the number of sequences packed seq-major (1 on the single-sequence
    /// path).
    fn conv_apply(
        &self,
        ctx: &ForwardContext,
        x: DevicePtr,
        base: DevicePtr,
        out: DevicePtr,
        application: u32,
        n_seq: u32,
        stream: u64,
    ) -> Result<()> {
        let block_g = self.block_g() as u32;
        let g = block_g * n_seq.max(1);
        let h = self.hidden_size as u32;
        let groups = h / self.conv_group_size as u32;
        let k = self.conv_kernel_size as u32;
        let dyn_stride = 2 * k * groups;
        let app_off = application * k * groups;
        // 2026-09-25: base_kernel is [2, k, h]; each application's slice is [k, h].
        let base_slice =
            base.offset((application as usize) * self.conv_kernel_size * self.hidden_size * 2);
        KernelLaunch::new(ctx.gpu, self.kernels.dflash2_conv2)
            .grid([g, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(self.scratch.conv_dyn)
            .arg_ptr(base_slice)
            .arg_ptr(out)
            .arg_u32(h)
            .arg_u32(self.conv_group_size as u32)
            .arg_u32(dyn_stride)
            .arg_u32(app_off)
            // 2026-09-25: Rows per band, so row 0 of each sequence has no predecessor.
            .arg_u32(block_g)
            .launch(stream)
    }

    /// 2026-09-25: Conv `prepare`: the dynamic-kernel GEMM on `hidden_in`, then the
    /// first conv application. Returns the buffer the sublayer's GEMMs read
    /// (`conv_tmp`), or `hidden_in` unchanged when DFlash2 is inactive or the layer
    /// has no conv weights for `site`. Overwrites `scratch.conv_dyn`, so the matching
    /// [`Self::conv_finish`] must run before the next prepare.
    pub(super) fn conv_prepare(
        &self,
        layer: &DflashLayer,
        site: ConvSite,
        hidden_in: DevicePtr,
        ctx: &ForwardContext,
        n_seq: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        if !self.dflash2_active() {
            return Ok(hidden_in);
        }
        let Some((base, proj, proj_q)) = self.conv_weights(layer, site) else {
            return Ok(hidden_in);
        };
        let g = self.block_g() as u32 * n_seq.max(1);
        let h = self.hidden_size as u32;
        let groups = h / self.conv_group_size as u32;
        let dyn_cols = 2 * self.conv_kernel_size as u32 * groups;
        // 2026-09-25: Dynamic kernels for both applications, from the pre-conv hidden.
        self.aux_gemm(
            ctx.gpu,
            hidden_in,
            &metrale_model_layers::weight_map::DenseWeight { weight: proj },
            proj_q.as_ref(),
            self.scratch.conv_dyn,
            g,
            dyn_cols,
            h,
            stream,
        )?;
        self.conv_apply(
            ctx,
            hidden_in,
            base,
            self.scratch.conv_tmp,
            0,
            n_seq,
            stream,
        )?;
        Ok(self.scratch.conv_tmp)
    }

    /// 2026-09-25: Conv `finish`: the second application on the sublayer output,
    /// using the dynamic slice computed at prepare. Returns the buffer the residual
    /// add should consume (`sublayer_out` unchanged when prepare was a no-op).
    pub(super) fn conv_finish(
        &self,
        layer: &DflashLayer,
        site: ConvSite,
        sublayer_out: DevicePtr,
        ctx: &ForwardContext,
        n_seq: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        if !self.dflash2_active() {
            return Ok(sublayer_out);
        }
        let Some((base, _proj, _)) = self.conv_weights(layer, site) else {
            return Ok(sublayer_out);
        };
        self.conv_apply(
            ctx,
            sublayer_out,
            base,
            self.scratch.conv_tmp,
            1,
            n_seq,
            stream,
        )?;
        Ok(self.scratch.conv_tmp)
    }

    /// 2026-10-10: `dflash2_topk16_part` over `rows` rows of `n` logits at `logits` holding
    /// global entries `[base, base + n)`, into `sel_cand_vals` / `sel_cand_idx`
    /// (`[rows, topk16_parts(n), 16]`).
    pub(super) fn launch_topk16_part(
        &self,
        gpu: &dyn GpuBackend,
        logits: DevicePtr,
        rows: u32,
        n: u32,
        base: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.kernels.dflash2_topk16_part)
            .grid([rows, topk16_parts(n as usize) as u32, 1])
            .block([TOPK16_THREADS, 1, 1])
            .arg_ptr(logits)
            .arg_ptr(self.scratch.sel_cand_vals)
            .arg_ptr(self.scratch.sel_cand_idx)
            .arg_u32(n)
            .arg_u32(base)
            .launch(stream)
    }

    /// 2026-10-10: `dflash2_topk16_merge` of `rows` rows of candidates `cands` (values,
    /// indices) laid out by `m`, into `out` (values, indices; `[rows, 16]` each); the -1e30
    /// overwrites go to `logits` when given (`Topk16Merge::pad` only).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_topk16_merge(
        &self,
        gpu: &dyn GpuBackend,
        logits: Option<DevicePtr>,
        cands: (DevicePtr, DevicePtr),
        out: (DevicePtr, DevicePtr),
        rows: u32,
        m: Topk16Merge,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            topk16_merge_fits_parts(m.parts as usize),
            "DFlash2 top-16 merge: {} parts exceed the one-block capacity of {} parts",
            m.parts,
            TOPK16_MERGE_CAPACITY / TOPK16_WIDTH
        );
        KernelLaunch::new(gpu, self.kernels.dflash2_topk16_merge)
            .grid([rows, 1, 1])
            .block([TOPK16_THREADS, 1, 1])
            .arg_ptr(logits.unwrap_or(DevicePtr::NULL))
            .arg_ptr(cands.0)
            .arg_ptr(cands.1)
            .arg_ptr(out.0)
            .arg_ptr(out.1)
            .arg_u32(m.n)
            .arg_u32(m.parts)
            .arg_u32(m.base)
            .arg_u32(m.part_stride)
            .arg_u32(m.row_stride)
            .arg_u32(u32::from(m.pad))
            .launch(stream)
    }

    /// 2026-10-10: Top-16 of each of `rows` logit rows into `sel_vals` / `sel_idx`, the
    /// selected logits overwritten: the two-stage kernels when they resolved, the lever
    /// is on and the vocabulary fits the merge ([`topk16_parts`]), else `dflash2_topk16`.
    /// Both write the same bytes (`kernels/gb10/common/dflash2.cu`).
    fn topk16(&self, gpu: &dyn GpuBackend, rows: u32, vocab: u32, stream: u64) -> Result<()> {
        let parts = topk16_parts(vocab as usize) as u32;
        if self.levers.topk16_split
            && self.kernels.dflash2_topk16_part.0 != 0
            && self.kernels.dflash2_topk16_merge.0 != 0
            && topk16_merge_fits_parts(parts as usize)
        {
            self.launch_topk16_part(gpu, self.scratch.logits, rows, vocab, 0, stream)?;
            return self.launch_topk16_merge(
                gpu,
                Some(self.scratch.logits),
                (self.scratch.sel_cand_vals, self.scratch.sel_cand_idx),
                (self.scratch.sel_vals, self.scratch.sel_idx),
                rows,
                Topk16Merge {
                    n: vocab,
                    parts,
                    base: 0,
                    part_stride: 16,
                    row_stride: parts * 16,
                    pad: true,
                },
                stream,
            );
        }
        KernelLaunch::new(gpu, self.kernels.dflash2_topk16)
            .grid([rows, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(self.scratch.logits)
            .arg_ptr(self.scratch.sel_vals)
            .arg_ptr(self.scratch.sel_idx)
            .arg_u32(vocab)
            .launch(stream)
    }

    /// 2026-09-25: Selector tail over `n_seq` bands: top-16 per row (overwriting the
    /// selected entries of `scratch.logits`), the H(h_t) projection, then one
    /// chain-walk launch writing rows 1.. of each band of `draft_tokens_dev`. Row 0 of
    /// each band is not written.
    pub(super) fn dflash2_select_block(
        &self,
        ctx: &ForwardContext,
        norm_noise: DevicePtr,
        n_seq: u32,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let n = n_seq.max(1);
        let g = self.block_g() as u32;
        let rows = g * n;
        let vocab = self.vocab_size as u32;
        let rank = self.selector_rank as u32;
        let hproj = self
            .selector_hidden_proj
            .as_ref()
            .expect("dflash2_active() checked selector_hidden_proj");
        let pred = self
            .selector_pred
            .as_ref()
            .expect("dflash2_active() checked selector_pred");
        let succ = self
            .selector_succ
            .as_ref()
            .expect("dflash2_active() checked selector_succ");

        // 2026-09-25: Order: top-16 (writes sel_vals and sel_idx, leaves
        // draft_tokens_dev alone), the H(h_t) GEMM, then the walk.
        // 2026-10-10: Under tensor parallelism the rows hold this rank's vocab slice: each
        // rank's lists are gathered and merged (`tp_propose.rs`).
        if self.tp.is_some() {
            self.tp_topk16(ctx, rows, stream)?;
        } else {
            self.topk16(gpu, rows, vocab, stream)?;
        }

        // 2026-09-25: H(h_t): [rows, hidden] → [rows, rank].
        self.aux_gemm(
            gpu,
            norm_noise,
            hproj,
            self.aux_q
                .as_ref()
                .and_then(|a| a.selector_hidden_proj.as_ref()),
            self.scratch.sel_hproj,
            rows,
            rank,
            self.hidden_size as u32,
            stream,
        )?;

        // 2026-09-25: Chain walk from row 1 of each band (`first_row` = 1).
        KernelLaunch::new(gpu, self.kernels.dflash2_selector_walk)
            // 2026-09-25: One block per sequence; each starts from its own row 0.
            .grid([n, 1, 1])
            .block([512, 1, 1])
            .arg_ptr(self.scratch.sel_vals)
            .arg_ptr(self.scratch.sel_idx)
            .arg_ptr(self.scratch.sel_hproj)
            .arg_ptr(pred.weight)
            .arg_ptr(succ.weight)
            .arg_ptr(self.scratch.draft_tokens_dev)
            .arg_u32(g)
            .arg_u32(rank)
            .arg_u32(1)
            .launch(stream)?;

        // 2026-09-25: Row 0 of each band is not rewritten: propose (propose.rs) and
        // the batched split (proposer.rs) drop row 0 whenever `mask_token_id != 0`.
        Ok(())
    }
}
