// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The PLE layer: ids -> NVMe row gather -> projections -> gate -> dilated
//! conv -> highway add.
//!
//! Owner: model-layers (PLE).
//! Invariants:
//! - `forward` advances the token history only on the path where it hashes
//!   the ids itself; a consumed prestage has already advanced it.
//! - Under CUDA-graph capture, `forward` returns an error instead of taking
//!   the D2H id readback or the un-prestaged gather.
//! - The multi-sequence form is `rows::forward_rows`; the kernel half is the
//!   same four steps (`project`, `gate`, `conv`, `add_highway`).
//!
//! It runs on the layers `ple_layer_ids` names and adds into the
//! `hc_mult`-wide highway before the layer's first hyper-connection site, as
//! `Qwen4ExpTextDecoderLayer.forward` adds `self.ple(...)` before
//! `attn_hyper_connection`.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::ids::PleIdDims;
use crate::layer::ForwardContext;
use crate::layers::ngram_embed::NgramTable;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// 2026-09-25: Per-sequence carry: the dilated conv's state and the token history the
/// id hash needs. Owned by the sequence's [`crate::layer::SsmLayerState`].
pub struct PleSeqState {
    /// 2026-09-25: `[(k-1)*dilation, hc_mult*hidden]` FP32, device.
    conv: DevicePtr,
    /// 2026-09-25: The last `context_len` token ids, EOS-filled at a sequence start.
    history: Vec<u32>,
    /// 2026-09-25: Set by `prestage`: the n-gram table's device VA, recorded when the
    /// step's host work (hash, fault-in, slot upload) has already run before
    /// graph replay or capture. `forward` consumes it and enqueues kernels only.
    prestaged_va: Option<u64>,
    /// 2026-09-25: The last VA `prestage` staged; only `release_seq_state` clears it.
    /// `rearm` restores it when a failed capture attempt re-runs the step
    /// eagerly: the slots are still in `slots_dev` and history has already
    /// advanced, so hashing again would count the token twice.
    last_staged_va: u64,
    /// 2026-10-13: The row of the last stage: `None` for `prestage` (row 0 of one
    /// sequence's step), `Some(i)` for row `i` of `prestage_rows`. Each consumer takes
    /// only its own kind, since the two put this carry's slots at different rows.
    staged_row: Option<usize>,
}

pub struct PleLayer {
    dims: PleIdDims,
    head_dim: usize,
    hidden: usize,
    hc_mult: usize,
    state_len: usize,
    k_size: usize,
    dilation: usize,
    eps: f32,

    key_proj: DenseWeight,
    value_proj: DenseWeight,
    norm_key: DenseWeight,
    norm_query: DenseWeight,
    norm_conv: DenseWeight,
    conv1d: DenseWeight,
    /// 2026-09-25: Behind a mutex because the NVMe cache resolves (and faults, and
    /// evicts) on the forward path, which needs `&mut`, while layers are
    /// invoked through `&self`.
    table: std::sync::Mutex<NgramTable>,

    embed_k: KernelHandle,
    gemm_k: KernelHandle,
    gate_k: KernelHandle,
    conv_k: KernelHandle,
    add_k: KernelHandle,

    /// 2026-09-25: Scratch, sized once for `max_tokens`. 2026-10-13: a longer pass runs
    /// in sub-chunks of `max_tokens` (`forward_with_ids`).
    emb: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gated: DevicePtr,
    gated_normed: DevicePtr,
    out: DevicePtr,
    slots_dev: DevicePtr,
    max_tokens: usize,
}

impl PleLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dims: PleIdDims,
        head_dim: usize,
        hidden: usize,
        hc_mult: usize,
        k_size: usize,
        dilation: usize,
        eps: f32,
        weights: PleWeights,
        table: NgramTable,
        max_tokens: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        dims.validate()?;
        anyhow::ensure!(max_tokens > 0, "PLE: zero-token scratch");
        let heads = dims.ngram_heads();
        anyhow::ensure!(
            heads * head_dim == hidden,
            "PLE: {heads} heads x {head_dim} dims = {} != ple_embed_dim {hidden}. \
             The head slices are CONCATENATED (not summed as LongCat's are), so \
             this product is the embedding width and a mismatch means the \
             geometry is not what we think.",
            heads * head_dim
        );
        let c = hc_mult * hidden;
        let state_len = (k_size - 1) * dilation;
        Ok(Self {
            dims,
            head_dim,
            hidden,
            hc_mult,
            state_len,
            k_size,
            dilation,
            eps,
            key_proj: weights.key_proj,
            value_proj: weights.value_proj,
            norm_key: weights.norm_key,
            norm_query: weights.norm_query,
            norm_conv: weights.norm_conv,
            conv1d: weights.conv1d,
            table: std::sync::Mutex::new(table),
            embed_k: gpu.kernel("embed_from_argmax", "batched_embed")?,
            gemm_k: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            gate_k: gpu.kernel("ple", "ple_gate")?,
            conv_k: gpu.kernel("ple", "ple_conv")?,
            add_k: gpu.kernel("ple", "ple_add_highway")?,
            emb: gpu.alloc(max_tokens * hidden * 2)?,
            key: gpu.alloc(max_tokens * c * 2)?,
            value: gpu.alloc(max_tokens * hidden * 2)?,
            gated: gpu.alloc(max_tokens * c * 4)?,
            gated_normed: gpu.alloc(max_tokens * c * 4)?,
            out: gpu.alloc(max_tokens * c * 4)?,
            slots_dev: gpu.alloc(max_tokens * heads * 4)?,
            max_tokens,
        })
    }

    /// 2026-09-25: Allocate one sequence's PLE carry (conv buffer + empty history).
    /// The conv buffer is not initialised here; the first `forward` or
    /// `prestage` sees the empty history and runs `reset`.
    pub fn new_seq_state(&self, gpu: &dyn GpuBackend) -> Result<PleSeqState> {
        Ok(PleSeqState {
            conv: gpu.alloc(self.state_len * self.hc_mult * self.hidden * 4)?,
            history: Vec::new(),
            prestaged_va: None,
            last_staged_va: 0,
            staged_row: None,
        })
    }

    // 2026-09-25: `reset`, `prestage`, `release_seq_state`, `snapshot_aux` and
    // `restore_aux` are in `aux_state.rs`.

    /// 2026-09-25: Restore the prestaged state after a failed CUDA-graph capture attempt
    /// (the eager re-run calls `forward` again, and the first call consumed
    /// `prestaged_va`).
    pub fn rearm(&self, st: &mut PleSeqState) {
        if st.last_staged_va != 0 {
            st.prestaged_va = Some(st.last_staged_va);
        }
    }

    /// 2026-09-25: One highway row with an explicit id: the multi-seq decode entry
    /// (`ctx.host_token_ids` holds the whole batch; the caller slices). It
    /// never consumes a prestage.
    pub fn forward_row(
        &self,
        st: &mut PleSeqState,
        highway_row: DevicePtr,
        ids: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_with_ids(st, highway_row, 1, false, Some(ids), ctx, stream)
    }

    /// 2026-09-25: Inject into `highway` `[T, hc_mult*hidden]` FP32, in place.
    /// `fresh` starts a new sequence (prefill from position 0).
    pub fn forward(
        &self,
        st: &mut PleSeqState,
        highway: DevicePtr,
        num_tokens: usize,
        fresh: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_with_ids(st, highway, num_tokens, fresh, None, ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_ids(
        &self,
        st: &mut PleSeqState,
        highway: DevicePtr,
        num_tokens: usize,
        fresh: bool,
        ids_override: Option<&[u32]>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let heads = self.dims.ngram_heads();
        let gpu = ctx.gpu;

        // 2026-09-25: The ids are hashed on the host from the token ids. The
        // caller's `ctx.host_token_ids` is preferred over reading the device
        // copy back: the readback is a blocking copy, and it cannot run inside
        // a CUDA-graph capture.
        let tokens: Vec<u32> = if let Some(ov) = ids_override {
            anyhow::ensure!(ov.len() == num_tokens, "PLE: ids_override length");
            ov.to_vec()
        } else if let Some(host) = ctx.host_token_ids {
            anyhow::ensure!(
                host.len() >= num_tokens,
                "PLE: host_token_ids has {} ids for {num_tokens} tokens",
                host.len()
            );
            host[..num_tokens].to_vec()
        } else {
            // 2026-09-25: Fallback for passes that did not pass the host slice.
            // Refused under capture rather than recorded.
            anyhow::ensure!(
                !ctx.graph_capture,
                "PLE: no host_token_ids and a D2H readback is \
                 capture-unsupported; thread the host ids through this pass"
            );
            let tok_dev = ctx.token_ids.ok_or_else(|| {
                anyhow::anyhow!(
                    "PLE needs token ids (host or device); this pass staged \
                     neither"
                )
            })?;
            let mut raw = vec![0u8; num_tokens * 4];
            gpu.copy_d2h(tok_dev, &mut raw)?;
            raw.chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };

        // 2026-09-25: Only a single-token, non-fresh decode without an ids
        // override consumes a prestage. A replayed decode step never runs this
        // forward, so a `Some` seen by a prefill or by `forward_row` is a
        // leftover staged for an earlier token. Consuming it would inject that
        // token's rows and skip the history advance. `.take()` clears it in
        // every case.
        let prestaged = st.prestaged_va.take().filter(|_| {
            num_tokens == 1 && !fresh && ids_override.is_none() && st.staged_row.is_none()
        });
        if fresh || st.history.len() != self.dims.context_len() {
            self.reset(st, gpu, stream)?;
        }

        if let Some(table_va) = prestaged {
            // 2026-09-25: The host half already ran from `decode_prestage`, before
            // graph replay or capture: slots sit in `slots_dev`, history has
            // advanced. Only the capture-safe kernel half remains.
            self.gather_embed(table_va, num_tokens, heads, gpu, stream)?;
            return self.kernel_steps(st.conv, highway, num_tokens, gpu, stream);
        }
        anyhow::ensure!(
            !ctx.graph_capture,
            "PLE: un-prestaged forward inside CUDA graph capture — the \
             pageable slot upload would invalidate the recording (901); \
             the scheduler must call decode_prestage every step"
        );
        // 2026-09-25: history ++ tokens, hashed together, then keep the new
        // tokens' rows: the same slice the reference takes with
        // `[:, -input_ids.shape[1]:]` (`rows::next_ids`, shared with
        // `prestage` and `forward_rows`).
        let flat = rows::next_ids(&self.dims, &mut st.history, &tokens);
        // 2026-10-13: A pass longer than the scratch runs in sub-chunks of `max_tokens`
        // rows, each carrying the conv state to the next as consecutive prefill chunks
        // do; every step is per row or row-independent, so the result is the
        // one-pass result. A faulting row cache may evict a released row on the next
        // resolve, before the previous sub-chunk's gather has read it, so that table
        // waits for the stream between sub-chunks.
        let faulting = self.table_faults_rows()?;
        let c = self.hc_mult * self.hidden;
        for start in (0..num_tokens).step_by(self.max_tokens) {
            let n = self.max_tokens.min(num_tokens - start);
            if start > 0 && faulting {
                gpu.synchronize(stream)?;
            }
            self.gather(
                &flat[start * heads..(start + n) * heads],
                n,
                heads,
                gpu,
                stream,
            )?;
            self.kernel_steps(st.conv, highway.offset(start * c * 4), n, gpu, stream)?;
        }
        Ok(())
    }

    /// 2026-10-13: Whether the n-gram table faults rows in on demand (anything but a
    /// fully resident table).
    fn table_faults_rows(&self) -> Result<bool> {
        let table = self
            .table
            .lock()
            .map_err(|_| anyhow::anyhow!("PLE table mutex poisoned"))?;
        Ok(!matches!(&*table, NgramTable::Bf16(_)))
    }

    /// 2026-09-25: Resolve row ids to cache slots and gather them into `self.emb`.
    ///
    /// `T * ngram_heads` rows of `head_dim` land contiguously, which is the
    /// `[T, ngram_heads * head_dim]` concatenation the projections read, so
    /// the generic `batched_embed` kernel serves.
    fn gather(
        &self,
        ids: &[u64],
        num_tokens: usize,
        heads: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let table_va = self.gather_host(ids, gpu, stream)?;
        self.gather_embed(table_va, num_tokens, heads, gpu, stream)
    }

    /// 2026-09-25: The host half of `gather`: NVMe fault-in + slot upload into the
    /// stable `slots_dev` buffer. The upload reads pageable host memory, so
    /// under CUDA graphs it runs from `prestage`, before replay or capture.
    /// Returns the table's device VA for the kernel half.
    fn gather_host(&self, ids: &[u64], gpu: &dyn GpuBackend, stream: u64) -> Result<u64> {
        let mut table = self
            .table
            .lock()
            .map_err(|_| anyhow::anyhow!("PLE table mutex poisoned"))?;
        let table_va = match &mut *table {
            #[cfg(feature = "cuda")]
            NgramTable::Cached(cache) => {
                // 2026-09-25: The host resolves row -> slot and faults missing rows
                // off NVMe into the pinned, GPU-addressable arena. The gather
                // kernel then reads the arena by slot.
                let mut slots = Vec::with_capacity(ids.len());
                let (h0, m0, _) = cache.stats();
                let t0 = std::time::Instant::now();
                cache.resolve(ids, &mut slots)?;
                // 2026-09-25: Gathers of more than 64 ids (prefill scale) log the
                // hit/miss profile at info; smaller ones (a decode step is
                // `ngram_heads` ids) at debug.
                let (h1, m1, _) = cache.stats();
                let (dh, dm) = (h1 - h0, m1 - m0);
                let us = t0.elapsed().as_micros();
                if ids.len() > 64 {
                    tracing::info!(
                        "PLE gather: {} ids, {dh} hits / {dm} misses, resolve {us}us",
                        ids.len()
                    );
                } else {
                    tracing::debug!(
                        "PLE gather: {} ids, {dh} hits / {dm} misses, resolve {us}us",
                        ids.len()
                    );
                }
                let bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
                gpu.copy_h2d_async(&bytes, self.slots_dev, stream)?;
                let va = cache.table_dev_va()?;
                // 2026-09-25: This releases the pins before `gather_embed` issues
                // the kernel that reads these slots (under CUDA graphs, before
                // the replay). `NgramRowCache::resolve` documents `end_batch` as
                // "once the gather has been issued", the order
                // `ngram_embed/embed.rs` uses. A later `resolve` may evict one of
                // these slots and write its new row from the host, which is not
                // ordered with the stream.
                cache.end_batch();
                DevicePtr(va)
            }
            NgramTable::Bf16(w) => {
                // 2026-09-25: Fully resident table: the slot is the row id, so the
                // ids are uploaded truncated to u32.
                let bytes: Vec<u8> = ids.iter().flat_map(|r| (*r as u32).to_le_bytes()).collect();
                gpu.copy_h2d_async(&bytes, self.slots_dev, stream)?;
                w.weight
            }
            NgramTable::Fp8(_) => anyhow::bail!(
                "PLE: FP8 n-gram tables are not wired. This checkpoint ships BF16 \
                 rows, which are both simpler and more accurate (on LongCat, BF16 \
                 measured 0.0050 error vs FP8's 0.0247)."
            ),
        };
        Ok(table_va.0)
    }

    /// 2026-09-25: The kernel half of `gather`: it reads `slots_dev` and the table
    /// arena, both stable device addresses, so it is graph-capture-safe.
    fn gather_embed(
        &self,
        table_va: u64,
        num_tokens: usize,
        heads: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ops::batched_embed(
            gpu,
            self.embed_k,
            self.slots_dev,
            DevicePtr(table_va),
            self.emb,
            (num_tokens * heads) as u32,
            self.head_dim as u32,
            stream,
        )
        .context("PLE row gather")
    }
}

/// 2026-09-25: The dense weights of one PLE site.
pub struct PleWeights {
    pub key_proj: DenseWeight,
    pub value_proj: DenseWeight,
    pub norm_key: DenseWeight,
    pub norm_query: DenseWeight,
    pub norm_conv: DenseWeight,
    pub conv1d: DenseWeight,
}

// 2026-09-25: A child module (not a sibling), because the aux fns read
// `PleLayer`'s private fields.
#[path = "aux_state.rs"]
mod aux_state;
#[path = "kernel_half.rs"]
mod kernel_half;
#[path = "rows.rs"]
mod rows;
#[cfg(test)]
#[path = "rows_tests.rs"]
mod rows_tests;
