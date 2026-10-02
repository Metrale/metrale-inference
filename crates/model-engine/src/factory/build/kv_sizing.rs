// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Step 5 of `build_model`: the slot count and the KV pool, sized together. The KV
//! budget is the util budget less what is already placed, the pre-load reserve at the slot count
//! (`SlotPlan::reserve_for`), the DFlash reserve and the MTP propose pool. `--max-batch-size N`
//! sizes for N; `auto` takes the count `slots::balance_slots` chooses. Split from `build.rs`.
//!
//! Owner: metrale-model-engine.
//! Invariants:
//! - The pool is sized once, at the resolved slot count, by the same arithmetic the balance
//!   evaluated for that count.

use anyhow::{Result, ensure};
use metrale_cache::kv_cache::KvCacheConfig;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::layers::MtpQuantization;
use metrale_model_layers::weight_map::MtpWeights;
use metrale_model_weights::weights::WeightStore;
use metrale_telemetry::prefix_cache::PrefixCache;

use super::slots::{AUTO_MAX_SLOTS, SlotPlan, SlotRequest, balance_slots};
use super::{kv_blocks, kv_budget};

/// 2026-10-01: What sizing the slots and the KV pool reads.
pub(super) struct KvInputs<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub store: &'a WeightStore,
    pub build_entry_free: Option<usize>,
    pub config: &'a ModelConfig,
    pub kv_config: &'a KvCacheConfig,
    pub prefix_cache: &'a dyn PrefixCache,
    pub dflash_reserve: usize,
    pub use_speculative: bool,
    pub mtp_weights: &'a [MtpWeights],
    pub effective_mtp_quant: MtpQuantization,
    pub max_seq_len: usize,
    pub kv_block_size: usize,
    pub hss_cache_blocks_per_seq: Option<u32>,
    pub gpu_memory_utilization: f64,
}

/// 2026-10-01: The resolved slot count and the pool's block count.
pub(super) struct KvSized {
    pub slots: usize,
    pub num_kv_blocks: usize,
}

/// 2026-10-01: Resolve the slot count and size the KV pool (logged as before, at the resolved
/// count).
pub(super) fn size_kv(inp: &KvInputs<'_>, plan: &SlotPlan<'_>) -> Result<KvSized> {
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let total_mem = inp.gpu.total_memory()?;
    let actual_free = inp.gpu.free_memory()?;
    let used_so_far = kv_budget::self_relative_used(
        inp.gpu,
        inp.store,
        inp.build_entry_free,
        actual_free,
        total_mem.saturating_sub(actual_free),
        gib,
    );
    let total_budget = (total_mem as f64 * inp.gpu_memory_utilization) as usize;
    // 2026-10-01: The KV budget at a slot count: what `build_model` computed before, with the
    // reserve evaluated at that count.
    let budget_at = |slots: usize| -> Result<(usize, usize, usize)> {
        let reserve = (plan.reserve_for)(slots)?;
        let kv_budget = total_budget
            .saturating_sub(used_so_far)
            .saturating_sub(reserve)
            .saturating_sub(inp.dflash_reserve)
            .min(
                actual_free
                    .saturating_sub(reserve)
                    .saturating_sub(inp.dflash_reserve),
            );
        let mtp_pool = kv_budget::mtp_pool_reserve_bytes(
            inp.use_speculative,
            inp.mtp_weights,
            inp.effective_mtp_quant,
            inp.config,
            inp.kv_config,
            kv_budget,
            inp.max_seq_len,
        );
        Ok((kv_budget.saturating_sub(mtp_pool), reserve, mtp_pool))
    };
    let blocks_at = |slots: usize| -> Result<usize> {
        let (kv_budget, _, _) = budget_at(slots)?;
        Ok(kv_blocks::pool_blocks(
            inp.kv_config,
            kv_budget,
            inp.prefix_cache.is_active(),
            inp.max_seq_len,
            inp.kv_block_size,
            slots,
        )?
        .0)
    };
    // 2026-10-01: A full-length sequence's blocks as the scheduler's KV admission reserves them
    // (`max_seq_len / block_size + 1`), at least the boot check's `full_length_blocks`, so a
    // count `auto` chooses passes both.
    let full_length = inp.max_seq_len / inp.kv_block_size.max(1) + 1;
    let slots = match plan.request {
        SlotRequest::Count(n) => n,
        SlotRequest::Auto => {
            ensure!(
                inp.hss_cache_blocks_per_seq.is_none(),
                "--max-batch-size auto balances slots against a budget-sized KV pool; \
                 --high-speed-swap sizes the pool from its per-sequence cap. Pass a count."
            );
            let b = balance_slots(AUTO_MAX_SLOTS, full_length, blocks_at)?;
            tracing::info!(target: "metrale_model_engine::factory::build", "--max-batch-size auto: {} slot(s) admit {} sequence(s) at the full \
                 --max-seq-len={} ({} KV blocks, {} block(s)/seq); {} slots would admit {}",
                b.slots,
                b.admitted,
                inp.max_seq_len,
                b.kv_blocks,
                full_length,
                AUTO_MAX_SLOTS,
                blocks_at(AUTO_MAX_SLOTS).map_or(0, |k| AUTO_MAX_SLOTS.min(k / full_length)),
            );
            b.slots
        }
    };
    let (kv_budget, reserve, mtp_pool) = budget_at(slots)?;
    if mtp_pool > 0 {
        tracing::info!(target: "metrale_model_engine::factory::build", "KV budget: reserving {:.1} GB for the MTP propose pool (post-sizing alloc in MtpHead::new)",
            gib(mtp_pool),
        );
    }
    let num_kv_blocks = match inp.hss_cache_blocks_per_seq {
        Some(cap) => kv_blocks::hss_kv_blocks(cap, inp.max_seq_len, inp.kv_block_size, slots),
        None => kv_blocks::budget_kv_blocks(
            inp.kv_config,
            kv_budget,
            inp.prefix_cache,
            inp.max_seq_len,
            inp.kv_block_size,
            slots,
            total_mem,
            inp.gpu_memory_utilization,
            total_budget,
            used_so_far,
            reserve,
        )?,
    };
    // 2026-10-01: The fix the overcommit warning names, only when the pool cannot admit the
    // requested count at full length.
    let balanced = (plan.request != SlotRequest::Auto
        && inp.hss_cache_blocks_per_seq.is_none()
        && num_kv_blocks / kv_blocks::full_length_blocks(inp.max_seq_len, inp.kv_block_size)
            < slots)
        .then(|| balance_slots(AUTO_MAX_SLOTS, full_length, blocks_at).ok())
        .flatten()
        .map(|b| b.slots);
    kv_blocks::check_kv_concurrency(
        num_kv_blocks,
        inp.hss_cache_blocks_per_seq,
        inp.max_seq_len,
        inp.kv_block_size,
        slots,
        balanced,
    )?;
    Ok(KvSized {
        slots,
        num_kv_blocks,
    })
}
