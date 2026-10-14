// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The reserve for device memory the process uses beyond the allocations the other
//! reserve terms plan: the GDN carry stash the model binds after the build (sized by its
//! allocator's own arithmetic), and what the driver itself takes (the CUDA context, loaded
//! modules, the local-memory reservation, CUDA graphs, and its per-allocation bookkeeping), which
//! no allocation ledger sees. Until 2026-10-01 this was one flat `cuda_headroom`: 4 GiB with a
//! speculative proposer, 512 MiB without.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - The carry stash term is `GdnCarrySizes::total`, the bytes `gdn_carry_bind` allocates.
//! - The driver terms are the build class's HARDWARE.toml `[memory]` table, calibrated per
//!   class from measured serves (GB10 below, H100 in kernels/hopper/HARDWARE.toml); a change to
//!   them cites a new measurement.
//!
//! Calibration (GB10, 2026-10-01; a serve's device footprint is the drop of host
//! `MemAvailable` from an idle box, less the allocation ledger and the serve's anonymous host
//! memory; the idle reading itself varies by about 0.9 GB between runs):
//!
//! | serve | ledger | driver use | of which |
//! |---|---|---|---|
//! | Qwen3.8-27B, declared tier, 128 slots, MTP K=4, C128 | 102.8 GB | 2.1-3.1 GB | +0.75 GB kernel slab, +0.15 GB page tables, +0.5 GB vmalloc |
//! | Qwen3.8-27B throughput recipe, worst cases | 103.5 GB | 1.6-2.4 GB | |
//! | Qwen3.6-35B-A3B, 128 slots, MTP K=2, worst cases | 101.4 GB | 2.7-2.8 GB | |
//! | Nemotron-3-Nano, nvfp4, no speculation, 8 slots | 40.5 GB | 2.3-2.9 GB | +0.88 GB kernel slab, 0.43 GB chunk slack |
//! | Nemotron-3-Nano, as above with prefix caching (pool fills the budget) | 102.9 GB | 3.5-3.6 GB | +0.96 GB kernel slab, +0.63 GB vmalloc, 0.43 GB chunk slack |
//!
//! Probed directly: context 250 MiB, the 218 modules of the 27B target 94 MiB, the local-memory
//! reservation 131 MiB (1,864 B/thread x 48 SMs x 1,536 threads, the same for the 27B, the
//! 35B-A3B and Nemotron-3-Nano kernel sets); the rest is the driver's per-allocation bookkeeping,
//! about 2% of the bytes allocated. It does not depend on speculation, so the no-speculation
//! serve needs the same headroom: 512 MiB under-reserved it by about 2.1-2.6 GB whenever its KV
//! pool fills the budget.
//!
//! One part is not proportional to bytes and is not a reserve term: the driver packs
//! allocations under 2 MiB into 2 MiB chunks and keeps a chunk resident while any of them lives.
//! Nemotron-3-Nano's 24,456 small weight allocations leave 429 MiB of that chunk slack (the dense
//! 27B's 2,274 leave 52 MiB). The KV sizing charges it from the allocation ledger's addresses
//! once the weights are placed (`model-engine` `kv_sizing.rs`), so it applies only where it occurs.
//! Less that slack, `driver_fixed_bytes + driver_budget_per_mille` of the util budget covers every
//! measured serve, the largest being the dense default tier's 3.13 GB, with 0.12 GB to spare at
//! util 0.85 on GB10. A KV pool the prefix cache leaves unclamped also keeps a small share of its
//! own budget for the driver (`model-engine` `kv_blocks::PREFIX_POOL_DRIVER_PER_MILLE`), the one
//! term calibrated rather than attributed.

use metrale_circuit::memory::DriverTerms;
use metrale_config::ModelConfig;

/// 2026-10-13: The driver terms of the class this binary's kernels were built for: its own
/// HARDWARE.toml `[memory]` table, or the nearest ancestor's (`metrale_kernels::TARGET_MEMORY_TABLE`),
/// read by the memory model's own parser. Until 2026-10-13 they were constants here, gb10's
/// calibration for every class; an H100 then reserved 0.45 GB more than its driver uses.
pub(crate) fn target_driver_terms() -> anyhow::Result<DriverTerms> {
    let (class, table) = metrale_kernels::TARGET_MEMORY_TABLE.ok_or_else(|| {
        anyhow::anyhow!(
            "this build carries no HARDWARE.toml [memory] table (built without a kernels/ tree): \
             the driver's share of the device cannot be reserved"
        )
    })?;
    Ok(DriverTerms::parse(class, table)?)
}

/// 2026-10-01: The headroom's terms, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RuntimeHeadroom {
    /// 2026-10-01: The GDN carry stash and its tables (`GdnCarrySizes::total`); 0 when the model
    /// cannot carry.
    pub carry_stash: usize,
    /// 2026-10-01: The class's `driver_fixed_bytes`.
    pub driver_fixed: usize,
    /// 2026-10-01: The class's `driver_budget_per_mille` of the util budget.
    pub driver_bookkeeping: usize,
}

impl RuntimeHeadroom {
    /// 2026-10-01: The terms for a util budget of `budget_bytes`, a carry stash of
    /// `carry_stash` bytes and the class's `driver` terms (2026-10-13).
    pub(crate) fn new(budget_bytes: usize, carry_stash: usize, driver: &DriverTerms) -> Self {
        Self {
            carry_stash,
            driver_fixed: driver.driver_fixed_bytes as usize,
            driver_bookkeeping: budget_bytes / 1000 * driver.driver_budget_per_mille as usize,
        }
    }

    /// 2026-10-01: The sum.
    pub(crate) fn total(&self) -> usize {
        self.carry_stash + self.driver_fixed + self.driver_bookkeeping
    }
}

/// 2026-10-01: The bytes `TransformerModel::gdn_carry_bind` allocates for `verify_slots` carry
/// slots (the MTP slots plus the dummy), or 0 when it binds nothing: no verify pools, an f16 h
/// pool, or GDN heads other than the 128 x 128 the carry kernels serve. Assumes the target ships
/// the carry kernels (gb10 does); without them the model binds nothing and this over-reserves.
pub(crate) fn carry_stash_bytes(
    config: &ModelConfig,
    verify_slots: Option<usize>,
    h_f16_pool: bool,
) -> usize {
    let Some(slots) = verify_slots else {
        return 0;
    };
    let (nv, kd, vd) = (
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    );
    if h_f16_pool || nv == 0 || kd != 128 || vd != 128 {
        return 0;
    }
    let layers = (0..config.num_hidden_layers)
        .filter(|&i| config.layer_type(i) == metrale_config::LayerType::LinearAttention)
        .count();
    let conv_dim = config.linear_num_key_heads * kd * 2 + nv * vd;
    use metrale_model_layers::layers::ops;
    ops::GdnCarrySizes::new(
        layers,
        slots,
        ops::gdn_carry_seq_floats(nv, kd, vd),
        ops::gdn_carry_conv_seq_elems(conv_dim),
        metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS,
    )
    .total()
}
