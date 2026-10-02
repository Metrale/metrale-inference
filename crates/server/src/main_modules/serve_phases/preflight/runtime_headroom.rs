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
//! - The driver terms are calibrated on GB10 from measured serves (below); a change to them
//!   cites a new measurement.
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
//! | Nemotron-3-Nano, nvfp4, no speculation, 8 slots | 40.5 GB | 2.7-2.9 GB | +0.89 GB kernel slab |
//! | Nemotron-3-Nano, as above with prefix caching (pool fills the budget) | 102.9 GB | 3.5-3.6 GB | |
//!
//! Probed directly: context 250 MiB, the 218 modules of the 27B target 94 MiB, the local-memory
//! reservation 131 MiB (1,864 B/thread x 48 SMs x 1,536 threads); the rest is the driver's
//! per-allocation bookkeeping. It does not depend on speculation, so the no-speculation serve
//! needs the same headroom: 512 MiB under-reserved it by about 2.1-2.6 GB whenever its KV pool
//! fills the budget. `DRIVER_FIXED_BYTES + DRIVER_BUDGET_PER_MILLE` of the util budget covers
//! the dense default tier's worst case, 3.13 GB, with 0.12 GB to spare at util 0.85 on GB10, and
//! every certified recipe. It does not cover Nemotron-3-Nano with prefix caching on (not a
//! certified configuration: its recipe runs without it, the pool clamped to demand), which reads
//! 0.4 GB above it; covering that would take 26 per mille, which leaves the dense default tier
//! no KV pool at 128 slots.

use metrale_config::ModelConfig;

/// 2026-10-01: The context, the loaded modules, the local-memory reservation and the CUDA
/// graphs: 475 MiB measured without the graphs; 1 GiB.
pub(super) const DRIVER_FIXED_BYTES: usize = 1 << 30;

/// 2026-10-01: The driver's bookkeeping for the allocations that fill the budget, per mille of
/// the util budget: 21 (2.1%), so that with the fixed term it covers 3.13 GB at a 105.9 GB budget.
pub(super) const DRIVER_BUDGET_PER_MILLE: usize = 21;

/// 2026-10-01: The headroom's terms, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RuntimeHeadroom {
    /// 2026-10-01: The GDN carry stash and its tables (`GdnCarrySizes::total`); 0 when the model
    /// cannot carry.
    pub carry_stash: usize,
    /// 2026-10-01: [`DRIVER_FIXED_BYTES`].
    pub driver_fixed: usize,
    /// 2026-10-01: [`DRIVER_BUDGET_PER_MILLE`] of the util budget.
    pub driver_bookkeeping: usize,
}

impl RuntimeHeadroom {
    /// 2026-10-01: The terms for a util budget of `budget_bytes` and a carry stash of
    /// `carry_stash` bytes.
    pub(crate) fn new(budget_bytes: usize, carry_stash: usize) -> Self {
        Self {
            carry_stash,
            driver_fixed: DRIVER_FIXED_BYTES,
            driver_bookkeeping: budget_bytes / 1000 * DRIVER_BUDGET_PER_MILLE,
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
