// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The byte sizes of the carried-state GDN verify's device allocations, split from
//! `ssm_gdn_carry.rs` unchanged: the binder (`TransformerModel::gdn_carry_bind`) allocates exactly
//! these and the preflight reserve plans their total.
//!
//! Owner: model-layers ops (GDN).
//! Invariants: none beyond the types.

/// 2026-10-01: The device allocations `TransformerModel::gdn_carry_bind` makes for `layers`
/// GDN layers and `slots` carry slots (the MTP slots plus the dummy), in bytes, one field per
/// allocation. The binder allocates exactly these and the preflight reserve plans `total()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnCarrySizes {
    /// 2026-10-01: `[layers][slots][seq_floats]` f32.
    pub stash: usize,
    /// 2026-10-01: `[layers][slots][conv_seq_elems]` bf16.
    pub conv_stash: usize,
    /// 2026-10-01: `[layers][table_seqs]` u64 conv pointers, for the verify and for the flush.
    pub conv_tab: usize,
    pub flush_conv_tab: usize,
    /// 2026-10-01: `[layers][slots]` u32 pending counts.
    pub pend: usize,
    /// 2026-10-01: `[table_seqs]` u32 slot table.
    pub slot_tab: usize,
    /// 2026-10-01: `[layers][table_seqs]` u32 flags.
    pub flags: usize,
    /// 2026-10-01: `[layers][table_seqs]` u64 flush pointers and `[table_seqs]` u32 flush slots.
    pub flush_tab: usize,
    pub flush_slots: usize,
}

impl GdnCarrySizes {
    /// 2026-10-01: The sizes for `layers` layers, `slots` slots, the per-slot widths and the
    /// verify table's `table_seqs` rows (`VERIFY_WY_TABLE_SEQS`).
    pub const fn new(
        layers: usize,
        slots: usize,
        seq_floats: usize,
        conv_seq_elems: usize,
        table_seqs: usize,
    ) -> Self {
        Self {
            stash: layers * slots * seq_floats * 4,
            conv_stash: layers * slots * conv_seq_elems * 2,
            conv_tab: layers * table_seqs * 8,
            flush_conv_tab: layers * table_seqs * 8,
            pend: layers * slots * 4,
            slot_tab: table_seqs * 4,
            flags: layers * table_seqs * 4,
            flush_tab: layers * table_seqs * 8,
            flush_slots: table_seqs * 4,
        }
    }

    /// 2026-10-01: Every allocation's bytes, summed.
    pub const fn total(&self) -> usize {
        self.stash
            + self.conv_stash
            + self.conv_tab
            + self.flush_conv_tab
            + self.pend
            + self.slot_tab
            + self.flags
            + self.flush_tab
            + self.flush_slots
    }
}
