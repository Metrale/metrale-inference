// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The tile-geometry variants of `moe_w4a16_grouped_gemm_ptrtable` that the bench
//! times against the base kernel. Data only, no GPU calls.
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Invariants:
//! - Each entry's `m_tile`, `n_tile` and `threads` are the `MT`, `NTILE` and `WARPS * 32` of
//!   the kernel it names.

/// 2026-09-25: One kernel under test. `m_tile` sets the grid height and `n_tile` the grid
/// width; either value wrong is silent (dropped rows or unwritten columns), not an error.
pub(crate) struct Variant {
    pub name: &'static str,
    pub kernel: &'static str,
    pub m_tile: u32,
    pub n_tile: u32,
    pub threads: u32,
}

const fn v(
    name: &'static str,
    kernel: &'static str,
    m_tile: u32,
    n_tile: u32,
    threads: u32,
) -> Variant {
    Variant {
        name,
        kernel,
        m_tile,
        n_tile,
        threads,
    }
}

pub(crate) const VARIANTS: &[Variant] = &[
    v(
        "base M64 N64 K16",
        "moe_w4a16_grouped_gemm_ptrtable",
        64,
        64,
        128,
    ),
    v(
        "     M64 N64 K32",
        "moe_w4a16_grouped_gemm_ptrtable_k32",
        64,
        64,
        128,
    ),
    v(
        "     M64 N64 K64",
        "moe_w4a16_grouped_gemm_ptrtable_k64",
        64,
        64,
        128,
    ),
    v(
        "     M16 N64 K64",
        "moe_w4a16_grouped_gemm_ptrtable_m16_k64",
        16,
        64,
        128,
    ),
    v(
        "al+km M16 N64 K128",
        "moe_w4a16_grouped_gemm_ptrtable_alkm_m16_k128",
        16,
        64,
        128,
    ),
    v(
        "BT M16 N64 K128",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128",
        16,
        64,
        128,
    ),
    v(
        "BT M16 N128 K128",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m16_n128_k128",
        16,
        128,
        256,
    ),
];
