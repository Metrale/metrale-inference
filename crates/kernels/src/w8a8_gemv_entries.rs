// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-05: The schedule points of the W8A8 skinny GEMV (`kernels/gb10/common/w8a8_gemv.cu`,
// `W8A8_TPL`) per token-tile band, which `[defaults] w8a8_gemv_entries` picks from.
//
// Owner: kernels crate.
// Invariants:
// - `lib.rs` compiles this file as a module and `build_defaults.rs` `include!`s it, so the
//   build-time parse and the runtime resolver accept the same names. `//!` cannot appear at an
//   `include!` site, which is why this header uses `//`.
// - Band `b` launches at most `8 << b` rows (8, 16, 32, 64, 128); every point of a band has
//   that many token tiles (`mb{1 << b}`). The points of a band differ only in K unroll and
//   blocks per SM, which do not change the arithmetic, so a band's points give the same bits.
// - The first point of each band is the baseline every class without a measurement declares.

/// 2026-10-05: Per band, the compiled entry suffixes (`w8a8_gemv_{rowscale,blk128}_{point}`).
pub const W8A8_GEMV_POINTS: [&[&str]; 5] = [
    &["mb1_ku8", "mb1_ku2_o4"],
    &["mb2", "mb2_ku2_o2"],
    &["mb4", "mb4_ku2_o2"],
    &["mb8", "mb8_ku2_o2"],
    &["mb16", "mb16_ku1_o2"],
];

/// 2026-10-05: The baseline point of every band.
pub const W8A8_GEMV_BASELINE: [&str; 5] = ["mb1_ku8", "mb2", "mb4", "mb8", "mb16"];

/// 2026-10-05: `name` as a compiled point of band `band`, or `None`.
pub fn w8a8_gemv_point(band: usize, name: &str) -> Option<&'static str> {
    W8A8_GEMV_POINTS
        .get(band)?
        .iter()
        .copied()
        .find(|p| *p == name)
}

/// 2026-10-05: Five names, one per band in band order, each a compiled point of its band; `None`
/// when the count or any name is wrong.
pub fn w8a8_gemv_entries<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Option<[&'static str; 5]> {
    let mut out = W8A8_GEMV_BASELINE;
    let mut count = 0;
    for (band, name) in names.into_iter().enumerate() {
        *out.get_mut(band)? = w8a8_gemv_point(band, name.trim())?;
        count += 1;
    }
    (count == out.len()).then_some(out)
}
