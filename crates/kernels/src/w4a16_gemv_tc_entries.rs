// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-09: The schedule points of the NVFP4 W4A16 tensor-core GEMV (`kernels/gb10/common/
// w4a16_gemv_tc.cu`, `W4TC_ENTRY`) per row tier, which `[defaults] w4a16_gemv_tc_entries` picks
// from.
//
// Owner: kernels crate.
// Invariants:
// - `lib.rs` compiles this file as a module and `build_defaults.rs` `include!`s it, so the
//   build-time parse and the runtime resolver accept the same names. `//!` cannot appear at an
//   `include!` site, which is why this header uses `//`.
// - Tier 0 serves 1..=8 rows (`tc8` points), tier 1 serves 1..=16 (`tc16` points). The points of
//   a tier differ only in NT, K unroll and blocks per SM, which do not change a column's
//   arithmetic, so a tier's points give the same bits.
// - The first point of each tier is the baseline every class without a measurement declares.

/// 2026-10-09: Per tier, the compiled entry suffixes (`w4a16_gemv_{point}`).
pub const W4A16_GEMV_TC_POINTS: [&[&str]; 2] = [
    &[
        "tc8",
        "tc8_nt1_ku4_o2",
        "tc8_nt1_ku8_o2",
        "tc8_nt2_ku4_o2",
        "tc8_nt2_ku2_o3",
        "tc8_nt1_ku4_o3",
        "tc8_nt4_ku2_o2",
        "tc8_nt2_ku2_o4",
        "tc8_nt4_ku1_o3",
        "tc8_nt2_ku1_o4",
        "tc8_nt4_ku2_o3",
    ],
    &[
        "tc16",
        "tc16_nt2_ku2_o2",
        "tc16_nt1_ku4_o2",
        "tc16_nt4_ku1_o2",
        "tc16_nt2_ku4_o2",
        "tc16_nt4_ku1_o3",
        "tc16_nt8_ku1_o2",
        "tc16_nt4_ku2_o2",
    ],
];

/// 2026-10-09: The wide row tiers (`tc32`: 17..=32 rows, `tc64`: 33..=64), m16 tiles sharing each
/// decoded weight fragment; a row's bits equal tc16's. Benched and identity-checked; not yet a
/// `[defaults]` tier.
pub const W4A16_GEMV_TC_WIDE_POINTS: [&[&str]; 2] = [
    &["tc32", "tc32_nt4_ku1_o2", "tc32_nt2_ku1_o2"],
    &["tc64", "tc64_nt2_ku1_o2", "tc64_nt4_ku1_o1"],
];

/// 2026-10-09: The baseline point of every tier.
pub const W4A16_GEMV_TC_BASELINE: [&str; 2] = ["tc8", "tc16"];

/// 2026-10-09: `name` as a compiled point of tier `tier`, or `None`.
pub fn w4a16_gemv_tc_point(tier: usize, name: &str) -> Option<&'static str> {
    W4A16_GEMV_TC_POINTS
        .get(tier)?
        .iter()
        .copied()
        .find(|p| *p == name)
}

/// 2026-10-09: Two names, tc8 tier then tc16 tier, each a compiled point of its tier; `None`
/// when the count or any name is wrong.
pub fn w4a16_gemv_tc_entries<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Option<[&'static str; 2]> {
    let mut out = W4A16_GEMV_TC_BASELINE;
    let mut count = 0;
    for (tier, name) in names.into_iter().enumerate() {
        *out.get_mut(tier)? = w4a16_gemv_tc_point(tier, name.trim())?;
        count += 1;
    }
    (count == out.len()).then_some(out)
}

/// 2026-10-09: The 8-column tiles per CTA (`NT`) of a point: its `_nt{NT}` field, else the bare
/// tier's (tc8: 1; tc16, tc32, tc64: 2). The launch grid is `ceil(N / (8 * NT))`.
// 2026-10-09: The build script includes this file for the parse only.
#[allow(dead_code)]
pub fn w4a16_gemv_tc_nt(point: &str) -> u32 {
    point
        .split('_')
        .find_map(|t| t.strip_prefix("nt").and_then(|v| v.parse().ok()))
        .unwrap_or(if point.starts_with("tc8") { 1 } else { 2 })
}
