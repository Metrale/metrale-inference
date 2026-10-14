// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The extra-compiler-flag layers a kernel target is built with, and the rule that merges them.
//!
//! Owner: metrale-kernels build.
//! Invariants:
//! - `merge_extra_flags` returns each flag once, at the position of its
//!   first occurrence, hardware layer first.
//!
//! Included via `#[path = "build_flags.rs"] mod build_flags;`. Its own file,
//! with no `super::` dependencies, so `tests/kernel_build_flags.rs` can
//! compile the same code: cargo never runs a build script's `#[cfg(test)]`
//! modules.

/// 2026-09-25: The `[build]` key a vendor's extra flags are declared under.
///
/// `apple` / `metal` read `extra_metal_flags`; every other vendor (NVIDIA,
/// SCALE, HIP) reads `extra_nvcc_flags`. A file may declare both; only the
/// vendor-matching list is forwarded, so flags do not cross toolchains.
///
/// Both `build_parse::parse_kernel_toml` and [`hardware_extra_flags`] call
/// this, so HARDWARE.toml and KERNEL.toml cannot disagree about the key.
pub(crate) fn flag_key(vendor: &str) -> &'static str {
    match vendor {
        "apple" | "metal" => "extra_metal_flags",
        _ => "extra_nvcc_flags",
    }
}

/// 2026-09-25: `[build] extra_*_flags` from a parsed `kernels/<hw>/HARDWARE.toml`.
///
/// The hardware layer states facts about the architecture, not the model.
/// `kernels/hopper` (sm_90a), `kernels/b200` (sm_100a) and `kernels/b300`
/// (sm_103a) use it to define `METRALE_NO_WARP_BLOCKSCALE_MMA`, which compiles
/// out the kernels built on the warp-level
/// `mma.sync ... .kind::mxf4nvf4.block_scale`; kernels/b200/HARDWARE.toml
/// records ptxas refusing that instruction for sm_100a.
///
/// Why here and not in a KERNEL.toml: hopper and b200 inherit gb10's kernel
/// tree (`inherits = "gb10"`) and with it gb10's KERNEL.tomls, so a per-model
/// declaration would mean forking those shared files. Callers pass only this
/// hardware's own HARDWARE.toml, so an overlay does not inherit these flags.
///
/// Panics on a non-string entry, naming the key.
pub(crate) fn hardware_extra_flags(hw_toml: &toml::Value, vendor: &str) -> Vec<String> {
    let key = flag_key(vendor);
    let Some(arr) = hw_toml
        .get("build")
        .and_then(|b| b.get(key))
        .and_then(|f| f.as_array())
    else {
        return Vec::new();
    };
    arr.iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("HARDWARE.toml: [build] {key} entries must be strings"))
                .to_string()
        })
        .collect()
}

/// 2026-09-25: The flag list a target is compiled with, from its three
/// declaration layers.
///
/// Least specific first (hardware, then the common-role KERNEL.tomls, then the
/// leaf-role ones), appended in order and deduped, so a flag declared more
/// than once is passed once, at the position its first layer gave it.
pub(crate) fn merge_extra_flags(
    hardware: &[String],
    common: &[String],
    model: &[String],
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for flag in hardware.iter().chain(common).chain(model) {
        if !out.contains(flag) {
            out.push(flag.clone());
        }
    }
    out
}

/// 2026-10-14: The value a `-D<name>=<value>` in `flags` gives `name`; a bare
/// `-D<name>` reads as `"1"`, and the last definition wins, as on the
/// compiler's command line. `None` when no flag defines `name`.
pub(crate) fn define_value<'a>(flags: &'a [String], name: &str) -> Option<&'a str> {
    flags.iter().rev().find_map(|f| {
        let rest = f.strip_prefix("-D")?.strip_prefix(name)?;
        if rest.is_empty() {
            Some("1")
        } else {
            rest.strip_prefix('=')
        }
    })
}

/// 2026-10-14: The R that `-DMETRALE_BATCHM_DIRECT_R=<R>` compiled the
/// `dense_gemv_bf16_batchm_r<R>_m<M>` entries with, read from the flags each
/// target was built with (one slice per target). `None` when no target
/// defines it.
///
/// Panics when R is not a positive integer, or when two targets of one build
/// disagree: the binary carries one constant, so it must name the one set the
/// kernels were compiled with.
pub(crate) fn batchm_direct_r(per_target: &[&[String]]) -> Option<u32> {
    const NAME: &str = "METRALE_BATCHM_DIRECT_R";
    let mut found: Option<u32> = None;
    for (i, flags) in per_target.iter().enumerate() {
        let r = define_value(flags, NAME).map(|v| {
            v.parse::<u32>()
                .ok()
                .filter(|&r| r > 0)
                .unwrap_or_else(|| panic!("-D{NAME}={v}: R must be a positive integer"))
        });
        if i == 0 {
            found = r;
        } else {
            assert_eq!(
                r, found,
                "-D{NAME} differs between the targets of one build ({r:?} vs {found:?})"
            );
        }
    }
    found
}
