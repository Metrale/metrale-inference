// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Tests of the GLM MLP precision plan: which kernel each group runs at each row
//! count under the two quantization flags, and the refusals.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};

use super::{GroupPrecision, MlpGroup, MlpKernel};

/// 2026-10-08: The W4A4 row cap the tests assume for the routed experts.
const CAP: usize = 16;

fn ladder(flag: &str, family: ProjFamily) -> metrale_config::activation_quantization::Ladder {
    ActivationQuantization::parse(flag)
        .expect("flag parses")
        .ladder(family)
        .clone()
}

fn experts(flag: &str, stamp: Nvfp4Act) -> GroupPrecision {
    GroupPrecision::resolve(
        MlpGroup::RoutedExperts,
        ladder(flag, ProjFamily::Moe),
        stamp,
        CAP,
        true,
    )
    .expect("resolves")
}

fn dense(flag: &str, stamp: Nvfp4Act, cap: usize) -> GroupPrecision {
    GroupPrecision::resolve(
        MlpGroup::DenseMlp,
        ladder(flag, ProjFamily::Ffn),
        stamp,
        cap,
        true,
    )
    .expect("resolves")
}

/// 2026-10-08: The defaults (`declared` weights, `declared` activations) on this checkpoint:
/// W4A4 for both groups at every decode width, the experts' grouped prefill above the cap on
/// the W4A16 kernels, the dense MLP W4A4 at every width (its W4A4 runs in chunks).
#[test]
fn declared_runs_w4a4_for_both_groups() {
    let e = experts("declared", Nvfp4Act::A4);
    for rows in [1, 2, 8, 16] {
        assert_eq!(e.kernel(rows), MlpKernel::W4a4Static, "{rows} rows");
    }
    assert_eq!(e.kernel(17), MlpKernel::W4a16);
    assert_eq!(e.kernel(2048), MlpKernel::W4a16);
    let d = dense("declared", Nvfp4Act::A4, usize::MAX);
    for rows in [1, 16, 64, 4096] {
        assert_eq!(d.kernel(rows), MlpKernel::W4a4Static, "{rows} rows");
    }
}

/// 2026-10-08: Under `--weight-quantization nvfp4` the stamps are absent, and under
/// `--activation-quantization adaptive` or `bf16` no rung asks for FP4: both groups run the
/// pre-W4A4 kernels, byte for byte the path before this plan existed.
#[test]
fn above_declared_tiers_run_the_pre_w4a4_kernels() {
    for (flag, stamp) in [
        ("declared", Nvfp4Act::Unstamped),
        ("adaptive", Nvfp4Act::A4),
        ("bf16", Nvfp4Act::A4),
    ] {
        let e = experts(flag, stamp);
        let d = dense(flag, stamp, usize::MAX);
        for rows in [1, 8, 16, 100] {
            assert_eq!(e.kernel(rows), MlpKernel::W4a16, "{flag} {stamp:?} {rows}");
            assert_eq!(d.kernel(rows), MlpKernel::Bf16, "{flag} {stamp:?} {rows}");
        }
        assert!(
            !d.reaches(MlpKernel::W4a4Static, 64),
            "{flag}: no packed dense copy"
        );
    }
}

/// 2026-10-08: A module the checkpoint declares wider (`Wide`, e.g. the BF16 MTP layer's
/// experts quantized at load) runs W4A16 under `declared`, and FP4 activations only when the
/// operator asks for `nvfp4` by name.
#[test]
fn a_wide_declaration_runs_fp4_only_when_named() {
    assert_eq!(
        experts("declared", Nvfp4Act::Wide).kernel(1),
        MlpKernel::W4a16
    );
    assert_eq!(
        experts("nvfp4", Nvfp4Act::Wide).kernel(1),
        MlpKernel::W4a4Static
    );
}

/// 2026-10-08: A ladder splits the groups by row count, and a family override reaches only its
/// own group: `ffn:bf16` leaves the experts on W4A4.
#[test]
fn ladders_and_family_overrides_route_per_group_and_width() {
    let e = experts("1-4=declared;5-=bf16", Nvfp4Act::A4);
    assert_eq!(e.kernel(4), MlpKernel::W4a4Static);
    assert_eq!(e.kernel(5), MlpKernel::W4a16);

    let flag = "declared,ffn:bf16";
    assert_eq!(experts(flag, Nvfp4Act::A4).kernel(1), MlpKernel::W4a4Static);
    let d = dense(flag, Nvfp4Act::A4, usize::MAX);
    assert_eq!(d.kernel(1), MlpKernel::Bf16);

    // 2026-10-08: A mixed dense ladder needs both weight forms.
    let mixed = dense("1-8=declared;9-=bf16", Nvfp4Act::A4, usize::MAX);
    assert!(mixed.reaches(MlpKernel::W4a4Static, 64));
    assert!(mixed.reaches(MlpKernel::Bf16, 64));
    assert!(
        !mixed.reaches(MlpKernel::Bf16, 8),
        "rows 1-8 never reach BF16"
    );
}

/// 2026-10-08: Without the W4A4 kernels (a target compiled without the FP4 block-scale MMA)
/// `declared` falls back to the 16-bit path and the log says it runs above declared; `nvfp4`
/// named explicitly is refused instead.
#[test]
fn missing_kernels_fall_back_under_declared_and_refuse_under_nvfp4() {
    let d = dense("declared", Nvfp4Act::A4, 0);
    assert_eq!(d.kernel(1), MlpKernel::Bf16);
    assert!(
        d.describe(4).contains("ABOVE declared"),
        "{}",
        d.describe(4)
    );

    let err = GroupPrecision::resolve(
        MlpGroup::DenseMlp,
        ladder("nvfp4", ProjFamily::Ffn),
        Nvfp4Act::A4,
        0,
        true,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("no W4A4"), "{err}");
}

/// 2026-10-08: `fp8` is refused for either group, and FP4 activations without the checkpoint's
/// scales are refused whether asked for by name or by the declaration.
#[test]
fn fp8_and_missing_scales_are_refused() {
    for group in [MlpGroup::RoutedExperts, MlpGroup::DenseMlp] {
        let family = match group {
            MlpGroup::RoutedExperts => ProjFamily::Moe,
            MlpGroup::DenseMlp => ProjFamily::Ffn,
        };
        let r = |flag: &str, scales: bool| {
            GroupPrecision::resolve(group, ladder(flag, family), Nvfp4Act::A4, CAP, scales)
                .map(|_| ())
                .map_err(|e| e.to_string())
        };
        assert!(r("fp8", true).unwrap_err().contains("fp8"));
        assert!(r("1-2=declared;3-=fp8", true).unwrap_err().contains("fp8"));
        assert!(r("nvfp4", false).unwrap_err().contains("input_scale"));
        assert!(r("declared", false).unwrap_err().contains("input_scale"));
        // 2026-10-08: No FP4 rung, no scales needed.
        assert!(r("bf16", false).is_ok());
        assert!(r("adaptive", false).is_ok());
    }
}

/// 2026-10-08: The load-log line names every range once, in order, and marks only the 16-bit
/// ranges of a W4A4-declared group.
#[test]
fn describe_lists_the_ranges_and_marks_the_ones_above_declared() {
    let e = experts("declared", Nvfp4Act::A4);
    assert_eq!(
        e.describe(64),
        "GLM routed experts [rows 1-16: W4a4Static, rows 17-64: W4a16 (ABOVE declared W4A4)]"
    );
    let wide = experts("declared", Nvfp4Act::Wide);
    assert_eq!(wide.describe(8), "GLM routed experts [rows 1-8: W4a16]");
}
