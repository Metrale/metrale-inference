// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: Which expert copies a qwen35 MoE layer builds: the NVFP4 projections
//! (`load_moe_qwen35`), the native-FP8 experts (`install_native_fp8_experts`) and the NVFP4
//! prefill copies (transposes and predequant), from the checkpoint's format, the diagnostic
//! `METRALE_FORCE_NVFP4_MOE` / `METRALE_FORCE_NVFP4_ALL` and the `--expert-quantization` tier.
//!
//! Owner: model-arch (qwen35 loader).
//! Invariants: none beyond the types.

use metrale_model_layers::layers::ExpertQuantization;
use metrale_model_layers::weight_map::Nvfp4MoeCopies;

/// 2026-09-27: The copies one MoE layer builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExpertLoadPlan {
    /// 2026-09-27: The NVFP4 projections `load_moe_qwen35` builds.
    pub nvfp4: Nvfp4MoeCopies,
    /// 2026-09-27: Whether the native-FP8 experts are installed (`set_fp8_experts`).
    pub fp8_experts: bool,
    /// 2026-09-27: Whether the NVFP4 prefill copies are built.
    pub nvfp4_prefill_copies: bool,
}

/// 2026-09-27: The plan for a layer. `native_fp8`: the checkpoint's experts are native FP8;
/// `fused`: they are in the fused `experts.gate_up_proj` layout; `force_env`: one of the force
/// levers is set.
///
/// - A native-FP8 layer under `fp8` without a force lever keeps its FP8 experts and builds only
///   the NVFP4 shared expert, as before the tiers.
/// - Under an NVFP4 tier it keeps its FP8 experts (prefill, the shared expert and, under
///   `nvfp4-gate-up`, the routed down projections decode from them) and builds only the routed
///   NVFP4 projections the tier decodes: gate and up, and down under `nvfp4`.
/// - Every other layer (a force lever, a fused or NVFP4 checkpoint) builds every NVFP4 projection
///   and the prefill copies (the latter not for a native-FP8 fused layer), and no FP8 experts.
pub(super) fn expert_load_plan(
    native_fp8: bool,
    fused: bool,
    force_env: bool,
    tier: ExpertQuantization,
) -> ExpertLoadPlan {
    let decode_tier = native_fp8 && !fused && tier.nvfp4_decode();
    let force = force_env || tier.nvfp4_decode();
    let fp8_only = native_fp8 && !force && !fused;
    let nvfp4 = if fp8_only {
        Nvfp4MoeCopies::SHARED_ONLY
    } else if decode_tier {
        Nvfp4MoeCopies {
            routed_gate_up: true,
            routed_down: tier.nvfp4_down(),
            shared: false,
        }
    } else {
        Nvfp4MoeCopies::ALL
    };
    ExpertLoadPlan {
        nvfp4,
        fp8_experts: native_fp8 && !fused && (!force || decode_tier),
        nvfp4_prefill_copies: (!native_fp8 || force) && !decode_tier,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ExpertQuantization as Q;

    fn plan(native_fp8: bool, fused: bool, force_env: bool, tier: Q) -> ExpertLoadPlan {
        expert_load_plan(native_fp8, fused, force_env, tier)
    }

    /// 2026-09-27: A native-FP8 checkpoint per tier: which copies are resident.
    #[test]
    fn native_fp8_copies_per_tier() {
        assert_eq!(
            plan(true, false, false, Q::Fp8),
            ExpertLoadPlan {
                nvfp4: Nvfp4MoeCopies::SHARED_ONLY,
                fp8_experts: true,
                nvfp4_prefill_copies: false,
            }
        );
        let gate_up = Nvfp4MoeCopies {
            routed_gate_up: true,
            routed_down: false,
            shared: false,
        };
        assert_eq!(
            plan(true, false, false, Q::Nvfp4GateUp),
            ExpertLoadPlan {
                nvfp4: gate_up,
                fp8_experts: true,
                nvfp4_prefill_copies: false,
            }
        );
        assert_eq!(
            plan(true, false, false, Q::Nvfp4),
            ExpertLoadPlan {
                nvfp4: Nvfp4MoeCopies {
                    routed_down: true,
                    ..gate_up
                },
                fp8_experts: true,
                nvfp4_prefill_copies: false,
            }
        );
        // 2026-09-27: A force lever does not change what a tier builds.
        for q in [Q::Nvfp4GateUp, Q::Nvfp4] {
            assert_eq!(plan(true, false, true, q), plan(true, false, false, q));
        }
    }

    /// 2026-09-27: The force levers and the non-native layouts keep their NVFP4-only loads.
    #[test]
    fn force_and_other_layouts_build_every_nvfp4_copy() {
        let nvfp4_only = ExpertLoadPlan {
            nvfp4: Nvfp4MoeCopies::ALL,
            fp8_experts: false,
            nvfp4_prefill_copies: true,
        };
        assert_eq!(plan(true, false, true, Q::Fp8), nvfp4_only);
        for q in Q::ALL {
            assert_eq!(plan(false, false, false, q), nvfp4_only);
            assert_eq!(
                plan(true, true, false, q),
                ExpertLoadPlan {
                    nvfp4_prefill_copies: q.nvfp4_decode(),
                    ..ExpertLoadPlan {
                        nvfp4: Nvfp4MoeCopies::ALL,
                        fp8_experts: false,
                        nvfp4_prefill_copies: false,
                    }
                }
            );
        }
    }
}
