// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: What the `met serve` command line publishes into the kernel-path
//! cells; `serve_flags::publish_kernel_flags` performs the plan.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - `KernelFlagPlan::from_args` reads only its argument: no environment, no
//!   process cells, no logging.
//! - A top-level `None` publishes nothing, so that cell resolves from its
//!   `METRALE_*` variable on first read; publishing a default would seal it.

use crate::cli::ServeArgs;
use crate::cli::flag_values::Tristate;
use metrale_model_layers::layers::{Phased, phased_name};

/// 2026-09-26: The GDN cell, which the command line owns as a whole once any GDN flag is
/// given (`--ssm-h-dtype`, `--gdn-fused-norm`, `--ssm-batched-recurrent`
/// `on|off`, `--exact-verify`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GdnPlan {
    pub h_f16: bool,
    pub h_f16_pool: bool,
    pub fused_norm: bool,
    /// 2026-09-26: `None`: the compiled target's `HARDWARE.toml` default decides,
    /// with `METRALE_SSM_BATCHED_RECURRENT` over it. A hard `false` would turn
    /// off `kernels/hopper`'s default on every serve that names another GDN flag.
    pub batched_recurrent: Option<bool>,
    pub exact_verify: bool,
}

/// 2026-09-26: Every kernel-path selection the command line makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KernelFlagPlan {
    /// 2026-09-26: `None` when no GDN flag was given.
    pub gdn: Option<GdnPlan>,
    /// 2026-09-28: `--weight-quantization` and its `--w4a4-downcast` lever. Always published:
    /// no environment fallback exists.
    pub weight_quant: metrale_config::WeightQuantTier,
    /// 2026-09-27: Always published: no environment fallback exists.
    pub expert_quantization: metrale_model_layers::layers::ExpertQuantization,
    /// 2026-10-09: `--dense-quantization`. Always published: no environment fallback exists.
    /// 2026-10-14: Per phase.
    pub dense_quantization: Phased<metrale_model_layers::layers::DenseQuantization>,
    /// 2026-10-11: `--dsa-quantization`. Always published: no environment fallback exists.
    /// 2026-10-14: Per phase.
    pub dsa_quantization: Phased<metrale_model_layers::layers::DsaQuantization>,
    pub prefill_codispatch: Option<bool>,
    pub prefill_varlen: Option<bool>,
    pub ssm_tail_midchunk: Option<bool>,
    pub hermetic: bool,
}

impl KernelFlagPlan {
    /// 2026-09-26: The plan for a command line that passed `validate_serve_args`.
    pub(crate) fn from_args(args: &ServeArgs) -> Self {
        let batched_recurrent = Tristate::validated(&args.ssm_batched_recurrent).pinned();
        let gdn_given = args.ssm_h_dtype.is_some()
            || args.gdn_fused_norm
            || batched_recurrent.is_some()
            || args.exact_verify;
        let gdn = gdn_given.then(|| {
            // 2026-09-26: The decode `validate_serve_args` checks, so the
            // validator and the published cell read the flag alike; `f16-pool`
            // sets both bits.
            let (h_f16, h_f16_pool) = metrale_model_layers::layers::qwen3_ssm::ssm_h_dtype_bits(
                args.ssm_h_dtype.as_deref(),
            );
            GdnPlan {
                h_f16,
                h_f16_pool,
                fused_norm: args.gdn_fused_norm,
                batched_recurrent,
                exact_verify: args.exact_verify,
            }
        });
        Self {
            gdn,
            weight_quant: args
                .weight_quant_tier()
                .expect("validated by validate_serve_args"),
            expert_quantization: args.expert_quantization.0,
            dense_quantization: args.dense_quantization.0,
            dsa_quantization: args.dsa_quantization.0,
            prefill_codispatch: args.prefill_codispatch.then_some(true),
            prefill_varlen: args.prefill_varlen_batch.then_some(true),
            ssm_tail_midchunk: args.no_ssm_tail_midchunk.then_some(false),
            hermetic: args.hermetic,
        }
    }
}

/// 2026-10-11: The boot log's dense-precision disclosure: both dense tiers as served.
/// 2026-10-14: Each per phase; the DSA sites' description is each phase's.
pub(crate) fn precision_line(
    dense: Phased<metrale_model_layers::layers::DenseQuantization>,
    dsa: Phased<metrale_model_layers::layers::DsaQuantization>,
) -> String {
    use metrale_model_layers::layers::{DenseQuantization as D, DsaQuantization as S};
    let sites = if dsa.is_split() {
        format!(
            "decode {}; prefill {}",
            dsa.decode.describe(),
            dsa.prefill.describe()
        )
    } else {
        dsa.decode.describe().to_string()
    };
    format!(
        "dense precision: --dense-quantization {} --dsa-quantization {} (DSA q_a/q_absorb/o_absorb: {sites})",
        phased_name(dense, D::name),
        phased_name(dsa, S::name),
    )
}

#[cfg(test)]
#[path = "kernel_flag_plan_tests.rs"]
mod tests;
