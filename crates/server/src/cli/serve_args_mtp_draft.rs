// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The MTP drafter's serve options, `--draft-confidence-stop` and
//! `--mtp-experts-nvfp4`, flattened into `ServeSchedulingArgs` after `--mtp-quantization`.
//! Split out of `serve_args/scheduling.rs`, verbatim, to keep it under the 500-line cap.
//!
//! Owner: server CLI.
//! Invariants: the `///` text on the struct's fields is the `--help` output and
//! carries no date.

use clap::Args;

#[derive(Args, Debug, Clone, PartialEq)]
pub struct ServeMtpDraftArgs {
    /// Per-position confidence stop for MTP draft chains (default: off). The
    /// drafter extends a chain past a draft only while that draft's top-1
    /// probability is at least TAU, so a chain of up to `--num-drafts` ends
    /// with its first draft below TAU (still verified). Unconfident chains
    /// cost fewer drafter steps and fewer verify rows; confident ones run to
    /// full depth. It changes which drafts are verified, never the emitted
    /// tokens. 0 < TAU < 1. Requires `--speculative`; not with `--dflash`.
    #[arg(long, value_name = "TAU")]
    pub draft_confidence_stop: Option<f32>,

    /// Draft on NVFP4 copies of a BF16 MoE MTP head's experts (default: false).
    ///
    /// The head's routed and shared experts are requantized from BF16 to NVFP4 at load and
    /// run the grouped NVFP4 tensor-core decode; the router, attention, fc and draft LM head
    /// keep the head's precision. Draft-only: the target model verifies every drafted token,
    /// so only how many drafts are accepted moves. Under --exact-verify greedy output is
    /// byte-identical with it on or off (measured on nvidia/Qwen3.6-35B-A3B-NVFP4); under the
    /// default chunkwise verify, like any change in acceptance (--num-drafts included), it can
    /// move greedy text. Refused on any other head (dense FFN, FP8 experts, or
    /// --mtp-quantization other than bf16).
    #[arg(long, default_value_t = false)]
    pub mtp_experts_nvfp4: bool,
}
