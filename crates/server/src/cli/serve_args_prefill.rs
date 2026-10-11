// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The batched-prefill serve flags (`--prefill-varlen-batch`,
//! `--prefill-codispatch`, `--prefill-varlen-with-decode`), flattened into
//! `ServeArgs` where the first two stood, so `--help` keeps their order.
//!
//! Owner: server CLI.
//! Invariants: the `///` text on the struct's fields is the `--help` output and
//! carries no date.

use clap::Args;

#[derive(Args, Debug, Clone, PartialEq)]
pub struct ServePrefillBatchArgs {
    /// Varlen (ragged) batched prefill, opt-in (default: off).
    ///
    /// Concurrently queued prompts of different lengths are prefilled together,
    /// one forward per wave, so each projection GEMM launches once over the wave's
    /// summed tokens instead of once per request. The scheduler defers chunk 0 of
    /// new requests so they can join a wave, and a wave holds at most
    /// min(`--max-prefill-tokens`, the max batch tokens) tokens. With
    /// `--prefill-codispatch` also set, this path is used.
    ///
    /// Batching changes GEMM row counts, and kernels are selected on row count, so
    /// per-request outputs can differ from the serial path.
    ///
    /// Environment fallback: `METRALE_PREFILL_VARLEN=1` (or `true`) turns it on
    /// when this flag is absent; a given flag wins over it.
    #[arg(long)]
    pub prefill_varlen_batch: bool,

    /// Co-dispatch fresh prompts: when >=2 requests without images are admitted
    /// together with nothing decoding or prefilling, defer their chunk-0 prefill so
    /// they batch into one forward.
    ///
    /// Only with chunked prefill, and not on an expert-parallel model.
    ///
    /// Environment fallback: `METRALE_PREFILL_CODISPATCH=1` (or `true`) turns it on
    /// when this flag is absent.
    #[arg(long)]
    pub prefill_codispatch: bool,

    /// Varlen batched prefill while sequences are decoding, opt-in (default:
    /// off). Needs the varlen batched prefill on (`--prefill-varlen-batch` or its
    /// environment fallback); the serve refuses to start without it.
    ///
    /// Without this flag the varlen waves form only when nothing is decoding: a
    /// prompt that arrives while others decode runs its first chunk alone, one
    /// request after another, so a burst of N prompts waits N prefills. With it,
    /// two or more such prompts (or one, while another is still prefilling) defer
    /// their first chunk and run as varlen waves, each its own forward, ahead of
    /// the tick's decode step, which keeps its speculative step.
    ///
    /// Not on an expert-parallel model, and not with `METRALE_HOLO_ALWAYS_MIXED`,
    /// which fuses prefill into decode instead. Batching changes GEMM row counts,
    /// so per-request outputs can differ from the serial path, as for
    /// `--prefill-varlen-batch`.
    #[arg(long)]
    pub prefill_varlen_with_decode: bool,
}
