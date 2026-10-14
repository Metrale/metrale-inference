// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `--require-kv-tokens`: refuse a boot whose KV pool holds fewer tokens than the
//! run asks for, instead of failing a long prompt later with "KV cache exhausted".
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - Without the flag the check passes and logs nothing; it has no default.

use anyhow::{Result, bail};

/// 2026-10-14: Fails when `required` is set and the pool, `blocks` of `block_size` tokens
/// (`None`: no paged pool), holds fewer tokens. `block_bytes` is one block across every layer,
/// K and V, for the shortfall in bytes.
pub(crate) fn require_kv_tokens(
    required: Option<usize>,
    blocks: usize,
    block_size: Option<usize>,
    block_bytes: usize,
) -> Result<()> {
    let Some(required) = required else {
        return Ok(());
    };
    let Some(block_size) = block_size else {
        bail!("--require-kv-tokens {required}: this model has no paged KV pool to check");
    };
    let tokens = blocks.saturating_mul(block_size);
    if tokens < required {
        let short_blocks = (required - tokens).div_ceil(block_size.max(1));
        bail!(
            "--require-kv-tokens {required}: the KV pool holds {tokens} tokens ({blocks} blocks \
             x {block_size} tokens, {:.2} GiB), {} tokens short ({short_blocks} blocks, \
             {:.2} GiB). The \"KV budget\" and \"KV cache:\" lines above give the accounting \
             that sized the pool.",
            gib(blocks.saturating_mul(block_bytes)),
            required - tokens,
            gib(short_blocks.saturating_mul(block_bytes)),
        );
    }
    tracing::info!(target: "met::main_modules::serve_load", "--require-kv-tokens {required}: the KV pool holds {tokens} tokens ({blocks} blocks x \
         {block_size}), {} to spare",
        tokens - required,
    );
    Ok(())
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
#[path = "kv_require_tests.rs"]
mod tests;
