// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_PREFILL_PIECE_ROWS=P` (unset: off): a prefill sub-chunk of more than
//! `P` rows keeps the launches whose arithmetic depends on the row count at `P`-row pieces, so a
//! serve whose chunk is a multiple of `P` (`--max-prefill-tokens 16384` with `P = 8192`) computes
//! every row with the bits of a serve chunked at `P`, while the row-invariant launches (the mHC
//! and norm kernels, the top-k, the grouped routed experts, the combine) run once over the whole
//! sub-chunk, so the routed-expert weights are read once per sub-chunk instead of once per `P`.
//!
//! Per piece (cut from the sub-chunk's first row, the last one shorter): the mixer (the KDA scan,
//! anchored at its call start, and DSA with its cuBLASLt projections), the router GEMM, the
//! shared expert, a dense MLP site, and the non-SP all-reduces (so each NCCL collective has the
//! size a `P`-row chunk gives it). Pieces share a call only when the call's routed path and
//! SP-mHC decision for the whole equal those for each piece ([`plan_calls`]); any other piece runs
//! alone as the `P`-chunked serve runs it.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `P` is a multiple of [`PIECE_ALIGN`], so a piece boundary is a boundary of the KDA chunked
//!   scan's 256-row slabs, whose state carries in FP32 either way.
//! - [`plan_calls`] covers `0..k` once, in order, with every piece `P` rows except the last.
//! - At load ([`check_piece_budget`]) the scheduler's chunk is a multiple of `P` wider than `P`,
//!   and `GlmLevers::prefill_rows` covers the highway, so the layer never re-splits a chunk off the grid.
//! - 2026-10-17: `P` is read from the startup settings ([`piece_rows`]), never the process
//!   environment.

use anyhow::Result;

/// 2026-10-14: The env lever's name.
pub const PIECE_ROWS_ENV: &str = "METRALE_GLM_PREFILL_PIECE_ROWS";

/// 2026-10-14: `P` must be a multiple of this: the KDA chunked scan's slab (`glm5next_kda`), so a
/// piece boundary lands where the `P`-chunked serve's call boundary does in the scan's grid.
pub const PIECE_ALIGN: usize = 256;

/// 2026-10-17: [`PIECE_ROWS_ENV`] of the startup settings: unset is off (`None`); otherwise a
/// positive multiple of [`PIECE_ALIGN`], anything else an error.
pub fn piece_rows(s: &metrale_settings::InferenceServerSettings) -> Result<Option<usize>> {
    checked(s.scheduler.glm_prefill_piece_rows).map_err(anyhow::Error::msg)
}

/// 2026-10-17: [`piece_rows`]'s check of the resolved value, pure.
fn checked(v: Option<i64>) -> Result<Option<usize>, String> {
    let Some(p) = v else {
        return Ok(None);
    };
    match usize::try_from(p) {
        Ok(rows) if rows != 0 && rows.is_multiple_of(PIECE_ALIGN) => Ok(Some(rows)),
        _ => Err(format!(
            "{PIECE_ROWS_ENV}={p}: must be a positive multiple of {PIECE_ALIGN} (the KDA scan's \
             slab), or unset"
        )),
    }
}

/// 2026-10-14: The calls a `k`-row sub-chunk runs as, each a list of piece row counts: `k` cut
/// into `p`-row pieces from row 0 (the last shorter), consecutive pieces grouped while
/// `share(piece)` and `share(call total)` are both `Some` and equal to `share` of the call's
/// first piece. A piece whose `share` is `None` runs as a call of its own. `share` names the
/// row-count-dependent decisions a call makes (its routed path, SP-mHC); an error from it is
/// returned unchanged. A `p` of 0 is an error.
pub fn plan_calls<S: PartialEq>(
    k: usize,
    p: usize,
    mut share: impl FnMut(usize) -> Result<Option<S>>,
) -> Result<Vec<Vec<usize>>> {
    anyhow::ensure!(p > 0, "prefill piece of 0 rows for {k} rows");
    let mut calls: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_sig: Option<S> = None;
    let mut t = 0usize;
    while t < k {
        let n = p.min(k - t);
        t += n;
        let sig = share(n)?;
        let joins = match (&cur_sig, &sig) {
            (Some(first), Some(own)) if !cur.is_empty() && own == first => {
                share(cur.iter().sum::<usize>() + n)?.as_ref() == Some(first)
            }
            _ => false,
        };
        if joins {
            cur.push(n);
            continue;
        }
        if !cur.is_empty() {
            calls.push(std::mem::take(&mut cur));
        }
        match sig {
            Some(s) => {
                cur.push(n);
                cur_sig = Some(s);
            }
            None => {
                calls.push(vec![n]);
                cur_sig = None;
            }
        }
    }
    if !cur.is_empty() {
        calls.push(cur);
    }
    Ok(calls)
}

/// 2026-10-14: The load-time check of piece mode against the serve's prefill sizing: the
/// scheduler's chunk (`prefill_budget`) a multiple of `piece` and wider than it (otherwise the
/// lever buys nothing, or cuts a chunk off the `P` grid), and `prefill_rows` at least the
/// highway (`max_batch_tokens`), so the layer runs each chunk in one sub-chunk.
pub fn check_piece_budget(
    piece: usize,
    prefill_rows: usize,
    prefill_budget: usize,
    max_batch_tokens: usize,
) -> Result<(), String> {
    if prefill_budget <= piece || !prefill_budget.is_multiple_of(piece) {
        return Err(format!(
            "{PIECE_ROWS_ENV}={piece}: the prefill budget ({prefill_budget}, \
             --max-prefill-tokens) must be a multiple of the piece wider than it"
        ));
    }
    if prefill_rows < max_batch_tokens {
        return Err(format!(
            "{PIECE_ROWS_ENV}={piece}: METRALE_GLM_PREFILL_ROWS ({prefill_rows}) must cover the \
             {max_batch_tokens}-token batch, or the layer re-splits each chunk"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm5next_prefill_piece_tests.rs"]
mod tests;
