// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `METRALE_GLM_MHC_PRE_SLICE=<rows>`: an mHC site's pre-mix and norm over more than
//! `rows` rows launches per slice of at most `rows` rows instead of once over all of them.
//!
//! The pre-mix blocks of one token each re-read that token's FP32 stream row (64 KB at H 4096,
//! hc 4); over a 2048-row prefill sub-chunk the rows no longer stay in L2 between those reads,
//! over 256 they do. Every pre-mix kernel computes each token from that token's row alone and
//! indexes its `mix` and `done` scratch from the launch's first token, and a slice's launches
//! finish before the next slice's start (one stream), so every output byte is the unsliced
//! launch's; the kernel choice by row count (per-row grid below 64 rows) keeps the bits too
//! (`glm5next_mhc_fused`, `glm5next_mhc`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Unset or `0`: one call over all rows, as before.

use super::*;

/// 2026-10-10: `METRALE_GLM_MHC_PRE_SLICE`, read once: `None` when unset or `0`, else the most
/// rows one pre-mix launch takes. Unparsable is an error.
pub(crate) fn mhc_pre_slice() -> Result<Option<usize>> {
    static V: std::sync::OnceLock<Result<Option<usize>, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| parse_slice(std::env::var("METRALE_GLM_MHC_PRE_SLICE").ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 2026-10-10: [`mhc_pre_slice`]'s parse, pure.
fn parse_slice(v: Option<&str>) -> Result<Option<usize>, String> {
    match v {
        None | Some("0") => Ok(None),
        Some(s) => s
            .parse::<usize>()
            .map(Some)
            .map_err(|_| format!("METRALE_GLM_MHC_PRE_SLICE={s:?}: expected a row count")),
    }
}

/// 2026-10-10: The `(start, rows)` launches of `k` rows under the slice `slice`.
fn slices(k: usize, slice: Option<usize>) -> Vec<(usize, usize)> {
    match slice {
        Some(s) if k > s => crate::glm5next_layer::multi_seq_chunks(k, s),
        _ => vec![(0, k)],
    }
}

impl Glm5NextLayer {
    /// 2026-10-10: [`Self::mhc_pre_norm_one`] per slice of `mhc_pre_slice()` rows: row `r` of the
    /// streams, `y`, `post`, `comb` and `normed` is row `r - start` of its slice's call. The
    /// caller's `profile::MHC` span ends with the last slice.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn mhc_pre_norm(
        &self,
        gpu: &dyn GpuBackend,
        mhc: &Glm5NextMhc,
        site: &Glm5NextMhcSiteWeights,
        norm_w: DevicePtr,
        streams: DevicePtr,
        y: DevicePtr,
        post: DevicePtr,
        comb: DevicePtr,
        normed: DevicePtr,
        k: usize,
        t_mhc: Option<std::time::Instant>,
        stream: u64,
    ) -> Result<()> {
        let parts = slices(k, mhc_pre_slice()?);
        let (h, hc) = (self.hidden, mhc.hc_mult);
        let last = parts.len() - 1;
        for (i, &(r0, n)) in parts.iter().enumerate() {
            self.mhc_pre_norm_one(
                gpu,
                mhc,
                site,
                norm_w,
                streams.offset(r0 * hc * h * 4),
                y.offset(r0 * h * 2),
                post.offset(r0 * hc * 4),
                comb.offset(r0 * hc * hc * 4),
                normed.offset(r0 * h * 2),
                n,
                if i == last { t_mhc } else { None },
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-10: Unset, `0` or a row count no wider than the launch: one launch over all rows;
    /// wider: consecutive slices of the slice width, the last one shorter.
    #[test]
    fn slices_cover_the_rows_in_order() {
        assert_eq!(slices(2048, None), vec![(0, 2048)]);
        assert_eq!(slices(200, Some(256)), vec![(0, 200)]);
        assert_eq!(slices(256, Some(256)), vec![(0, 256)]);
        assert_eq!(
            slices(600, Some(256)),
            vec![(0, 256), (256, 256), (512, 88)]
        );
        assert_eq!(parse_slice(None), Ok(None));
        assert_eq!(parse_slice(Some("0")), Ok(None));
        assert_eq!(parse_slice(Some("256")), Ok(Some(256)));
        assert!(parse_slice(Some("x")).is_err());
    }
}
