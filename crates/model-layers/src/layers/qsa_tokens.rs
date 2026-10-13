// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: How many tokens a QSA indexer sizes its per-sequence key buffers
//! for. The single source is the serve's `--max-seq-len`
//! (`ModelConfig::serve_max_seq_len`); the `METRALE_QSA_MAX_TOKENS` lever may
//! restate it, never override it, and a tool without a serve may set it alone.
//!
//! Owner: model-layers (QSA).
//! Invariants:
//! - A set lever that disagrees with a set `--max-seq-len` is an error, never
//!   a silent choice.
//! - Neither value set is an error: there is no built-in default.

use anyhow::{Context, Result, bail};

/// 2026-10-13: The indexer's `max_tokens` from the serve's `--max-seq-len`
/// (`0` when there is no serve) and the `METRALE_QSA_MAX_TOKENS` lever's text.
pub fn qsa_max_tokens(serve_max_seq_len: usize, lever: Option<&str>) -> Result<usize> {
    let lever = lever
        .map(|v| {
            v.trim()
                .parse::<usize>()
                .with_context(|| format!("METRALE_QSA_MAX_TOKENS={v:?} is not a token count"))
        })
        .transpose()?;
    match (serve_max_seq_len, lever) {
        (0, None) => bail!(
            "QSA: no --max-seq-len to size the indexer's per-sequence keys from, and \
             METRALE_QSA_MAX_TOKENS is unset"
        ),
        (0, Some(v)) | (_, Some(v)) if v == 0 => {
            bail!("QSA: a token count of 0 cannot hold a sequence")
        }
        (0, Some(v)) => Ok(v),
        (serve, None) => Ok(serve),
        (serve, Some(v)) if v == serve => Ok(v),
        (serve, Some(v)) => bail!(
            "QSA: METRALE_QSA_MAX_TOKENS={v} disagrees with --max-seq-len {serve}; the \
             indexer is sized from --max-seq-len, so unset the lever or set it equal"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::qsa_max_tokens;

    #[test]
    fn the_serve_value_is_the_source() {
        assert_eq!(qsa_max_tokens(36_864, None).unwrap(), 36_864);
        assert_eq!(qsa_max_tokens(36_864, Some("36864")).unwrap(), 36_864);
        assert_eq!(qsa_max_tokens(36_864, Some(" 36864 ")).unwrap(), 36_864);
    }

    #[test]
    fn a_disagreeing_lever_fails_fast() {
        let err = qsa_max_tokens(36_864, Some("32768"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("32768") && err.contains("36864"), "{err}");
        // Control: the same lever agrees with a matching serve value.
        assert_eq!(qsa_max_tokens(32_768, Some("32768")).unwrap(), 32_768);
    }

    #[test]
    fn without_a_serve_the_lever_alone_sizes_it_and_nothing_is_a_default() {
        assert_eq!(qsa_max_tokens(0, Some("4096")).unwrap(), 4096);
        assert!(qsa_max_tokens(0, None).is_err());
    }

    #[test]
    fn garbage_and_zero_are_refused() {
        assert!(qsa_max_tokens(36_864, Some("lots")).is_err());
        assert!(qsa_max_tokens(0, Some("0")).is_err());
        assert!(qsa_max_tokens(36_864, Some("0")).is_err());
    }
}
