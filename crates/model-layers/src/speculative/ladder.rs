// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-step MTP draft count as a function of the number of active
//! sequences (the K-vs-batch ladder), and the multi-sequence MTP width cap.
//!
//! Owner: model-layers (speculative).
//! Invariants:
//! - `mtp_ladder_drafts` returns 0 when `num_drafts` is 0, and a value in
//!   `1..=num_drafts` otherwise.
//! - The ladder, its kill switch and the cap are read from the environment
//!   once per process.
//!
//! The default ladder is `4:3,8:3,16:1,32:1`: three drafts up to 8 active
//! sequences, one draft above that. A width above the last step takes the last
//! step's count. `metrale_speculative::adaptive_rung` uses this count as its
//! floor and may raise widths 9..=16 to two drafts; it keeps the static count
//! when `METRALE_MTP_STATIC_RUNG` or `METRALE_MTP_K_LADDER` is present.
//!
//! Overrides:
//! - `METRALE_MTP_K_LADDER`, e.g. `"4:3,8:2,16:1"`: comma-separated
//!   `n_max:drafts` steps, sorted by `n_max`. A value that does not parse as a
//!   whole gives the default ladder. Draft counts are clamped to
//!   `1..=num_drafts`.
//! - `METRALE_NO_MTP_K_LADDER`, present with any value (`0` included): every
//!   width gets `num_drafts`, and the default of [`mtp_max_seqs`] drops from
//!   32 to 4.

/// 2026-09-25: Whether `METRALE_NO_MTP_K_LADDER` is present, with any value.
/// Read once per process.
pub fn mtp_ladder_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("METRALE_NO_MTP_K_LADDER").is_some())
}

/// 2026-09-25: Parses `n_max:drafts,...` into steps sorted by `n_max`. `None`
/// when any step fails to parse; the caller then uses the default ladder.
fn parse_ladder(value: &str) -> Option<Vec<(usize, usize)>> {
    let mut steps = Vec::new();
    for part in value.split(',') {
        let (n, k) = part.trim().split_once(':')?;
        steps.push((n.trim().parse().ok()?, k.trim().parse().ok()?));
    }
    if steps.is_empty() {
        return None;
    }
    steps.sort_by_key(|&(n, _)| n);
    Some(steps)
}

fn mtp_ladder_steps() -> &'static [(usize, usize)] {
    static STEPS: std::sync::OnceLock<Vec<(usize, usize)>> = std::sync::OnceLock::new();
    STEPS.get_or_init(|| {
        let parsed = std::env::var("METRALE_MTP_K_LADDER")
            .ok()
            .and_then(|value| parse_ladder(&value));
        // 2026-09-25: The default ladder. Measured 2026-07-31 on dgx1 at C=16,
        // one binary, three reps per arm: 16:1 181.63 tok/s, 16:2 172.70,
        // 16:3 152.97.
        parsed.unwrap_or_else(|| vec![(4, 3), (8, 3), (16, 1), (32, 1)])
    })
}

fn ladder_drafts_from_steps(steps: &[(usize, usize)], n_active: usize, num_drafts: usize) -> usize {
    if num_drafts == 0 {
        return 0;
    }
    steps
        .iter()
        .find(|&&(n_max, _)| n_active <= n_max)
        .or(steps.last())
        .map(|&(_, k)| k.clamp(1, num_drafts))
        .unwrap_or(num_drafts)
}

/// 2026-09-25: The per-step draft count for `n_active` concurrent sequences.
///
/// `num_drafts` is the configured ceiling. The result is in `1..=num_drafts`,
/// or 0 when `num_drafts` is 0. With the ladder disabled it is `num_drafts`.
/// A width above the last step uses the last step's count.
pub fn mtp_ladder_drafts(n_active: usize, num_drafts: usize) -> usize {
    if num_drafts == 0 {
        return 0;
    }
    if mtp_ladder_disabled() {
        return num_drafts;
    }
    ladder_drafts_from_steps(mtp_ladder_steps(), n_active, num_drafts)
}

/// 2026-09-29: Engine default of the multi-sequence MTP dispatch cap, when neither
/// `--mtp-max-seqs` nor MODEL.toml `[behavior] mtp_max_seqs` gives one. 4 when
/// `METRALE_NO_MTP_K_LADDER` is present.
pub const DEFAULT_MTP_MAX_SEQS: usize = 32;

/// 2026-09-29: The widest cap a serve accepts: one batched verify's sequence width
/// (`layer::VERIFY_WY_TABLE_SEQS`), which is also the serve's widest batch.
pub const MAX_MTP_MAX_SEQS: usize = crate::layer::VERIFY_WY_TABLE_SEQS;

/// 2026-09-29: Where the serve's dispatch cap came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpMaxSeqsSource {
    Cli,
    ModelDefault,
    EngineDefault,
}

/// 2026-09-29: Resolve the multi-sequence MTP dispatch cap: `--mtp-max-seqs` when given, else
/// MODEL.toml `[behavior] mtp_max_seqs` when above 0, else the engine default
/// ([`DEFAULT_MTP_MAX_SEQS`], 4 under `METRALE_NO_MTP_K_LADDER`). A cap outside
/// `1..=MAX_MTP_MAX_SEQS` is refused, from either source.
pub fn resolve_mtp_max_seqs(
    cli: Option<usize>,
    model_default: u32,
    ladder_disabled: bool,
) -> anyhow::Result<(usize, MtpMaxSeqsSource)> {
    let (n, source) = match (cli, model_default) {
        (Some(n), _) => (n, MtpMaxSeqsSource::Cli),
        (None, m) if m > 0 => (m as usize, MtpMaxSeqsSource::ModelDefault),
        (None, _) => (
            if ladder_disabled {
                4
            } else {
                DEFAULT_MTP_MAX_SEQS
            },
            MtpMaxSeqsSource::EngineDefault,
        ),
    };
    anyhow::ensure!(
        (1..=MAX_MTP_MAX_SEQS).contains(&n),
        "MTP dispatch cap {n} ({source:?}) is outside 1..={MAX_MTP_MAX_SEQS}"
    );
    Ok((n, source))
}

static PUBLISHED_MTP_MAX_SEQS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// 2026-09-29: Publish the serve's dispatch cap once, before the model sizes its verify pools
/// and the scheduler reads its levers. A second publication of a different value is refused.
pub fn set_mtp_max_seqs(n: usize) -> anyhow::Result<()> {
    let got = *PUBLISHED_MTP_MAX_SEQS.get_or_init(|| n);
    anyhow::ensure!(
        got == n,
        "MTP dispatch cap already published as {got}, refusing {n}"
    );
    Ok(())
}

/// 2026-09-29: The multi-sequence MTP dispatch cap the serve published
/// ([`set_mtp_max_seqs`]); before publication (unit tests, tools) the engine default.
///
/// The scheduler runs a speculative step only while the active count is at
/// most this (`SchedLevers::mtp_max_seqs`, checked in `lane_decode.rs`), and
/// the model sizes its verify pools and its single-sequence MTP structures
/// from the same value (`ssm_reserve::mtp_state_slots`,
/// `speculative::mtp_multi_seq_mode`).
pub fn mtp_max_seqs() -> usize {
    PUBLISHED_MTP_MAX_SEQS
        .get()
        .copied()
        .unwrap_or(if mtp_ladder_disabled() {
            4
        } else {
            DEFAULT_MTP_MAX_SEQS
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2026-09-25: Assumes the test process sets neither `METRALE_MTP_K_LADDER`
    // nor `METRALE_NO_MTP_K_LADDER`, which are latched once per process.
    #[test]
    fn default_ladder_holds_depth_to_the_cap() {
        assert_eq!(mtp_ladder_drafts(1, 3), 3);
        assert_eq!(mtp_ladder_drafts(4, 3), 3);
        assert_eq!(mtp_ladder_drafts(5, 3), 3);
        assert_eq!(mtp_ladder_drafts(8, 3), 3);
        assert_eq!(mtp_ladder_drafts(9, 3), 1);
        assert_eq!(mtp_ladder_drafts(16, 3), 1);
        assert_eq!(mtp_ladder_drafts(17, 3), 1);
        assert_eq!(mtp_ladder_drafts(32, 3), 1);
        assert_eq!(mtp_ladder_drafts(64, 3), 1);
        assert_eq!(mtp_ladder_drafts(16, 1), 1);
    }

    #[test]
    fn depth_at_width_env_rungs_parse_shape() {
        let steps = parse_ladder("32:2, 4:3,8:3, 16:2,24:2").unwrap();
        assert_eq!(steps, [(4, 3), (8, 3), (16, 2), (24, 2), (32, 2)]);
        assert_eq!(ladder_drafts_from_steps(&steps, 17, 3), 2);
        assert_eq!(ladder_drafts_from_steps(&steps, 24, 3), 2);
        assert_eq!(ladder_drafts_from_steps(&steps, 25, 3), 2);
        assert_eq!(ladder_drafts_from_steps(&steps, 32, 3), 2);
    }

    /// 2026-09-29: The flag beats MODEL.toml, MODEL.toml beats the engine default, 0 in
    /// MODEL.toml means unset, and a cap outside 1..=128 is refused from either source.
    #[test]
    fn mtp_max_seqs_resolution_order_and_bounds() {
        use MtpMaxSeqsSource::*;
        assert_eq!(resolve_mtp_max_seqs(Some(8), 128, false).unwrap(), (8, Cli));
        assert_eq!(
            resolve_mtp_max_seqs(None, 128, false).unwrap(),
            (128, ModelDefault)
        );
        assert_eq!(
            resolve_mtp_max_seqs(None, 0, false).unwrap(),
            (32, EngineDefault)
        );
        assert_eq!(
            resolve_mtp_max_seqs(None, 0, true).unwrap(),
            (4, EngineDefault)
        );
        assert_eq!(
            resolve_mtp_max_seqs(Some(32), 128, false).unwrap(),
            (32, Cli)
        );
        assert!(resolve_mtp_max_seqs(Some(0), 128, false).is_err());
        assert!(resolve_mtp_max_seqs(Some(129), 0, false).is_err());
        assert!(resolve_mtp_max_seqs(None, 129, false).is_err());
        assert_eq!(MAX_MTP_MAX_SEQS, 128);
    }

    #[test]
    fn explicit_steps_are_honored() {
        let steps = parse_ladder("4:3,8:2").unwrap();
        assert_eq!(ladder_drafts_from_steps(&steps, 4, 3), 3);
        assert_eq!(ladder_drafts_from_steps(&steps, 5, 3), 2);
        assert_eq!(ladder_drafts_from_steps(&steps, 8, 3), 2);
        assert_eq!(ladder_drafts_from_steps(&steps, 9, 3), 2);
    }

    #[test]
    fn malformed_ladder_is_rejected_as_a_unit() {
        assert_eq!(parse_ladder(""), None);
        assert_eq!(parse_ladder("4:3,broken,8:2"), None);
        assert_eq!(parse_ladder("4:three"), None);
    }

    #[test]
    fn ladder_clamps_to_configured_ceiling() {
        assert_eq!(mtp_ladder_drafts(2, 1), 1);
        assert_eq!(mtp_ladder_drafts(2, 0), 0);
    }
}
