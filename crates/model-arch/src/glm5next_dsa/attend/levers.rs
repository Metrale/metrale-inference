// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The environment levers of the GLM-5.3 DSA attention launcher, each read once.
//! Moved out of `attend.rs` unchanged (that file passed 500 lines), plus
//! `METRALE_GLM_DSA_HB_PREFILL_PIPE` and `METRALE_GLM_DSA_HB_PREFILL_TC`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: a malformed value is an error at resolve time, never a silent default.

use anyhow::Result;

use super::HB_MAX_SPLITS;

/// 2026-10-09: `METRALE_GLM_DSA_DECODE_HB=N` (1..=[`HB_MAX_SPLITS`]): decode the DSA attention
/// on the head-batched kernel, each row's selection split into `N` slices merged afterwards.
/// Unset keeps the per-head kernel. Opt-in: it changes the summation order (accuracy-gated).
/// Read once; any other value is an error at resolve time.
pub fn hb_splits_lever() -> Result<Option<usize>> {
    static V: std::sync::OnceLock<Result<Option<usize>, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| parse_hb_splits(std::env::var("METRALE_GLM_DSA_DECODE_HB").ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 2026-10-12: `METRALE_GLM_DSA_HB_PREFILL_ONE_SPLIT=1`: a prefill sub-chunk's head-batched
/// attend runs at one split (each CTA walks its row's whole selection, writing BF16 directly: no
/// partials, no merge) in a single launch over all its rows, which already fill the GPU. Decode
/// keeps `METRALE_GLM_DSA_DECODE_HB`'s split count. Off unless `1`; read once.
pub fn hb_prefill_one_split() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("METRALE_GLM_DSA_HB_PREFILL_ONE_SPLIT").as_deref() == Ok("1"))
}

/// 2026-10-12: `METRALE_GLM_DSA_HB_PREFILL_P_HI`, read once: `1` runs a prefill sub-chunk's
/// head-batched attend on `glm5next_dsa_mla_decode_hb_fp8_phi`, which carries the softmax
/// weights as one bf16 instead of bf16 hi + lo and so runs two thirds of the MMAs; unset or `0`
/// keeps the hi + lo entry; another value is an error. Decode is never affected. Opt-in and
/// accuracy-gated: a weight's relative error grows from 2^-16 to 2^-9.
pub fn hb_prefill_p_hi() -> Result<bool> {
    static V: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        parse_switch(
            "METRALE_GLM_DSA_HB_PREFILL_P_HI",
            std::env::var("METRALE_GLM_DSA_HB_PREFILL_P_HI")
                .ok()
                .as_deref(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-14: `METRALE_GLM_DSA_HB_PREFILL_PIPE`, read once: `1` runs the P-hi prefill attend
/// (`METRALE_GLM_DSA_HB_PREFILL_P_HI=1`) on `glm5next_dsa_mla_prefill_pipe_phi`, the same
/// arithmetic rescheduled (double-buffered tile, QK on all eight warps, two barriers per tile
/// instead of four): byte-identical output per row, so no new numerics point. Unset or `0` keeps
/// `glm5next_dsa_mla_decode_hb_fp8_phi`; another value is an error. Decode is never affected.
pub fn hb_prefill_pipe() -> Result<bool> {
    static V: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        parse_switch(
            "METRALE_GLM_DSA_HB_PREFILL_PIPE",
            std::env::var("METRALE_GLM_DSA_HB_PREFILL_PIPE")
                .ok()
                .as_deref(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-14: `METRALE_GLM_DSA_HB_PREFILL_TC`, read once: `1` runs the P-hi prefill attend
/// (`METRALE_GLM_DSA_HB_PREFILL_P_HI=1`, one split) on `glm5next_dsa_mla_prefill_tc_phi`, the same
/// arithmetic on a warp-specialized CTA per (row, head group) (producer warps gather and widen the
/// next tiles while consumer warps run QK, softmax and PV): byte-identical output per row, so no
/// new numerics point. Unset or `0` keeps the P-hi entry; another value is an error; exclusive
/// with `METRALE_GLM_DSA_HB_PREFILL_PIPE`. Decode is never affected.
pub fn hb_prefill_tc() -> Result<bool> {
    static V: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        parse_switch(
            "METRALE_GLM_DSA_HB_PREFILL_TC",
            std::env::var("METRALE_GLM_DSA_HB_PREFILL_TC")
                .ok()
                .as_deref(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-14: The readback of the prefill attend levers, logged once per process after the first
/// prefill sub-chunk's attend resolved them without error, so every rank's serve log shows which
/// entry its prefill ran (each set lever is refused unless it was swapped in).
pub(super) fn readback_prefill_levers(one_split: bool, p_hi: bool, pipe: bool, tc: bool) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(
            "DSA prefill attend levers: one_split={} p_hi={} pipe={} tc={}",
            u8::from(one_split),
            u8::from(p_hi),
            u8::from(pipe),
            u8::from(tc)
        )
    });
}

/// 2026-10-14: A `0`/`1` switch lever's parse, pure: unset or `0` off, `1` on, else an error
/// naming `env`.
pub(super) fn parse_switch(env: &str, v: Option<&str>) -> Result<bool, String> {
    match v {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{env}={other:?}: expected 0 or 1")),
    }
}

/// 2026-10-09: [`hb_splits_lever`]'s parse, pure.
pub(super) fn parse_hb_splits(v: Option<&str>) -> Result<Option<usize>, String> {
    let Some(v) = v else { return Ok(None) };
    match v.trim().parse::<usize>() {
        Ok(n) if (1..=HB_MAX_SPLITS).contains(&n) => Ok(Some(n)),
        _ => Err(format!(
            "METRALE_GLM_DSA_DECODE_HB={v:?}: expected a split count in 1..={HB_MAX_SPLITS}"
        )),
    }
}
