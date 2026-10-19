// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: What the DeepSeek-V4.1 safetensors loader reads itself, and what each tensor costs
//! the device, for the load's pre-flight. The loader reads every tensor of the checkpoint itself
//! (each rank only its part), so the checkpoint loader uploads nothing and records every
//! location.
//!
//! The charge of a tensor is the device bytes the bind allocates for it ([`charge`]): a rank's
//! kept part of a per-layer projection, routed expert table or scale plane in its stored
//! precision; twice the stored bytes for the engram projection, decoded to bf16; the whole
//! embedding, head and final norm. Tensors read on the host per step (the engram row tables) or
//! never read (the MTP head, the vision tower, the aligner, the image markers, the vision-language
//! router bias) charge nothing. The f32 widening of the norms, a few MiB, is not charged.
//!
//! Owner: model-arch weight loader.
//! Invariants: a tensor name is matched only against [`name`], [`expert_name`]'s tables and
//! [`scale_of`]; nothing here spells a per-layer tensor name.

use std::sync::Arc;

use metrale_config::ModelConfig;
use metrale_model_weights::weights::{DeferCharge, DeferHook, WeightDtype};

use super::source::{EXPERT_TABLES, Format, ROLES, T, name, scale_of};
use super::tp_cut::{BLOCK, Cut, Local, cut_of, expert_cut, read_plan};

/// 2026-10-19: The top-level tensors the loader uploads whole.
pub const WHOLE: [&str; 3] = ["embed.weight", "head.weight", "norm.weight"];

/// 2026-10-19: Where a checkpoint tensor goes at bind time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// 2026-10-19: Uploaded whole.
    Whole,
    /// 2026-10-19: Routed expert table `table` ([`EXPERT_TABLES`] order), this rank's part.
    Expert(usize),
    /// 2026-10-19: Per-layer role `t`, this rank's part; `scale`: its E8M0 tile plane.
    Role { t: T, scale: bool },
    /// 2026-10-19: Read on the host, or never read.
    Host,
}

/// 2026-10-19: Where `n` goes.
pub fn placement(n: &str) -> Placement {
    if WHOLE.contains(&n) {
        return Placement::Whole;
    }
    let Some((l, rest)) = n
        .strip_prefix("layers.")
        .and_then(|r| r.split_once('.'))
        .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
    else {
        return Placement::Host;
    };
    if let Some((e, table)) = rest
        .strip_prefix("ffn.experts.")
        .and_then(|r| r.split_once('.'))
    {
        return match (
            e.parse::<usize>(),
            EXPERT_TABLES.iter().position(|&x| x == table),
        ) {
            (Ok(_), Some(t)) => Placement::Expert(t),
            _ => Placement::Host,
        };
    }
    for t in ROLES {
        let w = name(Format::Safetensors, l, t);
        if w == n {
            return Placement::Role { t, scale: false };
        }
        if scale_of(&w).is_ok_and(|s| s == n) {
            return Placement::Role { t, scale: true };
        }
    }
    Placement::Host
}

/// 2026-10-19: The bytes of a `shape` tensor of `bytes` stored bytes that `cut` keeps; the whole
/// tensor when the cut does not fit it (the load refuses that tensor itself).
fn kept(shape: &[usize], bytes: usize, cut: &Cut) -> usize {
    match read_plan(shape, bytes, cut) {
        Ok(p) => match p.cols {
            Some(c) => p.len / p.row_bytes * c.len(),
            None => p.len,
        },
        Err(_) => bytes,
    }
}

/// 2026-10-19: The device bytes `local`'s rank allocates for the tensor `n`.
pub fn charge(local: &Local, n: &str, dtype: WeightDtype, shape: &[usize]) -> usize {
    let Some(bytes) = shape
        .iter()
        .try_fold(dtype.byte_size(), |a, &d| a.checked_mul(d))
    else {
        return usize::MAX;
    };
    match placement(n) {
        Placement::Whole => bytes,
        Placement::Host => 0,
        Placement::Expert(table) => match expert_cut(table, local) {
            Ok(cut) => kept(shape, bytes, &cut),
            Err(_) => bytes,
        },
        Placement::Role {
            t: T::EngramWkv,
            scale,
        } => match scale {
            true => 0,
            false => bytes.saturating_mul(2),
        },
        Placement::Role { t, scale: false } => kept(shape, bytes, &cut_of(t, local)),
        Placement::Role { t, scale: true } => match cut_of(t, local).div(BLOCK) {
            Ok(cut) => kept(shape, bytes, &cut),
            Err(_) => bytes,
        },
    }
}

/// 2026-10-19: Every tensor: the loader reads the whole checkpoint itself.
pub fn defer_hook() -> DeferHook {
    Arc::new(|_: &str, _: WeightDtype| true)
}

/// 2026-10-19: [`charge`] for `config`'s rank. A config whose rank split fails is charged whole
/// here, the larger charge; the load itself then refuses it with the split's error.
pub fn defer_charge(config: &ModelConfig) -> DeferCharge {
    let local = Local::of(config).unwrap_or_else(|e| {
        tracing::warn!("deepseek-v4.1: pre-flight charges every tensor whole: {e:#}");
        Local::whole(config)
    });
    Arc::new(move |n: &str, dtype: WeightDtype, shape: &[usize]| charge(&local, n, dtype, shape))
}

#[cfg(test)]
#[path = "defer_tests.rs"]
mod tests;
