// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: What each layer's projection groups were actually built to decode from, as the
//! arms installed them, and the one load line that states it against the checkpoint's
//! declared precision.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants:
//! - Business logic only: the arms [`ServedFormats::record`] what they built, the W8A8 install
//!   [`ServedFormats::upgrade_w8a8`]s what it installed, and the loader logs
//!   [`ServedFormats::summary`]. Nothing here reads the store, the policy or the environment.
//! - The summary is derived from the recorded builds and nothing else, so it cannot describe a
//!   load path the loader did not take. (It replaced a line computed from the policy before
//!   the layer loop, which said every FP8-declared projection ran as NVFP4 while the GDN
//!   projections had loaded native FP8.)

use std::collections::BTreeMap;
use std::fmt::Write as _;

use metrale_config::Nvfp4Act;
use metrale_config::precision_plan::LayerPrecision;

/// 2026-09-30: A layer's projection group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Group {
    /// 2026-09-30: q/k/v/o of a full-attention layer.
    Attention,
    /// 2026-09-30: in_proj_qkv/in_proj_z/out_proj of a GDN (linear-attention) layer.
    Gdn,
    /// 2026-09-30: gate/up/down of the dense FFN.
    Ffn,
}

impl Group {
    fn name(self) -> &'static str {
        match self {
            Group::Attention => "attention",
            Group::Gdn => "GDN",
            Group::Ffn => "dense FFN",
        }
    }
}

/// 2026-09-30: The weight a group's decode reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Served {
    /// 2026-09-30: The checkpoint's FP8 weight, 16-bit activations (W8A16).
    Fp8,
    /// 2026-09-30: The checkpoint's FP8 weight with FP8 activations (the W8A8 decode install).
    W8a8,
    /// 2026-09-30: An NVFP4 weight: `from_checkpoint` when the checkpoint stores it as NVFP4,
    /// otherwise requantized at load from a wider format. `act` is its activation stamp.
    Nvfp4 {
        from_checkpoint: bool,
        act: Nvfp4Act,
    },
    /// 2026-09-30: 16-bit weights.
    Bf16,
    /// 2026-09-30: The checkpoint's keep-packed 2-bit weights.
    Q2,
}

impl Served {
    /// 2026-09-30: Weight bits, the axis "below declared" is judged on.
    fn weight_bits(self) -> u8 {
        match self {
            Served::Fp8 | Served::W8a8 => 8,
            Served::Nvfp4 { .. } => 4,
            Served::Bf16 => 16,
            Served::Q2 => 2,
        }
    }

    fn label(self) -> String {
        match self {
            Served::Fp8 => "FP8 W8A16".into(),
            Served::W8a8 => "FP8 W8A8".into(),
            Served::Nvfp4 {
                from_checkpoint,
                act,
            } => {
                let src = if from_checkpoint {
                    "from the checkpoint"
                } else {
                    "requantized at load"
                };
                let act = match act {
                    Nvfp4Act::A4 => ", W4A4",
                    Nvfp4Act::Wide => ", W4A16",
                    Nvfp4Act::Unstamped => "",
                };
                format!("NVFP4 {src}{act}")
            }
            Served::Bf16 => "BF16".into(),
            Served::Q2 => "Q2_0 packed".into(),
        }
    }
}

/// 2026-09-30: One group of one layer: what it serves and the weight bits it declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    served: Served,
    declared_bits: u8,
}

/// 2026-09-30: The declared weight bits of `declared`: 16 when unquantized.
pub(super) fn declared_weight_bits(declared: LayerPrecision) -> u8 {
    declared.weight.map_or(16, |w| w.bits)
}

/// 2026-09-30: The served format of every recorded (group, layer).
#[derive(Debug, Default)]
pub(super) struct ServedFormats {
    rows: BTreeMap<(Group, usize), Entry>,
}

impl ServedFormats {
    /// 2026-09-30: Layer `layer`'s `group` was built to decode from `served`; the checkpoint
    /// declares `declared_bits` for its weight. A second record for the same pair replaces
    /// the first.
    pub(super) fn record(&mut self, group: Group, layer: usize, served: Served, declared_bits: u8) {
        self.rows.insert(
            (group, layer),
            Entry {
                served,
                declared_bits,
            },
        );
    }

    /// 2026-09-30: The W8A8 decode install put FP8 W8A8 weights on this group. A group never
    /// recorded is an error in the caller, so it is refused rather than invented.
    pub(super) fn upgrade_w8a8(&mut self, group: Group, layer: usize) -> anyhow::Result<()> {
        let e = self.rows.get_mut(&(group, layer)).ok_or_else(|| {
            anyhow::anyhow!(
                "W8A8 installed on layer {layer} {}, which no arm recorded",
                group.name()
            )
        })?;
        e.served = Served::W8a8;
        Ok(())
    }

    /// 2026-09-30: The load line: per group, how many layers serve each format, then how many
    /// serve a weight narrower than the checkpoint declares.
    pub(super) fn summary(&self, tier: &str) -> String {
        let mut by: BTreeMap<Group, BTreeMap<String, usize>> = BTreeMap::new();
        let mut below: BTreeMap<Group, usize> = BTreeMap::new();
        for (&(g, _), e) in &self.rows {
            *by.entry(g)
                .or_default()
                .entry(e.served.label())
                .or_default() += 1;
            if e.served.weight_bits() < e.declared_bits {
                *below.entry(g).or_default() += 1;
            }
        }
        let mut s = format!("--weight-quantization {tier}: decode weights by group (layers):");
        for (g, forms) in &by {
            let parts: Vec<String> = forms.iter().map(|(f, n)| format!("{n} {f}")).collect();
            let _ = write!(s, " {}: {};", g.name(), parts.join(", "));
        }
        if below.is_empty() {
            s.push_str(" none below the checkpoint's declared weight precision.");
        } else {
            let parts: Vec<String> = below
                .iter()
                .map(|(g, n)| format!("{} {n}", g.name()))
                .collect();
            let _ = write!(
                s,
                " below the checkpoint's declared weight precision: {}.",
                parts.join(", ")
            );
        }
        s
    }
}

#[cfg(test)]
#[path = "served_formats_tests.rs"]
mod served_formats_tests;
