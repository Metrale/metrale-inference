// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: `--activation-quantization`: which activation precision each decode projection
//! runs at, as a function of the projection family and of the number of rows in the launch
//! (decode batch rows, MTP verify rows included).
//!
//! A value is either one format for every row count, or a ladder that maps row ranges to
//! formats, optionally overridden per projection family:
//!
//! ```text
//! spec   := ladder ("," family ":" ladder)*
//! ladder := format | rung (";" rung)*
//! rung   := range "=" format
//! range  := N | N "-" | N "-" M              (1-based rows, inclusive)
//! format := "bf16" | "fp8" | "nvfp4" | "declared" | "adaptive"
//! family := "gdn" | "attn" | "ffn" | "moe" | "lm_head"
//! ```
//!
//! - `bf16`, `fp8` and `nvfp4` run that activation format through one fixed-order kernel family
//!   per weight format, so a row's output bits do not depend on the row count or on the other
//!   rows. `declared` is the checkpoint's declared input-activation format per layer
//!   (`quantization_config`), run the same way.
//! - `adaptive` is today's routing: each dispatch site picks its kernel by row count as it did
//!   before this flag, byte for byte. The named value `adaptive` is the ladder `1-=adaptive`; a
//!   rung may also hand only its range to it (`1=bf16;2-=adaptive`).
//! - A ladder's rungs start at row 1, leave no gap and do not overlap, and the last one is
//!   open-ended. A family override replaces the base ladder for that family only.
//!
//! The same shape, a rule per (family, row range), is what the auto-fuser's circuit rules
//! express per row range (kernels/circuits); a later step makes the circuit read this value.
//!
//! Owner: config (quantization).
//! Invariants:
//! - Pure: no environment, no I/O, no process state. The serve parses the flag here once and
//!   publishes the result; every dispatch site asks [`ActivationQuantization::route`].
//! - Its `Display` form is canonical and parses back to an equal value.

use std::collections::BTreeMap;
use std::fmt;

use anyhow::{Result, bail, ensure};

/// 2026-09-30: The activation format of one rung.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActQuantFormat {
    /// 2026-09-30: 16-bit activations.
    Bf16,
    /// 2026-09-30: FP8 E4M3 activations, one scale per row (and per 128-column group where
    /// the weights are block-scaled).
    Fp8,
    /// 2026-09-30: NVFP4 activations (E2M1, one E4M3 scale per 16 columns of a row).
    Nvfp4,
    /// 2026-09-30: The checkpoint's declared input-activation format of the layer.
    Declared,
    /// 2026-09-30: Today's per-row-count routing.
    Adaptive,
}

impl ActQuantFormat {
    /// 2026-09-30: Every format, in the order the flag lists them.
    pub const ALL: [Self; 5] = [
        Self::Bf16,
        Self::Fp8,
        Self::Nvfp4,
        Self::Declared,
        Self::Adaptive,
    ];

    /// 2026-09-30: The flag value, recipe value and record value.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Nvfp4 => "nvfp4",
            Self::Declared => "declared",
            Self::Adaptive => "adaptive",
        }
    }

    fn parse(s: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|f| f.name() == s)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown activation format {s:?} (one of {})",
                    Self::ALL.map(Self::name).join(", ")
                )
            })
    }
}

/// 2026-09-30: The projection families a family override can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProjFamily {
    /// 2026-09-30: GDN (linear attention) in_proj_qkvz, in_proj_ba and out_proj.
    Gdn,
    /// 2026-09-30: Full attention q (+gate), k, v and o.
    Attn,
    /// 2026-09-30: Dense MLP gate, up and down.
    Ffn,
    /// 2026-09-30: MoE router, routed experts and shared expert.
    Moe,
    /// 2026-09-30: The LM head.
    LmHead,
}

impl ProjFamily {
    /// 2026-09-30: Every family, in the order records list them.
    pub const ALL: [Self; 5] = [Self::Gdn, Self::Attn, Self::Ffn, Self::Moe, Self::LmHead];

    pub fn name(self) -> &'static str {
        match self {
            Self::Gdn => "gdn",
            Self::Attn => "attn",
            Self::Ffn => "ffn",
            Self::Moe => "moe",
            Self::LmHead => "lm_head",
        }
    }

    fn parse(s: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|f| f.name() == s)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown projection family {s:?} (one of {})",
                    Self::ALL.map(Self::name).join(", ")
                )
            })
    }
}

/// 2026-09-30: One rung: rows `lo..=hi` (`hi` `None`: no upper bound) run `format`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rung {
    pub lo: u32,
    pub hi: Option<u32>,
    pub format: ActQuantFormat,
}

/// 2026-09-30: Rungs covering every row count from 1, in order, without gaps or overlaps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ladder(Vec<Rung>);

impl Ladder {
    /// 2026-09-30: One format at every row count.
    pub fn uniform(format: ActQuantFormat) -> Self {
        Self(vec![Rung {
            lo: 1,
            hi: None,
            format,
        }])
    }

    pub fn rungs(&self) -> &[Rung] {
        &self.0
    }

    /// 2026-09-30: The format of `rows` (at least 1).
    pub fn format(&self, rows: u32) -> ActQuantFormat {
        let rows = rows.max(1);
        self.0
            .iter()
            .find(|r| rows >= r.lo && r.hi.is_none_or(|hi| rows <= hi))
            .map(|r| r.format)
            .expect("a validated ladder covers every row count from 1")
    }

    /// 2026-09-30: Whether every rung runs one format (no `adaptive` rung and one rung).
    pub fn is_uniform_invariant(&self) -> bool {
        self.0.len() == 1 && self.0[0].format != ActQuantFormat::Adaptive
    }

    fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        ensure!(!s.is_empty(), "empty ladder");
        if !s.contains('=') {
            return Ok(Self::uniform(ActQuantFormat::parse(s)?));
        }
        let mut rungs = Vec::new();
        for part in s.split(';') {
            let (range, format) = part
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("rung {part:?} is not RANGE=FORMAT"))?;
            let (lo, hi) = parse_range(range.trim())?;
            rungs.push(Rung {
                lo,
                hi,
                format: ActQuantFormat::parse(format.trim())?,
            });
        }
        Self::validated(rungs)
    }

    fn validated(rungs: Vec<Rung>) -> Result<Self> {
        let mut next = 1u32;
        for (i, r) in rungs.iter().enumerate() {
            ensure!(
                r.lo == next,
                "rung {} starts at row {} but must start at row {next} (rungs run from row 1 \
                 in order, without gaps or overlaps)",
                i + 1,
                r.lo
            );
            match r.hi {
                Some(hi) => {
                    ensure!(hi >= r.lo, "rung {} ends at {hi}, before it starts", i + 1);
                    next = hi
                        .checked_add(1)
                        .ok_or_else(|| anyhow::anyhow!("rung {} ends past u32", i + 1))?;
                }
                None => {
                    ensure!(
                        i + 1 == rungs.len(),
                        "rung {} is open-ended but rungs follow it",
                        i + 1
                    );
                    return Ok(Self(merge_equal(rungs)));
                }
            }
        }
        bail!("the last rung must be open-ended (N-), so every row count has a format")
    }
}

/// 2026-09-30: Adjacent rungs of one format merged, so equal ladders compare and print equal.
fn merge_equal(rungs: Vec<Rung>) -> Vec<Rung> {
    let mut out: Vec<Rung> = Vec::with_capacity(rungs.len());
    for r in rungs {
        match out.last_mut() {
            Some(prev) if prev.format == r.format => prev.hi = r.hi,
            _ => out.push(r),
        }
    }
    out
}

fn parse_row(s: &str) -> Result<u32> {
    let n: u32 = s
        .parse()
        .map_err(|_| anyhow::anyhow!("row bound {s:?} is not a whole number"))?;
    ensure!(n >= 1, "rows are counted from 1");
    Ok(n)
}

fn parse_range(s: &str) -> Result<(u32, Option<u32>)> {
    match s.split_once('-') {
        None => {
            let n = parse_row(s)?;
            Ok((n, Some(n)))
        }
        Some((lo, "")) => Ok((parse_row(lo)?, None)),
        Some((lo, hi)) => Ok((parse_row(lo)?, Some(parse_row(hi)?))),
    }
}

impl fmt::Display for Ladder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.len() == 1 {
            return f.write_str(self.0[0].format.name());
        }
        for (i, r) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(";")?;
            }
            match r.hi {
                Some(hi) if hi == r.lo => write!(f, "{}", r.lo)?,
                Some(hi) => write!(f, "{}-{hi}", r.lo)?,
                None => write!(f, "{}-", r.lo)?,
            }
            write!(f, "={}", r.format.name())?;
        }
        Ok(())
    }
}

/// 2026-09-30: A parsed `--activation-quantization` value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivationQuantization {
    base: Ladder,
    families: BTreeMap<ProjFamily, Ladder>,
}

impl Default for ActivationQuantization {
    /// 2026-09-30: The flag's default: the checkpoint's declared formats, row-invariant.
    fn default() -> Self {
        Self::uniform(ActQuantFormat::Declared)
    }
}

impl ActivationQuantization {
    /// 2026-09-30: One format for every family and row count.
    pub fn uniform(format: ActQuantFormat) -> Self {
        Self {
            base: Ladder::uniform(format),
            families: BTreeMap::new(),
        }
    }

    /// 2026-09-30: Today's routing everywhere, as recipes pin it.
    pub fn adaptive() -> Self {
        Self::uniform(ActQuantFormat::Adaptive)
    }

    pub fn parse(s: &str) -> Result<Self> {
        let mut parts = s.split(',');
        let base = Ladder::parse(parts.next().unwrap_or(""))
            .map_err(|e| anyhow::anyhow!("--activation-quantization {s:?}: {e}"))?;
        let mut families = BTreeMap::new();
        for part in parts {
            let (fam, ladder) = part.split_once(':').ok_or_else(|| {
                anyhow::anyhow!(
                    "--activation-quantization {s:?}: override {part:?} is not FAMILY:LADDER"
                )
            })?;
            let fam = ProjFamily::parse(fam.trim())
                .map_err(|e| anyhow::anyhow!("--activation-quantization {s:?}: {e}"))?;
            let ladder = Ladder::parse(ladder).map_err(|e| {
                anyhow::anyhow!("--activation-quantization {s:?}, {}: {e}", fam.name())
            })?;
            ensure!(
                families.insert(fam, ladder).is_none(),
                "--activation-quantization {s:?}: family {} is overridden twice",
                fam.name()
            );
        }
        // 2026-09-30: An override equal to the base says nothing; drop it so the canonical
        // form is unique.
        families.retain(|_, l| *l != base);
        Ok(Self { base, families })
    }

    /// 2026-09-30: The ladder `family` runs.
    pub fn ladder(&self, family: ProjFamily) -> &Ladder {
        self.families.get(&family).unwrap_or(&self.base)
    }

    /// 2026-09-30: The format a `family` projection of `rows` rows runs.
    pub fn route(&self, family: ProjFamily, rows: u32) -> ActQuantFormat {
        self.ladder(family).format(rows)
    }

    /// 2026-09-30: Whether every family runs `adaptive` at every row count (today's routing).
    pub fn is_adaptive(&self) -> bool {
        ProjFamily::ALL.iter().all(|&f| {
            self.ladder(f)
                .rungs()
                .iter()
                .all(|r| r.format == ActQuantFormat::Adaptive)
        })
    }

    /// 2026-09-30: Whether `family` is row-invariant: one non-adaptive format at every row
    /// count.
    pub fn is_invariant(&self, family: ProjFamily) -> bool {
        self.ladder(family).is_uniform_invariant()
    }
}

impl fmt::Display for ActivationQuantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base)?;
        for (fam, l) in &self.families {
            write!(f, ",{}:{l}", fam.name())?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "activation_quantization_tests.rs"]
mod activation_quantization_tests;
