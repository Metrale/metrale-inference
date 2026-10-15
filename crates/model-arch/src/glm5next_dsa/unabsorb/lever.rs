// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `METRALE_GLM_MLA_UNABSORB`, the unabsorbed-MLA prefill lever, and the DSA site
//! tier table that says, as data, which precision each projection of the unabsorbed prefill
//! follows.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Read once. Unset or `0` is off; `declared` and `dsa` are on; any other value is an error at
//!   load, never a silent default.
//! - Every (mode, DSA site) pair the loader can list has exactly one row in [`SITE_RULES`]; a
//!   pair without a row is refused ([`site_rule`]), so a site added to the layer gets a decision
//!   here before it loads.

use std::sync::OnceLock;

use anyhow::{Result, bail};

use super::super::Glm5NextDsaConfig;

/// 2026-10-15: The lever's name.
pub const ENV: &str = "METRALE_GLM_MLA_UNABSORB";

/// 2026-10-15: How a DSA prefill computes its query and output projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlaUnabsorb {
    /// 2026-10-15: Absorbed, as before the lever: `q_absorb` and `o_absorb` on every row.
    Off,
    /// 2026-10-15: Option B. Prefill runs `q_b_proj`, the per-head `kv_b_proj` halves and
    /// `o_proj`; `q_b_proj` and `o_proj` stay at the checkpoint-declared precision (BF16)
    /// whatever `--dsa-quantization` is. Decode stays absorbed.
    Declared,
    /// 2026-10-15: Option A. As `Declared`, but `q_b_proj` and `o_proj` take
    /// `--dsa-quantization`'s tier like the absorbed sites.
    Dsa,
}

impl MlaUnabsorb {
    /// 2026-10-15: The lever value.
    pub fn name(self) -> &'static str {
        match self {
            MlaUnabsorb::Off => "0",
            MlaUnabsorb::Declared => "declared",
            MlaUnabsorb::Dsa => "dsa",
        }
    }

    pub fn is_on(self) -> bool {
        self != MlaUnabsorb::Off
    }
}

/// 2026-10-15: [`mode`]'s parse, pure.
pub fn parse(v: Option<&str>) -> Result<MlaUnabsorb, String> {
    match v {
        None | Some("0") => Ok(MlaUnabsorb::Off),
        Some("declared") => Ok(MlaUnabsorb::Declared),
        Some("dsa") => Ok(MlaUnabsorb::Dsa),
        Some(s) => Err(format!("{ENV}={s:?}: expected 0, declared or dsa")),
    }
}

/// 2026-10-15: `METRALE_GLM_MLA_UNABSORB`, read once.
pub fn mode() -> Result<MlaUnabsorb> {
    static V: OnceLock<Result<MlaUnabsorb, String>> = OnceLock::new();
    V.get_or_init(|| parse(std::env::var(ENV).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 2026-10-15: The geometry the unabsorbed prefill is built for: NoPE, `qk_nope_head_dim` and
/// `v_head_dim` 256 (`glm5next_mla_head_bmm`'s `_q` is K 256, N 512; `_o` is K 512, N 256), and
/// `kv_lora_rank` 512. Anything else is refused at load.
pub fn check_cfg(cfg: &Glm5NextDsaConfig) -> Result<()> {
    if cfg.qk_rope_head_dim != 0
        || cfg.qk_nope_head_dim != HEAD_DIM
        || cfg.v_head_dim != HEAD_DIM
        || cfg.kv_lora_rank != LATENT
    {
        bail!(
            "{ENV}: the unabsorbed prefill is built for NoPE, qk_nope_head_dim = v_head_dim = \
             {HEAD_DIM} and kv_lora_rank {LATENT}; this checkpoint has rope {}, nope {}, v {}, \
             kv_lora_rank {}",
            cfg.qk_rope_head_dim,
            cfg.qk_nope_head_dim,
            cfg.v_head_dim,
            cfg.kv_lora_rank
        );
    }
    Ok(())
}

/// 2026-10-15: `glm5next_mla_head_bmm`'s per-head projection width (`_q` K, `_o` N).
pub const HEAD_DIM: usize = 256;
/// 2026-10-15: `glm5next_mla_head_bmm`'s latent width (`_q` N, `_o` K).
pub const LATENT: usize = 512;

/// 2026-10-15: The names `Glm5NextDsaLayer::unabsorb_projections` lists the unabsorbed
/// prefill's two dense projections under.
pub const Q_B_SITE: &str = "dsa.q_b_proj";
pub const O_PROJ_SITE: &str = "dsa.o_proj";

/// 2026-10-15: Which precision a DSA site follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteRule {
    /// 2026-10-15: `--dsa-quantization`'s tier.
    DsaKnob,
    /// 2026-10-15: The checkpoint-declared precision (BF16; left unregistered).
    Checkpoint,
}

/// 2026-10-15: The DSA site tier table: for every lever mode, each DSA site the loader lists and
/// the precision it follows. The absorbed sites follow `--dsa-quantization` in every mode; the
/// unabsorbed prefill's `q_b_proj` and `o_proj` exist only with the lever on and follow the
/// checkpoint under `declared` (Option B), the knob under `dsa` (Option A).
pub const SITE_RULES: [(MlaUnabsorb, &str, SiteRule); 13] = [
    (MlaUnabsorb::Off, "dsa.q_a_proj", SiteRule::DsaKnob),
    (MlaUnabsorb::Off, "dsa.q_absorb", SiteRule::DsaKnob),
    (MlaUnabsorb::Off, "dsa.o_absorb", SiteRule::DsaKnob),
    (MlaUnabsorb::Declared, "dsa.q_a_proj", SiteRule::DsaKnob),
    (MlaUnabsorb::Declared, "dsa.q_absorb", SiteRule::DsaKnob),
    (MlaUnabsorb::Declared, "dsa.o_absorb", SiteRule::DsaKnob),
    (MlaUnabsorb::Declared, Q_B_SITE, SiteRule::Checkpoint),
    (MlaUnabsorb::Declared, O_PROJ_SITE, SiteRule::Checkpoint),
    (MlaUnabsorb::Dsa, "dsa.q_a_proj", SiteRule::DsaKnob),
    (MlaUnabsorb::Dsa, "dsa.q_absorb", SiteRule::DsaKnob),
    (MlaUnabsorb::Dsa, "dsa.o_absorb", SiteRule::DsaKnob),
    (MlaUnabsorb::Dsa, Q_B_SITE, SiteRule::DsaKnob),
    (MlaUnabsorb::Dsa, O_PROJ_SITE, SiteRule::DsaKnob),
];

/// 2026-10-15: The rule [`SITE_RULES`] holds for DSA site `name` under `mode`; a pair without a
/// row (an unabsorbed site with the lever off, or an unknown site) is an error.
pub fn site_rule(name: &str, mode: MlaUnabsorb) -> Result<SiteRule> {
    SITE_RULES
        .iter()
        .find(|(m, n, _)| *m == mode && *n == name)
        .map(|r| r.2)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "DSA site {name:?} has no tier rule under {ENV}={}",
                mode.name()
            )
        })
}
